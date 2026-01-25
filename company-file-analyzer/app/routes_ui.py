"""
UI Routes - HTML pages for file upload and job management.
"""
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import UPLOADS_DIR, RESULTS_DIR, ALLOWED_EXTENSIONS, ALLOWED_AUDIO, ALLOWED_DOCS, ALLOWED_IMAGES
from app.db import create_job, get_job, get_all_jobs, JobType
from app.workers.queue import enqueue_job

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def get_file_type(filename: str) -> JobType | None:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_AUDIO:
        return JobType.AUDIO
    elif ext in ALLOWED_DOCS:
        return JobType.DOCUMENT
    elif ext in ALLOWED_IMAGES:
        return JobType.IMAGE
    return None


@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Main upload page."""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "allowed_extensions": ", ".join(sorted(ALLOWED_EXTENSIONS))
    })


@router.post("/upload")
async def handle_upload(
    request: Request,
    file: UploadFile = File(...),
    user_email: str = Form(None)
):
    """Handle file upload."""
    # Validate file type
    file_type = get_file_type(file.filename)
    if not file_type:
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "error": f"Niedozwolony typ pliku. Dozwolone: {', '.join(ALLOWED_EXTENSIONS)}",
            "allowed_extensions": ", ".join(sorted(ALLOWED_EXTENSIONS))
        })

    # Create job
    job_id = create_job(file.filename, file_type, user_email)

    # Save file
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Enqueue for processing
    enqueue_job(job_id)

    # Redirect to job detail
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_list(request: Request):
    """List all jobs."""
    jobs = get_all_jobs()
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "jobs": jobs
    })


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    """Show job details and result."""
    job = get_job(job_id)
    if not job:
        return templates.TemplateResponse("job_detail.html", {
            "request": request,
            "error": "Zadanie nie znalezione"
        })

    # Load result if available
    result_text = None
    if job.result_path:
        result_path = Path(job.result_path)
        if result_path.exists():
            result_text = result_path.read_text(encoding="utf-8")

    return templates.TemplateResponse("job_detail.html", {
        "request": request,
        "job": job,
        "result": result_text
    })
