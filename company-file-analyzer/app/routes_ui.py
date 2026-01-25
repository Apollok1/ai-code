from fastapi import APIRouter, Request, UploadFile, Form, File
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app import db
from app.services.storage import save_upload
from app.workers.queue import get_queue
from app.workers.tasks import process_job

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/")
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": settings.app_title
    })

@router.post("/upload")
async def upload_file(
    request: Request,
    profile: str = Form(...),
    file: UploadFile = File(...)
):
    job_id, upload_path = await save_upload(settings.data_dir, file)
    db.create_job(
        settings.db_path,
        job_id=job_id,
        filename=file.filename or "upload.bin",
        content_type=file.content_type or "",
        profile=profile,
        upload_path=upload_path
    )

    q = get_queue()
    q.enqueue(process_job, job_id)

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

@router.get("/jobs")
def jobs_page(request: Request):
    jobs = db.list_jobs(settings.db_path, limit=200)
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "title": settings.app_title,
        "jobs": jobs
    })

@router.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: str):
    job = db.get_job(settings.db_path, job_id)
    return templates.TemplateResponse("job_detail.html", {
        "request": request,
        "title": settings.app_title,
        "job": job
    })
