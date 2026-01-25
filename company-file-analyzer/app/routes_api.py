"""
API Routes - REST API for programmatic access.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.db import get_job, get_all_jobs, JobStatus

router = APIRouter()


class JobResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    created_at: str
    result_path: str | None = None
    error: str | None = None


@router.get("/jobs")
async def list_jobs(limit: int = 50) -> list[JobResponse]:
    """List all jobs."""
    jobs = get_all_jobs(limit)
    return [
        JobResponse(
            id=j.id,
            filename=j.filename,
            file_type=j.file_type.value,
            status=j.status.value,
            created_at=j.created_at.isoformat(),
            result_path=j.result_path,
            error=j.error
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job_detail(job_id: str) -> JobResponse:
    """Get job by ID."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        id=job.id,
        filename=job.filename,
        file_type=job.file_type.value,
        status=job.status.value,
        created_at=job.created_at.isoformat(),
        result_path=job.result_path,
        error=job.error
    )


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    """Get job result content."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Job status: {job.status.value}")
    if not job.result_path:
        raise HTTPException(status_code=404, detail="No result available")

    from pathlib import Path
    result_path = Path(job.result_path)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return {
        "job_id": job_id,
        "filename": job.filename,
        "result": result_path.read_text(encoding="utf-8")
    }
