"""
File storage operations.
"""
from pathlib import Path

from app.config import UPLOADS_DIR, RESULTS_DIR


def get_upload_path(job_id: str) -> Path:
    """Get upload directory for a job."""
    return UPLOADS_DIR / job_id


def get_uploaded_file(job_id: str) -> Path | None:
    """Get the uploaded file path for a job."""
    job_dir = get_upload_path(job_id)
    if not job_dir.exists():
        return None
    files = list(job_dir.iterdir())
    return files[0] if files else None


def save_result(job_id: str, content: str) -> Path:
    """Save job result to file."""
    result_path = RESULTS_DIR / f"{job_id}.txt"
    result_path.write_text(content, encoding="utf-8")
    return result_path


def get_result_path(job_id: str) -> Path:
    """Get result file path."""
    return RESULTS_DIR / f"{job_id}.txt"
