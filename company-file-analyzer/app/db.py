"""
Database models and operations (SQLite for MVP).
"""
import sqlite3
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.config import DATA_DIR


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobType(str, Enum):
    AUDIO = "audio"
    DOCUMENT = "document"
    IMAGE = "image"


@dataclass
class Job:
    id: str
    filename: str
    file_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result_path: Optional[str] = None
    error: Optional[str] = None
    user_email: Optional[str] = None


DB_PATH = DATA_DIR / "jobs.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result_path TEXT,
                error TEXT,
                user_email TEXT
            )
        """)
        conn.commit()


def create_job(filename: str, file_type: JobType, user_email: Optional[str] = None) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())[:8]
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, filename, file_type, status, user_email)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, filename, file_type.value, JobStatus.QUEUED.value, user_email)
        )
        conn.commit()
    return job_id


def get_job(job_id: str) -> Optional[Job]:
    """Get job by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row:
            return Job(
                id=row["id"],
                filename=row["filename"],
                file_type=JobType(row["file_type"]),
                status=JobStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                result_path=row["result_path"],
                error=row["error"],
                user_email=row["user_email"]
            )
    return None


def get_all_jobs(limit: int = 50) -> list[Job]:
    """Get all jobs, most recent first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Job(
                id=row["id"],
                filename=row["filename"],
                file_type=JobType(row["file_type"]),
                status=JobStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                result_path=row["result_path"],
                error=row["error"],
                user_email=row["user_email"]
            )
            for row in rows
        ]


def update_job_status(job_id: str, status: JobStatus, result_path: Optional[str] = None, error: Optional[str] = None):
    """Update job status."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, result_path = ?, error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status.value, result_path, error, job_id)
        )
        conn.commit()
