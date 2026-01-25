import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str) -> None:
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY,
      filename TEXT NOT NULL,
      content_type TEXT,
      profile TEXT NOT NULL,
      status TEXT NOT NULL,
      upload_path TEXT NOT NULL,
      result_path TEXT,
      error TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()

def create_job(db_path: str, job_id: str, filename: str, content_type: str, profile: str, upload_path: str) -> None:
    conn = get_conn(db_path)
    cur = conn.cursor()
    now = utc_now()
    cur.execute("""
      INSERT INTO jobs (id, filename, content_type, profile, status, upload_path, created_at, updated_at)
      VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
    """, (job_id, filename, content_type, profile, upload_path, now, now))
    conn.commit()
    conn.close()

def update_job(db_path: str, job_id: str, **fields: Any) -> None:
    allowed = {"status", "result_path", "error", "updated_at"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    fields["updated_at"] = utc_now()

    set_clause = ", ".join([f"{k}=?" for k in fields.keys()])
    values = list(fields.values()) + [job_id]

    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute(f"UPDATE jobs SET {set_clause} WHERE id=?", values)
    conn.commit()
    conn.close()

def get_job(db_path: str, job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def list_jobs(db_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
