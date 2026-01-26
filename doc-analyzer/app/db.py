"""
Database: SQLite schema for document analysis and offer configuration.

Tables:
  documents     - uploaded source documents (historical offers)
  clauses       - extracted clauses (scope / exclusion)
  offers        - offers created by constructors
  offer_clauses - M2M: clauses selected for an offer
  jobs          - background analysis jobs
"""
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
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    conn = get_conn(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        upload_path TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'uploaded',
        extracted_text TEXT,
        clause_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS clauses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        text TEXT NOT NULL,
        source_doc_id TEXT,
        frequency INTEGER DEFAULT 1,
        is_critical INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (source_doc_id) REFERENCES documents(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS offers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        client TEXT,
        notes TEXT,
        safety_ok INTEGER DEFAULT 0,
        missing_critical TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS offer_clauses (
        offer_id TEXT NOT NULL,
        clause_id INTEGER NOT NULL,
        PRIMARY KEY (offer_id, clause_id),
        FOREIGN KEY (offer_id) REFERENCES offers(id),
        FOREIGN KEY (clause_id) REFERENCES clauses(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        detail TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# ── Documents ──────────────────────────────────────────────

def create_document(db_path: str, doc_id: str, filename: str, upload_path: str) -> None:
    conn = get_conn(db_path)
    now = utc_now()
    conn.execute(
        "INSERT INTO documents (id, filename, upload_path, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (doc_id, filename, upload_path, "uploaded", now, now),
    )
    conn.commit()
    conn.close()


def get_document(db_path: str, doc_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_documents(db_path: str) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_document(db_path: str, doc_id: str, **fields: Any) -> None:
    allowed = {"status", "extracted_text", "clause_count", "updated_at"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    fields["updated_at"] = utc_now()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [doc_id]
    conn = get_conn(db_path)
    conn.execute(f"UPDATE documents SET {set_clause} WHERE id=?", vals)
    conn.commit()
    conn.close()


# ── Clauses ────────────────────────────────────────────────

def add_clause(db_path: str, category: str, text: str, source_doc_id: str = None, is_critical: int = 0) -> int:
    conn = get_conn(db_path)
    now = utc_now()
    cur = conn.execute(
        "INSERT INTO clauses (category, text, source_doc_id, is_critical, created_at) VALUES (?,?,?,?,?)",
        (category, text, source_doc_id, is_critical, now),
    )
    clause_id = cur.lastrowid
    conn.commit()
    conn.close()
    return clause_id


def find_similar_clause(db_path: str, category: str, text: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
    """Find existing clause with similar text (simple substring match for MVP)."""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM clauses WHERE category=?", (category,)
    ).fetchall()
    conn.close()

    text_lower = text.lower().strip()
    for row in rows:
        existing = row["text"].lower().strip()
        # Simple similarity: if one contains the other
        if text_lower in existing or existing in text_lower:
            return dict(row)
        # Or if they share >85% of words
        words_new = set(text_lower.split())
        words_old = set(existing.split())
        if words_new and words_old:
            overlap = len(words_new & words_old) / max(len(words_new), len(words_old))
            if overlap >= threshold:
                return dict(row)
    return None


def increment_clause_frequency(db_path: str, clause_id: int) -> None:
    conn = get_conn(db_path)
    conn.execute("UPDATE clauses SET frequency = frequency + 1 WHERE id=?", (clause_id,))
    conn.commit()
    conn.close()


def list_clauses(db_path: str, category: str = None) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    if category:
        rows = conn.execute(
            "SELECT * FROM clauses WHERE category=? ORDER BY frequency DESC, id", (category,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM clauses ORDER BY category, frequency DESC, id"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_critical_clauses(db_path: str) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM clauses WHERE is_critical=1 ORDER BY category, frequency DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_clause(db_path: str, clause_id: int, **fields: Any) -> None:
    allowed = {"text", "category", "is_critical", "frequency"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [clause_id]
    conn = get_conn(db_path)
    conn.execute(f"UPDATE clauses SET {set_clause} WHERE id=?", vals)
    conn.commit()
    conn.close()


def delete_clause(db_path: str, clause_id: int) -> None:
    conn = get_conn(db_path)
    conn.execute("DELETE FROM offer_clauses WHERE clause_id=?", (clause_id,))
    conn.execute("DELETE FROM clauses WHERE id=?", (clause_id,))
    conn.commit()
    conn.close()


# ── Offers ─────────────────────────────────────────────────

def create_offer(db_path: str, offer_id: str, name: str, client: str = "", notes: str = "") -> None:
    conn = get_conn(db_path)
    now = utc_now()
    conn.execute(
        "INSERT INTO offers (id, name, client, notes, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (offer_id, name, client, notes, now, now),
    )
    conn.commit()
    conn.close()


def get_offer(db_path: str, offer_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM offers WHERE id=?", (offer_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_offers(db_path: str) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM offers ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_clause_to_offer(db_path: str, offer_id: str, clause_id: int) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO offer_clauses (offer_id, clause_id) VALUES (?,?)",
        (offer_id, clause_id),
    )
    conn.commit()
    conn.close()


def remove_clause_from_offer(db_path: str, offer_id: str, clause_id: int) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "DELETE FROM offer_clauses WHERE offer_id=? AND clause_id=?",
        (offer_id, clause_id),
    )
    conn.commit()
    conn.close()


def get_offer_clauses(db_path: str, offer_id: str) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute("""
        SELECT c.* FROM clauses c
        JOIN offer_clauses oc ON oc.clause_id = c.id
        WHERE oc.offer_id=?
        ORDER BY c.category, c.id
    """, (offer_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_offer_safety(db_path: str, offer_id: str, safety_ok: bool, missing: List[str]) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE offers SET safety_ok=?, missing_critical=?, updated_at=? WHERE id=?",
        (1 if safety_ok else 0, "\n".join(missing) if missing else None, utc_now(), offer_id),
    )
    conn.commit()
    conn.close()


# ── Jobs ───────────────────────────────────────────────────

def create_job(db_path: str, job_id: str, job_type: str, detail: str = "") -> None:
    conn = get_conn(db_path)
    now = utc_now()
    conn.execute(
        "INSERT INTO jobs (id, job_type, status, detail, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (job_id, job_type, "queued", detail, now, now),
    )
    conn.commit()
    conn.close()


def update_job(db_path: str, job_id: str, **fields: Any) -> None:
    allowed = {"status", "error", "detail", "updated_at"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    fields["updated_at"] = utc_now()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [job_id]
    conn = get_conn(db_path)
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE id=?", vals)
    conn.commit()
    conn.close()


def get_job(db_path: str, job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_jobs(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
