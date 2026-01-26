"""
Background tasks for document analysis and clause extraction.

Job types:
  analyze_document  - extract text + AI clause extraction for single document
  analyze_batch     - process all uploaded documents
"""
import logging

from app.config import settings
from app import db
from app.services.extract_text import extract_text
from app.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def analyze_document(job_id: str, doc_id: str) -> None:
    """
    Analyze a single document:
    1. Extract text (PDF/DOCX/TXT)
    2. Send to AI for clause extraction
    3. Store clauses in DB (deduplicate by similarity)
    """
    db.update_job(settings.db_path, job_id, status="running")

    try:
        doc = db.get_document(settings.db_path, doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")

        # Step 1: Extract text
        logger.info(f"Extracting text from: {doc['filename']}")
        db.update_document(settings.db_path, doc_id, status="extracting")

        text = extract_text(doc["upload_path"])
        if not text.strip():
            db.update_document(settings.db_path, doc_id, status="empty")
            db.update_job(settings.db_path, job_id, status="done",
                          detail=f"Document {doc['filename']} is empty")
            return

        db.update_document(settings.db_path, doc_id, extracted_text=text)

        # Step 2: AI clause extraction
        logger.info(f"Extracting clauses from: {doc['filename']}")
        db.update_document(settings.db_path, doc_id, status="analyzing")

        client = OllamaClient(settings.ollama_host, settings.ollama_model)
        result = client.extract_clauses(text)

        # Step 3: Store clauses (with deduplication)
        clause_count = 0

        for scope_item in result.get("scope", []):
            if not scope_item.strip():
                continue
            existing = db.find_similar_clause(settings.db_path, "scope", scope_item)
            if existing:
                db.increment_clause_frequency(settings.db_path, existing["id"])
            else:
                db.add_clause(settings.db_path, "scope", scope_item.strip(), source_doc_id=doc_id)
                clause_count += 1

        for excl in result.get("exclusions", []):
            if not excl.strip():
                continue
            is_critical = 1 if excl in result.get("critical_exclusions", []) else 0
            existing = db.find_similar_clause(settings.db_path, "exclusion", excl)
            if existing:
                db.increment_clause_frequency(settings.db_path, existing["id"])
                # Promote to critical if AI thinks so
                if is_critical and not existing["is_critical"]:
                    db.update_clause(settings.db_path, existing["id"], is_critical=1)
            else:
                db.add_clause(settings.db_path, "exclusion", excl.strip(),
                              source_doc_id=doc_id, is_critical=is_critical)
                clause_count += 1

        # Handle critical exclusions that weren't in the general exclusions list
        for crit in result.get("critical_exclusions", []):
            if not crit.strip():
                continue
            if crit not in result.get("exclusions", []):
                existing = db.find_similar_clause(settings.db_path, "exclusion", crit)
                if existing:
                    db.increment_clause_frequency(settings.db_path, existing["id"])
                    if not existing["is_critical"]:
                        db.update_clause(settings.db_path, existing["id"], is_critical=1)
                else:
                    db.add_clause(settings.db_path, "exclusion", crit.strip(),
                                  source_doc_id=doc_id, is_critical=1)
                    clause_count += 1

        db.update_document(settings.db_path, doc_id, status="done", clause_count=clause_count)
        db.update_job(settings.db_path, job_id, status="done",
                      detail=f"Extracted {clause_count} new clauses from {doc['filename']}")

        logger.info(f"Done: {doc['filename']} → {clause_count} new clauses")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        db.update_job(settings.db_path, job_id, status="failed", error=str(e))
        db.update_document(settings.db_path, doc_id, status="failed")


def analyze_batch(job_id: str) -> None:
    """
    Analyze all uploaded documents that haven't been processed yet.
    """
    db.update_job(settings.db_path, job_id, status="running")

    try:
        docs = db.list_documents(settings.db_path)
        pending = [d for d in docs if d["status"] in ("uploaded", "failed")]

        if not pending:
            db.update_job(settings.db_path, job_id, status="done",
                          detail="No documents to process")
            return

        client = OllamaClient(settings.ollama_host, settings.ollama_model)
        processed = 0
        errors = 0

        for doc in pending:
            try:
                logger.info(f"Batch: processing {doc['filename']} ({processed+1}/{len(pending)})")

                text = extract_text(doc["upload_path"])
                if not text.strip():
                    db.update_document(settings.db_path, doc["id"], status="empty")
                    continue

                db.update_document(settings.db_path, doc["id"],
                                   extracted_text=text, status="analyzing")

                result = client.extract_clauses(text)
                clause_count = _store_clauses(doc["id"], result)

                db.update_document(settings.db_path, doc["id"],
                                   status="done", clause_count=clause_count)
                processed += 1

            except Exception as e:
                logger.error(f"Batch error on {doc['filename']}: {e}")
                db.update_document(settings.db_path, doc["id"], status="failed")
                errors += 1

        detail = f"Processed {processed}/{len(pending)} documents"
        if errors:
            detail += f" ({errors} errors)"

        db.update_job(settings.db_path, job_id, status="done", detail=detail)

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}", exc_info=True)
        db.update_job(settings.db_path, job_id, status="failed", error=str(e))


def _store_clauses(doc_id: str, result: dict) -> int:
    """Store extracted clauses with deduplication. Returns count of NEW clauses."""
    count = 0

    for scope_item in result.get("scope", []):
        if not scope_item.strip():
            continue
        existing = db.find_similar_clause(settings.db_path, "scope", scope_item)
        if existing:
            db.increment_clause_frequency(settings.db_path, existing["id"])
        else:
            db.add_clause(settings.db_path, "scope", scope_item.strip(), source_doc_id=doc_id)
            count += 1

    critical_set = set(result.get("critical_exclusions", []))

    for excl in result.get("exclusions", []):
        if not excl.strip():
            continue
        is_crit = 1 if excl in critical_set else 0
        existing = db.find_similar_clause(settings.db_path, "exclusion", excl)
        if existing:
            db.increment_clause_frequency(settings.db_path, existing["id"])
            if is_crit and not existing["is_critical"]:
                db.update_clause(settings.db_path, existing["id"], is_critical=1)
        else:
            db.add_clause(settings.db_path, "exclusion", excl.strip(),
                          source_doc_id=doc_id, is_critical=is_crit)
            count += 1

    # Critical exclusions not in general list
    for crit in critical_set:
        if crit not in result.get("exclusions", []) and crit.strip():
            existing = db.find_similar_clause(settings.db_path, "exclusion", crit)
            if existing:
                db.increment_clause_frequency(settings.db_path, existing["id"])
                if not existing["is_critical"]:
                    db.update_clause(settings.db_path, existing["id"], is_critical=1)
            else:
                db.add_clause(settings.db_path, "exclusion", crit.strip(),
                              source_doc_id=doc_id, is_critical=1)
                count += 1

    return count
