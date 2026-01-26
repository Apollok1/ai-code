"""
UI Routes - HTML pages for Doc Analyzer.

Pages:
  /                  Dashboard (stats)
  /archive/upload    Upload historical documents
  /archive           List of analyzed documents
  /clauses           Clause library (scope + exclusions)
  /clauses/{id}/toggle-critical   Toggle critical flag
  /offers/new        New offer configurator
  /offers            List of offers
  /offers/{id}       Offer detail + safety check
"""
import uuid

from fastapi import APIRouter, Request, UploadFile, Form, File
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app import db
from app.services.storage import save_upload
from app.workers.queue import get_queue
from app.workers.tasks import analyze_document, analyze_batch

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# ── Dashboard ──────────────────────────────────────────────

@router.get("/")
def dashboard(request: Request):
    docs = db.list_documents(settings.db_path)
    clauses = db.list_clauses(settings.db_path)
    offers = db.list_offers(settings.db_path)
    jobs = db.list_jobs(settings.db_path, limit=10)

    scope_count = len([c for c in clauses if c["category"] == "scope"])
    excl_count = len([c for c in clauses if c["category"] == "exclusion"])
    crit_count = len([c for c in clauses if c["is_critical"]])

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": settings.app_title,
        "doc_count": len(docs),
        "scope_count": scope_count,
        "excl_count": excl_count,
        "crit_count": crit_count,
        "offer_count": len(offers),
        "jobs": jobs,
    })


# ── Archive (upload & list documents) ─────────────────────

@router.get("/archive/upload")
def archive_upload_page(request: Request):
    return templates.TemplateResponse("archive_upload.html", {
        "request": request,
        "title": settings.app_title,
    })


@router.post("/archive/upload")
async def archive_upload(request: Request, files: list[UploadFile] = File(...)):
    """Upload one or more historical offer documents."""
    doc_ids = []
    for file in files:
        doc_id, upload_path = await save_upload(settings.data_dir, file)
        db.create_document(settings.db_path, doc_id, file.filename or "upload.bin", upload_path)
        doc_ids.append(doc_id)

        # Create analysis job for each document
        job_id = str(uuid.uuid4())[:8]
        db.create_job(settings.db_path, job_id, "analyze_document", detail=file.filename)
        q = get_queue()
        q.enqueue(analyze_document, job_id, doc_id, job_timeout="30m")

    return RedirectResponse(url="/archive", status_code=303)


@router.post("/archive/analyze-all")
def analyze_all(request: Request):
    """Re-analyze all unprocessed documents."""
    job_id = str(uuid.uuid4())[:8]
    db.create_job(settings.db_path, job_id, "analyze_batch")
    q = get_queue()
    q.enqueue(analyze_batch, job_id, job_timeout="2h")
    return RedirectResponse(url="/archive", status_code=303)


@router.get("/archive")
def archive_list(request: Request):
    docs = db.list_documents(settings.db_path)
    return templates.TemplateResponse("archive.html", {
        "request": request,
        "title": settings.app_title,
        "docs": docs,
    })


# ── Clauses (library) ─────────────────────────────────────

@router.get("/clauses")
def clauses_page(request: Request, category: str = None):
    clauses = db.list_clauses(settings.db_path, category=category)
    return templates.TemplateResponse("clauses.html", {
        "request": request,
        "title": settings.app_title,
        "clauses": clauses,
        "filter_category": category,
    })


@router.post("/clauses/{clause_id}/toggle-critical")
def toggle_critical(clause_id: int):
    """Toggle the critical flag on a clause."""
    clauses = db.list_clauses(settings.db_path)
    for c in clauses:
        if c["id"] == clause_id:
            db.update_clause(settings.db_path, clause_id,
                             is_critical=0 if c["is_critical"] else 1)
            break
    return RedirectResponse(url="/clauses", status_code=303)


@router.post("/clauses/{clause_id}/delete")
def delete_clause(clause_id: int):
    db.delete_clause(settings.db_path, clause_id)
    return RedirectResponse(url="/clauses", status_code=303)


@router.post("/clauses/add")
def add_clause_manual(category: str = Form(...), text: str = Form(...), is_critical: int = Form(0)):
    """Manually add a clause."""
    db.add_clause(settings.db_path, category, text.strip(), is_critical=is_critical)
    return RedirectResponse(url="/clauses", status_code=303)


# ── Offers (configurator) ─────────────────────────────────

@router.get("/offers")
def offers_list(request: Request):
    offers = db.list_offers(settings.db_path)
    return templates.TemplateResponse("offers.html", {
        "request": request,
        "title": settings.app_title,
        "offers": offers,
    })


@router.get("/offers/new")
def new_offer_page(request: Request):
    scope_clauses = db.list_clauses(settings.db_path, category="scope")
    excl_clauses = db.list_clauses(settings.db_path, category="exclusion")
    return templates.TemplateResponse("offer_new.html", {
        "request": request,
        "title": settings.app_title,
        "scope_clauses": scope_clauses,
        "excl_clauses": excl_clauses,
    })


@router.post("/offers/new")
def create_offer(
    request: Request,
    name: str = Form(...),
    client: str = Form(""),
    notes: str = Form(""),
    clause_ids: list[int] = Form(default=[]),
):
    offer_id = str(uuid.uuid4())[:8]
    db.create_offer(settings.db_path, offer_id, name, client, notes)

    for cid in clause_ids:
        db.add_clause_to_offer(settings.db_path, offer_id, cid)

    # Run safety check
    _run_safety_check(offer_id)

    return RedirectResponse(url=f"/offers/{offer_id}", status_code=303)


@router.get("/offers/{offer_id}")
def offer_detail(request: Request, offer_id: str):
    offer = db.get_offer(settings.db_path, offer_id)
    if not offer:
        return templates.TemplateResponse("offer_detail.html", {
            "request": request,
            "title": settings.app_title,
            "offer": None,
        })

    offer_clauses = db.get_offer_clauses(settings.db_path, offer_id)
    scope = [c for c in offer_clauses if c["category"] == "scope"]
    exclusions = [c for c in offer_clauses if c["category"] == "exclusion"]

    # Parse missing_critical
    missing = offer.get("missing_critical", "") or ""
    missing_list = [m.strip() for m in missing.split("\n") if m.strip()]

    return templates.TemplateResponse("offer_detail.html", {
        "request": request,
        "title": settings.app_title,
        "offer": offer,
        "scope": scope,
        "exclusions": exclusions,
        "missing_list": missing_list,
    })


@router.post("/offers/{offer_id}/recheck")
def recheck_safety(offer_id: str):
    """Re-run safety check with AI."""
    _run_safety_check(offer_id)
    return RedirectResponse(url=f"/offers/{offer_id}", status_code=303)


def _run_safety_check(offer_id: str) -> None:
    """Run safety check: compare selected clauses against critical exclusions."""
    offer_clauses = db.get_offer_clauses(settings.db_path, offer_id)
    selected_texts = [c["text"] for c in offer_clauses]

    critical = db.get_critical_clauses(settings.db_path)
    critical_texts = [c["text"] for c in critical]

    if not critical_texts:
        db.update_offer_safety(settings.db_path, offer_id, safety_ok=True, missing=[])
        return

    # Simple check: which critical clauses are NOT in the selected set
    selected_lower = {t.lower().strip() for t in selected_texts}
    missing = []
    for crit in critical_texts:
        crit_lower = crit.lower().strip()
        # Check if any selected clause is similar enough
        found = False
        for sel in selected_lower:
            words_crit = set(crit_lower.split())
            words_sel = set(sel.split())
            if words_crit and words_sel:
                overlap = len(words_crit & words_sel) / max(len(words_crit), len(words_sel))
                if overlap >= 0.7:
                    found = True
                    break
        if not found:
            missing.append(crit)

    db.update_offer_safety(
        settings.db_path,
        offer_id,
        safety_ok=len(missing) == 0,
        missing=missing,
    )
