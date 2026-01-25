"""
Company File Analyzer - FastAPI Application
MVP: Upload files, process in background, view results
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes_ui import router as ui_router
from app.routes_api import router as api_router
from app.db import init_db

app = FastAPI(
    title="Company File Analyzer",
    description="Upload and analyze company files (audio, PDF, documents, images)",
    version="0.1.0"
)

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Routers
app.include_router(ui_router, tags=["UI"])
app.include_router(api_router, prefix="/api", tags=["API"])


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    init_db()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
