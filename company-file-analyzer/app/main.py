from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app import db
from app.routes_ui import router as ui_router

def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_title)
    db.init_db(settings.db_path)

    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(ui_router)

    return app

app = create_app()
