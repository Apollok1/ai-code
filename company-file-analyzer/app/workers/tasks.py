from pathlib import Path
from app.config import settings
from app import db
from app.services.extract_text import extract_text
from app.services.ollama_client import OllamaClient
from app.services.storage import save_result_text

def process_job(job_id: str) -> None:
    job = db.get_job(settings.db_path, job_id)
    if not job:
        return

    db.update_job(settings.db_path, job_id, status="running", error=None)

    try:
        upload_path = job["upload_path"]
        profile = job["profile"]

        # MVP profiles
        if profile == "document_summary":
            text = extract_text(upload_path)
            client = OllamaClient(settings.ollama_host, settings.ollama_model)
            summary = client.summarize_pl(text)
            result_path = save_result_text(settings.data_dir, job_id, summary)
            db.update_job(settings.db_path, job_id, status="done", result_path=result_path)

        elif profile == "audio_transcribe_summary":
            # MVP: placeholder (audio pipeline dodamy w etapie 2)
            msg = (
                "Profil audio jest w MVP jeszcze nieaktywny.\n"
                "Następny krok: dodamy serwis Whisper (docker) lub bibliotekę faster-whisper.\n"
            )
            result_path = save_result_text(settings.data_dir, job_id, msg)
            db.update_job(settings.db_path, job_id, status="done", result_path=result_path)

        else:
            raise ValueError(f"Nieznany profil: {profile}")

    except Exception as e:
        db.update_job(settings.db_path, job_id, status="failed", error=str(e))
