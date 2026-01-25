"""
Background task definitions.
"""
import logging
from pathlib import Path

from app.db import get_job, update_job_status, JobStatus, JobType
from app.services.storage import get_uploaded_file, save_result
from app.services.transcribe_audio import transcribe_audio, format_transcription
from app.services.extract_text import extract_text
from app.services.ollama_client import summarize_text

logger = logging.getLogger(__name__)


def process_job(job_id: str) -> None:
    """
    Process a job based on its type.

    Pipeline:
    1. Audio → transcribe → summarize
    2. Document → extract text → summarize
    3. Image → OCR → summarize

    Args:
        job_id: The job ID to process
    """
    logger.info(f"Processing job: {job_id}")

    job = get_job(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return

    # Update status to running
    update_job_status(job_id, JobStatus.RUNNING)

    try:
        # Get uploaded file
        file_path = get_uploaded_file(job_id)
        if not file_path:
            raise FileNotFoundError(f"No uploaded file for job {job_id}")

        logger.info(f"Processing file: {file_path.name} (type: {job.file_type})")

        # Process based on type
        if job.file_type == JobType.AUDIO:
            result = process_audio(file_path)
        elif job.file_type == JobType.DOCUMENT:
            result = process_document(file_path)
        elif job.file_type == JobType.IMAGE:
            result = process_image(file_path)
        else:
            raise ValueError(f"Unknown job type: {job.file_type}")

        # Save result
        result_path = save_result(job_id, result)
        update_job_status(job_id, JobStatus.DONE, result_path=str(result_path))

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED, error=str(e))


def process_audio(file_path: Path) -> str:
    """
    Process audio file: transcribe + summarize.

    Args:
        file_path: Path to audio file

    Returns:
        Formatted result with transcription and summary
    """
    logger.info("Transcribing audio...")
    transcription = transcribe_audio(file_path)
    transcription = format_transcription(transcription)

    logger.info("Generating summary...")
    summary = summarize_text(transcription)

    return f"""# Transkrypcja: {file_path.name}

## Podsumowanie
{summary}

---

## Pełna transkrypcja
{transcription}
"""


def process_document(file_path: Path) -> str:
    """
    Process document: extract text + summarize.

    Args:
        file_path: Path to document

    Returns:
        Formatted result with extracted text and summary
    """
    logger.info("Extracting text from document...")
    text = extract_text(file_path)

    if not text.strip():
        return f"""# Dokument: {file_path.name}

Nie udało się wyodrębnić tekstu z dokumentu.
Plik może być pusty lub zawierać tylko obrazy (wymagane OCR).
"""

    logger.info("Generating summary...")
    summary = summarize_text(text)

    # Truncate full text if very long
    text_preview = text[:5000] + "\n\n[... tekst skrócony ...]" if len(text) > 5000 else text

    return f"""# Dokument: {file_path.name}

## Podsumowanie
{summary}

---

## Wyodrębniony tekst
{text_preview}
"""


def process_image(file_path: Path) -> str:
    """
    Process image: OCR + summarize.

    Args:
        file_path: Path to image

    Returns:
        Formatted result with OCR text and summary
    """
    logger.info("Extracting text from image (OCR)...")
    text = extract_text(file_path)

    if not text.strip():
        return f"""# Obraz: {file_path.name}

Nie wykryto tekstu na obrazie.
"""

    logger.info("Generating summary...")
    summary = summarize_text(text)

    return f"""# Obraz: {file_path.name}

## Podsumowanie
{summary}

---

## Rozpoznany tekst (OCR)
{text}
"""
