from pathlib import Path
import uuid
from fastapi import UploadFile
import aiofiles


def ensure_dirs(base: str) -> None:
    Path(base).mkdir(parents=True, exist_ok=True)
    (Path(base) / "uploads").mkdir(parents=True, exist_ok=True)
    (Path(base) / "results").mkdir(parents=True, exist_ok=True)


async def save_upload(base_dir: str, file: UploadFile) -> tuple[str, str]:
    """
    Zwraca: (doc_id, upload_path)
    """
    ensure_dirs(base_dir)
    doc_id = str(uuid.uuid4())[:8]
    safe_name = file.filename or "upload.bin"
    upload_dir = Path(base_dir) / "uploads" / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / safe_name

    async with aiofiles.open(upload_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)

    return doc_id, str(upload_path)


def save_result_text(base_dir: str, job_id: str, text: str) -> str:
    ensure_dirs(base_dir)
    path = Path(base_dir) / "results" / f"{job_id}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)
