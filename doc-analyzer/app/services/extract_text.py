from pathlib import Path
from pypdf import PdfReader
import docx


def extract_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in [".txt", ".md", ".log"]:
        return p.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

    if ext == ".docx":
        d = docx.Document(path)
        return "\n".join([para.text for para in d.paragraphs]).strip()

    raise ValueError(f"Nieobsługiwany typ pliku: {ext}")
