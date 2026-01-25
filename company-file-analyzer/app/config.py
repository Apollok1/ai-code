"""
Configuration from environment variables.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/jobs.db")

# Redis (for RQ queue)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# Whisper (for audio transcription)
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:9000")

# File limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

# Allowed file types
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
ALLOWED_DOCS = {".pdf", ".docx", ".doc", ".txt", ".rtf"}
ALLOWED_IMAGES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO | ALLOWED_DOCS | ALLOWED_IMAGES
