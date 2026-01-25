# Company File Analyzer

Firmowa aplikacja do analizy plików (audio, PDF, dokumenty, obrazy).

## Architektura MVP

```
User → FastAPI (upload) → Redis Queue → Worker → Ollama/Whisper → Result
                ↓
          SQLite (jobs)
```

## Quick Start

### 1. Uruchom serwisy (Redis, Ollama)

```bash
docker compose up -d
```

### 2. Zainstaluj zależności Python

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Pobierz model Ollama

```bash
docker exec -it cfa-ollama ollama pull llama3:latest
```

### 4. Uruchom aplikację

**Terminal 1 - API:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 - Worker:**
```bash
rq worker
```

### 5. Otwórz przeglądarkę

http://localhost:8000

## Struktura projektu

```
company-file-analyzer/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration (env vars)
│   ├── db.py                # SQLite models
│   ├── routes_ui.py         # HTML pages
│   ├── routes_api.py        # REST API
│   ├── services/
│   │   ├── storage.py       # File storage
│   │   ├── ollama_client.py # LLM calls
│   │   ├── extract_text.py  # PDF/DOC/IMG → text
│   │   └── transcribe_audio.py  # Audio → text
│   └── workers/
│       ├── queue.py         # RQ config
│       └── tasks.py         # Background jobs
├── templates/               # HTML (Jinja2)
├── static/                  # CSS/JS
├── data/                    # Uploads & results (gitignored)
├── docker-compose.yml
└── requirements.txt
```

## Obsługiwane formaty

| Typ | Formaty | Pipeline |
|-----|---------|----------|
| Audio | .mp3, .wav, .m4a, .ogg, .flac, .webm | Whisper → Ollama |
| Dokumenty | .pdf, .docx, .doc, .txt, .rtf | Extract text → Ollama |
| Obrazy | .png, .jpg, .jpeg, .tiff, .bmp | OCR (Tesseract) → Ollama |

## Konfiguracja (ENV)

```bash
# .env
DATABASE_URL=sqlite:///data/jobs.db
REDIS_URL=redis://localhost:6379/0
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest
WHISPER_URL=http://localhost:9000
```

## Następne kroki (po MVP)

- [ ] Logowanie Microsoft (SSO Entra ID)
- [ ] PostgreSQL zamiast SQLite
- [ ] Speaker diarization (pyannote)
- [ ] Email notifications
- [ ] MinIO dla plików
