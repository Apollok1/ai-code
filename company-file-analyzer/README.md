# Company File Analyzer (MVP) - FastAPI + HTML + SQLite + Redis(RQ) + Ollama

## Start (lokalnie / VM)
1) Skopiuj env:
   cp .env.example .env

2) Uruchom Redis + Ollama:
   docker compose up -d

3) (Ważne) Pobierz model w Ollama (w kontenerze):
   docker compose exec ollama ollama pull qwen2.5:7b

4) Python venv:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

5) Start web:
   uvicorn app.main:app --reload

6) W drugim terminalu (ten sam venv):
   source .venv/bin/activate
   rq worker -u redis://127.0.0.1:6379/0 default

Wejdź: http://127.0.0.1:8000
