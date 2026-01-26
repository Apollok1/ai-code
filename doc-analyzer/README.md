# Doc Analyzer - Konfigurator Ofert (MVP)

System standaryzacji ofert: analiza archiwum → baza klauzul → konfigurator → safety net.

## Flow

```
1. Wgraj ~50 historycznych ofert (PDF/DOCX)
2. AI wyodrębnia klauzule: zakres prac + wykluczenia
3. Konstruktor tworzy nową ofertę wybierając z gotowej listy
4. Safety check: czy nie pominięto krytycznych wykluczeń
```

## Start

```bash
cp .env.example .env
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1:
uvicorn app.main:app --reload

# Terminal 2:
rq worker -u redis://127.0.0.1:6379/0 doc-analyzer
```

Wejdź: http://127.0.0.1:8000

## Strony

| URL | Opis |
|-----|------|
| `/` | Dashboard (statystyki) |
| `/archive/upload` | Upload historycznych ofert |
| `/archive` | Lista przeanalizowanych dokumentów |
| `/clauses` | Baza klauzul (zakres + wykluczenia) |
| `/offers/new` | Konfigurator nowej oferty |
| `/offers` | Lista ofert |
| `/offers/{id}` | Szczegóły oferty + safety check |
