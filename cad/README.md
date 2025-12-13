# CAD Estimator Panel Service

Service CAD Estimator z interfejsem Streamlit i integracją PostgreSQL + pgvector.

## Struktura

```
cad/
├── Dockerfile              # Definicja obrazu Docker
├── requirements.txt        # Zależności Python
└── README.md              # Ten plik

# Kod montowany z ai-code/src/
src/cad/                    # ← montowane jako volume
├── __init__.py
├── application/            # Logika biznesowa
├── domain/                 # Modele domenowe
├── infrastructure/         # AI, database, parsery
│   ├── ai/
│   ├── database/
│   ├── embeddings/
│   ├── learning/
│   └── multi_model/       # Multi-model pipeline
└── presentation/          # UI Streamlit
    ├── app.py            # ← Entry point
    ├── components/
    └── state/
```

## Automatyczne wdrożenie

Katalog `cad/` używa **direct mounting** - kod jest montowany bezpośrednio z `ai-code/src/`.

### Jak używać:

**1. Deploy (z nowym skryptem):**
```bash
cd /home/michal/ai-code
./deploy-direct.sh
```

**2. Push (wysyłanie zmian):**
```bash
cd /home/michal/ai-code
git add .
git commit -m "update: CAD changes"
git push
```

## Co robi Docker Compose:

```yaml
cad-panel:
  build:
    context: ${AI_CODE_PATH}/cad
  volumes:
    # MONTUJ BEZPOŚREDNIO źródła z ai-code
    - ${AI_CODE_PATH}/src:/app/src:ro
    # Dane projektu w moj-asystent-ai
    - ./cad/data:/data
```

## Zmienne środowiskowe

```yaml
environment:
  - DB_HOST=cad-postgres
  - DB_NAME=cad_estimator
  - DB_USER=cad_user
  - DB_PASSWORD=cad_password_2024
  - OLLAMA_URL=http://ollama:11434
  - EMBED_MODEL=nomic-embed-text
  - EMBED_DIM=768
  - STRICT_OFFLINE=1
```

## Porty

- **8501** - interfejs webowy Streamlit

## Wolumeny

- `${AI_CODE_PATH}/src:/app/src:ro` - kod źródłowy (read-only, montowany z ai-code)
- `./cad/data:/data` - dane projektu (persystentne w moj-asystent-ai)

## Zależności

### Usługi:
- **cad-postgres** - PostgreSQL z pgvector dla embeddings
- **ollama** - LLM dla AI estimacji

### Multi-Model Pipeline:
Aplikacja wspiera multi-model pipeline (4 etapy):
1. Technical Analysis
2. Structural Decomposition
3. Hours Estimation
4. Risk Analysis

Konfiguracja w sidebar UI.

## Rozwój

### Edycja kodu:
```bash
cd /home/michal/ai-code/src/cad
nano presentation/app.py
# lub
nano infrastructure/multi_model/orchestrator.py
```

### Testowanie zmian:
```bash
# Streamlit ma auto-reload - wystarczy odświeżyć przeglądarkę!
# Lub restart:
cd /home/michal/ai-code
./deploy-direct.sh
```

### Sprawdzenie logów:
```bash
cd /home/michal/moj-asystent-ai
docker compose logs -f cad-panel
```
