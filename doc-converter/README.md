# Document Converter Service

Serwis do konwersji dokumentów z interfejsem Streamlit.

## Struktura

```
doc-converter/
├── Dockerfile              # Definicja obrazu Docker
├── requirements.txt        # Zależności Python
├── README.md              # Ten plik
└── app/
    └── converter.py       # Aplikacja Streamlit (kopiowana z doc_converter.py)
```

## Automatyczne wdrożenie

Katalog `doc-converter/` jest automatycznie kopiowany do `moj-asystent-ai/` przez skrypt deploy.

### Jak używać skryptu deploy:

**1. Deploy (pobieranie i aktualizacja):**
```bash
cd /home/michal/ai-code
./deploy-ai-updated.sh
```

lub z rebuild:
```bash
REBUILD=1 ./deploy-ai-updated.sh
```

**2. Push (wysyłanie zmian na GitHub):**
```bash
ACTION=push ./deploy-ai-updated.sh
```

## Co robi skrypt deploy:

1. **Aktualizuje kod** - `git pull` w repozytorium ai-code
2. **Kopiuje strukturę** - rsync całego katalogu `doc-converter/` do `moj-asystent-ai/doc-converter/`
3. **Aktualizuje kontenery** - restartuje lub przebudowuje kontener `doc-converter`
4. **Sprawdza status** - testuje endpoint health check

## Ręczna aktualizacja

Jeśli chcesz ręcznie zaktualizować tylko doc-converter:

```bash
cd /home/michal/ai-code
git pull

# Kopiuj strukturę
rsync -av --delete \
  /home/michal/ai-code/doc-converter/ \
  /home/michal/moj-asystent-ai/doc-converter/

# Przebuduj i uruchom
cd /home/michal/moj-asystent-ai
docker compose up -d --build doc-converter

# Sprawdź logi
docker compose logs -f doc-converter
```

## Zależności systemowe (w Dockerfile)

- **tesseract-ocr** - OCR dla skanowanych dokumentów
- **poppler-utils** - konwersja PDF do obrazów
- **ffmpeg** - przetwarzanie audio/video
- **libgl1-mesa-glx** - OpenCV

## Zmienne środowiskowe (w docker-compose.yml)

```yaml
environment:
  - OLLAMA_URL=http://ollama:11434
  - WHISPER_URL=http://whisper:9000
  - PYANNOTE_URL=http://pyannote:8000
  - ANYTHINGLLM_URL=${ANYTHINGLLM_URL}
  - ANYTHINGLLM_API_KEY=${ANYTHINGLLM_API_KEY}
  - STRICT_OFFLINE=1
  - FFMPEG_BINARY=/usr/bin/ffmpeg
```

## Porty

- **8502** - interfejs webowy Streamlit

## Wolumeny

- `./doc-converter/app:/app/app:ro` - kod aplikacji (read-only)
- `./outputs:/app/outputs` - pliki wyjściowe
