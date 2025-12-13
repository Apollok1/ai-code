# Pyannote Speaker Diarization Server

Service do rozpoznawania mówców (speaker diarization) w plikach audio.

## FIX: Hugging Face API

**Problem:** Stara wersja używała `use_auth_token`, nowa używa `token`.

**Rozwiązanie:** Kod próbuje obu wersji API automatically.

## Wymagania

- HF_TOKEN (Hugging Face token)
- Akceptacja licencji modelu: https://huggingface.co/pyannote/speaker-diarization-3.1

## Konfiguracja

```bash
# W .env:
HF_TOKEN=hf_xxxxxxxxxxxxx
```

## Build i uruchomienie

```bash
cd /home/michal/moj-asystent-ai
docker compose build pyannote
docker compose up -d pyannote

# Sprawdź logi
docker compose logs -f pyannote

# Health check
curl http://localhost:8001/health
```

## API

### GET /health
Zwraca status i czy model jest załadowany.

### POST /diarize
Upload pliku audio → zwraca segmenty z identyfikacją mówców.

## Troubleshooting

### Model not loading
1. Sprawdź HF_TOKEN: `docker compose exec pyannote env | grep HF_TOKEN`
2. Sprawdź czy zaakceptowałeś licencję modelu na Hugging Face
3. Sprawdź logi: `docker compose logs pyannote`

### Timeout podczas ładowania
Model jest duży (~1GB). Pierwsze uruchomienie może zająć 2-3 minuty.
