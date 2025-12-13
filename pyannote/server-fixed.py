import os, torch, logging, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pyannote.audio import Pipeline
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyannote-server")

HF_TOKEN = os.getenv("HF_TOKEN")
app = FastAPI()
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Ładowanie modelu na urządzeniu: {device}")

        # TRY BOTH APIs - kompatybilność z różnymi wersjami!
        try:
            # Próba NOWEGO API (huggingface_hub >= 0.16)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            )
            logger.info("Model załadowany (nowe API: token=)")
        except TypeError as e:
            if "token" in str(e):
                # Fallback do STAREGO API (pyannote.audio < 3.1)
                logger.info("Próba ze starym API (use_auth_token=)...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                )
                logger.info("Model załadowany (stare API: use_auth_token=)")
            else:
                raise

        pipeline.to(torch.device(device))
        logger.info(f"✅ Model pyannote załadowany na {device}")

    except Exception as e:
        logger.error(f"Ładowanie modelu nie powiodło się: {e}", exc_info=True)
        pipeline = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(pipeline)}

@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model nie jest gotowy.")

    try:
        # Zapisz plik tymczasowo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Przetwarzanie: {file.filename}, rozmiar: {len(content)} bytes")

        # Przetwórz (przekaż ŚCIEŻKĘ, nie bajty!)
        diarization = pipeline(tmp_path)

        # Usuń tymczasowy plik
        os.unlink(tmp_path)

        # Konwertuj do JSON
        segments = [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        logger.info(f"✅ Znaleziono {len(segments)} segmentów")
        return {"segments": segments}

    except Exception as e:
        logger.error(f"Błąd diaryzacji: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
