#!/usr/bin/env python3
"""
Pyannote diarization server - FIXED for new huggingface_hub API
"""
import os
import logging
from fastapi import FastAPI, UploadFile, File
from pyannote.audio import Pipeline
import torch
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyannote-server")

app = FastAPI()

# Global pipeline
pipeline = None
model_loaded = False

@app.on_event("startup")
async def load_model():
    global pipeline, model_loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Ładowanie modelu na urządzeniu: {device}")

        # FIXED: używamy 'token' zamiast 'use_auth_token'
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN nie ustawiony!")
            return

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token  # Dla starszych wersji
        )
        pipeline.to(torch.device(device))
        model_loaded = True
        logger.info("Model załadowany pomyślnie!")
    except TypeError as e:
        # Jeśli 'use_auth_token' nie działa, spróbuj 'token'
        if "use_auth_token" in str(e):
            logger.info("Próba z nowym API (token=...)")
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token  # NOWE API
                )
                pipeline.to(torch.device(device))
                model_loaded = True
                logger.info("Model załadowany pomyślnie (nowe API)!")
            except Exception as e2:
                logger.error(f"Ładowanie modelu nie powiodło się (nowe API): {e2}")
        else:
            logger.error(f"Ładowanie modelu nie powiodło się: {e}")
    except Exception as e:
        logger.error(f"Ładowanie modelu nie powiodło się: {e}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@app.post("/diarize")
async def diarize(audio_file: UploadFile = File(...)):
    if not model_loaded:
        return {"error": "Model not loaded"}

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Run diarization
        output = pipeline(tmp_path)

        # Convert to JSON-serializable format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # Cleanup
        os.unlink(tmp_path)

        return {
            "segments": segments,
            "num_speakers": len(set(s["speaker"] for s in segments))
        }
    except Exception as e:
        logger.error(f"Diarization error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
