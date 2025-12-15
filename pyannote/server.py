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

    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Run diarization
        result = pipeline(tmp_path)

        # --- Rozpakowanie wyniku pipeline ---

        diarization = None

        # 0) Jeśli sam wynik ma itertracks (stare pyannote) – użyj go
        if hasattr(result, "itertracks"):
            diarization = result

        # 1) Jeśli to DiarizeOutput / dataclass – spróbuj znanych pól
        if diarization is None:
            candidate_attrs = [
                "speaker_diarization",
                "exclusive_speaker_diarization",
                "diarization",
                "annotation",
            ]
            for name in candidate_attrs:
                value = getattr(result, name, None)
                if value is not None and hasattr(value, "itertracks"):
                    diarization = value
                    logger.info(f"Używam pola '{name}' jako anotacji diarization.")
                    break

        # 2) Jako ostatnia deska ratunku: przeleć wszystkie pola dataclass
        if diarization is None and hasattr(result, "__dataclass_fields__"):
            for field_name in result.__dataclass_fields__.keys():
                value = getattr(result, field_name, None)
                if value is not None and hasattr(value, "itertracks"):
                    diarization = value
                    logger.info(
                        f"Używam pola dataclass '{field_name}' jako anotacji diarization."
                    )
                    break

        # 3) Jeszcze fallback: jeśli to lista/krotka – sprawdź elementy
        if diarization is None and isinstance(result, (list, tuple)):
            for item in result:
                if hasattr(item, "itertracks"):
                    diarization = item
                    logger.info("Używam pierwszego elementu wyników jako anotacji.")
                    break

        if diarization is None:
            logger.error(
                f"Nie mogę znaleźć anotacji diarization w wyniku pipeline: "
                f"type={type(result)}, dir={dir(result)}"
            )
            return {"error": "Unsupported diarization output format"}

        # --- Konwersja na JSON-serializable format ---

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })

        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        return {
            "segments": segments,
            "num_speakers": len(set(s["speaker"] for s in segments)),
        }

    except Exception as e:
        logger.error(f"Diarization error: {e}")
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
