#!/usr/bin/env python3
"""
Whisper ASR Server with ROCm (AMD GPU) support
Compatible API with openai-whisper-asr-webservice
"""
import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import torch
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-rocm")

app = FastAPI(title="Whisper ASR ROCm")

# Global model
model = None
device = None

@app.on_event("startup")
async def load_model():
    global model, device

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 Using CUDA/ROCm device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("⚠️  CUDA/ROCm not available, using CPU")

    # Load model
    model_name = os.getenv("ASR_MODEL", "medium")
    logger.info(f"Loading Whisper model: {model_name} on {device}")

    model = whisper.load_model(model_name, device=device)
    logger.info(f"✅ Model loaded successfully on {device}")


@app.get("/")
async def root():
    return {"message": "Whisper ASR ROCm API", "device": device}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }


@app.post("/asr")
async def transcribe(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe", regex="^(transcribe|translate)$"),
    language: str = Query("pl"),
    word_timestamps: bool = Query(False),
    output: str = Query("json")
):
    """
    Transcribe audio file using Whisper.

    Compatible with openai-whisper-asr-webservice API.
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing audio: {audio_file.filename}, size: {len(content)} bytes")

        # Transcribe
        result = model.transcribe(
            tmp_path,
            task=task,
            language=language if language != "auto" else None,
            word_timestamps=word_timestamps,
            verbose=False
        )

        # Cleanup
        os.unlink(tmp_path)

        # Format response (compatible with original API)
        response = {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", language),
            "duration": result.get("duration", 0)
        }

        logger.info(f"✅ Transcription complete: {len(result['text'])} chars")
        return response

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )
