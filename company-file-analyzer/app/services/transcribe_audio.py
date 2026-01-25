"""
Audio transcription service (Whisper API).
MVP: Basic transcription without speaker diarization.
"""
import logging
from pathlib import Path
import requests

from app.config import WHISPER_URL

logger = logging.getLogger(__name__)


def transcribe_audio(file_path: Path, language: str = "pl", timeout: int = 600) -> str:
    """
    Transcribe audio file using Whisper API.

    Args:
        file_path: Path to audio file
        language: Language code (pl, en, etc.)
        timeout: Request timeout in seconds (long files need more time)

    Returns:
        Transcription text

    Raises:
        Exception: If transcription fails
    """
    logger.info(f"Transcribing: {file_path.name}")

    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{WHISPER_URL}/asr",
                files={"audio_file": (file_path.name, f)},
                data={
                    "language": language,
                    "output": "json"
                },
                timeout=timeout
            )
            response.raise_for_status()

        result = response.json()

        # Handle different response formats
        if isinstance(result, list):
            # List of segments: [{start, end, text}, ...]
            segments = result
            text_parts = [seg.get("text", "") for seg in segments]
            return "\n".join(text_parts)
        elif isinstance(result, dict):
            # Single text response or segments in dict
            if "text" in result:
                return result["text"]
            elif "segments" in result:
                segments = result["segments"]
                return "\n".join(seg.get("text", "") for seg in segments)

        return str(result)

    except requests.exceptions.Timeout:
        logger.error(f"Transcription timeout for {file_path.name}")
        raise TimeoutError(f"Transcription timeout (>{timeout}s)")
    except requests.exceptions.RequestException as e:
        logger.error(f"Whisper request failed: {e}")
        raise


def format_transcription(text: str, include_timestamps: bool = False) -> str:
    """
    Format transcription for readability.

    Args:
        text: Raw transcription text
        include_timestamps: Whether to include timestamps (for future use)

    Returns:
        Formatted text
    """
    # Clean up whitespace
    lines = text.strip().split("\n")
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)
