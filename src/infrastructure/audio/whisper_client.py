"""
Whisper ASR client.
"""

import logging
import requests

from domain.models.audio import AudioSegment
from domain.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class WhisperASRClient:
    """
    Whisper Automatic Speech Recognition client.

    Implements WhisperClient protocol.
    """

    def __init__(self, base_url: str):
        """
        Initialize Whisper client.

        Args:
            base_url: Whisper API URL (e.g., http://localhost:9000)
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"WhisperASRClient initialized: {base_url}")

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        timeout: int = 300
    ) -> list[AudioSegment]:
        """
        Transcribe audio to text with timestamps.

        Args:
            audio_bytes: Audio file data
            language: Optional language code
            timeout: Request timeout

        Returns:
            List of audio segments with timestamps

        Raises:
            AudioProcessingError: If transcription fails
        """
        try:
            files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
            data = {}

            if language:
                data["language"] = language

            response = requests.post(
                f"{self.base_url}/asr",
                files=files,
                data=data,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()

            # Parse segments
            segments = []
            for seg in result.get("segments", []):
                segments.append(AudioSegment(
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                    text=seg.get("text", "").strip()
                ))

            logger.info(f"Whisper transcribed {len(segments)} segments")
            return segments

        except requests.exceptions.RequestException as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise AudioProcessingError(
                f"Whisper ASR failed: {e}",
                service="Whisper"
            ) from e

    def health_check(self) -> bool:
        """Check if Whisper service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3)
            return response.ok
        except Exception:
            # Try alternative endpoints
            try:
                response = requests.get(self.base_url, timeout=3)
                return response.ok
            except Exception:
                return False
