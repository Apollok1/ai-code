"""
Pyannote speaker diarization client.
"""

import logging
import requests

from domain.models.audio import DiarizationSegment
from domain.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class PyannoteClient:
    """
    Pyannote speaker diarization client.

    Implements DiarizationClient protocol.
    """

    def __init__(self, base_url: str):
        """
        Initialize Pyannote client.

        Args:
            base_url: Pyannote API URL (e.g., http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"PyannoteClient initialized: {base_url}")

    def diarize(
        self,
        audio_bytes: bytes,
        timeout: int = 300
    ) -> list[DiarizationSegment]:
        """
        Perform speaker diarization.

        Args:
            audio_bytes: Audio file data
            timeout: Request timeout

        Returns:
            List of diarization segments with speaker IDs

        Raises:
            AudioProcessingError: If diarization fails
        """
        try:
            files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}

            response = requests.post(
                f"{self.base_url}/diarize",
                files=files,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()

            # Parse diarization segments
            segments = []
            for seg in result.get("segments", []):
                segments.append(DiarizationSegment(
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                    speaker=seg.get("speaker", "SPEAKER_00")
                ))

            # Normalize speaker IDs (some APIs return different formats)
            segments = self._normalize_speakers(segments)

            logger.info(f"Pyannote diarized {len(segments)} segments")
            return segments

        except requests.exceptions.RequestException as e:
            logger.error(f"Pyannote diarization failed: {e}")
            raise AudioProcessingError(
                f"Pyannote diarization failed: {e}",
                service="Pyannote"
            ) from e

    def health_check(self) -> bool:
        """Check if Pyannote service is available"""
        try:
            for endpoint in ["/health", "/status", "/ping", "/"]:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                    if response.ok:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    @staticmethod
    def _normalize_speakers(segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
        """
        Normalize speaker IDs to SPEAKER_XX format.
        """
        # Build speaker mapping
        unique_speakers = sorted(set(seg.speaker for seg in segments))
        speaker_map = {
            old: f"SPEAKER_{i:02d}"
            for i, old in enumerate(unique_speakers)
        }

        # Apply mapping
        return [
            DiarizationSegment(
                start=seg.start,
                end=seg.end,
                speaker=speaker_map.get(seg.speaker, seg.speaker)
            )
            for seg in segments
        ]
