"""
Audio processing service protocols.
"""

from typing import Protocol
from domain.models.audio import AudioSegment, DiarizationSegment


class WhisperClient(Protocol):
    """Interface for Whisper ASR (Automatic Speech Recognition)"""

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        timeout: int = 300
    ) -> list[AudioSegment]:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: Audio file data
            language: Optional language code
            timeout: Request timeout in seconds

        Returns:
            List of audio segments with timestamps

        Raises:
            AudioProcessingError: If transcription fails
        """
        ...

    def health_check(self) -> bool:
        """Check if Whisper service is available"""
        ...


class DiarizationClient(Protocol):
    """Interface for speaker diarization (Pyannote)"""

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
        ...

    def health_check(self) -> bool:
        """Check if diarization service is available"""
        ...
