"""
Audio Extractor - Extract text from audio files using Whisper ASR.

Strategy:
- Transcribe with Whisper
- Optional speaker diarization with Pyannote
- Combine transcription + speakers
"""

import logging
from typing import BinaryIO

from domain.models.document import (
    ExtractionResult,
    Page,
    ExtractionMetadata,
    DocumentType
)
from domain.models.config import ExtractionConfig
from domain.models.audio import AudioSegment, DiarizationSegment
from domain.interfaces.audio_service import WhisperClient, DiarizationClient
from domain.exceptions import ExtractionError, AudioProcessingError

logger = logging.getLogger(__name__)


class AudioExtractor:
    """
    Audio extractor with Whisper ASR and optional speaker diarization.
    """

    def __init__(
        self,
        whisper_client: WhisperClient,
        diarization_client: DiarizationClient | None = None
    ):
        """
        Initialize audio extractor.

        Args:
            whisper_client: Whisper ASR client
            diarization_client: Optional Pyannote diarization client
        """
        self.whisper = whisper_client
        self.diarization = diarization_client
        logger.info(
            f"AudioExtractor initialized "
            f"(diarization={'enabled' if diarization_client else 'disabled'})"
        )

    @property
    def name(self) -> str:
        return "Audio Extractor"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.MP3', '.WAV')

    def can_handle(self, file_name: str) -> bool:
        """Check if file is audio"""
        lower_name = file_name.lower()
        return any(
            lower_name.endswith(ext)
            for ext in ('.mp3', '.wav', '.m4a', '.ogg', '.flac')
        )

    def extract(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Extract text from audio.

        Args:
            file: Audio file object
            file_name: Original file name
            config: Extraction configuration

        Returns:
            ExtractionResult with transcription (optionally with speakers)

        Raises:
            ExtractionError: If extraction fails
        """
        import time
        start_time = time.time()

        try:
            file.seek(0)
            audio_bytes = file.read()

            logger.info(f"Extracting audio: {file_name}")

            # Calculate timeout based on file size
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            timeout = int(max(240, file_size_mb * 25))

            # Transcribe with Whisper
            segments = self.whisper.transcribe(
                audio_bytes,
                language=None,  # Auto-detect
                timeout=timeout
            )

            # Optional diarization
            has_speakers = False
            if self.diarization and config.enable_diarization:
                try:
                    if self.diarization.health_check():
                        segments = self._add_speakers(segments, audio_bytes, timeout)
                        has_speakers = True
                    else:
                        logger.warning("Diarization service not healthy, skipping")
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}, continuing without speakers")

            # Format text
            text_lines = [str(seg) for seg in segments]
            full_text = "\n".join(text_lines)

            processing_time = time.time() - start_time

            # Calculate audio duration
            audio_duration = max((seg.end for seg in segments), default=0.0)

            metadata = ExtractionMetadata(
                document_type=DocumentType.AUDIO,
                pages_count=1,
                extraction_method="whisper" + (" + pyannote" if has_speakers else ""),
                processing_time_seconds=processing_time,
                file_size_bytes=len(audio_bytes),
                has_speakers=has_speakers,
                audio_duration_seconds=audio_duration
            )

            logger.info(
                f"Audio extraction complete: {file_name} | "
                f"{len(segments)} segments | {audio_duration:.1f}s audio | "
                f"{processing_time:.2f}s processing"
            )

            return ExtractionResult(
                file_name=file_name,
                pages=[Page(
                    number=1,
                    text=full_text,
                    metadata={"has_speakers": has_speakers, "segments_count": len(segments)}
                )],
                metadata=metadata
            )

        except Exception as e:
            logger.exception(f"Audio extraction failed: {file_name}")
            raise ExtractionError(
                f"Failed to extract audio: {e}",
                file_name=file_name
            ) from e

    def _add_speakers(
        self,
        transcription_segments: list[AudioSegment],
        audio_bytes: bytes,
        timeout: int
    ) -> list[AudioSegment]:
        """
        Add speaker labels to transcription segments.

        Args:
            transcription_segments: Segments from Whisper
            audio_bytes: Original audio data
            timeout: Request timeout

        Returns:
            Segments with speaker labels
        """
        # Get diarization
        diar_segments = self.diarization.diarize(audio_bytes, timeout=timeout)

        # Match transcription segments with speakers
        result = []
        for trans_seg in transcription_segments:
            # Find best matching speaker
            speaker = self._find_speaker(trans_seg, diar_segments)

            # Create new segment with speaker
            new_seg = AudioSegment(
                start=trans_seg.start,
                end=trans_seg.end,
                text=trans_seg.text,
                speaker=speaker
            )
            result.append(new_seg)

        return result

    @staticmethod
    def _find_speaker(
        audio_seg: AudioSegment,
        diar_segments: list[DiarizationSegment]
    ) -> str:
        """
        Find best matching speaker for audio segment.

        Uses maximum overlap strategy.
        """
        if not diar_segments:
            return "SPEAKER_00"

        max_overlap = 0.0
        best_speaker = "SPEAKER_00"

        for diar_seg in diar_segments:
            if diar_seg.overlaps_with(audio_seg.start, audio_seg.end):
                overlap = diar_seg.overlap_duration(audio_seg.start, audio_seg.end)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = diar_seg.speaker

        return best_speaker
