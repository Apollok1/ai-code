"""
Audio-specific domain models.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AudioSegment:
    """Single audio segment with timing and text"""
    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None  # Speaker ID (e.g., "SPEAKER_00")

    @property
    def duration(self) -> float:
        """Segment duration in seconds"""
        return self.end - self.start

    def format_timestamp(self) -> str:
        """Format as [start-end]"""
        return f"[{self.start:.1f}s - {self.end:.1f}s]"

    def __str__(self) -> str:
        if self.speaker:
            return f"{self.format_timestamp()} {self.speaker}: {self.text}"
        return f"{self.format_timestamp()} {self.text}"


@dataclass
class DiarizationSegment:
    """Speaker diarization segment"""
    start: float
    end: float
    speaker: str  # Speaker ID

    def overlaps_with(self, audio_start: float, audio_end: float) -> bool:
        """Check if this diarization overlaps with audio segment"""
        return not (self.end <= audio_start or self.start >= audio_end)

    def overlap_duration(self, audio_start: float, audio_end: float) -> float:
        """Calculate overlap duration"""
        overlap_start = max(self.start, audio_start)
        overlap_end = min(self.end, audio_end)
        return max(0, overlap_end - overlap_start)
