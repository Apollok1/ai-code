"""
Session Manager - Clean wrapper around Streamlit session_state.

This provides a typed, ergonomic API instead of raw dictionary access.
"""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

try:
    import streamlit as st
except ImportError:
    st = None  # For testing without Streamlit

from domain.models.document import ExtractionResult
from domain.models.config import ExtractionConfig


@dataclass
class ConversionStats:
    """Statistics about conversion process"""
    processed: int = 0
    errors: int = 0
    total_pages: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate conversion duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class ConversionState:
    """
    Complete state of the conversion process.

    This replaces the scattered st.session_state usage with typed structure.
    """
    # Results
    results: list[ExtractionResult] = field(default_factory=list)
    converted: bool = False

    # Audio-specific
    audio_summaries: list[dict[str, Any]] = field(default_factory=list)
    speaker_maps: dict[str, dict[str, str]] = field(default_factory=dict)

    # Statistics
    stats: ConversionStats = field(default_factory=ConversionStats)

    # UI state
    converting: bool = False
    cancel_requested: bool = False

    # File signature (for caching)
    file_signature: int | None = None

    def reset(self) -> None:
        """Reset conversion state"""
        self.results.clear()
        self.audio_summaries.clear()
        self.speaker_maps.clear()
        self.converted = False
        self.converting = False
        self.cancel_requested = False
        self.file_signature = None
        self.stats = ConversionStats()

    def start_conversion(self) -> None:
        """Mark conversion as started"""
        self.converting = True
        self.cancel_requested = False
        self.stats.start_time = datetime.now()

    def end_conversion(self) -> None:
        """Mark conversion as ended"""
        self.converting = False
        self.converted = True
        self.stats.end_time = datetime.now()

    def add_result(self, result: ExtractionResult) -> None:
        """Add extraction result"""
        self.results.append(result)
        self.stats.processed += 1
        self.stats.total_pages += result.metadata.pages_count

        if result.metadata.has_errors():
            self.stats.errors += 1

    def get_result(self, file_name: str) -> ExtractionResult | None:
        """Get result by file name"""
        for result in self.results:
            if result.file_name == file_name:
                return result
        return None

    def has_results(self) -> bool:
        """Check if we have any results"""
        return len(self.results) > 0

    @property
    def combined_text(self) -> str:
        """Get all results combined as text"""
        lines = []
        for result in self.results:
            lines.append(f"\n{'='*80}\n")
            lines.append(f"FILE: {result.file_name}\n")
            lines.append(f"{'='*80}\n")
            lines.append(result.full_text)
            lines.append(f"\n[Pages: {result.metadata.pages_count}]\n")
        return "\n".join(lines)


class SessionManager:
    """
    Manager for Streamlit session state with typed access.

    Usage:
        session = SessionManager()
        session.state.add_result(result)
        if session.state.has_results():
            # ...
    """

    STATE_KEY = "conversion_state_v2"

    def __init__(self):
        if st is None:
            raise RuntimeError("Streamlit not available")
        self._init_state()

    def _init_state(self) -> None:
        """Initialize session state if not exists"""
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = ConversionState()

    @property
    def state(self) -> ConversionState:
        """Get current conversion state"""
        return st.session_state[self.STATE_KEY]

    def reset(self) -> None:
        """Reset session state"""
        self.state.reset()

    # === CONVENIENCE METHODS ===

    def is_converting(self) -> bool:
        """Check if conversion is in progress"""
        return self.state.converting

    def has_results(self) -> bool:
        """Check if we have results"""
        return self.state.has_results()

    def start_conversion(self) -> None:
        """Start conversion process"""
        self.state.start_conversion()

    def end_conversion(self) -> None:
        """End conversion process"""
        self.state.end_conversion()

    def request_cancel(self) -> None:
        """Request cancellation"""
        self.state.cancel_requested = True

    def is_cancel_requested(self) -> bool:
        """Check if cancellation requested"""
        return self.state.cancel_requested

    def add_result(self, result: ExtractionResult) -> None:
        """Add extraction result"""
        self.state.add_result(result)

    def get_results(self) -> list[ExtractionResult]:
        """Get all results"""
        return self.state.results

    def save_speaker_map(self, file_name: str, speaker_map: dict[str, str]) -> None:
        """Save speaker mapping for a file"""
        self.state.speaker_maps[file_name] = speaker_map

    def get_speaker_map(self, file_name: str) -> dict[str, str]:
        """Get speaker mapping for a file"""
        return self.state.speaker_maps.get(file_name, {})

    # === LEGACY COMPATIBILITY (for gradual migration) ===

    def get_legacy(self, key: str, default: Any = None) -> Any:
        """
        Get value from old session_state format.

        This allows gradual migration from old code.
        """
        return st.session_state.get(key, default)

    def set_legacy(self, key: str, value: Any) -> None:
        """
        Set value in old session_state format.

        For backward compatibility during migration.
        """
        st.session_state[key] = value

    # === CONFIG PERSISTENCE ===

    def save_config(self, config: ExtractionConfig) -> None:
        """Save extraction config to session"""
        st.session_state["extraction_config"] = config

    def load_config(self) -> ExtractionConfig | None:
        """Load extraction config from session"""
        return st.session_state.get("extraction_config")

    # === FILE SIGNATURE (for caching) ===

    def calculate_file_signature(self, files: list) -> int:
        """Calculate signature for file list"""
        try:
            items = [
                (f.name, getattr(f, 'size', None) or len(f.getvalue()))
                for f in files
            ]
            return hash(tuple(items))
        except Exception:
            return 0

    def files_changed(self, files: list) -> bool:
        """Check if files changed since last conversion"""
        new_sig = self.calculate_file_signature(files)
        old_sig = self.state.file_signature

        if new_sig != old_sig:
            self.state.file_signature = new_sig
            return True
        return False
