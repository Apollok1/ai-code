"""
Core document domain models.

These models represent the essential business entities and are framework-agnostic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    IMAGE = "image"
    AUDIO = "audio"
    EMAIL = "email"
    TEXT = "text"
    UNKNOWN = "unknown"

    @classmethod
    def from_filename(cls, filename: str) -> "DocumentType":
        """Detect document type from filename extension"""
        lower_name = filename.lower()

        if lower_name.endswith('.pdf'):
            return cls.PDF
        elif lower_name.endswith('.docx'):
            return cls.DOCX
        elif lower_name.endswith(('.pptx', '.ppt')):
            return cls.PPTX
        elif lower_name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            return cls.IMAGE
        elif lower_name.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
            return cls.AUDIO
        elif lower_name.endswith(('.eml', '.msg')):
            return cls.EMAIL
        elif lower_name.endswith('.txt'):
            return cls.TEXT
        else:
            return cls.UNKNOWN


@dataclass(frozen=True)
class Page:
    """
    Represents a single page or section of a document.

    Immutable to prevent accidental modifications.
    """
    number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate page data"""
        if self.number < 1:
            raise ValueError(f"Page number must be >= 1, got {self.number}")

    def is_empty(self) -> bool:
        """Check if page has no meaningful text"""
        return len(self.text.strip()) == 0

    def word_count(self) -> int:
        """Count words in page text"""
        return len(self.text.split())

    def char_count(self) -> int:
        """Count characters (excluding whitespace)"""
        return len(self.text.strip())

    def preview(self, max_chars: int = 100) -> str:
        """Get a preview of page text"""
        text = self.text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


@dataclass
class ExtractionMetadata:
    """
    Metadata about the extraction process.

    Mutable to allow updates during processing.
    """
    document_type: DocumentType
    pages_count: int
    extraction_method: str
    processing_time_seconds: float
    file_size_bytes: int

    # Optional fields
    has_speakers: bool = False
    vision_model: str | None = None
    ocr_language: str | None = None
    audio_duration_seconds: float | None = None

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message"""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "document_type": self.document_type.value,
            "pages_count": self.pages_count,
            "extraction_method": self.extraction_method,
            "processing_time_seconds": self.processing_time_seconds,
            "file_size_bytes": self.file_size_bytes,
            "has_speakers": self.has_speakers,
            "vision_model": self.vision_model,
            "ocr_language": self.ocr_language,
            "audio_duration_seconds": self.audio_duration_seconds,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ExtractionResult:
    """
    Complete result of document extraction.

    This is the main output of any extractor.
    """
    file_name: str
    pages: list[Page]
    metadata: ExtractionMetadata
    extracted_at: datetime = field(default_factory=datetime.now)

    # Original text preserved for speaker mapping
    original_text: str | None = None

    def __post_init__(self):
        """Set original_text if not provided"""
        if self.original_text is None:
            self.original_text = self.full_text

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages"""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def total_words(self) -> int:
        """Count total words across all pages"""
        return sum(page.word_count() for page in self.pages)

    @property
    def total_chars(self) -> int:
        """Count total characters across all pages"""
        return sum(page.char_count() for page in self.pages)

    def is_successful(self) -> bool:
        """Check if extraction was successful (no errors, has content)"""
        return not self.metadata.has_errors() and len(self.pages) > 0

    def get_page(self, page_number: int) -> Page | None:
        """Get specific page by number"""
        for page in self.pages:
            if page.number == page_number:
                return page
        return None

    def filter_empty_pages(self) -> list[Page]:
        """Get only non-empty pages"""
        return [page for page in self.pages if not page.is_empty()]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_name": self.file_name,
            "pages": [
                {
                    "number": page.number,
                    "text": page.text,
                    "metadata": page.metadata,
                    "word_count": page.word_count(),
                }
                for page in self.pages
            ],
            "metadata": self.metadata.to_dict(),
            "extracted_at": self.extracted_at.isoformat(),
            "total_words": self.total_words,
            "total_chars": self.total_chars,
        }

    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        lines = [
            f"# {self.file_name}",
            "",
            f"**Extracted:** {self.extracted_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Type:** {self.metadata.document_type.value}",
            f"**Pages:** {self.metadata.pages_count}",
            f"**Method:** {self.metadata.extraction_method}",
            f"**Words:** {self.total_words:,}",
            "",
            "---",
            "",
        ]

        for page in self.pages:
            lines.append(f"## Page {page.number}")
            lines.append("")
            lines.append(page.text)
            lines.append("")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation"""
        return (
            f"ExtractionResult(file='{self.file_name}', "
            f"pages={len(self.pages)}, words={self.total_words})"
        )

    def __repr__(self) -> str:
        return self.__str__()
