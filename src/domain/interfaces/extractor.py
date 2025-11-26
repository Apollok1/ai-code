"""
Extractor Protocol - Interface for all document extractors.

Using Protocol (PEP 544) for structural subtyping (duck typing with type checking).
"""

from typing import Protocol, BinaryIO
from domain.models.document import ExtractionResult
from domain.models.config import ExtractionConfig


class Extractor(Protocol):
    """
    Common interface for all document extractors.

    All extractors must implement these methods to be used in the pipeline.
    No inheritance required - just implement the methods (structural typing).
    """

    def can_handle(self, file_name: str) -> bool:
        """
        Check if this extractor can handle the given file.

        Args:
            file_name: Name of the file (with extension)

        Returns:
            True if this extractor supports the file type
        """
        ...

    def extract(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Extract content from the file.

        Args:
            file: Binary file object
            file_name: Original file name
            config: Extraction configuration

        Returns:
            ExtractionResult with extracted content

        Raises:
            ExtractionError: If extraction fails
            UnsupportedFormatError: If file format not supported
        """
        ...

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """
        File extensions supported by this extractor.

        Returns:
            Tuple of supported extensions (e.g., ('.pdf', '.PDF'))
        """
        ...

    @property
    def name(self) -> str:
        """
        Human-readable name of the extractor.

        Returns:
            Extractor name (e.g., "PDF Extractor")
        """
        ...
