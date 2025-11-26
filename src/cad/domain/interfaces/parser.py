"""
CAD Estimator Pro - Parser Protocol Interfaces

Protocols for file parsers (Excel, PDF, JSON).
"""
from typing import Protocol, BinaryIO
from ..models import Component


class ExcelParser(Protocol):
    """
    Protocol for Excel file parsers.

    Parses CAD project Excel files with component hierarchy and hours.
    """

    def parse(self, file: BinaryIO) -> dict:
        """
        Parse Excel file.

        Args:
            file: Excel file stream (bytes)

        Returns:
            Dict with keys:
            - components: list[dict] - Component data
            - totals: dict - Total hours (layout, detail, documentation)
            - multipliers: dict - Multipliers from Excel
            - statistics: dict - Stats (parts_count, assemblies_count)

        Raises:
            ExcelParsingError: If parsing fails
        """
        ...

    def extract_description_from_a1(self, file: BinaryIO) -> str:
        """
        Extract project description from cell A1 (first sheet).

        Args:
            file: Excel file stream (bytes)

        Returns:
            Description text (or empty string if not found)

        Raises:
            ExcelParsingError: If file read fails
        """
        ...


class PDFParser(Protocol):
    """
    Protocol for PDF file parsers.

    Extracts text from PDF specifications/drawings.
    """

    def extract_text(self, file: BinaryIO, max_pages: int = 200) -> str:
        """
        Extract text from PDF file.

        Args:
            file: PDF file stream (bytes)
            max_pages: Maximum pages to process

        Returns:
            Extracted text

        Raises:
            PDFParsingError: If extraction fails
        """
        ...


class ComponentParser(Protocol):
    """
    Protocol for component parsers.

    Parses component data from various sources (Excel comments, JSON, AI responses).
    """

    def parse_subcomponents_from_comment(self, comment: str) -> list[dict]:
        """
        Parse sub-components from Excel comment string.

        Example: "2x docisk, śruba trapezowa, 4x wspornik" ->
        [
            {"name": "docisk", "quantity": 2},
            {"name": "śruba trapezowa", "quantity": 1},
            {"name": "wspornik", "quantity": 4}
        ]

        Args:
            comment: Comment string from Excel cell

        Returns:
            List of sub-component dicts

        Raises:
            ParsingError: If parsing fails
        """
        ...

    def parse_ai_response(
        self,
        ai_text: str,
        fallback_components: list[Component] | None = None
    ) -> dict:
        """
        Parse AI response to estimate data.

        Args:
            ai_text: Raw AI response (JSON or text)
            fallback_components: Fallback components (if AI parsing fails)

        Returns:
            Dict with estimate data:
            - components: list[Component]
            - risks: list[Risk]
            - suggestions: list[Suggestion]
            - assumptions: list[str]
            - warnings: list[str]
            - overall_confidence: float

        Raises:
            AIResponseParsingError: If parsing fails completely
        """
        ...

    def canonicalize_component_name(self, name: str) -> str:
        """
        Canonicalize component name for matching/learning.

        Removes dimensions, numbers, stopwords, applies aliases (PL/DE/EN -> EN).

        Example: "Wspornik montażowy 400x300mm" -> "bracket mounting"

        Args:
            name: Raw component name

        Returns:
            Canonicalized name

        Raises:
            ParsingError: If normalization fails
        """
        ...
