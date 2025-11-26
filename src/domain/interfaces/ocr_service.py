"""
OCR Service Protocol - Interface for OCR engines.
"""

from typing import Protocol


class OCRService(Protocol):
    """Interface for OCR (Optical Character Recognition) services"""

    def extract_text(
        self,
        image_bytes: bytes,
        language: str = "pol+eng",
        preprocess: bool = True
    ) -> str:
        """
        Extract text from image using OCR.

        Args:
            image_bytes: Image data
            language: OCR language (Tesseract format)
            preprocess: Apply image preprocessing

        Returns:
            Extracted text

        Raises:
            OCRError: If OCR processing fails
        """
        ...

    def get_available_languages(self) -> list[str]:
        """
        Get list of available OCR languages.

        Returns:
            List of language codes
        """
        ...
