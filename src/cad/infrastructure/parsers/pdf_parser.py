"""
CAD Estimator Pro - PDF Parser

Implementation of PDFParser protocol.
"""
import logging
from typing import BinaryIO
from PyPDF2 import PdfReader

from ...domain.exceptions import PDFParsingError

logger = logging.getLogger(__name__)


class CADPDFParser:
    """PDF parser for CAD specifications."""

    def extract_text(self, file: BinaryIO, max_pages: int = 200) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(file)
            text = ""

            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    text += f"\n[... {len(reader.pages)} pages total, processed {max_pages} ...]"
                    break
                text += (page.extract_text() or "") + "\n"

            logger.info(f"✅ Extracted {len(text)} chars from PDF ({min(len(reader.pages), max_pages)} pages)")
            return text

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}", exc_info=True)
            raise PDFParsingError(f"Failed to extract text from PDF: {e}")
