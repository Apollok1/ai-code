"""
PDF Extractor - Extract text from PDF files.

Strategy:
1. Try pdfplumber (fast, direct text extraction)
2. If insufficient text → OCR fallback
3. Optionally use Vision for better quality
"""

import io
import time
import logging
from typing import BinaryIO

import pdfplumber
from pdf2image import convert_from_bytes

from domain.models.document import (
    ExtractionResult,
    Page,
    ExtractionMetadata,
    DocumentType
)
from domain.models.config import ExtractionConfig
from domain.interfaces.ocr_service import OCRService
from domain.interfaces.llm_client import VisionLLMClient
from domain.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    PDF document extractor with multiple strategies.

    Implements the Extractor protocol (structural typing).
    """

    def __init__(
        self,
        ocr_service: OCRService,
        vision_client: VisionLLMClient | None = None
    ):
        """
        Initialize PDF extractor.

        Args:
            ocr_service: OCR service for image-based PDFs
            vision_client: Optional vision model for enhanced extraction
        """
        self.ocr = ocr_service
        self.vision = vision_client
        logger.info(f"PDFExtractor initialized (vision={'enabled' if vision_client else 'disabled'})")

    @property
    def name(self) -> str:
        return "PDF Extractor"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.pdf', '.PDF')

    def can_handle(self, file_name: str) -> bool:
        """Check if file is a PDF"""
        return file_name.lower().endswith('.pdf')

    def extract(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Extract text from PDF using the best available strategy.

        Args:
            file: PDF file object
            file_name: Original file name
            config: Extraction configuration

        Returns:
            ExtractionResult with extracted pages

        Raises:
            ExtractionError: If extraction fails
        """
        start_time = time.time()
        file.seek(0)

        try:
            logger.info(f"Extracting PDF: {file_name}")

            # Strategy 1: Try direct text extraction with pdfplumber
            pages, text_quality = self._extract_with_pdfplumber(file, config)

            total_text = "".join(p.text for p in pages)
            extraction_method = "pdfplumber"

            # Strategy 2: OCR/Vision fallback for scanned PDFs
            if len(total_text.strip()) < config.min_text_length_for_ocr:
                logger.info(f"Low text quality ({len(total_text)} chars), using OCR/Vision fallback")
                pages = self._extract_with_ocr_or_vision(file, file_name, config)
                extraction_method = "vision" if (self.vision and config.use_vision) else "ocr"

            processing_time = time.time() - start_time

            # Build metadata
            metadata = ExtractionMetadata(
                document_type=DocumentType.PDF,
                pages_count=len(pages),
                extraction_method=extraction_method,
                processing_time_seconds=processing_time,
                file_size_bytes=self._get_file_size(file),
                vision_model=config.vision_model if extraction_method == "vision" else None,
                ocr_language=config.ocr_language if extraction_method == "ocr" else None
            )

            logger.info(
                f"PDF extraction complete: {file_name} | "
                f"{len(pages)} pages | {processing_time:.2f}s | {extraction_method}"
            )

            return ExtractionResult(
                file_name=file_name,
                pages=pages,
                metadata=metadata
            )

        except Exception as e:
            logger.exception(f"PDF extraction failed: {file_name}")
            raise ExtractionError(
                f"Failed to extract PDF: {e}",
                file_name=file_name
            ) from e

    def _extract_with_pdfplumber(
        self,
        file: BinaryIO,
        config: ExtractionConfig
    ) -> tuple[list[Page], str]:
        """
        Extract text using pdfplumber.

        Returns:
            Tuple of (pages, quality_indicator)
        """
        file.seek(0)
        pages = []

        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                if i >= config.max_pages:
                    logger.warning(f"Reached page limit ({config.max_pages}), stopping")
                    break

                text = page.extract_text() or ""
                pages.append(Page(number=i + 1, text=text))

            logger.debug(
                f"pdfplumber extracted {len(pages)}/{total_pages} pages, "
                f"total chars: {sum(len(p.text) for p in pages)}"
            )

        return pages, "good"

    def _extract_with_ocr_or_vision(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> list[Page]:
        """
        Fallback: Convert PDF to images and use OCR or Vision.

        Returns:
            List of extracted pages
        """
        file.seek(0)
        pdf_bytes = file.read()

        # Convert PDF pages to images
        logger.info(f"Converting PDF to images (DPI={config.ocr_dpi})")

        try:
            images = convert_from_bytes(
                pdf_bytes,
                fmt="jpeg",
                dpi=config.ocr_dpi,
                first_page=1,
                last_page=min(config.max_pages, 999)
            )
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            raise ExtractionError(
                f"Cannot convert PDF to images: {e}",
                file_name=file_name
            ) from e

        pages = []

        # Use Vision if available and enabled
        if self.vision and config.use_vision:
            logger.info(f"Processing {len(images)} pages with Vision model: {config.vision_model}")

            for i, img in enumerate(images):
                img_bytes = self._image_to_bytes(img)

                try:
                    text = self.vision.analyze_image(
                        img_bytes,
                        config.vision_prompt,
                        model=config.vision_model
                    )
                    pages.append(Page(number=i + 1, text=text))
                    logger.debug(f"Vision processed page {i+1}: {len(text)} chars")
                except Exception as e:
                    logger.warning(f"Vision failed for page {i+1}: {e}")
                    # Fallback to empty page or OCR
                    pages.append(Page(number=i + 1, text=f"[Vision error: {e}]"))

        else:
            # Fallback: OCR
            logger.info(f"Processing {len(images)} pages with OCR ({config.ocr_language})")

            for i, img in enumerate(images):
                img_bytes = self._image_to_bytes(img)

                try:
                    text = self.ocr.extract_text(
                        img_bytes,
                        language=config.ocr_language,
                        preprocess=True
                    )
                    pages.append(Page(number=i + 1, text=text))
                    logger.debug(f"OCR processed page {i+1}: {len(text)} chars")
                except Exception as e:
                    logger.warning(f"OCR failed for page {i+1}: {e}")
                    pages.append(Page(number=i + 1, text=f"[OCR error: {e}]"))

        return pages

    @staticmethod
    def _image_to_bytes(img) -> bytes:
        """Convert PIL Image to JPEG bytes"""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @staticmethod
    def _get_file_size(file: BinaryIO) -> int:
        """Get file size in bytes"""
        current_pos = file.tell()
        file.seek(0, 2)  # End of file
        size = file.tell()
        file.seek(current_pos)  # Restore position
        return size
