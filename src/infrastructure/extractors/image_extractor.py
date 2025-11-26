"""
Image Extractor - Extract text from images using OCR or Vision.

Strategy:
- OCR only
- Vision only (transcribe or describe)
- OCR + Vision (combined)
"""

import logging
from typing import BinaryIO, Literal
import base64

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


class ImageExtractor:
    """
    Image extractor with multiple strategies.

    Supports: OCR, Vision (transcribe/describe), OCR+Vision combined
    """

    def __init__(
        self,
        ocr_service: OCRService,
        vision_client: VisionLLMClient | None = None
    ):
        """
        Initialize image extractor.

        Args:
            ocr_service: OCR service for text extraction
            vision_client: Optional vision model
        """
        self.ocr = ocr_service
        self.vision = vision_client
        logger.info(f"ImageExtractor initialized (vision={'enabled' if vision_client else 'disabled'})")

    @property
    def name(self) -> str:
        return "Image Extractor"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG')

    def can_handle(self, file_name: str) -> bool:
        """Check if file is an image"""
        lower_name = file_name.lower()
        return any(lower_name.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'))

    def extract(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Extract text from image using configured method.

        Args:
            file: Image file object
            file_name: Original file name
            config: Extraction configuration (vision_mode determines strategy)

        Returns:
            ExtractionResult with extracted text

        Raises:
            ExtractionError: If extraction fails
        """
        import time
        start_time = time.time()

        try:
            file.seek(0)
            img_bytes = file.read()

            logger.info(f"Extracting image: {file_name} (mode: {config.vision_mode})")

            results = []
            extraction_method = config.vision_mode

            # Strategy 1: OCR
            if config.vision_mode in ("ocr", "ocr_plus_desc"):
                ocr_text = self.ocr.extract_text(img_bytes, config.ocr_language)
                results.append(f"=== OCR ===\n{ocr_text}")

            # Strategy 2: Vision
            if config.vision_mode in ("transcribe", "describe", "ocr_plus_desc"):
                if self.vision and config.use_vision:
                    img_b64 = base64.b64encode(img_bytes).decode()

                    # Choose prompt based on mode
                    if config.vision_mode == "transcribe":
                        prompt = config.vision_prompt  # Transcription prompt
                        tag = "Vision (transcription)"
                    else:
                        # Use describe prompt (should be in config, but we'll use vision_prompt as fallback)
                        prompt = config.vision_prompt
                        tag = "Vision (description)"

                    vision_text = self.vision.analyze_image(
                        img_bytes,
                        prompt,
                        model=config.vision_model
                    )
                    results.append(f"=== {tag} ===\n{vision_text}")
                else:
                    results.append("[Vision not available]")

            full_text = "\n\n".join(results).strip()

            processing_time = time.time() - start_time

            metadata = ExtractionMetadata(
                document_type=DocumentType.IMAGE,
                pages_count=1,
                extraction_method=extraction_method,
                processing_time_seconds=processing_time,
                file_size_bytes=len(img_bytes),
                vision_model=config.vision_model if (self.vision and config.use_vision) else None,
                ocr_language=config.ocr_language if config.vision_mode in ("ocr", "ocr_plus_desc") else None
            )

            logger.info(
                f"Image extraction complete: {file_name} | "
                f"{extraction_method} | {processing_time:.2f}s"
            )

            return ExtractionResult(
                file_name=file_name,
                pages=[Page(number=1, text=full_text)],
                metadata=metadata
            )

        except Exception as e:
            logger.exception(f"Image extraction failed: {file_name}")
            raise ExtractionError(
                f"Failed to extract image: {e}",
                file_name=file_name
            ) from e
