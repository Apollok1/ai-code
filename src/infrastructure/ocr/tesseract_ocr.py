"""
Tesseract OCR service with adaptive preprocessing.
"""

import logging
import io
import subprocess

from PIL import Image
import numpy as np
import cv2
import pytesseract

from domain.exceptions import OCRError

logger = logging.getLogger(__name__)


class TesseractOCR:
    """
    Tesseract OCR service with intelligent preprocessing.

    Implements OCRService protocol.
    """

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
            language: Tesseract language code
            preprocess: Apply adaptive preprocessing

        Returns:
            Extracted text

        Raises:
            OCRError: If OCR fails
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Adaptive preprocessing
            if preprocess:
                img = self._preprocess_adaptive(img)

            # Run OCR
            text = pytesseract.image_to_string(img, lang=language)
            return text or ""

        except Exception as e:
            logger.exception(f"OCR failed: {e}")
            raise OCRError(f"OCR processing failed: {e}") from e

    def get_available_languages(self) -> list[str]:
        """
        Get list of available Tesseract languages.

        Returns:
            List of language codes
        """
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=4
            )

            output = result.stdout or ""
            langs = [
                line.strip()
                for line in output.splitlines()
                if line.strip() and not line.lower().startswith("list of")
            ]
            return langs

        except Exception as e:
            logger.warning(f"Failed to get Tesseract languages: {e}")
            return []

    @staticmethod
    def _preprocess_adaptive(img: Image.Image) -> Image.Image:
        """
        Adaptive preprocessing based on image quality.

        High quality → no preprocessing
        Low quality → Otsu thresholding
        """
        # Convert to grayscale
        gray = img.convert('L')
        np_img = np.array(gray)

        # Check image variance (quality indicator)
        variance = np_img.var()

        # High quality image - no preprocessing needed
        if variance > 1000:
            logger.debug("High quality image detected, skipping preprocessing")
            return gray

        # Low quality - apply Otsu thresholding
        logger.debug("Low quality image detected, applying Otsu thresholding")
        _, binary = cv2.threshold(
            np_img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return Image.fromarray(binary)
