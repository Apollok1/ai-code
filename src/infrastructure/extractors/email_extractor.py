"""
Email Extractor - Extract text from email files (EML, MSG).

Strategy:
- EML: Use mailparser library
- MSG: Use extract-msg library
- Fallback: Raw text decode
"""

import logging
from typing import BinaryIO
import tempfile
import os

from domain.models.document import (
    ExtractionResult,
    Page,
    ExtractionMetadata,
    DocumentType
)
from domain.models.config import ExtractionConfig
from domain.exceptions import ExtractionError

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import mailparser
except ImportError:
    mailparser = None

try:
    import extract_msg
except ImportError:
    extract_msg = None


class EmailExtractor:
    """
    Email extractor for EML and MSG formats.

    Gracefully handles missing optional dependencies.
    """

    @property
    def name(self) -> str:
        return "Email Extractor"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.eml', '.msg', '.EML', '.MSG')

    def can_handle(self, file_name: str) -> bool:
        """Check if file is an email"""
        lower_name = file_name.lower()
        return lower_name.endswith('.eml') or lower_name.endswith('.msg')

    def extract(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Extract text from email.

        Args:
            file: Email file object
            file_name: Original file name
            config: Extraction configuration

        Returns:
            ExtractionResult with email content

        Raises:
            ExtractionError: If extraction fails
        """
        import time
        start_time = time.time()

        try:
            lower_name = file_name.lower()

            if lower_name.endswith('.eml'):
                text = self._extract_eml(file)
            elif lower_name.endswith('.msg'):
                text = self._extract_msg(file)
            else:
                raise ExtractionError(f"Unsupported email format: {file_name}")

            processing_time = time.time() - start_time

            metadata = ExtractionMetadata(
                document_type=DocumentType.EMAIL,
                pages_count=1,
                extraction_method="mailparser" if lower_name.endswith('.eml') else "extract-msg",
                processing_time_seconds=processing_time,
                file_size_bytes=self._get_file_size(file)
            )

            logger.info(
                f"Email extraction complete: {file_name} | {processing_time:.2f}s"
            )

            return ExtractionResult(
                file_name=file_name,
                pages=[Page(number=1, text=text)],
                metadata=metadata
            )

        except Exception as e:
            logger.exception(f"Email extraction failed: {file_name}")
            raise ExtractionError(
                f"Failed to extract email: {e}",
                file_name=file_name
            ) from e

    def _extract_eml(self, file: BinaryIO) -> str:
        """Extract text from EML file"""
        file.seek(0)
        raw = file.read()

        if mailparser is None:
            logger.warning("mailparser not installed, using fallback")
            return raw.decode("utf-8", errors="ignore")

        try:
            mail = mailparser.parse_from_bytes(raw)

            headers = [
                f"From: {mail.from_[0][1] if mail.from_ else ''}",
                f"To: {', '.join([x[1] for x in mail.to]) if mail.to else ''}",
                f"Subject: {mail.subject or ''}",
                f"Date: {mail.date.isoformat() if mail.date else ''}"
            ]

            body = (mail.text_plain[0] if mail.text_plain else mail.body) or ""

            attachments = [
                att.get('filename')
                for att in (mail.attachments or [])
                if att.get('filename')
            ]

            if attachments:
                headers.append(f"Attachments: {', '.join(attachments)}")

            return "\n".join(headers) + "\n\n" + body

        except Exception as e:
            logger.warning(f"mailparser failed: {e}, using fallback")
            return raw.decode("utf-8", errors="ignore")

    def _extract_msg(self, file: BinaryIO) -> str:
        """Extract text from MSG file"""
        file.seek(0)
        raw = file.read()

        if extract_msg is None:
            logger.warning("extract-msg not installed, using fallback")
            return raw.decode("utf-8", errors="ignore")

        try:
            # MSG requires file path, so write to temp file
            with tempfile.NamedTemporaryFile(suffix=".msg", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            msg = extract_msg.Message(tmp_path)

            headers = [
                f"From: {msg.sender or ''}",
                f"To: {msg.to or ''}",
                f"Subject: {msg.subject or ''}",
                f"Date: {msg.date or ''}"
            ]

            body = msg.body or ""

            # Cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            return "\n".join(headers) + "\n\n" + body

        except Exception as e:
            logger.warning(f"extract-msg failed: {e}, using fallback")
            return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _get_file_size(file: BinaryIO) -> int:
        """Get file size in bytes"""
        current_pos = file.tell()
        file.seek(0, 2)
        size = file.tell()
        file.seek(current_pos)
        return size
