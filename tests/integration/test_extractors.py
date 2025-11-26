"""
Integration tests for extractors with mocked services.
"""

import pytest
import io
from unittest.mock import Mock

from src.domain.models.config import ExtractionConfig
from src.domain.models.document import DocumentType, ExtractionResult
from src.infrastructure.extractors.docx_extractor import DOCXExtractor
from src.infrastructure.extractors.email_extractor import EmailExtractor


class TestDOCXExtractorIntegration:
    """Test DOCX extractor"""

    def test_docx_extractor_properties(self):
        """Test extractor properties"""
        extractor = DOCXExtractor()

        assert extractor.name == "DOCX Extractor"
        assert ".docx" in extractor.supported_extensions
        assert extractor.can_handle("test.docx")
        assert not extractor.can_handle("test.pdf")


class TestEmailExtractorIntegration:
    """Test email extractor"""

    def test_email_extractor_properties(self):
        """Test extractor properties"""
        extractor = EmailExtractor()

        assert extractor.name == "Email Extractor"
        assert ".eml" in extractor.supported_extensions
        assert ".msg" in extractor.supported_extensions
        assert extractor.can_handle("test.eml")
        assert extractor.can_handle("test.msg")
        assert not extractor.can_handle("test.txt")
