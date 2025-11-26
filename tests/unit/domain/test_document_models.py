"""
Unit tests for domain models.

These tests verify business logic without any external dependencies.
"""

import pytest
from datetime import datetime

from src.domain.models.document import (
    DocumentType,
    Page,
    ExtractionMetadata,
    ExtractionResult
)


class TestDocumentType:
    """Test DocumentType enum"""

    def test_from_filename_pdf(self):
        assert DocumentType.from_filename("document.pdf") == DocumentType.PDF
        assert DocumentType.from_filename("DOCUMENT.PDF") == DocumentType.PDF

    def test_from_filename_docx(self):
        assert DocumentType.from_filename("file.docx") == DocumentType.DOCX

    def test_from_filename_image(self):
        assert DocumentType.from_filename("photo.jpg") == DocumentType.IMAGE
        assert DocumentType.from_filename("scan.png") == DocumentType.IMAGE

    def test_from_filename_audio(self):
        assert DocumentType.from_filename("meeting.mp3") == DocumentType.AUDIO
        assert DocumentType.from_filename("call.wav") == DocumentType.AUDIO

    def test_from_filename_unknown(self):
        assert DocumentType.from_filename("data.xyz") == DocumentType.UNKNOWN


class TestPage:
    """Test Page model"""

    def test_create_page(self):
        page = Page(number=1, text="Hello World")
        assert page.number == 1
        assert page.text == "Hello World"
        assert not page.is_empty()

    def test_page_number_validation(self):
        """Page numbers must be >= 1"""
        with pytest.raises(ValueError, match="Page number must be >= 1"):
            Page(number=0, text="Invalid")

    def test_empty_page(self):
        page = Page(number=1, text="   ")
        assert page.is_empty()

    def test_word_count(self):
        page = Page(number=1, text="Hello World from tests")
        assert page.word_count() == 4

    def test_char_count(self):
        page = Page(number=1, text="  Hello  ")
        assert page.char_count() == 5

    def test_preview(self):
        long_text = "A" * 200
        page = Page(number=1, text=long_text)
        preview = page.preview(max_chars=100)
        assert len(preview) == 103  # 100 + "..."
        assert preview.endswith("...")

    def test_page_immutability(self):
        """Pages should be immutable (frozen dataclass)"""
        page = Page(number=1, text="Original")
        with pytest.raises(Exception):  # FrozenInstanceError
            page.text = "Modified"


class TestExtractionMetadata:
    """Test ExtractionMetadata model"""

    def test_create_metadata(self):
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=10,
            extraction_method="pdfplumber",
            processing_time_seconds=5.5,
            file_size_bytes=1024000
        )
        assert metadata.document_type == DocumentType.PDF
        assert metadata.pages_count == 10
        assert not metadata.has_errors()

    def test_add_error(self):
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=1,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        metadata.add_error("Something went wrong")
        assert metadata.has_errors()
        assert len(metadata.errors) == 1

    def test_add_warning(self):
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=1,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        metadata.add_warning("Low quality")
        assert len(metadata.warnings) == 1
        assert not metadata.has_errors()  # warnings != errors

    def test_to_dict(self):
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=5,
            extraction_method="ocr",
            processing_time_seconds=10.0,
            file_size_bytes=500000,
            ocr_language="pol+eng"
        )
        data = metadata.to_dict()
        assert data["document_type"] == "pdf"
        assert data["pages_count"] == 5
        assert data["ocr_language"] == "pol+eng"


class TestExtractionResult:
    """Test ExtractionResult model"""

    def test_create_result(self):
        pages = [
            Page(number=1, text="Page 1 content"),
            Page(number=2, text="Page 2 content")
        ]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=2,
            extraction_method="pdfplumber",
            processing_time_seconds=2.0,
            file_size_bytes=10000
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        assert result.file_name == "test.pdf"
        assert len(result.pages) == 2
        assert result.is_successful()

    def test_full_text(self):
        pages = [
            Page(number=1, text="First"),
            Page(number=2, text="Second")
        ]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=2,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        assert result.full_text == "First\n\nSecond"

    def test_total_words(self):
        pages = [
            Page(number=1, text="Hello World"),
            Page(number=2, text="From Tests")
        ]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=2,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        assert result.total_words == 4

    def test_get_page(self):
        pages = [
            Page(number=1, text="First"),
            Page(number=2, text="Second")
        ]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=2,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        page = result.get_page(2)
        assert page is not None
        assert page.text == "Second"

        assert result.get_page(99) is None

    def test_filter_empty_pages(self):
        pages = [
            Page(number=1, text="Content"),
            Page(number=2, text="   "),  # Empty
            Page(number=3, text="More content")
        ]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=3,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        non_empty = result.filter_empty_pages()
        assert len(non_empty) == 2

    def test_unsuccessful_result_with_errors(self):
        """Result with errors should not be successful"""
        pages = [Page(number=1, text="Content")]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=1,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        metadata.add_error("Test error")

        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        assert not result.is_successful()

    def test_unsuccessful_result_without_pages(self):
        """Result without pages should not be successful"""
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=0,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=[],
            metadata=metadata
        )

        assert not result.is_successful()

    def test_to_dict(self):
        pages = [Page(number=1, text="Test")]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=1,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        data = result.to_dict()
        assert data["file_name"] == "test.pdf"
        assert len(data["pages"]) == 1
        assert "metadata" in data
        assert "extracted_at" in data

    def test_to_markdown(self):
        pages = [
            Page(number=1, text="First page"),
            Page(number=2, text="Second page")
        ]
        metadata = ExtractionMetadata(
            document_type=DocumentType.PDF,
            pages_count=2,
            extraction_method="test",
            processing_time_seconds=1.0,
            file_size_bytes=100
        )
        result = ExtractionResult(
            file_name="test.pdf",
            pages=pages,
            metadata=metadata
        )

        markdown = result.to_markdown()
        assert "# test.pdf" in markdown
        assert "## Page 1" in markdown
        assert "## Page 2" in markdown
        assert "First page" in markdown
        assert "Second page" in markdown
