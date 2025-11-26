"""
Integration tests for extraction pipeline.

These tests verify that all components work together correctly.
"""

import pytest
import io
from unittest.mock import Mock, MagicMock

from src.domain.models.config import AppConfig, ExtractionConfig
from src.domain.models.document import ExtractionResult, DocumentType
from src.infrastructure.factory import create_pipeline, create_extractors
from src.application.pipeline import ExtractionPipeline


class TestPipelineIntegration:
    """Integration tests for extraction pipeline"""

    @pytest.fixture
    def mock_config(self):
        """Mock application config"""
        return AppConfig(
            ollama_url="http://localhost:11434",
            whisper_url="http://localhost:9000",
            pyannote_url="http://localhost:8000",
            max_workers=2
        )

    @pytest.fixture
    def extraction_config(self):
        """Extraction configuration"""
        return ExtractionConfig(
            use_vision=False,  # Disable for faster tests
            enable_diarization=False
        )

    def test_create_pipeline(self, mock_config):
        """Test pipeline creation"""
        pipeline = create_pipeline(
            mock_config,
            vision_enabled=False,
            audio_diarization_enabled=False
        )

        assert isinstance(pipeline, ExtractionPipeline)
        assert len(pipeline.extractors) == 6  # 6 extractors
        assert pipeline.max_workers == 2

    def test_pipeline_stats(self, mock_config):
        """Test pipeline statistics"""
        pipeline = create_pipeline(mock_config, vision_enabled=False)
        stats = pipeline.get_stats()

        assert stats["extractors_count"] == 6
        assert stats["max_workers"] == 2
        assert len(stats["supported_extensions"]) > 0
        assert ".pdf" in stats["supported_extensions"]
        assert ".docx" in stats["supported_extensions"]

    def test_pipeline_finds_extractor(self, mock_config):
        """Test that pipeline finds appropriate extractor"""
        pipeline = create_pipeline(mock_config, vision_enabled=False)

        # Test various file types
        assert pipeline._find_extractor("document.pdf") is not None
        assert pipeline._find_extractor("document.docx") is not None
        assert pipeline._find_extractor("presentation.pptx") is not None
        assert pipeline._find_extractor("image.jpg") is not None
        assert pipeline._find_extractor("audio.mp3") is not None
        assert pipeline._find_extractor("email.eml") is not None

        # Unknown format
        assert pipeline._find_extractor("unknown.xyz") is None

    def test_supported_extensions(self, mock_config):
        """Test listing supported extensions"""
        pipeline = create_pipeline(mock_config, vision_enabled=False)
        extensions = pipeline._list_supported_extensions()

        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".pptx" in extensions
        assert ".jpg" in extensions
        assert ".mp3" in extensions
        assert ".eml" in extensions


class TestExtractorsIntegration:
    """Test individual extractors"""

    def test_docx_extractor_created(self):
        """Test DOCX extractor can be created"""
        config = AppConfig()
        extractors = create_extractors(config, vision_enabled=False)

        docx_extractor = next(
            (e for e in extractors if e.name == "DOCX Extractor"),
            None
        )
        assert docx_extractor is not None
        assert docx_extractor.can_handle("test.docx")

    def test_pdf_extractor_created(self):
        """Test PDF extractor can be created"""
        config = AppConfig()
        extractors = create_extractors(config, vision_enabled=False)

        pdf_extractor = next(
            (e for e in extractors if e.name == "PDF Extractor"),
            None
        )
        assert pdf_extractor is not None
        assert pdf_extractor.can_handle("test.pdf")

    def test_all_extractors_have_unique_extensions(self):
        """Test that extractors don't overlap in file handling"""
        config = AppConfig()
        extractors = create_extractors(config, vision_enabled=False)

        # Check that each file type has exactly one handler
        test_files = [
            "doc.pdf", "doc.docx", "pres.pptx",
            "img.jpg", "audio.mp3", "mail.eml"
        ]

        for test_file in test_files:
            handlers = [e for e in extractors if e.can_handle(test_file)]
            assert len(handlers) == 1, f"{test_file} should have exactly 1 handler"


@pytest.mark.skip("Requires running services")
class TestEndToEnd:
    """End-to-end tests (require actual services)"""

    def test_process_sample_pdf(self):
        """Test processing a real PDF file"""
        # This would test with actual Ollama, Tesseract, etc.
        pass
