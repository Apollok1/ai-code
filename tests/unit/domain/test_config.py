"""
Unit tests for configuration models.
"""

import pytest
from pydantic import ValidationError

from src.domain.models.config import (
    OCRConfig,
    VisionConfig,
    AudioConfig,
    AppConfig,
    ExtractionConfig
)


class TestOCRConfig:
    """Test OCR configuration"""

    def test_default_config(self):
        config = OCRConfig()
        assert config.language == "pol+eng"
        assert config.dpi == 150
        assert config.min_text_length == 100

    def test_custom_config(self):
        config = OCRConfig(
            language="eng",
            dpi=300,
            max_pages=50
        )
        assert config.language == "eng"
        assert config.dpi == 300
        assert config.max_pages == 50

    def test_invalid_dpi(self):
        """DPI must be between 72 and 600"""
        with pytest.raises(ValidationError):
            OCRConfig(dpi=50)  # Too low

        with pytest.raises(ValidationError):
            OCRConfig(dpi=1000)  # Too high

    def test_config_immutable(self):
        """OCR config should be immutable"""
        config = OCRConfig()
        with pytest.raises(Exception):  # Pydantic frozen
            config.dpi = 300


class TestVisionConfig:
    """Test Vision configuration"""

    def test_default_config(self):
        config = VisionConfig()
        assert config.enabled is True
        assert config.model_name == "qwen2.5vl:7b"
        assert config.timeout_seconds == 120

    def test_custom_prompts(self):
        config = VisionConfig(
            transcribe_prompt="Custom transcribe",
            describe_prompt="Custom describe"
        )
        assert config.transcribe_prompt == "Custom transcribe"
        assert config.describe_prompt == "Custom describe"

    def test_timeout_validation(self):
        """Timeout must be between 10 and 600"""
        with pytest.raises(ValidationError):
            VisionConfig(timeout_seconds=5)  # Too low

        with pytest.raises(ValidationError):
            VisionConfig(timeout_seconds=1000)  # Too high


class TestAudioConfig:
    """Test Audio configuration"""

    def test_default_config(self):
        config = AudioConfig()
        assert config.enable_diarization is True
        assert config.enable_summarization is True
        assert config.chunk_size_chars == 6000

    def test_chunk_size_validation(self):
        """Chunk size must be between 1000 and 10000"""
        with pytest.raises(ValidationError):
            AudioConfig(chunk_size_chars=500)

        with pytest.raises(ValidationError):
            AudioConfig(chunk_size_chars=20000)


class TestAppConfig:
    """Test main application configuration"""

    def test_default_config(self):
        config = AppConfig()
        assert config.ollama_url == "http://127.0.0.1:11434"
        assert config.offline_mode is True
        assert config.max_workers == 4

    def test_url_validation(self):
        """URLs must start with http:// or https://"""
        with pytest.raises(ValidationError):
            AppConfig(ollama_url="invalid-url")

    def test_url_trailing_slash_removed(self):
        """Trailing slashes should be removed from URLs"""
        config = AppConfig(ollama_url="http://localhost:11434/")
        assert config.ollama_url == "http://localhost:11434"

    def test_workers_validation(self):
        """Workers must be between 1 and 16"""
        with pytest.raises(ValidationError):
            AppConfig(max_workers=0)

        with pytest.raises(ValidationError):
            AppConfig(max_workers=20)


class TestExtractionConfig:
    """Test extraction runtime configuration"""

    def test_default_config(self):
        config = ExtractionConfig()
        assert config.ocr_language == "pol+eng"
        assert config.use_vision is True
        assert config.enable_diarization is True

    def test_from_app_config(self):
        """Test creating ExtractionConfig from validated configs"""
        app_config = AppConfig()
        ocr_config = OCRConfig(language="eng", dpi=300)
        vision_config = VisionConfig(enabled=False)
        audio_config = AudioConfig(enable_diarization=False)

        extraction_config = ExtractionConfig.from_app_config(
            app_config,
            ocr_config,
            vision_config,
            audio_config
        )

        assert extraction_config.ocr_language == "eng"
        assert extraction_config.ocr_dpi == 300
        assert extraction_config.use_vision is False
        assert extraction_config.enable_diarization is False

    def test_calculate_timeout(self):
        """Test timeout calculation based on file size"""
        config = ExtractionConfig()

        # Small file
        config.file_size_bytes = 0
        assert config.calculate_timeout(base=240) == 240

        # 10 MB file
        config.file_size_bytes = 10 * 1024 * 1024
        timeout = config.calculate_timeout(base=240, per_mb=25)
        assert timeout == max(240, 10 * 25)

    def test_config_mutable(self):
        """ExtractionConfig should be mutable (dataclass, not frozen)"""
        config = ExtractionConfig()
        config.ocr_dpi = 300  # Should not raise
        assert config.ocr_dpi == 300
