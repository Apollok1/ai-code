"""
Configuration models using Pydantic for validation.

These models ensure type safety and validation for all configuration.
"""

from dataclasses import dataclass, field
from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
import os


# === PYDANTIC MODELS FOR VALIDATION ===

class OCRConfig(BaseModel):
    """OCR-specific configuration"""
    model_config = ConfigDict(frozen=True)  # Immutable

    language: str = Field(default="pol+eng", description="Tesseract language")
    dpi: int = Field(default=150, ge=72, le=600, description="Image DPI for OCR")
    min_text_length: int = Field(default=100, ge=0, description="Min text before OCR fallback")
    max_pages: int = Field(default=20, ge=1, le=1000, description="Max pages to OCR")

    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate OCR language format"""
        if not v or not all(c.isalpha() or c == '+' for c in v):
            raise ValueError(f"Invalid OCR language format: {v}")
        return v


class VisionConfig(BaseModel):
    """Vision model configuration"""
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable vision models")
    model_name: str = Field(default="qwen2.5vl:7b", description="Vision model to use")
    transcribe_prompt: str = Field(
        default="Przepisz DOKŁADNIE cały tekst z obrazu. Zachowaj pisownię, układ, symbole.",
        description="Prompt for text transcription"
    )
    describe_prompt: str = Field(
        default="Przeprowadź szczegółową analizę techniczną tego obrazu po polsku.",
        description="Prompt for image description"
    )
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    web_enhancement: bool = Field(default=False, description="Use web search to enhance vision")


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    model_config = ConfigDict(frozen=True)

    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")
    enable_summarization: bool = Field(default=True, description="Enable meeting summaries")
    summary_model: str = Field(default="qwen2.5:7b", description="Model for summarization")
    chunk_size_chars: int = Field(default=6000, ge=1000, le=10000, description="Chunk size for summarization")
    whisper_timeout_base: int = Field(default=240, description="Base timeout for Whisper")
    whisper_timeout_per_mb: int = Field(default=25, description="Additional timeout per MB")


class AppConfig(BaseSettings):
    """
    Main application configuration loaded from environment variables.

    Uses Pydantic Settings for automatic env loading.
    """
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Service URLs
    ollama_url: str = Field(default="http://127.0.0.1:11434", description="Ollama API URL")
    whisper_url: str = Field(default="http://127.0.0.1:9000", description="Whisper ASR URL")
    pyannote_url: str = Field(default="http://127.0.0.1:8000", description="Pyannote diarization URL")
    anythingllm_url: str = Field(default="", description="AnythingLLM URL (optional)")
    anythingllm_api_key: str = Field(default="", description="AnythingLLM API key")

    # Security
    offline_mode: bool = Field(default=True, description="Block non-local network requests")
    allow_web_search: bool = Field(default=False, description="Allow web lookup")

    # Performance
    max_workers: int = Field(default=4, ge=1, le=16, description="Max parallel workers")
    model_cache_ttl_seconds: int = Field(default=300, ge=60, le=3600, description="Model list cache TTL")

    # Models
    default_text_model: str = Field(default="qwen2.5:7b", description="Default text model")
    embed_model: str = Field(default="nomic-embed-text", description="Embedding model")

    # Storage
    output_directory: str = Field(default="outputs", description="Output directory for results")

    @field_validator('ollama_url', 'whisper_url', 'pyannote_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format"""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {v}")
        return v.rstrip('/')

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables"""
        return cls()


# === DATACLASS FOR RUNTIME EXTRACTION CONFIG ===

@dataclass
class ExtractionConfig:
    """
    Runtime configuration for extraction process.

    Mutable dataclass used during extraction.
    """
    # OCR settings
    ocr_language: str = "pol+eng"
    ocr_dpi: int = 150
    min_text_length_for_ocr: int = 100
    max_pages: int = 20

    # Vision settings
    use_vision: bool = True
    vision_model: str = "qwen2.5vl:7b"
    vision_mode: Literal["transcribe", "describe", "ocr", "ocr_plus_desc"] = "describe"
    vision_prompt: str = "Przepisz DOKŁADNIE cały tekst z obrazu."

    # Audio settings
    enable_diarization: bool = True
    enable_summarization: bool = True
    summary_model: str = "qwen2.5:7b"
    chunk_size: int = 6000

    # General
    file_size_bytes: int = 0

    @classmethod
    def from_app_config(
        cls,
        app_config: AppConfig,
        ocr_config: OCRConfig,
        vision_config: VisionConfig,
        audio_config: AudioConfig
    ) -> "ExtractionConfig":
        """Create ExtractionConfig from validated Pydantic configs"""
        return cls(
            # OCR
            ocr_language=ocr_config.language,
            ocr_dpi=ocr_config.dpi,
            min_text_length_for_ocr=ocr_config.min_text_length,
            max_pages=ocr_config.max_pages,
            # Vision
            use_vision=vision_config.enabled,
            vision_model=vision_config.model_name,
            vision_prompt=vision_config.transcribe_prompt,
            # Audio
            enable_diarization=audio_config.enable_diarization,
            enable_summarization=audio_config.enable_summarization,
            summary_model=audio_config.summary_model,
            chunk_size=audio_config.chunk_size_chars,
        )

    def calculate_timeout(self, base: int = 240, per_mb: int = 25) -> int:
        """Calculate timeout based on file size"""
        if self.file_size_bytes == 0:
            return base
        size_mb = max(1.0, self.file_size_bytes / 1024 / 1024)
        return int(max(base, size_mb * per_mb))
