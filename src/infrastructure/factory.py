"""
Factory functions for easy initialization of infrastructure components.

Makes it dead simple to create fully configured pipeline.
"""

import logging
from domain.models.config import AppConfig, OCRConfig, VisionConfig, AudioConfig
from domain.interfaces.extractor import Extractor
from application.pipeline import ExtractionPipeline

# Infrastructure imports
from .llm.ollama_client import OllamaClient
from .ocr.tesseract_ocr import TesseractOCR
from .audio.whisper_client import WhisperASRClient
from .audio.pyannote_client import PyannoteClient

# Extractors
from .extractors.pdf_extractor import PDFExtractor
from .extractors.docx_extractor import DOCXExtractor
from .extractors.pptx_extractor import PPTXExtractor
from .extractors.image_extractor import ImageExtractor
from .extractors.audio_extractor import AudioExtractor
from .extractors.email_extractor import EmailExtractor

logger = logging.getLogger(__name__)


def create_ollama_client(config: AppConfig) -> OllamaClient:
    """
    Create Ollama client.

    Args:
        config: Application configuration

    Returns:
        Configured Ollama client
    """
    return OllamaClient(
        base_url=config.ollama_url,
        cache_ttl_seconds=config.model_cache_ttl_seconds
    )


def create_ocr_service() -> TesseractOCR:
    """
    Create OCR service.

    Returns:
        Tesseract OCR service
    """
    return TesseractOCR()


def create_whisper_client(config: AppConfig) -> WhisperASRClient:
    """
    Create Whisper ASR client.

    Args:
        config: Application configuration

    Returns:
        Whisper client
    """
    return WhisperASRClient(base_url=config.whisper_url)


def create_pyannote_client(config: AppConfig) -> PyannoteClient:
    """
    Create Pyannote diarization client.

    Args:
        config: Application configuration

    Returns:
        Pyannote client
    """
    return PyannoteClient(base_url=config.pyannote_url)


def create_extractors(
    config: AppConfig,
    vision_enabled: bool = True,
    audio_diarization_enabled: bool = True
) -> list[Extractor]:
    """
    Create all extractors with dependencies.

    Args:
        config: Application configuration
        vision_enabled: Enable vision models for PDF/PPTX/Image
        audio_diarization_enabled: Enable speaker diarization

    Returns:
        List of configured extractors
    """
    # Create services
    ollama = create_ollama_client(config)
    ocr = create_ocr_service()
    whisper = create_whisper_client(config)
    pyannote = create_pyannote_client(config) if audio_diarization_enabled else None

    # Create extractors
    extractors: list[Extractor] = [
        PDFExtractor(
            ocr_service=ocr,
            vision_client=ollama if vision_enabled else None
        ),
        DOCXExtractor(),
        PPTXExtractor(
            vision_client=ollama if vision_enabled else None
        ),
        ImageExtractor(
            ocr_service=ocr,
            vision_client=ollama if vision_enabled else None
        ),
        AudioExtractor(
            whisper_client=whisper,
            diarization_client=pyannote
        ),
        EmailExtractor(),
    ]

    logger.info(
        f"Created {len(extractors)} extractors "
        f"(vision={vision_enabled}, diarization={audio_diarization_enabled})"
    )

    return extractors


def create_pipeline(
    config: AppConfig,
    vision_enabled: bool = True,
    audio_diarization_enabled: bool = True
) -> ExtractionPipeline:
    """
    Create fully configured extraction pipeline.

    This is the main factory function - one call to set up everything!

    Args:
        config: Application configuration
        vision_enabled: Enable vision models
        audio_diarization_enabled: Enable speaker diarization

    Returns:
        Ready-to-use extraction pipeline

    Example:
        >>> config = AppConfig.from_env()
        >>> pipeline = create_pipeline(config)
        >>> result = pipeline.process_single(file, "doc.pdf", extraction_config)
    """
    extractors = create_extractors(
        config,
        vision_enabled=vision_enabled,
        audio_diarization_enabled=audio_diarization_enabled
    )

    pipeline = ExtractionPipeline(
        extractors=extractors,
        max_workers=config.max_workers
    )

    logger.info("✓ Extraction pipeline ready")
    return pipeline


# Convenience function for quick setup
def quick_pipeline(
    ollama_url: str = "http://localhost:11434",
    whisper_url: str = "http://localhost:9000",
    pyannote_url: str = "http://localhost:8000",
    max_workers: int = 4
) -> ExtractionPipeline:
    """
    Quick pipeline setup with default configuration.

    Args:
        ollama_url: Ollama API URL
        whisper_url: Whisper API URL
        pyannote_url: Pyannote API URL
        max_workers: Max parallel workers

    Returns:
        Configured pipeline

    Example:
        >>> pipeline = quick_pipeline()
        >>> results = pipeline.process_batch(files, config)
    """
    config = AppConfig(
        ollama_url=ollama_url,
        whisper_url=whisper_url,
        pyannote_url=pyannote_url,
        max_workers=max_workers
    )

    return create_pipeline(config)
