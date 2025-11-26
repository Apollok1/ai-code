"""Domain interfaces (Ports) - Abstract protocols for external dependencies"""

from .extractor import Extractor
from .llm_client import LLMClient, VisionLLMClient
from .ocr_service import OCRService
from .storage import Storage

__all__ = [
    "Extractor",
    "LLMClient",
    "VisionLLMClient",
    "OCRService",
    "Storage",
]
