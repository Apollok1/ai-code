"""Document extractors - Implementations of Extractor protocol"""

from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor
from .pptx_extractor import PPTXExtractor
from .image_extractor import ImageExtractor
from .audio_extractor import AudioExtractor
from .email_extractor import EmailExtractor

__all__ = [
    "PDFExtractor",
    "DOCXExtractor",
    "PPTXExtractor",
    "ImageExtractor",
    "AudioExtractor",
    "EmailExtractor",
]
