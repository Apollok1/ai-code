"""
Text extraction from documents and images.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_from_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted text
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pdfplumber not installed, trying PyPDF2")
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text())
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError("Install pdfplumber or PyPDF2: pip install pdfplumber")


def extract_from_docx(file_path: Path) -> str:
    """
    Extract text from DOCX file.

    Args:
        file_path: Path to DOCX file

    Returns:
        Extracted text
    """
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")


def extract_from_image(file_path: Path, language: str = "pol+eng") -> str:
    """
    Extract text from image using OCR.

    Args:
        file_path: Path to image file
        language: Tesseract language code

    Returns:
        Extracted text
    """
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except ImportError:
        raise ImportError("Install pytesseract and Pillow: pip install pytesseract Pillow")


def extract_from_txt(file_path: Path) -> str:
    """Extract text from plain text file."""
    encodings = ["utf-8", "cp1250", "iso-8859-2", "latin-1"]
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode file: {file_path}")


def extract_text(file_path: Path) -> str:
    """
    Extract text from file based on extension.

    Args:
        file_path: Path to file

    Returns:
        Extracted text

    Raises:
        ValueError: If file type not supported
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext in {".docx", ".doc"}:
        return extract_from_docx(file_path)
    elif ext in {".txt", ".rtf"}:
        return extract_from_txt(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
        return extract_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
