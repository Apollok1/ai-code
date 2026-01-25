"""
Ollama client for text summarization and analysis.
"""
import logging
import requests
from typing import Optional

from app.config import OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def generate_text(prompt: str, model: Optional[str] = None, timeout: int = 120) -> str:
    """
    Generate text using Ollama.

    Args:
        prompt: The prompt to send
        model: Model to use (defaults to OLLAMA_MODEL)
        timeout: Request timeout in seconds

    Returns:
        Generated text

    Raises:
        Exception: If generation fails
    """
    model = model or OLLAMA_MODEL

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        raise


def summarize_text(text: str, language: str = "pl") -> str:
    """
    Summarize text using Ollama.

    Args:
        text: Text to summarize
        language: Output language (pl/en)

    Returns:
        Summary text
    """
    if language == "pl":
        prompt = f"""Przygotuj zwięzłe podsumowanie poniższego tekstu w języku polskim.
Wyodrębnij kluczowe punkty i najważniejsze informacje.

TEKST:
{text}

PODSUMOWANIE:"""
    else:
        prompt = f"""Prepare a concise summary of the following text.
Extract key points and most important information.

TEXT:
{text}

SUMMARY:"""

    return generate_text(prompt)


def analyze_document(text: str, analysis_type: str = "summary") -> str:
    """
    Analyze document content.

    Args:
        text: Document text
        analysis_type: Type of analysis (summary, key_points, action_items)

    Returns:
        Analysis result
    """
    prompts = {
        "summary": f"""Przygotuj zwięzłe podsumowanie dokumentu:

{text}

PODSUMOWANIE:""",

        "key_points": f"""Wyodrębnij kluczowe punkty z dokumentu jako listę:

{text}

KLUCZOWE PUNKTY:""",

        "action_items": f"""Wyodrębnij zadania do wykonania (action items) z dokumentu:

{text}

ZADANIA:"""
    }

    prompt = prompts.get(analysis_type, prompts["summary"])
    return generate_text(prompt)
