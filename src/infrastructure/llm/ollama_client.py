"""
Ollama client - LLM and Vision model client.
"""

import logging
import base64
import requests

from domain.exceptions import ServiceError, VisionError
from .model_cache import ModelCache

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama API client with caching.

    Implements both LLMClient and VisionLLMClient protocols.
    """

    # Known vision model prefixes
    VISION_PREFIXES = frozenset([
        "llava", "bakllava", "moondream", "llava-phi",
        "qwen2-vl", "qwen2.5vl", "qwen",
        "cogvlm", "internvl", "minicpm-v"
    ])

    def __init__(self, base_url: str, cache_ttl_seconds: int = 300):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API URL (e.g., http://localhost:11434)
            cache_ttl_seconds: Model list cache TTL
        """
        self.base_url = base_url.rstrip('/')
        self.cache = ModelCache(ttl_seconds=cache_ttl_seconds)
        logger.info(f"OllamaClient initialized: {base_url}")

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        timeout: int = 120
    ) -> str:
        """
        Generate text response.

        Args:
            prompt: Input prompt
            model: Model name (None = use first available)
            json_mode: Request JSON-formatted response
            timeout: Request timeout

        Returns:
            Generated text

        Raises:
            ServiceError: If request fails
        """
        if model is None:
            models = self.list_models()
            if not models:
                raise ServiceError("No models available", service_name="Ollama")
            model = models[0]

        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }

            if json_mode:
                payload["format"] = "json"

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama text generation failed: {e}")
            raise ServiceError(
                f"Ollama request failed: {e}",
                service_name="Ollama",
                model=model
            ) from e

    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str | None = None,
        timeout: int = 120
    ) -> str:
        """
        Analyze image with vision model.

        Args:
            image_bytes: Image data
            prompt: Vision prompt
            model: Vision model name
            timeout: Request timeout

        Returns:
            Generated description

        Raises:
            VisionError: If vision analysis fails
        """
        if model is None:
            vision_models = self.list_vision_models()
            if not vision_models:
                raise VisionError("No vision models available")
            model = vision_models[0]

        try:
            # Encode image to base64
            img_b64 = base64.b64encode(image_bytes).decode('utf-8')

            payload = {
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama vision analysis failed: {e}")
            raise VisionError(
                f"Vision model request failed: {e}",
                model_name=model
            ) from e

    def list_models(self, force_refresh: bool = False) -> list[str]:
        """
        List available models (cached).

        Args:
            force_refresh: Force refresh cache

        Returns:
            List of model names
        """
        return self.cache.get(
            "all_models",
            fetcher=self._fetch_models,
            force_refresh=force_refresh
        )

    def list_vision_models(self, force_refresh: bool = False) -> list[str]:
        """
        List available vision models (cached).

        Args:
            force_refresh: Force refresh cache

        Returns:
            List of vision model names
        """
        def fetch_vision():
            all_models = self.list_models(force_refresh=force_refresh)
            return [
                m for m in all_models
                if any(m.startswith(prefix) for prefix in self.VISION_PREFIXES)
            ]

        return self.cache.get(
            "vision_models",
            fetcher=fetch_vision,
            force_refresh=force_refresh
        )

    def _fetch_models(self) -> list[str]:
        """Fetch models from Ollama API"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            logger.info(f"Fetched {len(models)} models from Ollama")
            return models

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch models from Ollama: {e}")
            return []

    def health_check(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.ok
        except Exception:
            return False
