"""
CAD Estimator Pro - Ollama AI Client

Implementation of AIClient, VisionAIClient, and EmbeddingClient protocols for Ollama.
"""
import logging
import requests
from typing import Any
from datetime import datetime, timedelta

from cad.domain.models.config import OllamaConfig
from cad.domain.exceptions import AIGenerationError, AIResponseParsingError, EmbeddingError

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Time-based cache for Ollama models list.

    Reduces HTTP requests by caching model list with TTL.
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize ModelCache.

        Args:
            ttl_seconds: Time-to-live for cached data (default: 300s = 5min)
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[datetime, list[str]]] = {}

    def get(self, key: str, fetcher: callable, force_refresh: bool = False) -> list[str]:
        """
        Get cached data or fetch if expired.

        Args:
            key: Cache key
            fetcher: Function to fetch data if cache miss
            force_refresh: Force refresh even if cached

        Returns:
            Cached or freshly fetched data
        """
        now = datetime.now()

        if not force_refresh and key in self._cache:
            timestamp, data = self._cache[key]
            if now - timestamp < self.ttl:
                logger.debug(f"Cache HIT for '{key}' (age: {(now - timestamp).seconds}s)")
                return data

        logger.debug(f"Cache MISS for '{key}' - fetching...")
        data = fetcher()
        self._cache[key] = (now, data)
        return data

    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.debug("ModelCache cleared")


class OllamaClient:
    """
    Ollama AI client implementation.

    Implements AIClient, VisionAIClient, and EmbeddingClient protocols.
    Provides text generation, vision analysis, and embedding generation.
    """

    def __init__(self, config: OllamaConfig):
        """
        Initialize OllamaClient.

        Args:
            config: Ollama configuration

        Raises:
            AIGenerationError: If Ollama service is unreachable
        """
        self.config = config
        self.base_url = config.url.rstrip('/')
        self.cache = ModelCache(ttl_seconds=config.cache_ttl_seconds)

        # Test connection
        try:
            self.list_available_models()
            logger.info(f"✅ OllamaClient initialized (url={self.base_url})")
        except Exception as e:
            logger.warning(f"⚠️ Ollama service unreachable at {self.base_url}: {e}")
            # Don't raise - allow operation in degraded mode

    # AIClient implementation

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        timeout: int = 120
    ) -> str:
        """
        Generate text response from AI model.

        Args:
            prompt: Input prompt
            model: Model name (None = use default from config)
            json_mode: Enable JSON output format
            timeout: Request timeout in seconds

        Returns:
            Generated text response

        Raises:
            AIGenerationError: If generation fails
        """
        model = model or self.config.text_model

        if not self.is_model_available(model):
            logger.warning(f"Model '{model}' not available, falling back to default")
            model = self.config.text_model

        logger.info(f"🤖 Generating text with {model} (json={json_mode}, len={len(prompt)})")

        try:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 8192
                }
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
            text = data.get("response", "")

            if not text:
                raise AIGenerationError(
                    f"Empty response from Ollama",
                    model=model,
                    prompt_length=len(prompt)
                )

            logger.info(f"✅ Generated {len(text)} chars")
            return text

        except requests.exceptions.Timeout:
            raise AIGenerationError(
                f"Ollama request timed out after {timeout}s",
                model=model,
                prompt_length=len(prompt)
            )
        except requests.exceptions.RequestException as e:
            raise AIGenerationError(
                f"Ollama request failed: {e}",
                model=model,
                prompt_length=len(prompt)
            )
        except Exception as e:
            logger.error(f"Unexpected error in generate_text: {e}", exc_info=True)
            raise AIGenerationError(
                f"Text generation failed: {e}",
                model=model,
                prompt_length=len(prompt)
            )

    def list_available_models(self) -> list[str]:
        """
        List available AI models.

        Returns:
            List of model names

        Raises:
            AIGenerationError: If listing fails
        """
        def _fetch_models() -> list[str]:
            try:
                response = requests.get(
                    f"{self.base_url}/api/tags",
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.debug(f"Found {len(models)} Ollama models")
                return models

            except Exception as e:
                logger.error(f"Failed to list models: {e}", exc_info=True)
                raise AIGenerationError(f"Failed to list Ollama models: {e}")

        return self.cache.get("models_list", _fetch_models)

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if model is available.

        Args:
            model_name: Model name to check

        Returns:
            True if available
        """
        try:
            models = self.list_available_models()
            return model_name in models
        except Exception:
            return False

    # VisionAIClient implementation

    def analyze_image(
        self,
        prompt: str,
        images_base64: list[str],
        model: str | None = None,
        json_mode: bool = False,
        timeout: int = 120
    ) -> str:
        """
        Analyze images with AI vision model.

        Args:
            prompt: Analysis prompt
            images_base64: List of base64-encoded images
            model: Vision model name (None = use default from config)
            json_mode: Enable JSON output format
            timeout: Request timeout in seconds

        Returns:
            Analysis result text

        Raises:
            AIGenerationError: If analysis fails
        """
        model = model or self.config.vision_model

        if not model:
            raise AIGenerationError("No vision model configured", model=None)

        if not self.is_model_available(model):
            raise AIGenerationError(f"Vision model '{model}' not available", model=model)

        logger.info(f"🖼️ Analyzing {len(images_base64)} images with {model} (json={json_mode})")

        try:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "images": images_base64,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 8192
                }
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
            text = data.get("response", "")

            if not text:
                raise AIGenerationError(
                    f"Empty response from vision model",
                    model=model,
                    prompt_length=len(prompt)
                )

            logger.info(f"✅ Vision analysis: {len(text)} chars")
            return text

        except requests.exceptions.Timeout:
            raise AIGenerationError(
                f"Vision request timed out after {timeout}s",
                model=model,
                prompt_length=len(prompt)
            )
        except requests.exceptions.RequestException as e:
            raise AIGenerationError(
                f"Vision request failed: {e}",
                model=model,
                prompt_length=len(prompt)
            )
        except Exception as e:
            logger.error(f"Unexpected error in analyze_image: {e}", exc_info=True)
            raise AIGenerationError(
                f"Vision analysis failed: {e}",
                model=model,
                prompt_length=len(prompt)
            )

    def is_vision_available(self) -> bool:
        """
        Check if vision capability is available.

        Returns:
            True if vision models are available
        """
        if not self.config.vision_model:
            return False

        return self.is_model_available(self.config.vision_model)

    # EmbeddingClient implementation

    def generate_embedding(
        self,
        text: str,
        model: str | None = None,
        timeout: int = 30
    ) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text
            model: Embedding model name (None = use default from config)
            timeout: Request timeout in seconds

        Returns:
            Embedding vector (list of floats)

        Raises:
            EmbeddingError: If generation fails
        """
        model = model or self.config.embed_model

        logger.debug(f"📊 Generating embedding with {model} (len={len(text)})")

        try:
            payload = {
                "model": model,
                "prompt": text
            }

            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()
            embedding = data.get("embedding", [])

            if not embedding:
                raise EmbeddingError(
                    f"Empty embedding from Ollama",
                    text=text,
                    model=model
                )

            logger.debug(f"✅ Generated embedding dim={len(embedding)}")
            return embedding

        except requests.exceptions.Timeout:
            raise EmbeddingError(
                f"Embedding request timed out after {timeout}s",
                text=text,
                model=model
            )
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(
                f"Embedding request failed: {e}",
                text=text,
                model=model
            )
        except Exception as e:
            logger.error(f"Unexpected error in generate_embedding: {e}", exc_info=True)
            raise EmbeddingError(
                f"Embedding generation failed: {e}",
                text=text,
                model=model
            )

    def get_embedding_dimension(self, model: str | None = None) -> int:
        """
        Get embedding vector dimension for model.

        Args:
            model: Model name (None = use default from config)

        Returns:
            Embedding dimension (e.g., 768)

        Raises:
            EmbeddingError: If dimension cannot be determined
        """
        model = model or self.config.embed_model

        try:
            # Generate test embedding to determine dimension
            embedding = self.generate_embedding("test", model=model)
            dim = len(embedding)
            logger.info(f"📐 Embedding dimension for '{model}': {dim}")
            return dim

        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}", exc_info=True)
            # Fallback to config
            return self.config.embed_dim
