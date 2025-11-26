"""
CAD Estimator Pro - AI Client Protocol Interface

Protocol for AI service clients (text and vision models).
"""
from typing import Protocol


class AIClient(Protocol):
    """
    Protocol for AI text model clients.

    Provides text generation capabilities.
    """

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
            model: Model name (None = use default)
            json_mode: Enable JSON output format
            timeout: Request timeout in seconds

        Returns:
            Generated text response

        Raises:
            AIGenerationError: If generation fails
        """
        ...

    def list_available_models(self) -> list[str]:
        """
        List available AI models.

        Returns:
            List of model names

        Raises:
            AIGenerationError: If listing fails
        """
        ...

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if model is available.

        Args:
            model_name: Model name to check

        Returns:
            True if available
        """
        ...


class VisionAIClient(Protocol):
    """
    Protocol for AI vision model clients.

    Provides image analysis capabilities.
    """

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
            model: Vision model name (None = use default)
            json_mode: Enable JSON output format
            timeout: Request timeout in seconds

        Returns:
            Analysis result text

        Raises:
            AIGenerationError: If analysis fails
        """
        ...

    def is_vision_available(self) -> bool:
        """
        Check if vision capability is available.

        Returns:
            True if vision models are available
        """
        ...


class EmbeddingClient(Protocol):
    """
    Protocol for embedding generation clients.

    Provides text embedding capabilities for semantic search.
    """

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
            model: Embedding model name (None = use default)
            timeout: Request timeout in seconds

        Returns:
            Embedding vector (list of floats)

        Raises:
            EmbeddingError: If generation fails
        """
        ...

    def get_embedding_dimension(self, model: str | None = None) -> int:
        """
        Get embedding vector dimension for model.

        Args:
            model: Model name (None = use default)

        Returns:
            Embedding dimension (e.g., 768)

        Raises:
            EmbeddingError: If dimension cannot be determined
        """
        ...
