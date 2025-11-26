"""
LLM Client Protocols - Interfaces for Language Model clients.
"""

from typing import Protocol


class LLMClient(Protocol):
    """Interface for text-based LLM clients (Ollama, OpenAI, etc.)"""

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        timeout: int = 120
    ) -> str:
        """
        Generate text response from LLM.

        Args:
            prompt: Input prompt
            model: Model name (None = use default)
            json_mode: Request JSON-formatted response
            timeout: Request timeout in seconds

        Returns:
            Generated text

        Raises:
            ServiceError: If LLM request fails
        """
        ...

    def list_models(self) -> list[str]:
        """
        List available models.

        Returns:
            List of model names

        Raises:
            ServiceError: If cannot fetch models
        """
        ...


class VisionLLMClient(Protocol):
    """Interface for vision-capable LLM clients"""

    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str | None = None,
        timeout: int = 120
    ) -> str:
        """
        Analyze image and generate description.

        Args:
            image_bytes: Image data (JPEG, PNG, etc.)
            prompt: Vision prompt
            model: Vision model name
            timeout: Request timeout

        Returns:
            Generated description

        Raises:
            VisionError: If vision analysis fails
        """
        ...

    def list_vision_models(self) -> list[str]:
        """
        List available vision models.

        Returns:
            List of vision model names
        """
        ...
