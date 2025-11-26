"""
Model list cache to avoid repeated HTTP requests.
"""

import logging
from datetime import datetime, timedelta
from typing import Callable

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Time-based cache for model lists.

    Reduces HTTP requests to Ollama by caching model list for TTL seconds.
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time to live for cache entries (default: 5 minutes)
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[datetime, list[str]]] = {}
        logger.debug(f"ModelCache initialized (TTL={ttl_seconds}s)")

    def get(
        self,
        key: str,
        fetcher: Callable[[], list[str]],
        force_refresh: bool = False
    ) -> list[str]:
        """
        Get cached models or fetch if expired.

        Args:
            key: Cache key (e.g., "all_models", "vision_models")
            fetcher: Function to fetch models if cache miss
            force_refresh: Force refresh even if cached

        Returns:
            List of model names
        """
        now = datetime.now()

        # Check cache
        if not force_refresh and key in self._cache:
            timestamp, models = self._cache[key]
            if now - timestamp < self.ttl:
                logger.debug(f"Cache HIT: {key} ({len(models)} models)")
                return models

        # Cache miss or expired - fetch
        logger.debug(f"Cache MISS: {key}, fetching...")
        models = fetcher()
        self._cache[key] = (now, models)
        logger.info(f"Cache updated: {key} ({len(models)} models)")
        return models

    def clear(self, key: str | None = None):
        """
        Clear cache.

        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            self._cache.pop(key, None)
            logger.debug(f"Cache cleared: {key}")
        else:
            self._cache.clear()
            logger.debug("Cache cleared: all")

    def __len__(self) -> int:
        """Number of cached entries"""
        return len(self._cache)
