"""
Storage Protocol - Interface for result persistence.
"""

from typing import Protocol, Any
from domain.models.document import ExtractionResult


class Storage(Protocol):
    """Interface for storing and retrieving extraction results"""

    def save_result(self, result: ExtractionResult) -> str:
        """
        Save extraction result.

        Args:
            result: Extraction result to save

        Returns:
            Storage ID or path

        Raises:
            StorageError: If save fails
        """
        ...

    def load_result(self, result_id: str) -> ExtractionResult:
        """
        Load extraction result by ID.

        Args:
            result_id: Storage ID or path

        Returns:
            Loaded extraction result

        Raises:
            StorageError: If load fails
        """
        ...

    def list_results(self) -> list[tuple[str, Any]]:
        """
        List all stored results.

        Returns:
            List of (id, metadata) tuples
        """
        ...

    def delete_result(self, result_id: str) -> bool:
        """
        Delete result by ID.

        Args:
            result_id: Storage ID

        Returns:
            True if deleted, False if not found
        """
        ...

    def clear_all(self) -> int:
        """
        Clear all stored results.

        Returns:
            Number of deleted results
        """
        ...
