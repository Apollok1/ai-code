"""
CAD Estimator Pro - Database Protocol Interface

Protocol-based interface for database operations (PEP 544).
"""
from typing import Protocol, ContextManager, Any
from cad.models import Project, ProjectVersion, ComponentPattern


class DatabaseClient(Protocol):
    """
    Protocol for database client implementations.

    Provides CRUD operations for projects, patterns, and versions.
    Uses Protocol (PEP 544) for structural subtyping (duck typing with type checking).
    """

    def get_connection(self) -> ContextManager[Any]:
        """
        Get database connection context manager.

        Returns:
            Context manager yielding connection object

        Raises:
            ConnectionError: If connection fails
        """
        ...

    def init_schema(self) -> bool:
        """
        Initialize database schema (tables, indexes, extensions).

        Returns:
            True if successful

        Raises:
            DatabaseError: If initialization fails
        """
        ...

    # Project operations
    def get_project(self, project_id: int) -> Project | None:
        """
        Get project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found

        Raises:
            DatabaseError: If query fails
        """
        ...

    def save_project(self, project: Project) -> int:
        """
        Save project (insert or update).

        Args:
            project: Project to save

        Returns:
            Project ID

        Raises:
            DatabaseError: If save fails
        """
        ...

    def delete_project(self, project_id: int) -> bool:
        """
        Delete project by ID.

        Args:
            project_id: Project ID

        Returns:
            True if deleted

        Raises:
            DatabaseError: If delete fails
        """
        ...

    def search_projects(
        self,
        query: str | None = None,
        department: str | None = None,
        is_historical: bool | None = None,
        limit: int = 100
    ) -> list[Project]:
        """
        Search projects with filters.

        Args:
            query: Text search query (full-text)
            department: Department filter (131-135)
            is_historical: Filter by historical flag
            limit: Max results

        Returns:
            List of matching projects

        Raises:
            DatabaseError: If search fails
        """
        ...

    # Project versions
    def save_project_version(self, version: ProjectVersion) -> int:
        """
        Save project version.

        Args:
            version: ProjectVersion to save

        Returns:
            Version ID

        Raises:
            DatabaseError: If save fails
        """
        ...

    def get_project_versions(self, project_id: int) -> list[ProjectVersion]:
        """
        Get all versions for a project.

        Args:
            project_id: Project ID

        Returns:
            List of versions (newest first)

        Raises:
            DatabaseError: If query fails
        """
        ...

    # Component patterns
    def get_pattern(self, pattern_key: str, department: str) -> ComponentPattern | None:
        """
        Get component pattern by key and department.

        Args:
            pattern_key: Canonicalized component name
            department: Department code (131-135)

        Returns:
            ComponentPattern or None

        Raises:
            DatabaseError: If query fails
        """
        ...

    def save_pattern(self, pattern: ComponentPattern) -> bool:
        """
        Save or update component pattern.

        Args:
            pattern: ComponentPattern to save

        Returns:
            True if successful

        Raises:
            DatabaseError: If save fails
        """
        ...

    def get_patterns_by_department(self, department: str, min_occurrences: int = 1) -> list[ComponentPattern]:
        """
        Get all patterns for a department.

        Args:
            department: Department code (131-135)
            min_occurrences: Min occurrences filter

        Returns:
            List of patterns

        Raises:
            DatabaseError: If query fails
        """
        ...

    def delete_patterns_with_low_confidence(self, min_confidence: float = 0.1) -> int:
        """
        Delete patterns with confidence below threshold.

        Args:
            min_confidence: Min confidence threshold

        Returns:
            Number of deleted patterns

        Raises:
            DatabaseError: If delete fails
        """
        ...
