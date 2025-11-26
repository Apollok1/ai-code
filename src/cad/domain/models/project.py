"""
CAD Estimator Pro - Project Domain Model

Represents complete CAD project with estimation and versions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from .department import DepartmentCode
from .estimate import Estimate


@dataclass(frozen=True)
class Project:
    """
    CAD Project with estimation and metadata.

    Aggregate root for project domain.
    """

    id: int | None  # Database ID (None for new projects)
    name: str
    department: DepartmentCode
    estimate: Estimate
    description: str = ""
    client: str = ""
    cad_system: str = ""
    actual_hours: float | None = None  # Real hours (after completion)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_historical: bool = False  # True for imported historical projects
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Project name cannot be empty")
        if self.actual_hours is not None and self.actual_hours < 0:
            raise ValueError(f"Actual hours cannot be negative: {self.actual_hours}")

    @property
    def estimated_hours(self) -> float:
        """Total estimated hours."""
        return self.estimate.total_hours

    @property
    def accuracy(self) -> float | None:
        """
        Estimation accuracy (if actual_hours provided).

        Returns:
            Accuracy as a value between 0.0 and 1.0 (1.0 = perfect estimate)
            None if actual_hours not set
        """
        if self.actual_hours is None or self.estimated_hours == 0:
            return None

        error = abs(self.estimated_hours - self.actual_hours)
        return 1.0 - (error / self.estimated_hours)

    @property
    def accuracy_percentage(self) -> float | None:
        """Accuracy as percentage (e.g., 85.5%)."""
        acc = self.accuracy
        return acc * 100 if acc is not None else None

    def with_actual_hours(self, actual_hours: float) -> "Project":
        """
        Create new Project instance with actual hours set.

        Args:
            actual_hours: Real hours spent on project

        Returns:
            New Project instance with actual_hours
        """
        return Project(
            id=self.id,
            name=self.name,
            department=self.department,
            estimate=self.estimate,
            description=self.description,
            client=self.client,
            cad_system=self.cad_system,
            actual_hours=actual_hours,
            created_at=self.created_at,
            updated_at=datetime.now(),
            is_historical=self.is_historical,
            metadata=self.metadata
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (for database/JSON)."""
        return {
            "id": self.id,
            "name": self.name,
            "department": self.department.value,
            "description": self.description,
            "client": self.client,
            "cad_system": self.cad_system,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "accuracy": self.accuracy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_historical": self.is_historical,
            "estimate": self.estimate.to_dict(),
            **self.metadata
        }


@dataclass(frozen=True)
class ProjectVersion:
    """
    Project version snapshot (for versioning/history).

    Immutable snapshot of project state at a point in time.
    """

    id: int | None
    project_id: int
    version: str  # e.g., "v1.0", "v1.1", "v2.0"
    estimate: Estimate
    change_description: str = ""
    changed_by: str = "System"
    is_approved: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.version or not self.version.strip():
            raise ValueError("Version string cannot be empty")
        if self.project_id < 1:
            raise ValueError(f"Project ID must be >= 1, got {self.project_id}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (for database)."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "version": self.version,
            "estimated_hours": self.estimate.total_hours,
            "estimated_hours_3d_layout": self.estimate.phases.layout,
            "estimated_hours_3d_detail": self.estimate.phases.detail,
            "estimated_hours_2d": self.estimate.phases.documentation,
            "change_description": self.change_description,
            "changed_by": self.changed_by,
            "is_approved": self.is_approved,
            "created_at": self.created_at.isoformat(),
            "components": [c.to_dict() for c in self.estimate.components]
        }
