"""
CAD Estimator Pro - Component Domain Model

Represents CAD components with hours estimation and metadata.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SubComponent:
    """
    Sub-component reference (from Excel comments).

    Example: "2x docisk" -> quantity=2, name="docisk"
    """

    name: str
    quantity: int = 1

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("SubComponent name cannot be empty")
        if self.quantity < 1:
            raise ValueError(f"SubComponent quantity must be >= 1, got {self.quantity}")


@dataclass(frozen=True)
class Component:
    """
    CAD Component with hours breakdown and confidence score.

    Immutable value object representing a single CAD component or assembly.
    """

    name: str
    hours_3d_layout: float  # 3D Layout hours
    hours_3d_detail: float  # 3D Detail hours
    hours_2d: float  # 2D Documentation hours
    confidence: float = 0.5  # Confidence score (0.0-1.0)
    confidence_reason: str = ""  # Why this confidence level
    is_summary: bool = False  # True for assembly-level summaries
    category: str = ""  # Component category (modelowanie, spawanie, etc.)
    comment: str = ""  # Additional notes
    subcomponents: tuple[SubComponent, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Component name cannot be empty")
        if self.hours_3d_layout < 0:
            raise ValueError(f"Layout hours cannot be negative: {self.hours_3d_layout}")
        if self.hours_3d_detail < 0:
            raise ValueError(f"Detail hours cannot be negative: {self.hours_3d_detail}")
        if self.hours_2d < 0:
            raise ValueError(f"2D hours cannot be negative: {self.hours_2d}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

        # Ensure subcomponents is tuple (immutable)
        if not isinstance(self.subcomponents, tuple):
            object.__setattr__(self, "subcomponents", tuple(self.subcomponents))

    @property
    def total_hours(self) -> float:
        """Total hours (layout + detail + 2D)."""
        return self.hours_3d_layout + self.hours_3d_detail + self.hours_2d

    @property
    def confidence_level(self) -> str:
        """Human-readable confidence level: HIGH/MEDIUM/LOW."""
        if self.confidence > 0.7:
            return "HIGH"
        elif self.confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def accuracy_estimate(self) -> str:
        """Estimated accuracy range based on confidence."""
        if self.confidence > 0.7:
            return "±10%"
        elif self.confidence > 0.4:
            return "±20%"
        else:
            return "±40%"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (for JSON serialization)."""
        return {
            "name": self.name,
            "hours_3d_layout": self.hours_3d_layout,
            "hours_3d_detail": self.hours_3d_detail,
            "hours_2d": self.hours_2d,
            "hours": self.total_hours,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "is_summary": self.is_summary,
            "category": self.category,
            "comment": self.comment,
            "subcomponents": [
                {"name": sub.name, "quantity": sub.quantity}
                for sub in self.subcomponents
            ],
            **self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Component":
        """
        Create Component from dict (from JSON/database).

        Args:
            data: Dict with component data

        Returns:
            Component instance

        Raises:
            ValueError: If required fields are missing
        """
        name = data.get("name", "").strip()
        if not name:
            raise ValueError("Component dict must have 'name' field")

        # Hours (with fallbacks)
        layout = float(data.get("hours_3d_layout", 0.0))
        detail = float(data.get("hours_3d_detail", 0.0))
        doc = float(data.get("hours_2d", 0.0))

        # Confidence
        confidence = float(data.get("confidence", 0.5))
        confidence_reason = data.get("confidence_reason", "")

        # Flags
        is_summary = bool(data.get("is_summary", False))
        category = data.get("category", "")
        comment = data.get("comment", "")

        # Subcomponents
        subs_data = data.get("subcomponents", [])
        subcomponents = []
        for sub in subs_data:
            if isinstance(sub, dict):
                subcomponents.append(SubComponent(
                    name=sub.get("name", ""),
                    quantity=int(sub.get("quantity", 1))
                ))

        # Metadata (extra fields)
        metadata = {}
        ignore_keys = {
            "name", "hours_3d_layout", "hours_3d_detail", "hours_2d", "hours",
            "confidence", "confidence_reason", "is_summary", "category",
            "comment", "subcomponents"
        }
        for key, value in data.items():
            if key not in ignore_keys:
                metadata[key] = value

        return cls(
            name=name,
            hours_3d_layout=layout,
            hours_3d_detail=detail,
            hours_2d=doc,
            confidence=confidence,
            confidence_reason=confidence_reason,
            is_summary=is_summary,
            category=category,
            comment=comment,
            subcomponents=tuple(subcomponents),
            metadata=metadata
        )


@dataclass(frozen=True)
class ComponentPattern:
    """
    Learned pattern for a component type (stored in database).

    Represents historical data for a canonical component name.
    Uses full Welford's algorithm with M2 (sum of squared deviations) for variance tracking.
    """

    name: str
    pattern_key: str  # Canonicalized name (for matching)
    department_code: str  # Department (131-135)
    avg_hours_layout: float
    avg_hours_detail: float
    avg_hours_doc: float

    # Welford's M2 (sum of squared deviations) for variance tracking
    m2_hours_layout: float = 0.0
    m2_hours_detail: float = 0.0
    m2_hours_doc: float = 0.0

    confidence: float = 0.3
    occurrences: int = 1  # Number of times seen
    source: str = "actual"  # Source: 'actual', 'historical_excel', etc.

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Pattern name cannot be empty")
        if not self.pattern_key or not self.pattern_key.strip():
            raise ValueError("Pattern key cannot be empty")
        if self.occurrences < 1:
            raise ValueError(f"Pattern occurrences must be >= 1, got {self.occurrences}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

    @property
    def total_hours(self) -> float:
        """Total average hours."""
        return self.avg_hours_layout + self.avg_hours_detail + self.avg_hours_doc

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict (for database)."""
        return {
            "name": self.name,
            "pattern_key": self.pattern_key,
            "department": self.department_code,
            "avg_hours_3d_layout": self.avg_hours_layout,
            "avg_hours_3d_detail": self.avg_hours_detail,
            "avg_hours_2d": self.avg_hours_doc,
            "m2_hours_layout": self.m2_hours_layout,
            "m2_hours_detail": self.m2_hours_detail,
            "m2_hours_doc": self.m2_hours_doc,
            "avg_hours_total": self.total_hours,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "source": self.source
        }
