"""
CAD Estimator Pro - Estimation Feedback Model

Domain model for collecting actual vs estimated hours feedback.
Used for continuous model improvement and calibration.
"""
from dataclasses import dataclass
from .estimate import EstimatePhases


@dataclass(frozen=True)
class EstimationFeedback:
    """
    Feedback for a single component estimation.

    Compares AI-estimated hours with actual hours spent.
    Used for model retraining and accuracy metrics.
    """

    # Identification
    component_name: str
    component_category: str | None = None
    department_code: str | None = None

    # Estimated (what AI predicted)
    estimated_hours: EstimatePhases

    # Actual (what really happened)
    actual_hours: EstimatePhases | None = None

    # Model metadata
    model_used: str | None = None
    complexity_level: str | None = None
    complexity_multiplier: float = 1.0
    pattern_matched: bool = False
    estimated_confidence: float = 0.5

    # Notes from user
    notes: str | None = None

    def __post_init__(self):
        """Validate feedback data."""
        if not (0.0 <= self.estimated_confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.estimated_confidence}")

        if self.complexity_multiplier < 0:
            raise ValueError(f"Complexity multiplier must be >= 0, got {self.complexity_multiplier}")

    @property
    def has_actuals(self) -> bool:
        """Check if actual hours have been filled in."""
        return self.actual_hours is not None

    @property
    def mae(self) -> float | None:
        """
        Mean Absolute Error (MAE) across all phases.

        Returns None if actuals not available.
        """
        if not self.has_actuals:
            return None

        return (
            abs(self.estimated_hours.layout - self.actual_hours.layout) +
            abs(self.estimated_hours.detail - self.actual_hours.detail) +
            abs(self.estimated_hours.documentation - self.actual_hours.documentation)
        ) / 3.0

    @property
    def error_percentage(self) -> float | None:
        """
        Overall error percentage.

        Formula: |estimated - actual| / actual * 100

        Returns None if actuals not available or actual is zero.
        """
        if not self.has_actuals:
            return None

        actual_total = self.actual_hours.total
        if actual_total == 0:
            return None

        estimated_total = self.estimated_hours.total
        return abs(estimated_total - actual_total) / actual_total * 100.0

    @property
    def accuracy(self) -> float | None:
        """
        Accuracy percentage (inverse of error).

        Returns: 100% - error_percentage
        Returns None if error cannot be calculated.
        """
        if self.error_percentage is None:
            return None

        return 100.0 - self.error_percentage

    @property
    def should_retrain(self) -> bool:
        """
        Should this feedback trigger model retraining?

        Criteria: Error > 25%
        """
        if not self.has_actuals:
            return False

        return self.error_percentage is not None and self.error_percentage > 25.0

    @property
    def is_high_quality_example(self) -> bool:
        """
        Is this a high-quality example for few-shot learning?

        Criteria: Error < 10% (accuracy > 90%)
        """
        if not self.has_actuals:
            return False

        return self.error_percentage is not None and self.error_percentage < 10.0

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "component_name": self.component_name,
            "component_category": self.component_category,
            "department_code": self.department_code,

            # Estimated
            "estimated_hours_3d_layout": self.estimated_hours.layout,
            "estimated_hours_3d_detail": self.estimated_hours.detail,
            "estimated_hours_2d": self.estimated_hours.documentation,
            "estimated_confidence": self.estimated_confidence,

            # Actual (may be None)
            "actual_hours_3d_layout": self.actual_hours.layout if self.has_actuals else None,
            "actual_hours_3d_detail": self.actual_hours.detail if self.has_actuals else None,
            "actual_hours_2d": self.actual_hours.documentation if self.has_actuals else None,

            # Metadata
            "model_used": self.model_used,
            "complexity_level": self.complexity_level,
            "complexity_multiplier": self.complexity_multiplier,
            "pattern_matched": self.pattern_matched,

            # Notes
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EstimationFeedback":
        """Create from dictionary (e.g., from database)."""
        estimated_hours = EstimatePhases(
            layout=data["estimated_hours_3d_layout"],
            detail=data["estimated_hours_3d_detail"],
            documentation=data["estimated_hours_2d"]
        )

        actual_hours = None
        if data.get("actual_hours_3d_layout") is not None:
            actual_hours = EstimatePhases(
                layout=data["actual_hours_3d_layout"],
                detail=data["actual_hours_3d_detail"],
                documentation=data["actual_hours_2d"]
            )

        return cls(
            component_name=data["component_name"],
            component_category=data.get("component_category"),
            department_code=data.get("department_code"),
            estimated_hours=estimated_hours,
            actual_hours=actual_hours,
            model_used=data.get("model_used"),
            complexity_level=data.get("complexity_level"),
            complexity_multiplier=data.get("complexity_multiplier", 1.0),
            pattern_matched=data.get("pattern_matched", False),
            estimated_confidence=data.get("estimated_confidence", 0.5),
            notes=data.get("notes")
        )

    def __repr__(self) -> str:
        """String representation."""
        status = "complete" if self.has_actuals else "pending"
        accuracy_str = f", accuracy={self.accuracy:.1f}%" if self.has_actuals else ""

        return (
            f"EstimationFeedback("
            f"component='{self.component_name}', "
            f"status={status}, "
            f"estimated={self.estimated_hours.total:.1f}h"
            f"{accuracy_str})"
        )
