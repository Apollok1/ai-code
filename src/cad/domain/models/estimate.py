"""
CAD Estimator Pro - Estimate Domain Model

Represents complete project estimation with components, risks, and suggestions.
"""
from dataclasses import dataclass, field
from .component import Component
from .risk import Risk
from .suggestion import Suggestion


@dataclass(frozen=True)
class EstimatePhases:
    """
    Breakdown of hours by project phase.

    Represents 3D Layout, 3D Detail, and 2D Documentation phases.
    """

    layout: float  # 3D Layout hours
    detail: float  # 3D Detail hours
    documentation: float  # 2D Documentation hours

    def __post_init__(self):
        if self.layout < 0:
            raise ValueError(f"Layout hours cannot be negative: {self.layout}")
        if self.detail < 0:
            raise ValueError(f"Detail hours cannot be negative: {self.detail}")
        if self.documentation < 0:
            raise ValueError(f"Documentation hours cannot be negative: {self.documentation}")

    @property
    def total(self) -> float:
        """Total hours across all phases."""
        return self.layout + self.detail + self.documentation

    def to_dict(self) -> dict[str, float]:
        """Convert to dict."""
        return {
            "layout": self.layout,
            "detail": self.detail,
            "documentation": self.documentation,
            "total": self.total
        }


@dataclass(frozen=True)
class Estimate:
    """
    Complete project estimation.

    Aggregates components, phases, risks, suggestions, and metadata.
    Immutable aggregate root for estimation domain.
    """

    components: tuple[Component, ...]
    phases: EstimatePhases
    overall_confidence: float
    risks: tuple[Risk, ...] = field(default_factory=tuple)
    suggestions: tuple[Suggestion, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    raw_ai_response: str = ""

    def __post_init__(self):
        if not self.components:
            raise ValueError("Estimate must have at least one component")
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(f"Overall confidence must be 0.0-1.0, got {self.overall_confidence}")

        # Ensure tuples (immutable)
        if not isinstance(self.components, tuple):
            object.__setattr__(self, "components", tuple(self.components))
        if not isinstance(self.risks, tuple):
            object.__setattr__(self, "risks", tuple(self.risks))
        if not isinstance(self.suggestions, tuple):
            object.__setattr__(self, "suggestions", tuple(self.suggestions))
        if not isinstance(self.assumptions, tuple):
            object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def total_hours(self) -> float:
        """Total estimated hours."""
        return self.phases.total

    @property
    def non_summary_components(self) -> list[Component]:
        """Components excluding summaries (assemblies)."""
        return [c for c in self.components if not c.is_summary]

    @property
    def component_count(self) -> int:
        """Number of non-summary components."""
        return len(self.non_summary_components)

    @property
    def confidence_level(self) -> str:
        """Human-readable confidence level: HIGH/MEDIUM/LOW."""
        if self.overall_confidence > 0.7:
            return "HIGH"
        elif self.overall_confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def accuracy_estimate(self) -> str:
        """Estimated accuracy range based on confidence."""
        if self.overall_confidence > 0.7:
            return "±10%"
        elif self.overall_confidence > 0.4:
            return "±20%"
        else:
            return "±40%"

    @classmethod
    def from_components(
        cls,
        components: list[Component],
        risks: list[Risk] | None = None,
        suggestions: list[Suggestion] | None = None,
        assumptions: list[str] | None = None,
        warnings: list[str] | None = None,
        raw_ai_response: str = ""
    ) -> "Estimate":
        """
        Create Estimate from components list.

        Automatically calculates phases and overall confidence from components.

        Args:
            components: List of Component objects
            risks: Optional list of risks
            suggestions: Optional list of suggestions
            assumptions: Optional list of assumptions
            warnings: Optional list of warnings
            raw_ai_response: Raw AI response text

        Returns:
            Estimate instance

        Raises:
            ValueError: If components list is empty
        """
        if not components:
            raise ValueError("Cannot create estimate from empty components list")

        # Filter non-summary components
        non_summary = [c for c in components if not c.is_summary]

        # Calculate phases
        layout = sum(c.hours_3d_layout for c in non_summary)
        detail = sum(c.hours_3d_detail for c in non_summary)
        doc = sum(c.hours_2d for c in non_summary)

        phases = EstimatePhases(layout=layout, detail=detail, documentation=doc)

        # Calculate overall confidence (weighted by hours)
        total_hours = sum(c.total_hours for c in non_summary)
        if total_hours > 0:
            weighted_conf = sum(c.confidence * c.total_hours for c in non_summary)
            overall_confidence = weighted_conf / total_hours
        else:
            # Fallback: simple average
            overall_confidence = sum(c.confidence for c in non_summary) / len(non_summary)

        return cls(
            components=tuple(components),
            phases=phases,
            overall_confidence=overall_confidence,
            risks=tuple(risks or []),
            suggestions=tuple(suggestions or []),
            assumptions=tuple(assumptions or []),
            warnings=tuple(warnings or []),
            raw_ai_response=raw_ai_response
        )

    def to_dict(self) -> dict:
        """Convert to dict (for JSON serialization)."""
        return {
            "components": [c.to_dict() for c in self.components],
            "phases": self.phases.to_dict(),
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level,
            "accuracy_estimate": self.accuracy_estimate,
            "component_count": self.component_count,
            "risks": [r.to_dict() for r in self.risks],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "raw_ai_response": self.raw_ai_response
        }
