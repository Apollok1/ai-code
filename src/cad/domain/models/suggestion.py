"""
CAD Estimator Pro - Suggestion Domain Model

Represents AI suggestions for alternatives, improvements, and warnings.
"""
from dataclasses import dataclass, field
from enum import Enum


class SuggestionType(str, Enum):
    """Type of suggestion."""

    ALTERNATIVE = "alternative"  # Inny sposób realizacji
    IMPROVEMENT = "improvement"  # Ulepszenie/optymalizacja
    WARNING = "warning"  # Ostrzeżenie/ryzyko
    OTHER = "other"


class SuggestionPriority(str, Enum):
    """Suggestion priority level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class SuggestionImpact:
    """
    Impact of applying a suggestion.

    Represents changes in hours, cost, and quality.
    """

    hours_delta: float = 0.0  # Change in hours (+/-)
    cost_delta: float = 0.0  # Change in cost PLN (+/-)
    quality_info: str = ""  # Quality impact description

    def __post_init__(self):
        # Validate hours_delta
        object.__setattr__(self, "hours_delta", float(self.hours_delta))
        object.__setattr__(self, "cost_delta", float(self.cost_delta))


@dataclass(frozen=True)
class Suggestion:
    """
    AI-generated suggestion for project optimization.

    Immutable value object.
    """

    type: SuggestionType
    title: str
    description: str
    priority: SuggestionPriority
    impact: SuggestionImpact = field(default_factory=SuggestionImpact)
    components_to_add: list[str] = field(default_factory=list)
    components_to_remove: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.title or not self.title.strip():
            raise ValueError("Suggestion title cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Suggestion description cannot be empty")
        if len(self.title) > 100:
            raise ValueError(f"Suggestion title too long: {len(self.title)} chars (max 100)")

    def to_dict(self) -> dict:
        """Convert to dict (for JSON serialization)."""
        return {
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "impact": {
                "hours_delta": self.impact.hours_delta,
                "cost_delta": self.impact.cost_delta,
                "quality_info": self.impact.quality_info
            },
            "components_to_add": list(self.components_to_add),
            "components_to_remove": list(self.components_to_remove)
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Suggestion":
        """
        Create Suggestion from dict (from JSON/AI response).

        Args:
            data: Dict with keys: type, title, description, priority, impact, etc.

        Returns:
            Suggestion instance

        Raises:
            ValueError: If required fields are missing
        """
        title = data.get("title", "").strip()
        description = data.get("description", "").strip()

        if not title:
            raise ValueError("Suggestion dict must have 'title' field")
        if not description:
            raise ValueError("Suggestion dict must have 'description' field")

        # Parse type
        type_str = data.get("type", "other").lower()
        try:
            sug_type = SuggestionType(type_str)
        except ValueError:
            sug_type = SuggestionType.OTHER

        # Parse priority
        priority_str = data.get("priority", "medium").lower()
        try:
            priority = SuggestionPriority(priority_str)
        except ValueError:
            priority = SuggestionPriority.MEDIUM

        # Parse impact
        impact_data = data.get("impact", {})
        if isinstance(impact_data, dict):
            impact = SuggestionImpact(
                hours_delta=float(impact_data.get("hours_delta", 0.0)),
                cost_delta=float(impact_data.get("cost_delta", 0.0)),
                quality_info=impact_data.get("quality_info", "")
            )
        else:
            impact = SuggestionImpact()

        return cls(
            type=sug_type,
            title=title,
            description=description,
            priority=priority,
            impact=impact,
            components_to_add=data.get("components_to_add", []),
            components_to_remove=data.get("components_to_remove", [])
        )
