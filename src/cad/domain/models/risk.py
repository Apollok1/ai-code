"""
CAD Estimator Pro - Risk Domain Model

Represents project risks with impact level and mitigation strategies.
"""
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    """Risk impact level."""

    HIGH = "wysoki"
    MEDIUM = "średni"
    LOW = "niski"
    UNKNOWN = "nieznany"


@dataclass(frozen=True)
class Risk:
    """
    Project risk with description, impact level, and mitigation strategy.

    Immutable value object.
    """

    risk: str
    impact: RiskLevel
    mitigation: str

    def __post_init__(self):
        if not self.risk or not self.risk.strip():
            raise ValueError("Risk description cannot be empty")
        if not self.mitigation or not self.mitigation.strip():
            raise ValueError("Mitigation strategy cannot be empty")

    def to_dict(self) -> dict[str, str]:
        """Convert to dict (for JSON serialization)."""
        return {
            "risk": self.risk,
            "impact": self.impact.value,
            "mitigation": self.mitigation
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Risk":
        """
        Create Risk from dict (from JSON/AI response).

        Args:
            data: Dict with keys: risk, impact, mitigation

        Returns:
            Risk instance

        Raises:
            ValueError: If required fields are missing
        """
        risk_text = data.get("risk", "").strip()
        mitigation_text = data.get("mitigation", "").strip()

        if not risk_text:
            raise ValueError("Risk dict must have 'risk' field")
        if not mitigation_text:
            raise ValueError("Risk dict must have 'mitigation' field")

        # Parse impact level
        impact_str = data.get("impact", "nieznany").lower()
        if "wys" in impact_str or "high" in impact_str:
            impact = RiskLevel.HIGH
        elif "śred" in impact_str or "med" in impact_str:
            impact = RiskLevel.MEDIUM
        elif "nis" in impact_str or "low" in impact_str or "nisk" in impact_str:
            impact = RiskLevel.LOW
        else:
            impact = RiskLevel.UNKNOWN

        return cls(risk=risk_text, impact=impact, mitigation=mitigation_text)
