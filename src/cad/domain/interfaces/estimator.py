"""
CAD Estimator Pro - Estimator Protocol Interface

Protocol for estimation engines.
"""
from typing import Protocol
from cad.domain.models import Estimate, Component, DepartmentCode


class Estimator(Protocol):
    """
    Protocol for estimation engines.

    Provides component hours estimation from various sources (AI, patterns, heuristics).
    """

    def estimate(
        self,
        description: str,
        department: DepartmentCode,
        components_hint: list[Component] | None = None,
        pdf_text: str = "",
        images_base64: list[str] | None = None
    ) -> Estimate:
        """
        Generate project estimate.

        Args:
            description: Project description
            department: Department code
            components_hint: Optional component hints (from Excel/JSON)
            pdf_text: Optional PDF specification text
            images_base64: Optional images (for vision models)

        Returns:
            Estimate with components, risks, suggestions

        Raises:
            AIGenerationError: If estimation fails
        """
        ...

    def suggest_adjustments(
        self,
        components: list[Component],
        department: DepartmentCode,
        conservativeness: float = 1.0
    ) -> list[dict]:
        """
        Suggest additional components based on sub-components and patterns.

        Args:
            components: Existing components
            department: Department code
            conservativeness: Adjustment factor (0.5-1.5)

        Returns:
            List of adjustment proposals:
            [
                {
                    "parent": "Component name",
                    "adds": [
                        {
                            "name": "Sub-component",
                            "qty": 2,
                            "layout_add": 1.0,
                            "detail_add": 3.0,
                            "doc_add": 1.0,
                            "reason": "Pattern match",
                            "confidence": 0.8
                        }
                    ]
                }
            ]

        Raises:
            PatternLearningError: If pattern lookup fails
        """
        ...
