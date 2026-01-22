"""
CAD Estimator Pro - Stage 4: Risk Analysis & Optimization

Identifies risks and suggests optimizations for the estimate.
"""
import json
import logging
from typing import Any

from cad.domain.models.multi_model import StageContext
from cad.domain.models.estimate import Risk
from cad.domain.models.config import MultiModelConfig
from cad.domain.interfaces.ai_client import AIClient
from cad.domain.exceptions import AIGenerationError, ValidationError

logger = logging.getLogger(__name__)


class RiskOptimizationStage:
    """
    Stage 4: Risk Analysis & Optimization.

    Critically analyzes the complete estimate:
    - Identifies technical risks
    - Suggests optimizations
    - Documents assumptions made
    - Provides warnings about uncertainties
    """

    def __init__(self, ai_client: AIClient, config: MultiModelConfig):
        """
        Initialize risk optimization stage.

        Args:
            ai_client: AI client for generation
            config: Multi-model configuration
        """
        self.ai_client = ai_client
        self.config = config

    def analyze_risks(
        self,
        context: StageContext,
        model: str | None = None,
    ) -> tuple[list[Risk], list[str], list[str], list[str]]:
        """
        Analyze risks and generate suggestions.

        Args:
            context: Pipeline context (must have estimated_components)
            model: Optional model override

        Returns:
            Tuple of (risks, suggestions, assumptions, warnings)

        Raises:
            AIGenerationError: If analysis fails
            ValidationError: If estimated_components missing
        """
        if not context.estimated_components:
            raise ValidationError("Stage 4 requires estimated_components from Stage 3")

        model_to_use = model or self.config.stage4_model

        logger.info(f"Stage 4: Risk & Optimization Analysis (model: {model_to_use})")

        # Build comprehensive summary for analysis
        prompt = self._build_risk_analysis_prompt(context)

        try:
            # Generate risk analysis
            response = self.ai_client.generate_text(
                prompt=prompt,
                model=model_to_use,
                json_mode=True,
            )

            # Parse JSON response
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    analysis_data = json.loads(response[start_idx:end_idx])
                else:
                    raise AIGenerationError(
                        f"Invalid JSON response from model: {response[:200]}"
                    )

            # Parse risks
            risks: list[Risk] = []
            for risk_data in analysis_data.get("risks", []):
                # Map from AI response fields to Risk model fields
                # 'description' -> 'risk', 'category' is ignored (not used in Risk model)
                risk_dict = {
                    "risk": risk_data.get("description", "") or risk_data.get("risk", ""),
                    "impact": risk_data.get("impact", "medium"),
                    "mitigation": risk_data.get("mitigation", ""),
                }
                try:
                    risk = Risk.from_dict(risk_dict)
                    risks.append(risk)
                except (ValueError, KeyError) as e:
                    # Skip invalid risks
                    self._report_progress(
                        context, f"⚠️ Skipping invalid risk: {str(e)}", stage=4
                    )
                    continue

            # Extract other outputs (obsługujemy oba klucze: suggestions / optimization_suggestions)
            suggestions = (
                analysis_data.get("suggestions")
                or analysis_data.get("optimization_suggestions", [])
            )
            assumptions = analysis_data.get("assumptions", [])
            warnings = analysis_data.get("warnings", [])

            return risks, suggestions, assumptions, warnings

        except Exception as e:
            logger.error(f"Risk analysis failed: {e}", exc_info=True)
            raise AIGenerationError(f"Stage 4 failed: {e}")

    def _build_risk_analysis_prompt(self, context: StageContext) -> str:
        """
        Zbuduj prompt dla Stage 4 na podstawie:
        - opisu projektu
        - technical_analysis (Stage 1)
        - struktury komponentów (Stage 2)
        - wyceny komponentów (Stage 3)
        """
        tech = context.technical_analysis
        structure = context.structural_decomposition
        components = context.estimated_components or []

        # Podstawowe statystyki
        total_components = len(components)
        total_hours = sum(c.total_hours for c in components) if components else 0.0
        avg_confidence = (
            sum(c.confidence for c in components) / total_components
            if total_components > 0
            else 0.0
        )

        materials = ", ".join(tech.materials or []) if tech and tech.materials else "unknown"
        key_challenges = (
            "; ".join(tech.key_challenges or []) if tech and tech.key_challenges else "none"
        )

        structure_total_count = (
            structure.total_component_count if structure else total_components
        )
        structure_depth = structure.max_depth if structure else 1

        # TOP komponenty wg godzin
        if components:
            comps_sorted = sorted(
                components, key=lambda c: c.total_hours, reverse=True
            )
            top = comps_sorted[:10]
            comp_lines: list[str] = []
            for c in top:
                comp_lines.append(
                    f"- {c.name}: {c.total_hours:.1f}h "
                    f"(layout={c.hours_3d_layout:.1f}, detail={c.hours_3d_detail:.1f}, 2D={c.hours_2d:.1f}, "
                    f"confidence={c.confidence:.2f})"
                )
            components_summary = "\n".join(comp_lines)
        else:
            components_summary = "No components available."

        return f"""
You are a senior CAD/CAM project manager performing a CRITICAL RISK REVIEW of a CAD hours estimate before it is sent to the client.

PROJECT SUMMARY:
- Description: {context.description}
- Department code: {context.department_code}
- Total components: {total_components}
- Total estimated hours: {total_hours:.1f} h
- Average confidence (0–1): {avg_confidence:.2f}

TECHNICAL ANALYSIS (from Stage 1):
- Complexity level: {tech.project_complexity if tech else "unknown"}
- Main materials: {materials}
- Key technical challenges: {key_challenges}

COMPONENT STRUCTURE (from Stage 2):
- Total component count: {structure_total_count}
- Structure depth: {structure_depth} levels

TOP COMPONENTS BY HOURS:
{components_summary}

TASK:
Perform a STRICT, PRACTICAL PROJECT REVIEW and identify:

1. RISKS – what could realistically go wrong with this estimate?
   - Technical risks (design complexity, tolerances, new technologies).
   - Scope risks (unclear requirements, missing interfaces).
   - Planning risks (underestimated hours, missing tasks, learning curve).
   - Organizational risks (dependencies on client data, late decisions).

2. OPTIMIZATION SUGGESTIONS – realistic ways to reduce hours or improve reliability:
   - Reuse of existing designs / patterns.
   - Standardization and simplification opportunities.
   - Better splitting of work between team members (senior vs junior).

3. ASSUMPTIONS – what has been implicitly assumed in this estimate?
   - Data completeness, client collaboration, reuse of templates, etc.

4. WARNINGS – clear statements for the client / PM about important caveats.

CONSTRAINTS:
- Focus on 3–10 most important items in each category (do not generate huge lists).
- Be concrete and specific, not generic management buzzwords.

OUTPUT FORMAT:
Return ONE valid JSON object, and NOTHING else.

JSON SCHEMA:
{{
  "risks": [
    {{
      "description": "Concrete risk description",
      "impact": "low|medium|high|critical",
      "mitigation": "Concrete mitigation action"
    }}
  ],
  "suggestions": ["suggestion1", "suggestion2", "..."],
  "assumptions": ["assumption1", "assumption2", "..."],
  "warnings": ["warning1", "warning2", "..."]
}}

Output ONLY JSON, strictly following this structure.
"""
