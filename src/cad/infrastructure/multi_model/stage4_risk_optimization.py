"""
CAD Estimator Pro - Stage 4: Risk Analysis & Optimization

Identifies risks and suggests optimizations for the estimate.
"""
import json
import logging
from typing import Any

from ...domain.models.multi_model import StageContext
from ...domain.models.estimate import Risk
from ...domain.models.config import MultiModelConfig
from ...domain.interfaces.ai_client import AIClient
from ...domain.exceptions import AIGenerationError, ValidationError

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
        model: str | None = None
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
                json_mode=True
            )

            # Parse JSON response
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    analysis_data = json.loads(response[start_idx:end_idx])
                else:
                    raise AIGenerationError(f"Invalid JSON response from model: {response[:200]}")

            # Parse risks
            risks = []
            for risk_data in analysis_data.get('risks', []):
                risk = Risk(
                    category=risk_data.get('category', 'other'),
                    description=risk_data.get('description', ''),
                    impact=risk_data.get('impact', 'medium'),
                    mitigation=risk_data.get('mitigation', '')
                )
                risks.append(risk)

            # Extract other outputs
            suggestions = analysis_data.get('optimization_suggestions', [])
            assumptions = analysis_data.get('assumptions', [])
            warnings = analysis_data.get('warnings', [])

            return risks, suggestions, assumptions, warnings

        except Exception as e:
            logger.error(f"Risk analysis failed: {e}", exc_info=True)
            raise AIGenerationError(f"Stage 4 failed: {e}")

    def _build_risk_analysis_prompt(self, context: StageContext) -> str:
        """Build risk analysis prompt."""
        # Calculate totals
        total_hours = sum(comp.total_hours for comp in context.estimated_components)
        avg_confidence = sum(comp.confidence for comp in context.estimated_components) / len(context.estimated_components) if context.estimated_components else 0

        # Get top 10 components by hours
        top_components = sorted(
            context.estimated_components,
            key=lambda c: c.total_hours,
            reverse=True
        )[:10]

        components_summary = "\n".join([
            f"- {comp.name}: {comp.total_hours:.1f}h (confidence: {comp.confidence:.0%})"
            for comp in top_components
        ])

        tech_analysis = context.technical_analysis
        structure = context.structural_decomposition

        return f"""You are a senior CAD/CAM project manager performing CRITICAL RISK ANALYSIS on an estimate.

PROJECT SUMMARY:
- Description: {context.description}
- Department: {context.department_code}
- Total Components: {len(context.estimated_components)}
- Total Estimated Hours: {total_hours:.1f}h
- Average Confidence: {avg_confidence:.0%}

TECHNICAL ANALYSIS:
- Complexity: {tech_analysis.project_complexity if tech_analysis else 'unknown'}
- Materials: {', '.join(tech_analysis.materials[:3]) if tech_analysis and tech_analysis.materials else 'N/A'}
- Key Challenges: {', '.join(tech_analysis.key_challenges[:2]) if tech_analysis and tech_analysis.key_challenges else 'N/A'}

COMPONENT STRUCTURE:
- Total Component Count: {structure.total_component_count if structure else 'N/A'}
- Structure Depth: {structure.max_depth if structure else 'N/A'} levels

TOP 10 COMPONENTS BY HOURS:
{components_summary}

TASK: Perform a CRITICAL ANALYSIS to identify:
1. **Risks**: What could go wrong? What might cause this estimate to be inaccurate?
2. **Optimization Suggestions**: How could we reduce hours or improve accuracy?
3. **Assumptions**: What assumptions were made in this estimate?
4. **Warnings**: What should the client be warned about?

Think critically like a project manager reviewing an estimate before sending to client.

OUTPUT FORMAT (JSON):
{{
    "reasoning": "Your critical analysis (2-3 paragraphs)",
    "risks": [
        {{
            "category": "technical|schedule|cost|quality|scope",
            "description": "What the risk is",
            "impact": "low|medium|high|critical",
            "mitigation": "How to reduce this risk"
        }},
        ...
    ],
    "optimization_suggestions": [
        "Suggestion 1: Consider using standard parts for X to reduce modeling time",
        "Suggestion 2: ...",
        ...
    ],
    "assumptions": [
        "Assumption 1: Client will provide CAD files for standard components",
        "Assumption 2: ...",
        ...
    ],
    "warnings": [
        "Warning 1: Low confidence on complex assembly X - may need more research",
        "Warning 2: ...",
        ...
    ]
}}

Be honest and thorough. Better to identify risks now than be surprised later."""
