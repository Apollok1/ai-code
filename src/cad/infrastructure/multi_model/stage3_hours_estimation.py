"""
CAD Estimator Pro - Stage 3: Hours Estimation

Estimates hours for each component using patterns and AI.
"""
import json
import logging
from typing import Any

from cad.domain.models.multi_model import StageContext, ComponentNode
from cad.domain.models.component import Component
from cad.domain.models.config import MultiModelConfig
from cad.domain.interfaces.ai_client import AIClient
from cad.domain.interfaces.database import DatabaseClient
from cad.domain.exceptions import AIGenerationError, ValidationError

logger = logging.getLogger(__name__)


class HoursEstimationStage:
    """
    Stage 3: Hours Estimation.

    Estimates hours for each component using:
    - Historical pattern matching (from database)
    - AI reasoning about complexity
    - Multipliers from technical analysis
    """

    def __init__(
        self,
        ai_client: AIClient,
        db_client: DatabaseClient,
        config: MultiModelConfig
    ):
        """
        Initialize hours estimation stage.

        Args:
            ai_client: AI client for generation
            db_client: Database client for patterns
            config: Multi-model configuration
        """
        self.ai_client = ai_client
        self.db_client = db_client
        self.config = config

    def estimate_hours(self, context: StageContext, model: str | None = None) -> StageContext:
        """
        Estimate hours for all components.

        Args:
            context: Pipeline context (must have structural_decomposition)
            model: Optional model override

        Returns:
            Updated context with estimated_components populated

        Raises:
            AIGenerationError: If estimation fails
            ValidationError: If structural_decomposition missing
        """
        if context.structural_decomposition is None:
            raise ValidationError("Stage 3 requires structural_decomposition from Stage 2")

        model_to_use = model or self.config.stage3_model

        logger.info(f"Stage 3: Hours Estimation (model: {model_to_use})")

        # Get all components from structure
        all_nodes = context.structural_decomposition.all_components

        # Get complexity multiplier from technical analysis
        complexity_multiplier = 1.0
        if context.technical_analysis:
            complexity_multiplier = context.technical_analysis.complexity_score

        # Build estimation prompt
        prompt = self._build_estimation_prompt(context, all_nodes, complexity_multiplier)

        try:
            # Generate estimates
            response = self.ai_client.generate_text(
                prompt=prompt,
                model=model_to_use,
                json_mode=True
            )

            # Parse JSON response
            try:
                estimates_data = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    estimates_data = json.loads(response[start_idx:end_idx])
                else:
                    raise AIGenerationError(f"Invalid JSON response from model: {response[:200]}")

            # Convert to Component objects
            components = []
            estimates = estimates_data.get('estimates', [])

            for est in estimates:
                # Get pattern match if available
                pattern = self._find_pattern_for_component(
                    est['name'],
                    context.department_code
                )

                # Use pattern if available, otherwise use AI estimate
                if pattern:
                    hours_3d_layout = pattern.get('avg_hours_3d_layout', est['hours_3d_layout'])
                    hours_3d_detail = pattern.get('avg_hours_3d_detail', est['hours_3d_detail'])
                    hours_2d = pattern.get('avg_hours_2d', est['hours_2d'])
                    confidence = min(pattern.get('confidence', 0.5), 0.85)
                    confidence_reason = f"Based on {pattern.get('occurrence_count', 0)} historical examples"
                else:
                    hours_3d_layout = est['hours_3d_layout'] * complexity_multiplier
                    hours_3d_detail = est['hours_3d_detail'] * complexity_multiplier
                    hours_2d = est['hours_2d'] * complexity_multiplier
                    confidence = est.get('confidence', 0.5)
                    confidence_reason = est.get('reasoning', 'AI estimate')

                component = Component(
                    name=est['name'],
                    hours_3d_layout=hours_3d_layout,
                    hours_3d_detail=hours_3d_detail,
                    hours_2d=hours_2d,
                    confidence=confidence,
                    confidence_reason=confidence_reason,
                    is_summary=False,
                    subcomponents=()
                )
                components.append(component)

            # Return updated context
            return context.with_estimated_components(components)

        except Exception as e:
            logger.error(f"Hours estimation failed: {e}", exc_info=True)
            raise AIGenerationError(f"Stage 3 failed: {e}")

    

    def _build_estimation_prompt(context, tech_analysis, complexity_multiplier,
                             materials, components_list: str) -> str:
        return f"""
You are a CAD/CAM estimator calculating realistic engineering hours for each component of a mechanical project.

PROJECT CONTEXT:
- Description: {context.description}
- Department code: {context.department_code}
- Complexity level: {tech_analysis.project_complexity}
- Complexity multiplier: {complexity_multiplier}x
- Dominant materials: {materials}

COMPONENTS TO ESTIMATE (from previous stage):
{components_list}

Each component listed above ALREADY has a name. You MUST:
- Keep the component names EXACTLY as given (do not translate, do not rename).
- Provide hour estimates for THREE phases:
  1. hours_3d_layout  – initial 3D positioning and basic shapes in the assembly,
  2. hours_3d_detail  – full detailed 3D modeling with all relevant features,
  3. hours_2d         – 2D manufacturing drawings with dimensions and annotations.

CONSIDER:
- Geometric and functional complexity of each component.
- Manufacturing requirements (tolerances, surface finish, weld symbols, GD&T).
- Standard vs custom parts (standard = much less time).
- Repetition: if several components are very similar, later ones are faster than the first.
- The overall project complexity and department type (more safety-critical = more hours).

CONSTRAINTS:
- Use realistic ranges, e.g. 0.1–200 hours per component per phase (most parts will be much lower).
- Use decimal numbers (e.g. 1.5, 3.0), not strings.
- If there is almost no work for a phase, you can use 0.0 or a very small value (e.g. 0.2).
- If you are very uncertain, keep hours modest and reduce confidence.

OUTPUT FORMAT:
Return ONE valid JSON object, and NOTHING else.

JSON SCHEMA:
{{
  "reasoning": "Your global estimation strategy (1 short paragraph).",
  "estimates": [
    {{
      "name": "Component Name (exactly as in input)",
      "hours_3d_layout": 2.0,
      "hours_3d_detail": 8.0,
      "hours_2d": 4.0,
      "confidence": 0.7,
      "reasoning": "Why these hours make sense for this component."
    }}
  ]
}}

Output ONLY JSON, strictly following this structure.
"""

    def _find_pattern_for_component(self, component_name: str, department: str) -> dict | None:
        """Find historical pattern for component."""
        try:
            # Use fuzzy search to find similar components
            patterns = self.db_client.search_component_patterns(
                query=component_name,
                department=department,
                limit=1
            )
            if patterns:
                return patterns[0]
        except Exception as e:
            logger.warning(f"Pattern search failed for '{component_name}': {e}")

        return None
