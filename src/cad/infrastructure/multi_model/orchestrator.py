"""
CAD Estimator Pro - Multi-Model Pipeline Orchestrator

Coordinates execution of all 4 pipeline stages sequentially.
"""
import logging
from typing import Callable

from cad.domain.models.multi_model import (
    StageContext,
    PipelineStage,
    PipelineProgress
)
from cad.domain.models import Estimate, EstimatePhases
from cad.domain.models.config import MultiModelConfig
from cad.domain.interfaces.ai_client import AIClient
from cad.domain.interfaces.database import DatabaseClient
from cad.domain.exceptions import AIGenerationError, ValidationError

from .stage1_technical_analysis import TechnicalAnalysisStage
from .stage2_structural_decomposition import StructuralDecompositionStage
from .stage3_hours_estimation import HoursEstimationStage
from .stage4_risk_optimization import RiskOptimizationStage

logger = logging.getLogger(__name__)


class MultiModelOrchestrator:
    """
    Orchestrator for multi-model pipeline.

    Executes all 4 stages sequentially:
    1. Technical Analysis
    2. Structural Decomposition
    3. Hours Estimation
    4. Risk & Optimization
    """

    def __init__(
        self,
        ai_client: AIClient,
        db_client: DatabaseClient,
        config: MultiModelConfig
    ):
        """
        Initialize orchestrator.

        Args:
            ai_client: AI client for generation
            db_client: Database client for patterns
            config: Multi-model configuration
        """
        self.ai_client = ai_client
        self.db_client = db_client
        self.config = config

        # Initialize stages
        self.stage1 = TechnicalAnalysisStage(ai_client, config)
        self.stage2 = StructuralDecompositionStage(ai_client, config)
        self.stage3 = HoursEstimationStage(ai_client, db_client, config)
        self.stage4 = RiskOptimizationStage(ai_client, config)

    def execute_pipeline(
        self,
        context: StageContext,
        enable_multi_model: bool = True,
        progress_callback: Callable[[PipelineProgress], None] | None = None,
        stage1_model: str | None = None,
        stage2_model: str | None = None,
        stage3_model: str | None = None,
        stage4_model: str | None = None
    ) -> Estimate:
        """
        Execute complete multi-model pipeline.

        Args:
            context: Initial pipeline context
            enable_multi_model: If False, fallback to single-model
            progress_callback: Optional callback(PipelineProgress) for UI updates
            stage1_model: Optional model override for Stage 1
            stage2_model: Optional model override for Stage 2
            stage3_model: Optional model override for Stage 3
            stage4_model: Optional model override for Stage 4

        Returns:
            Complete Estimate object

        Raises:
            AIGenerationError: If any stage fails
            ValidationError: If context invalid
        """
        if not enable_multi_model or not self.config.enabled:
            logger.info("Multi-model disabled, using single-model fallback")
            return self._single_model_fallback(context)

        logger.info("Starting multi-model pipeline execution")

        # Log model configuration
        models_used = {
            'stage1': stage1_model or self.config.stage1_model,
            'stage2': stage2_model or self.config.stage2_model,
            'stage3': stage3_model or self.config.stage3_model,
            'stage4': stage4_model or self.config.stage4_model
        }
        logger.info(f"Models: Stage1={models_used['stage1']}, Stage2={models_used['stage2']}, "
                   f"Stage3={models_used['stage3']}, Stage4={models_used['stage4']}")

        completed_stages = []
        total_stages = 4

        try:
            # Stage 1: Technical Analysis
            self._report_progress(
                PipelineStage.TECHNICAL_ANALYSIS,
                completed_stages,
                total_stages,
                progress_callback
            )

            tech_analysis = self.stage1.analyze(context, model=stage1_model)

            # SANITY CHECK: Stage 1
            self._validate_stage1_output(tech_analysis)

            context = context.with_technical_analysis(tech_analysis)
            completed_stages.append(PipelineStage.TECHNICAL_ANALYSIS)

            logger.info(f"Stage 1 complete: Complexity={tech_analysis.project_complexity}, "
                       f"Materials={len(tech_analysis.materials)}")

            # Stage 2: Structural Decomposition
            self._report_progress(
                PipelineStage.STRUCTURAL_DECOMPOSITION,
                completed_stages,
                total_stages,
                progress_callback
            )

            structure = self.stage2.decompose(context, model=stage2_model)

            # SANITY CHECK: Stage 2 (CRITICAL - errors here propagate!)
            self._validate_stage2_output(structure)

            context = context.with_structural_decomposition(structure)
            completed_stages.append(PipelineStage.STRUCTURAL_DECOMPOSITION)

            logger.info(f"Stage 2 complete: Components={structure.total_component_count}, "
                       f"Depth={structure.max_depth}")

            # Stage 3: Hours Estimation
            self._report_progress(
                PipelineStage.HOURS_ESTIMATION,
                completed_stages,
                total_stages,
                progress_callback
            )

            context = self.stage3.estimate_hours(context, model=stage3_model)

            # SANITY CHECK: Stage 3
            total_hours = self._validate_stage3_output(context)

            completed_stages.append(PipelineStage.HOURS_ESTIMATION)

            logger.info(f"Stage 3 complete: Estimated {len(context.estimated_components)} components, "
                       f"Total hours={total_hours:.1f}")

            # Stage 4: Risk & Optimization
            self._report_progress(
                PipelineStage.RISK_OPTIMIZATION,
                completed_stages,
                total_stages,
                progress_callback
            )

            risks, suggestions, assumptions, warnings = self.stage4.analyze_risks(context, model=stage4_model)
            completed_stages.append(PipelineStage.RISK_OPTIMIZATION)

            logger.info(f"Stage 4 complete: Risks={len(risks)}, Suggestions={len(suggestions)}")

            # Build final Estimate
            estimate = self._build_estimate(
                context=context,
                risks=risks,
                suggestions=suggestions,
                assumptions=assumptions,
                warnings=warnings
            )

            # Report completion
            self._report_progress(
                PipelineStage.RISK_OPTIMIZATION,
                completed_stages,
                total_stages,
                progress_callback,
                stage_outputs={'estimate': estimate}
            )

            logger.info("Multi-model pipeline execution complete")
            return estimate

        except Exception as e:
            logger.error(f"Pipeline execution failed at stage {len(completed_stages) + 1}: {e}",
                        exc_info=True)
            raise

    def execute_stage(
        self,
        stage_name: str,
        context: StageContext,
        model: str | None = None
    ) -> StageContext:
        """
        Execute single stage of pipeline.

        Args:
            stage_name: Name of stage to execute
            context: Current pipeline context
            model: Optional model override

        Returns:
            Updated context with stage output

        Raises:
            ValueError: If stage_name invalid
            AIGenerationError: If stage fails
        """
        if stage_name == "technical_analysis":
            tech_analysis = self.stage1.analyze(context, model)
            return context.with_technical_analysis(tech_analysis)

        elif stage_name == "structural_decomposition":
            structure = self.stage2.decompose(context, model)
            return context.with_structural_decomposition(structure)

        elif stage_name == "hours_estimation":
            return self.stage3.estimate_hours(context, model)

        elif stage_name == "risk_optimization":
            # This stage doesn't update context, it only analyzes
            risks, suggestions, assumptions, warnings = self.stage4.analyze_risks(context, model)
            # Could add these to context if needed
            return context

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def _single_model_fallback(self, context: StageContext) -> Estimate:
        """
        Fallback to single-model estimation.

        This would use the existing EstimationPipeline logic.
        For now, we'll create a minimal estimate.
        """
        logger.warning("Single-model fallback not fully implemented yet")

        # Create minimal estimate with placeholder data
        from cad.domain.models.component import Component
        from cad.domain.models.estimate import Risk

        components = [
            Component(
                name="Placeholder Component",
                hours_3d_layout=10.0,
                hours_3d_detail=20.0,
                hours_2d=10.0,
                confidence=0.5,
                confidence_reason="Single-model fallback"
            )
        ]

        return Estimate.from_components(
            components=components,
            risks=[],
            generation_metadata={
                'multi_model': False,
                'fallback': True
            }
        )

    def _build_estimate(
        self,
        context: StageContext,
        risks: list,
        suggestions: list[str],
        assumptions: list[str],
        warnings: list[str]
    ) -> Estimate:
        """Build final Estimate object from pipeline results."""
        # Calculate phases
        total_layout = sum(c.hours_3d_layout for c in context.estimated_components)
        total_detail = sum(c.hours_3d_detail for c in context.estimated_components)
        total_2d = sum(c.hours_2d for c in context.estimated_components)

        phases = EstimatePhases(
            hours_3d_layout=total_layout,
            hours_3d_detail=total_detail,
            hours_2d=total_2d
        )

        # Calculate overall confidence
        if context.estimated_components:
            overall_confidence = sum(c.confidence for c in context.estimated_components) / len(context.estimated_components)
        else:
            overall_confidence = 0.5

        # Build metadata with detailed stage outputs
        metadata = {
            'multi_model': True,
            'stage1_complexity': context.technical_analysis.project_complexity if context.technical_analysis else None,
            'stage1_materials': context.technical_analysis.materials if context.technical_analysis else [],
            'stage1_standards': context.technical_analysis.applicable_standards if context.technical_analysis else [],
            'stage1_challenges': context.technical_analysis.key_challenges if context.technical_analysis else [],
            'stage2_component_count': context.structural_decomposition.total_component_count if context.structural_decomposition else None,
            'stage2_max_depth': context.structural_decomposition.max_depth if context.structural_decomposition else None,
            'suggestions': suggestions,
            'assumptions': assumptions,
            'warnings': warnings
        }

        return Estimate(
            components=tuple(context.estimated_components),
            phases=phases,
            overall_confidence=overall_confidence,
            risks=tuple(risks),
            generation_metadata=metadata
        )

    def _report_progress(
        self,
        current_stage: PipelineStage,
        completed_stages: list[PipelineStage],
        total_stages: int,
        callback: Callable[[PipelineProgress], None] | None,
        stage_outputs: dict | None = None
    ) -> None:
        """Report progress to callback if provided."""
        if callback is None:
            return

        progress = PipelineProgress(
            current_stage=current_stage,
            completed_stages=tuple(completed_stages),
            total_stages=total_stages,
            stage_outputs=stage_outputs or {}
        )

        try:
            callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    # =========================================================================
    # VALIDATION / SANITY CHECKS (added for production robustness)
    # =========================================================================

    def _validate_stage1_output(self, tech_analysis) -> None:
        """
        Validate Stage 1 (Technical Analysis) output.

        Ensures the model produced sensible technical understanding.

        Raises:
            ValidationError: If output invalid
        """
        if not tech_analysis:
            raise ValidationError("Stage 1: Technical analysis is None")

        # Check project complexity is reasonable
        if not hasattr(tech_analysis, 'project_complexity'):
            logger.warning("Stage 1: Missing project_complexity field")

        complexity = getattr(tech_analysis, 'project_complexity', None)
        if complexity and not isinstance(complexity, str):
            raise ValidationError(f"Stage 1: Invalid complexity type: {type(complexity)}")

        # Check materials list exists (can be empty)
        if not hasattr(tech_analysis, 'materials'):
            logger.warning("Stage 1: Missing materials field")

        # Check key challenges exist
        if not hasattr(tech_analysis, 'key_challenges'):
            logger.warning("Stage 1: Missing key_challenges field")

        logger.info("✓ Stage 1 validation passed")

    def _validate_stage2_output(self, structure) -> None:
        """
        Validate Stage 2 (Structural Decomposition) output.

        CRITICAL: Errors here propagate to all subsequent stages!

        Raises:
            ValidationError: If output invalid
        """
        if not structure:
            raise ValidationError("Stage 2: Structural decomposition is None")

        # Check total component count
        if not hasattr(structure, 'total_component_count'):
            raise ValidationError("Stage 2: Missing total_component_count field")

        component_count = structure.total_component_count
        if component_count < 1:
            raise ValidationError(f"Stage 2: Component count too low: {component_count}")

        if component_count > 1000:
            logger.warning(f"Stage 2: Very high component count: {component_count} - potential hallucination?")

        # Check depth is reasonable
        if hasattr(structure, 'max_depth'):
            depth = structure.max_depth
            if depth < 1:
                raise ValidationError(f"Stage 2: Invalid depth: {depth}")
            if depth > 10:
                logger.warning(f"Stage 2: Very deep hierarchy: {depth} levels - potential over-decomposition")

        # Check root components exist
        if hasattr(structure, 'root_components'):
            if not structure.root_components:
                raise ValidationError("Stage 2: No root components found")

        logger.info("✓ Stage 2 validation passed (CRITICAL stage validated)")

    def _validate_stage3_output(self, context) -> float:
        """
        Validate Stage 3 (Hours Estimation) output.

        Args:
            context: Stage context with estimated_components

        Returns:
            Total estimated hours

        Raises:
            ValidationError: If output invalid
        """
        if not hasattr(context, 'estimated_components'):
            raise ValidationError("Stage 3: Missing estimated_components")

        components = context.estimated_components
        if not components:
            raise ValidationError("Stage 3: No components estimated")

        # Check each component has valid hours
        total_hours = 0.0
        for i, comp in enumerate(components):
            if not hasattr(comp, 'total_hours'):
                raise ValidationError(f"Stage 3: Component {i} missing total_hours")

            hours = comp.total_hours
            if hours < 0:
                raise ValidationError(f"Stage 3: Component {i} has negative hours: {hours}")

            if hours > 10000:
                logger.warning(f"Stage 3: Component {i} has very high hours: {hours} - potential error")

            total_hours += hours

        # Sanity check: total hours should be reasonable for CAD project
        if total_hours < 0.1:
            raise ValidationError(f"Stage 3: Total hours too low: {total_hours}")

        if total_hours > 100000:
            logger.warning(f"Stage 3: Total hours very high: {total_hours} - potential hallucination")

        logger.info(f"✓ Stage 3 validation passed (total={total_hours:.1f}h)")
        return total_hours
