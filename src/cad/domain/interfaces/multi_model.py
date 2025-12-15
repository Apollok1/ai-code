"""
CAD Estimator Pro - Multi-Model Pipeline Interfaces

Protocol interfaces for multi-model estimation pipeline.
"""
from typing import Protocol

from cad.models.multi_model import (
    StageContext,
    TechnicalAnalysis,
    StructuralDecomposition,
    PipelineProgress
)
from cad.models import Estimate


class TechnicalAnalysisStage(Protocol):
    """
    Stage 1: Technical Analysis & Deep Thinking.

    Analyzes project requirements, identifies materials,
    constraints, and technical complexity.
    """

    def analyze(self, context: StageContext, model: str | None = None) -> TechnicalAnalysis:
        """
        Perform technical analysis of project.

        Args:
            context: Current pipeline context
            model: Optional model override

        Returns:
            TechnicalAnalysis with deep technical understanding

        Raises:
            AIGenerationError: If analysis fails
        """
        ...


class StructuralDecompositionStage(Protocol):
    """
    Stage 2: Structural Decomposition.

    Breaks down project into hierarchical component structure.
    """

    def decompose(self, context: StageContext, model: str | None = None) -> StructuralDecomposition:
        """
        Decompose project into component hierarchy.

        Args:
            context: Pipeline context (must have technical_analysis)
            model: Optional model override

        Returns:
            StructuralDecomposition with component tree

        Raises:
            AIGenerationError: If decomposition fails
            ValidationError: If technical_analysis missing from context
        """
        ...


class HoursEstimationStage(Protocol):
    """
    Stage 3: Hours Estimation.

    Estimates hours for each component using patterns and AI.
    """

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
        ...


class RiskOptimizationStage(Protocol):
    """
    Stage 4: Risk Analysis & Optimization.

    Identifies risks and suggests optimizations.
    """

    def analyze_risks(self, context: StageContext, model: str | None = None) -> tuple[list, list, list, list]:
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
        ...


class MultiModelOrchestrator(Protocol):
    """
    Orchestrator for multi-model pipeline.

    Coordinates execution of all 4 stages sequentially.
    """

    def execute_pipeline(
        self,
        context: StageContext,
        enable_multi_model: bool = True,
        progress_callback: callable | None = None
    ) -> Estimate:
        """
        Execute complete multi-model pipeline.

        Args:
            context: Initial pipeline context
            enable_multi_model: If False, fallback to single-model
            progress_callback: Optional callback(PipelineProgress) for UI updates

        Returns:
            Complete Estimate object

        Raises:
            AIGenerationError: If any stage fails
            ValidationError: If context invalid
        """
        ...

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
        ...
