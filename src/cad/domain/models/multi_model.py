"""
CAD Estimator Pro - Multi-Model Domain Models

Domain models for multi-model pipeline stages.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineStage(str, Enum):
    """Pipeline stage enumeration."""
    TECHNICAL_ANALYSIS = "technical_analysis"
    STRUCTURAL_DECOMPOSITION = "structural_decomposition"
    HOURS_ESTIMATION = "hours_estimation"
    RISK_OPTIMIZATION = "risk_optimization"


@dataclass(frozen=True)
class TechnicalAnalysis:
    """
    Output from Stage 1: Technical Analysis.

    Deep technical understanding of the project.
    """
    project_complexity: str  # "low", "medium", "high", "very_high"
    materials: list[str]
    manufacturing_methods: list[str]
    technical_constraints: list[str]
    applicable_standards: list[str]  # ISO, EN, DIN, etc.
    key_challenges: list[str]
    estimated_assembly_count: int | None = None
    raw_analysis: str = ""  # Full AI analysis text

    @property
    def complexity_score(self) -> float:
        """Convert complexity to numeric score."""
        mapping = {"low": 1.0, "medium": 1.3, "high": 1.6, "very_high": 2.0}
        return mapping.get(self.project_complexity, 1.3)


@dataclass(frozen=True)
class ComponentNode:
    """
    Component in structural hierarchy.

    Used in Stage 2: Structural Decomposition.
    """
    name: str
    category: str | None = None
    quantity: int = 1
    parent_name: str | None = None
    children: tuple["ComponentNode", ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """Check if component is a leaf node."""
        return len(self.children) == 0

    def flatten(self) -> list["ComponentNode"]:
        """Flatten tree to list of all nodes."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass(frozen=True)
class StructuralDecomposition:
    """
    Output from Stage 2: Structural Decomposition.

    Hierarchical component breakdown.
    """
    root_components: tuple[ComponentNode, ...]
    total_component_count: int
    max_depth: int
    assembly_relationships: dict[str, list[str]] = field(default_factory=dict)
    raw_structure: str = ""  # Full AI structure text

    @property
    def all_components(self) -> list[ComponentNode]:
        """Get flattened list of all components."""
        result = []
        for root in self.root_components:
            result.extend(root.flatten())
        return result


@dataclass(frozen=True)
class StageContext:
    """
    Context passed between pipeline stages.

    Accumulates information as pipeline progresses.
    """
    # Input data
    description: str
    department_code: str
    pdf_texts: list[str] = field(default_factory=list)
    excel_data: dict | None = None
    image_analyses: list[str] = field(default_factory=list)

    # Stage 1 output
    technical_analysis: TechnicalAnalysis | None = None

    # Stage 2 output
    structural_decomposition: StructuralDecomposition | None = None

    # Stage 3 output (Components from regular estimation)
    estimated_components: list[Any] = field(default_factory=list)

    # Historical context
    similar_projects: list[dict] = field(default_factory=list)
    available_patterns: dict[str, Any] = field(default_factory=dict)

    def with_technical_analysis(self, analysis: TechnicalAnalysis) -> "StageContext":
        """Create new context with technical analysis added."""
        return StageContext(
            description=self.description,
            department_code=self.department_code,
            pdf_texts=self.pdf_texts,
            excel_data=self.excel_data,
            image_analyses=self.image_analyses,
            technical_analysis=analysis,
            structural_decomposition=self.structural_decomposition,
            estimated_components=self.estimated_components,
            similar_projects=self.similar_projects,
            available_patterns=self.available_patterns
        )

    def with_structural_decomposition(self, structure: StructuralDecomposition) -> "StageContext":
        """Create new context with structural decomposition added."""
        return StageContext(
            description=self.description,
            department_code=self.department_code,
            pdf_texts=self.pdf_texts,
            excel_data=self.excel_data,
            image_analyses=self.image_analyses,
            technical_analysis=self.technical_analysis,
            structural_decomposition=structure,
            estimated_components=self.estimated_components,
            similar_projects=self.similar_projects,
            available_patterns=self.available_patterns
        )

    def with_estimated_components(self, components: list[Any]) -> "StageContext":
        """Create new context with estimated components added."""
        return StageContext(
            description=self.description,
            department_code=self.department_code,
            pdf_texts=self.pdf_texts,
            excel_data=self.excel_data,
            image_analyses=self.image_analyses,
            technical_analysis=self.technical_analysis,
            structural_decomposition=self.structural_decomposition,
            estimated_components=components,
            similar_projects=self.similar_projects,
            available_patterns=self.available_patterns
        )


@dataclass(frozen=True)
class PipelineProgress:
    """
    Progress tracking for multi-model pipeline.
    """
    current_stage: PipelineStage
    completed_stages: tuple[PipelineStage, ...]
    total_stages: int
    stage_outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        return (len(self.completed_stages) / self.total_stages) * 100 if self.total_stages > 0 else 0

    @property
    def is_complete(self) -> bool:
        """Check if all stages are complete."""
        return len(self.completed_stages) == self.total_stages
