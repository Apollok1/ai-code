"""
CAD Estimator Pro - Stage 2: Structural Decomposition

Breaks down project into hierarchical component structure.
"""
import json
import logging
from typing import Any

from ...domain.models.multi_model import (
    StageContext,
    StructuralDecomposition,
    ComponentNode
)
from ...domain.models.config import MultiModelConfig
from ...domain.interfaces.ai_client import AIClient
from ...domain.exceptions import AIGenerationError, ValidationError

logger = logging.getLogger(__name__)


class StructuralDecompositionStage:
    """
    Stage 2: Structural Decomposition.

    Uses AI to break down project into component hierarchy:
    - Identify major assemblies and subassemblies
    - Build parent-child relationships
    - Estimate quantities for each component
    - Create logical grouping/categories
    """

    def __init__(self, ai_client: AIClient, config: MultiModelConfig):
        """
        Initialize structural decomposition stage.

        Args:
            ai_client: AI client for generation
            config: Multi-model configuration
        """
        self.ai_client = ai_client
        self.config = config

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
        if context.technical_analysis is None:
            raise ValidationError("Stage 2 requires technical_analysis from Stage 1")

        model_to_use = model or self.config.stage2_model

        logger.info(f"Stage 2: Structural Decomposition (model: {model_to_use})")

        # Build prompt with context from Stage 1
        prompt = self._build_decomposition_prompt(context)

        try:
            # Generate structure
            response = self.ai_client.generate_text(
                prompt=prompt,
                model=model_to_use,
                json_mode=True
            )

            # Parse JSON response
            try:
                structure_data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    structure_data = json.loads(response[start_idx:end_idx])
                else:
                    raise AIGenerationError(f"Invalid JSON response from model: {response[:200]}")

            # Parse component tree
            root_components = self._parse_component_tree(structure_data.get('components', []))

            # Calculate metrics
            all_nodes = []
            for root in root_components:
                all_nodes.extend(root.flatten())

            max_depth = self._calculate_max_depth(root_components)

            # Extract assembly relationships
            assembly_rels = structure_data.get('assembly_relationships', {})

            return StructuralDecomposition(
                root_components=tuple(root_components),
                total_component_count=len(all_nodes),
                max_depth=max_depth,
                assembly_relationships=assembly_rels,
                raw_structure=structure_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Structural decomposition failed: {e}", exc_info=True)
            raise AIGenerationError(f"Stage 2 failed: {e}")

    def _build_decomposition_prompt(self, context: StageContext) -> str:
        """Build structural decomposition prompt."""
        tech_analysis = context.technical_analysis

        return f"""You are a senior CAD/CAM engineer breaking down a mechanical design project into its component structure.

PROJECT DESCRIPTION:
{context.description}

TECHNICAL ANALYSIS FROM STAGE 1:
- Complexity: {tech_analysis.project_complexity}
- Materials: {', '.join(tech_analysis.materials[:5])}
- Manufacturing: {', '.join(tech_analysis.manufacturing_methods[:3])}
- Estimated Assembly Count: {tech_analysis.estimated_assembly_count or 'Unknown'}
- Key Challenges: {', '.join(tech_analysis.key_challenges[:3])}

TASK: Break this project down into a HIERARCHICAL COMPONENT STRUCTURE.

Think about:
1. What are the major assemblies/modules?
2. What sub-assemblies does each major assembly contain?
3. What individual parts/components are in each sub-assembly?
4. How many of each component is needed (quantity)?
5. What category does each component belong to? (e.g., "Frame", "Mechanism", "Hydraulics", "Electronics", etc.)

OUTPUT FORMAT (JSON):
{{
    "reasoning": "Your thinking about how to break down this project (1-2 paragraphs)",
    "components": [
        {{
            "name": "Main Assembly 1",
            "category": "Frame",
            "quantity": 1,
            "children": [
                {{
                    "name": "Sub-assembly 1.1",
                    "category": "Support Structure",
                    "quantity": 2,
                    "children": [
                        {{
                            "name": "Bracket",
                            "category": "Fastener",
                            "quantity": 4,
                            "children": []
                        }}
                    ]
                }}
            ]
        }},
        {{
            "name": "Main Assembly 2",
            "category": "Mechanism",
            "quantity": 1,
            "children": [...]
        }}
    ],
    "assembly_relationships": {{
        "Main Assembly 1": ["Sub-assembly 1.1", "Sub-assembly 1.2"],
        "Sub-assembly 1.1": ["Bracket", "Bolt"]
    }}
}}

Be specific and realistic. Create a structure that makes sense for manufacturing and assembly."""

    def _parse_component_tree(self, components_data: list[dict]) -> list[ComponentNode]:
        """Parse component tree from JSON data."""
        result = []
        for comp_data in components_data:
            node = self._parse_component_node(comp_data, parent_name=None)
            result.append(node)
        return result

    def _parse_component_node(self, data: dict, parent_name: str | None) -> ComponentNode:
        """Recursively parse component node."""
        children_data = data.get('children', [])
        children = tuple(
            self._parse_component_node(child_data, parent_name=data['name'])
            for child_data in children_data
        )

        return ComponentNode(
            name=data['name'],
            category=data.get('category'),
            quantity=data.get('quantity', 1),
            parent_name=parent_name,
            children=children,
            metadata=data.get('metadata', {})
        )

    def _calculate_max_depth(self, roots: list[ComponentNode], current_depth: int = 1) -> int:
        """Calculate maximum depth of component tree."""
        if not roots:
            return current_depth - 1

        max_child_depth = current_depth
        for root in roots:
            if root.children:
                child_depth = self._calculate_max_depth(list(root.children), current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth
