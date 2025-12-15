"""
CAD Estimator Pro - Stage 1: Technical Analysis

Deep technical analysis and reasoning about project requirements.
"""
import json
import logging
from typing import Any

from cad.domain.models.multi_model import StageContext, TechnicalAnalysis
from cad.domain.models.config import MultiModelConfig
from cad.domain.interfaces.ai_client import AIClient
from cad.domain.exceptions import AIGenerationError, ValidationError

logger = logging.getLogger(__name__)


class TechnicalAnalysisStage:
    """
    Stage 1: Technical Analysis & Deep Thinking.

    Uses a reasoning model to analyze project deeply:
    - Identify materials and manufacturing methods
    - Assess technical complexity
    - Find applicable standards (ISO, EN, DIN)
    - Identify key challenges and constraints
    """

    def __init__(self, ai_client: AIClient, config: MultiModelConfig):
        """
        Initialize technical analysis stage.

        Args:
            ai_client: AI client for generation
            config: Multi-model configuration
        """
        self.ai_client = ai_client
        self.config = config

    def analyze(self, context: StageContext, model: str | None = None) -> TechnicalAnalysis:
        """
        Perform deep technical analysis of project.

        Args:
            context: Current pipeline context
            model: Optional model override

        Returns:
            TechnicalAnalysis with deep technical understanding

        Raises:
            AIGenerationError: If analysis fails
        """
        model_to_use = model or self.config.stage1_model

        logger.info(f"Stage 1: Technical Analysis (model: {model_to_use})")

        # Build comprehensive context
        context_parts = [
            f"Project Description: {context.description}",
            f"Department: {context.department_code}"
        ]

        if context.pdf_texts:
            context_parts.append("PDF Specifications:")
            for i, pdf_text in enumerate(context.pdf_texts[:3], 1):  # Limit to 3 PDFs
                context_parts.append(f"PDF {i}: {pdf_text[:2000]}...")  # Limit each PDF

        if context.image_analyses:
            context_parts.append("Image Analyses:")
            for i, img_analysis in enumerate(context.image_analyses[:3], 1):
                context_parts.append(f"Image {i}: {img_analysis[:500]}...")

        if context.excel_data:
            context_parts.append(f"Excel Data Available: {context.excel_data.get('statistics', {})}")

        full_context = "\n\n".join(context_parts)

        # Build technical analysis prompt
        prompt = self._build_technical_prompt(full_context)

        try:
            # Generate analysis
            response = self.ai_client.generate_text(
                prompt=prompt,
                model=model_to_use,
                json_mode=True
            )

            # Parse JSON response
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    analysis_data = json.loads(response[start_idx:end_idx])
                else:
                    raise AIGenerationError(f"Invalid JSON response from model: {response[:200]}")

            # Validate and create TechnicalAnalysis
            return TechnicalAnalysis(
                project_complexity=analysis_data.get('project_complexity', 'medium').lower(),
                materials=analysis_data.get('materials', []),
                manufacturing_methods=analysis_data.get('manufacturing_methods', []),
                technical_constraints=analysis_data.get('technical_constraints', []),
                applicable_standards=analysis_data.get('applicable_standards', []),
                key_challenges=analysis_data.get('key_challenges', []),
                estimated_assembly_count=analysis_data.get('estimated_assembly_count'),
                raw_analysis=analysis_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Technical analysis failed: {e}", exc_info=True)
            raise AIGenerationError(f"Stage 1 failed: {e}")

    def _build_technical_prompt(self, project_context: str) -> str:
        """Build technical analysis prompt."""
        return f"""You are a senior CAD/CAM engineer with 20+ years of experience analyzing mechanical design projects.

Perform a DEEP TECHNICAL ANALYSIS of the following CAD project. Think step-by-step about:
1. What materials would be used and why
2. How the parts would be manufactured (machining, welding, casting, etc.)
3. What technical constraints exist (tolerances, surface finish, assembly requirements)
4. What standards apply (ISO, EN, DIN, ASME, etc.)
5. What are the key technical challenges
6. Estimate the overall complexity (low/medium/high/very_high)
7. Roughly estimate how many unique assemblies/parts this project has

PROJECT CONTEXT:
{project_context}

OUTPUT FORMAT (JSON):
{{
    "reasoning": "Your detailed step-by-step technical reasoning (2-3 paragraphs)",
    "project_complexity": "low|medium|high|very_high",
    "materials": ["material1", "material2", ...],
    "manufacturing_methods": ["method1", "method2", ...],
    "technical_constraints": ["constraint1", "constraint2", ...],
    "applicable_standards": ["ISO 1234", "EN 5678", ...],
    "key_challenges": ["challenge1", "challenge2", ...],
    "estimated_assembly_count": 15
}}

Be thorough and specific. Think like an experienced engineer reviewing a project proposal."""
