"""
CAD Estimator Pro - Estimation Pipeline

Main orchestrator for CAD project estimation workflow.
"""
import logging
from typing import BinaryIO, Any

from ..domain.models import Estimate, Component, DepartmentCode, Risk, Suggestion
from ..domain.models.config import AppConfig
from ..domain.exceptions import AIGenerationError, ParsingError
from ..infrastructure.ai.ollama_client import OllamaClient
from ..infrastructure.parsers.excel_parser import CADExcelParser
from ..infrastructure.parsers.pdf_parser import CADPDFParser
from ..infrastructure.parsers.component_parser import CADComponentParser
from ..infrastructure.learning.pattern_learner import PatternLearner
from ..infrastructure.learning.bundle_learner import BundleLearner
from ..infrastructure.embeddings.pgvector_service import PgVectorService
from ..infrastructure.multi_model import MultiModelOrchestrator
from ..domain.models.multi_model import StageContext

logger = logging.getLogger(__name__)


class EstimationPipeline:
    """
    Estimation pipeline orchestrator.

    Coordinates:
    - File parsing (Excel, PDF)
    - AI analysis
    - Pattern matching
    - Bundle suggestions
    - Semantic search
    """

    def __init__(
        self,
        config: AppConfig,
        db_client: Any,
        ai_client: OllamaClient,
        excel_parser: CADExcelParser,
        pdf_parser: CADPDFParser,
        component_parser: CADComponentParser,
        pattern_learner: PatternLearner,
        bundle_learner: BundleLearner,
        pgvector_service: PgVectorService,
        multi_model_orchestrator: MultiModelOrchestrator | None = None
    ):
        """
        Initialize EstimationPipeline.

        Args:
            config: Application configuration
            db_client: Database client
            ai_client: AI client
            excel_parser: Excel parser
            pdf_parser: PDF parser
            component_parser: Component parser
            pattern_learner: Pattern learner
            bundle_learner: Bundle learner
            pgvector_service: PgVector service
            multi_model_orchestrator: Optional multi-model orchestrator
        """
        self.config = config
        self.db = db_client
        self.ai = ai_client
        self.excel_parser = excel_parser
        self.pdf_parser = pdf_parser
        self.component_parser = component_parser
        self.pattern_learner = pattern_learner
        self.bundle_learner = bundle_learner
        self.pgvector = pgvector_service
        self.multi_model = multi_model_orchestrator

    def estimate_from_description(
        self,
        description: str,
        department: DepartmentCode,
        pdf_files: list[BinaryIO] | None = None,
        excel_file: BinaryIO | None = None,
        use_multi_model: bool | None = None
    ) -> Estimate:
        """
        Generate estimate from project description.

        Args:
            description: Project description
            department: Department code
            pdf_files: Optional PDF specification files
            excel_file: Optional Excel component hints
            use_multi_model: If True, use multi-model pipeline; if None, use config default

        Returns:
            Estimate object

        Raises:
            AIGenerationError: If AI estimation fails
            ParsingError: If file parsing fails
        """
        # Determine whether to use multi-model
        should_use_multi_model = use_multi_model
        if should_use_multi_model is None:
            should_use_multi_model = self.config.multi_model.enabled

        # Route to appropriate method
        if should_use_multi_model and self.multi_model is not None:
            logger.info(f"🚀 Starting MULTI-MODEL estimation for department {department.value}")
            return self._estimate_multi_model(description, department, pdf_files, excel_file)
        else:
            logger.info(f"🚀 Starting SINGLE-MODEL estimation for department {department.value}")
            return self._estimate_single_model(description, department, pdf_files, excel_file)

    def _estimate_multi_model(
        self,
        description: str,
        department: DepartmentCode,
        pdf_files: list[BinaryIO] | None,
        excel_file: BinaryIO | None
    ) -> Estimate:
        """Execute multi-model pipeline estimation."""
        # Parse files
        excel_data = None
        if excel_file:
            try:
                excel_data = self.excel_parser.parse(excel_file)
                logger.info(f"📊 Parsed {len(excel_data['components'])} components from Excel")
            except Exception as e:
                logger.warning(f"Excel parsing failed: {e}")

        pdf_texts = []
        if pdf_files:
            try:
                pdf_texts = [self.pdf_parser.extract_text(f) for f in pdf_files]
                logger.info(f"📄 Extracted text from {len(pdf_files)} PDFs")
            except Exception as e:
                logger.warning(f"PDF parsing failed: {e}")

        # Find similar projects
        similar_projects = []
        if description:
            try:
                similar_projects = self.pgvector.find_similar_projects(
                    description,
                    department.value,
                    limit=5
                )
                if similar_projects:
                    logger.info(f"🔍 Found {len(similar_projects)} similar projects")
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Build context for multi-model pipeline
        context = StageContext(
            description=description,
            department_code=department.value,
            pdf_texts=pdf_texts,
            excel_data=excel_data,
            similar_projects=similar_projects
        )

        # Execute pipeline
        estimate = self.multi_model.execute_pipeline(context, enable_multi_model=True)

        logger.info(f"✅ Multi-model estimation complete: {estimate.total_hours:.1f}h, {estimate.component_count} components")
        return estimate

    def _estimate_single_model(
        self,
        description: str,
        department: DepartmentCode,
        pdf_files: list[BinaryIO] | None,
        excel_file: BinaryIO | None
    ) -> Estimate:
        """Execute single-model (legacy) estimation."""
        logger.info(f"🚀 Starting estimation for department {department.value}")

        # Parse Excel hints (if provided)
        excel_components = []
        if excel_file:
            try:
                excel_data = self.excel_parser.parse(excel_file)
                excel_components = excel_data['components']
                logger.info(f"📊 Parsed {len(excel_components)} components from Excel")
            except Exception as e:
                logger.warning(f"Excel parsing failed: {e}")

        # Parse PDF specifications (if provided)
        pdf_text = ""
        if pdf_files:
            try:
                pdf_texts = [self.pdf_parser.extract_text(f) for f in pdf_files]
                pdf_text = "\n\n".join(pdf_texts)
                logger.info(f"📄 Extracted {len(pdf_text)} chars from {len(pdf_files)} PDFs")
            except Exception as e:
                logger.warning(f"PDF parsing failed: {e}")

        # Find similar projects (semantic search)
        similar_projects = []
        if description:
            try:
                similar_projects = self.pgvector.find_similar_projects(
                    description,
                    department.value,
                    limit=5
                )
                if similar_projects:
                    logger.info(f"🔍 Found {len(similar_projects)} similar projects")
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Build AI prompt (simplified - full implementation would be more complex)
        prompt = self._build_estimation_prompt(
            description,
            department,
            excel_components,
            pdf_text,
            similar_projects
        )

        # Generate AI estimation
        try:
            ai_response = self.ai.generate_text(
                prompt,
                model=None,  # Use default
                json_mode=True,
                timeout=self.config.ollama.timeout_seconds
            )
            logger.info(f"✅ AI response received ({len(ai_response)} chars)")
        except Exception as e:
            logger.error(f"AI generation failed: {e}", exc_info=True)
            raise AIGenerationError(f"Failed to generate estimate: {e}")

        # Parse AI response to components
        # (Simplified - real implementation would use component_parser.parse_ai_response)
        components = self._parse_ai_components(ai_response, excel_components)

        # Enrich with pattern matching
        components = self._enrich_with_patterns(components, department)

        # Create estimate
        estimate = Estimate.from_components(
            components=components,
            risks=[],
            suggestions=[],
            assumptions=[],
            warnings=[],
            raw_ai_response=ai_response
        )

        logger.info(f"✅ Estimation complete: {estimate.total_hours:.1f}h, {estimate.component_count} components")
        return estimate

    def suggest_bundle_additions(
        self,
        components: list[Component],
        department: DepartmentCode
    ) -> list[dict]:
        """
        Suggest additional components based on bundles.

        Args:
            components: Current components
            department: Department code

        Returns:
            List of suggestions
        """
        suggestions = []

        for component in components:
            if component.is_summary:
                continue

            # Get typical bundles
            bundles = self.bundle_learner.get_typical_bundles(
                component.name,
                department.value
            )

            if bundles:
                for bundle in bundles:
                    # Get pattern for sub-component
                    pattern = self.pattern_learner.get_pattern_for_component(
                        bundle['sub_name'],
                        department.value
                    )

                    if pattern:
                        suggestions.append({
                            'parent': component.name,
                            'sub_name': bundle['sub_name'],
                            'avg_quantity': bundle['avg_quantity'],
                            'hours_layout': pattern.avg_hours_layout,
                            'hours_detail': pattern.avg_hours_detail,
                            'hours_doc': pattern.avg_hours_doc,
                            'confidence': min(pattern.confidence, bundle['confidence']),
                            'source': 'bundle'
                        })

        logger.info(f"💡 Generated {len(suggestions)} bundle suggestions")
        return suggestions

    def _build_estimation_prompt(
        self,
        description: str,
        department: DepartmentCode,
        excel_components: list[dict],
        pdf_text: str,
        similar_projects: list[dict]
    ) -> str:
        """Build AI estimation prompt (simplified)."""
        # This is a simplified version - real implementation would be much more detailed
        prompt = f"""You are a CAD estimation expert. Analyze this project and provide detailed component breakdown.

Department: {department.value}
Description: {description}

PDF Specifications: {pdf_text[:1000] if pdf_text else 'None'}

Excel Components (hints): {len(excel_components)} components provided

Similar Projects:
"""
        for proj in similar_projects[:3]:
            prompt += f"- {proj['name']}: {proj['estimated_hours']:.1f}h (similarity: {proj['similarity']:.0%})\n"

        prompt += """
Return JSON with:
{
  "components": [
    {"name": "Component name", "layout_h": 1.0, "detail_h": 3.0, "doc_h": 1.0, "confidence": 0.8}
  ],
  "overall_confidence": 0.75
}
"""
        return prompt

    def _parse_ai_components(self, ai_response: str, excel_components: list[dict]) -> list[Component]:
        """Parse AI response to Component objects (simplified)."""
        # Simplified - real implementation would use proper JSON parsing with fallbacks
        import json
        try:
            data = json.loads(ai_response)
            components = []

            for comp_data in data.get('components', []):
                component = Component(
                    name=comp_data.get('name', 'Unknown'),
                    hours_3d_layout=float(comp_data.get('layout_h', 0)),
                    hours_3d_detail=float(comp_data.get('detail_h', 0)),
                    hours_2d=float(comp_data.get('doc_h', 0)),
                    confidence=float(comp_data.get('confidence', 0.5))
                )
                components.append(component)

            return components
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            # Fallback: use Excel components
            return [
                Component(
                    name=c.get('name', 'Unknown'),
                    hours_3d_layout=float(c.get('hours_3d_layout', 0)),
                    hours_3d_detail=float(c.get('hours_3d_detail', 0)),
                    hours_2d=float(c.get('hours_2d', 0)),
                    confidence=0.5
                )
                for c in excel_components[:20]
            ]

    def _enrich_with_patterns(self, components: list[Component], department: DepartmentCode) -> list[Component]:
        """Enrich components with pattern data."""
        enriched = []

        for component in components:
            # Try to find pattern
            pattern = self.pattern_learner.get_pattern_for_component(
                component.name,
                department.value
            )

            if pattern and pattern.occurrences >= 3:
                # Use pattern data if high confidence
                enriched_component = Component(
                    name=component.name,
                    hours_3d_layout=pattern.avg_hours_layout,
                    hours_3d_detail=pattern.avg_hours_detail,
                    hours_2d=pattern.avg_hours_doc,
                    confidence=pattern.confidence,
                    confidence_reason=f"Pattern match (n={pattern.occurrences})",
                    category=component.category,
                    comment=component.comment,
                    subcomponents=component.subcomponents,
                    metadata={'pattern_key': pattern.pattern_key}
                )
                enriched.append(enriched_component)
            else:
                enriched.append(component)

        return enriched
