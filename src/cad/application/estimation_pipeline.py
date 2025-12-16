"""
CAD Estimator Pro - Estimation Pipeline

Main orchestrator for CAD project estimation workflow.
"""
import logging
from typing import BinaryIO, Any, Dict


from cad.domain.models import Estimate, Component, DepartmentCode, Risk, Suggestion
from cad.domain.models.config import AppConfig
from cad.domain.exceptions import AIGenerationError, ParsingError
from cad.infrastructure.ai.ollama_client import OllamaClient
from cad.infrastructure.parsers.excel_parser import CADExcelParser
from cad.infrastructure.parsers.pdf_parser import CADPDFParser
from cad.infrastructure.parsers.component_parser import CADComponentParser
from cad.infrastructure.learning.pattern_learner import PatternLearner
from cad.infrastructure.learning.bundle_learner import BundleLearner
from cad.infrastructure.embeddings.pgvector_service import PgVectorService
from cad.infrastructure.multi_model import MultiModelOrchestrator
from cad.domain.models.multi_model import StageContext

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
        multi_model_orchestrator: MultiModelOrchestrator | None = None,
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
        use_multi_model: bool | None = None,
        stage1_model: str | None = None,
        stage2_model: str | None = None,
        stage3_model: str | None = None,
        stage4_model: str | None = None,
    ) -> Estimate:
        """
        Generate estimate from project description.

        Args:
            description: Project description
            department: Department code
            pdf_files: Optional PDF specification files
            excel_file: Optional Excel component hints
            use_multi_model: If True, use multi-model pipeline; if None, use config default
            stage1_model: Optional model override for Stage 1
            stage2_model: Optional model override for Stage 2
            stage3_model: Optional model override for Stage 3
            stage4_model: Optional model override for Stage 4

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
            logger.info(
                f"🚀 Starting MULTI-MODEL estimation for department {department.value}"
            )
            return self._estimate_multi_model(
                description,
                department,
                pdf_files,
                excel_file,
                stage1_model,
                stage2_model,
                stage3_model,
                stage4_model,
            )
        else:
            logger.info(
                f"🚀 Starting SINGLE-MODEL estimation for department {department.value}"
            )
            return self._estimate_single_model(
                description, department, pdf_files, excel_file
            )

    # ================= MULTI-MODEL =================

    def _estimate_multi_model(
        self,
        description: str,
        department: DepartmentCode,
        pdf_files: list[BinaryIO] | None,
        excel_file: BinaryIO | None,
        stage1_model: str | None = None,
        stage2_model: str | None = None,
        stage3_model: str | None = None,
        stage4_model: str | None = None,
    ) -> Estimate:
        """Execute multi-model pipeline estimation."""
        # Parse files
        excel_data = None
        if excel_file:
            try:
                excel_data = self.excel_parser.parse(excel_file)
                logger.info(
                    f"📊 Parsed {len(excel_data['components'])} components from Excel"
                )
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
                    description, department.value, limit=5
                )
                if similar_projects:
                    logger.info(
                        f"🔍 Found {len(similar_projects)} similar projects (multi-model)"
                    )
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Build context for multi-model pipeline
        context = StageContext(
            description=description,
            department_code=department.value,
            pdf_texts=pdf_texts,
            excel_data=excel_data,
            similar_projects=similar_projects,
        )

        # Execute pipeline with model overrides
        estimate = self.multi_model.execute_pipeline(
            context,
            enable_multi_model=True,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            stage3_model=stage3_model,
            stage4_model=stage4_model,
        )

        logger.info(
            f"✅ Multi-model estimation complete: {estimate.total_hours:.1f}h, {estimate.component_count} components"
        )
        return estimate

    # ================= SINGLE-MODEL =================

    def _estimate_single_model(
        self,
        description: str,
        department: DepartmentCode,
        pdf_files: list[BinaryIO] | None,
        excel_file: BinaryIO | None,
    ) -> Estimate:
        """Execute single-model (legacy) estimation."""
        logger.info(f"🚀 Starting SINGLE-MODEL estimation for department {department.value}")

        # Parse Excel hints (if provided)
        excel_components = []
        if excel_file:
            try:
                excel_data = self.excel_parser.parse(excel_file)
                excel_components = excel_data["components"]
                logger.info(
                    f"📊 Parsed {len(excel_components)} components from Excel (single-model)"
                )
            except Exception as e:
                logger.warning(f"Excel parsing failed: {e}")

        # Parse PDF specifications (if provided)
        pdf_text = ""
        if pdf_files:
            try:
                pdf_texts = [self.pdf_parser.extract_text(f) for f in pdf_files]
                pdf_text = "\n\n".join(pdf_texts)
                logger.info(
                    f"📄 Extracted {len(pdf_text)} chars from {len(pdf_files)} PDFs (single-model)"
                )
            except Exception as e:
                logger.warning(f"PDF parsing failed: {e}")

        # Find similar projects (semantic search)
        similar_projects = []
        if description:
            try:
                similar_projects = self.pgvector.find_similar_projects(
                    description, department.value, limit=5
                )
                if similar_projects:
                    logger.info(
                        f"🔍 Found {len(similar_projects)} similar projects (single-model)"
                    )
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Build AI prompt
        prompt = self._build_estimation_prompt(
            description,
            department,
            excel_components,
            pdf_text,
            similar_projects,
        )

        # Generate AI estimation
        try:
            ai_response = self.ai.generate_text(
                prompt,
                model=None,  # Use default
                json_mode=True,
                timeout=self.config.ollama.timeout_seconds,
            )
            logger.info(f"✅ AI response received ({len(ai_response)} chars)")
        except Exception as e:
            logger.error(f"AI generation failed: {e}", exc_info=True)
            raise AIGenerationError(f"Failed to generate estimate: {e}")

        # Parse AI response to components
        components = self._parse_ai_components(ai_response, excel_components)

        # Enrich with pattern matching
        components = self._enrich_with_patterns(components, department)

        # Track pre-scaling total for metadata
        total_before_scaling = sum(
            c.total_hours for c in components
            if not getattr(c, "is_summary", False)
        )

        # Minimalne godziny — skalowanie w górę, jeśli total jest nienaturalnie niski
        components = self._apply_minimum_hours_to_components(components, department)

        # Check if scaling was applied
        total_after_scaling = sum(
            c.total_hours for c in components
            if not getattr(c, "is_summary", False)
        )
        was_scaled = abs(total_after_scaling - total_before_scaling) > 0.1

        # Create estimate
        estimate = Estimate.from_components(
            components=components,
            risks=[],
            suggestions=[],
            assumptions=[],
            warnings=[],
            raw_ai_response=ai_response,
        )

        # Enrich with single-model metadata
        if not estimate.generation_metadata:
            estimate.generation_metadata = {}

        estimate.generation_metadata.update({
            "multi_model": False,
            "pipeline_type": "single_model",
            "had_excel_file": excel_file is not None,
            "had_pdf_files": pdf_files is not None and len(pdf_files) > 0,
            "similar_projects": similar_projects if similar_projects else [],
            "similar_projects_count": len(similar_projects) if similar_projects else 0,
            "description": description,
            "department": department.value,
        })

        if was_scaled:
            scale_factor = total_after_scaling / total_before_scaling if total_before_scaling > 0 else 1.0
            estimate.generation_metadata["scaling_info"] = (
                f"Przeskalowano z {total_before_scaling:.1f}h do {total_after_scaling:.1f}h "
                f"(współczynnik: {scale_factor:.2f}x)"
            )

        logger.info(
            f"✅ Estimation complete (single-model): {estimate.total_hours:.1f}h, {estimate.component_count} components"
        )
        return estimate

    # ================= BUNDLES =================

    def suggest_bundle_additions(
        self, components: list[Component], department: DepartmentCode
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
                component.name, department.value
            )

            if bundles:
                for bundle in bundles:
                    # Get pattern for sub-component
                    pattern = self.pattern_learner.get_pattern_for_component(
                        bundle["sub_name"], department.value
                    )

                    if pattern:
                        suggestions.append(
                            {
                                "parent": component.name,
                                "sub_name": bundle["sub_name"],
                                "avg_quantity": bundle["avg_quantity"],
                                "hours_layout": pattern.avg_hours_layout,
                                "hours_detail": pattern.avg_hours_detail,
                                "hours_doc": pattern.avg_hours_doc,
                                "confidence": min(
                                    pattern.confidence, bundle["confidence"]
                                ),
                                "source": "bundle",
                            }
                        )

        logger.info(f"💡 Generated {len(suggestions)} bundle suggestions")
        return suggestions

    # ================= PROMPT DLA SINGLE-MODELU =================

    def _build_estimation_prompt(
        self,
        description: str,
        department: DepartmentCode,
        excel_components: list[dict],
        pdf_text: str,
        similar_projects: list[dict],
    ) -> str:
        """Build AI estimation prompt for SINGLE-MODEL pipeline."""
        # krótki przegląd komponentów z Excela (jeśli są)
        excel_preview = "none"
        if excel_components:
            names = [c.get("name", "Unknown") for c in excel_components[:10]]
            excel_preview = f"{len(excel_components)} components, e.g.: " + ", ".join(
                names
            )

        # podobne projekty
        if similar_projects:
            similar_block_lines = []
            for proj in similar_projects[:3]:
                name = proj.get("name", "N/A")
                est = proj.get("estimated_hours", 0.0) or 0.0
                sim = proj.get("similarity", 0.0) or 0.0
                similar_block_lines.append(
                    f"- {name}: {est:.1f}h (similarity: {sim:.0%})"
                )
            similar_block = "\n".join(similar_block_lines)
        else:
            similar_block = "none"

        pdf_preview = pdf_text[:1000] if pdf_text else "none"

        return f"""You are a senior CAD/CAM estimator. Analyze this project and provide a realistic, engineering-grade estimate.

PROJECT CONTEXT:
- Department code: {department.value}
- Description: {description}

PDF SPECIFICATIONS (truncated):
{pdf_preview}

EXCEL COMPONENT HINTS:
{excel_preview}

SIMILAR HISTORICAL PROJECTS (from internal database):
{similar_block}

TASK:
1. Propose a reasonable component breakdown for this project (4–12 unique components for typical mechanical projects).
2. For EACH component estimate three hour buckets:
   - layout_h  – 3D layout / positioning in assembly,
   - detail_h  – detailed 3D modelling with all relevant features,
   - doc_h     – 2D drawings with dimensions and annotations.
3. Provide a global overall_confidence between 0.3 and 0.9.

CONSTRAINTS AND RANGES:
- Use realistic ranges: 0.3–80 hours per component per phase (most parts will be lower).
- Total project hours should realistically NOT be below:
  - 25–30h for very simple welded frames or fixtures,
  - 40h+ for medium projects,
  - 60h+ for more complex projects.
- If in doubt, ERR ON THE SIDE OF OVERESTIMATION rather than underestimation.
- Keep all numbers as plain decimals (e.g. 1.5, 3.0), not strings.

OUTPUT FORMAT:
Return ONE valid JSON object and NOTHING else. No markdown, no comments.

EXPECTED JSON STRUCTURE:
{{
  "components": [
    {{
      "name": "Component name",
      "layout_h": 2.0,
      "detail_h": 8.0,
      "doc_h": 4.0,
      "confidence": 0.7
    }}
  ],
  "overall_confidence": 0.75
}}
"""

    # ================= PARSING/ENRICHMENT =================

    def _parse_ai_components(
        self, ai_response: str, excel_components: list[dict]
    ) -> list[Component]:
        """Parse AI response to Component objects (simplified)."""
        import json

        try:
            data = json.loads(ai_response)
            components: list[Component] = []

            for comp_data in data.get("components", []):
                component = Component(
                    name=comp_data.get("name", "Unknown"),
                    hours_3d_layout=float(comp_data.get("layout_h", 0)),
                    hours_3d_detail=float(comp_data.get("detail_h", 0)),
                    hours_2d=float(comp_data.get("doc_h", 0)),
                    confidence=float(comp_data.get("confidence", 0.5)),
                )
                components.append(component)

            return components
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            # Fallback: use Excel components
            return [
                Component(
                    name=c.get("name", "Unknown"),
                    hours_3d_layout=float(c.get("hours_3d_layout", 0)),
                    hours_3d_detail=float(c.get("hours_3d_detail", 0)),
                    hours_2d=float(c.get("hours_2d", 0)),
                    confidence=0.5,
                )
                for c in excel_components[:20]
            ]

    def _enrich_with_patterns(
        self, components: list[Component], department: DepartmentCode
    ) -> list[Component]:
        """
        Enrich components with pattern data using multi-strategy matching:
        1. Exact pattern key match (canonicalized name)
        2. Vector similarity search (semantic matching)
        3. Keep AI estimate if no matches found
        """
        enriched: list[Component] = []

        for component in components:
            # Strategy 1: Try exact pattern match
            pattern = self.pattern_learner.get_pattern_for_component(
                component.name, department.value
            )

            if pattern and pattern.occurrences >= 3:
                # High confidence exact match - use pattern fully
                enriched_component = Component(
                    name=component.name,
                    hours_3d_layout=pattern.avg_hours_layout,
                    hours_3d_detail=pattern.avg_hours_detail,
                    hours_2d=pattern.avg_hours_doc,
                    confidence=pattern.confidence,
                    confidence_reason=f"Exact pattern match (n={pattern.occurrences})",
                    category=component.category,
                    comment=component.comment,
                    subcomponents=component.subcomponents,
                    metadata={"pattern_key": pattern.pattern_key, "match_type": "exact"},
                )
                enriched.append(enriched_component)
                logger.debug(f"✓ Exact match: {component.name} → {pattern.pattern_key}")

            elif pattern and pattern.occurrences >= 1:
                # Low occurrence exact match - use weighted blending
                blended = self._blend_pattern_with_ai(
                    component, pattern, blend_strategy="low_occurrence"
                )
                enriched.append(blended)
                logger.debug(f"⚖️ Blended match: {component.name} (n={pattern.occurrences})")

            else:
                # Strategy 2: Try vector similarity search
                similar_patterns = self.pgvector.find_similar_components(
                    name=component.name,
                    department=department.value,
                    limit=3,
                    similarity_threshold=0.70  # 70% similarity minimum
                )

                if similar_patterns and len(similar_patterns) > 0:
                    best_match = similar_patterns[0]

                    # Check if best match has enough occurrences
                    if best_match.get('occurrences', 0) >= 3:
                        # Use similar pattern with adjusted confidence
                        similarity_score = best_match.get('similarity', 0.0)
                        base_confidence = best_match.get('confidence', 0.5)
                        # Adjust confidence based on similarity
                        adjusted_confidence = base_confidence * similarity_score

                        enriched_component = Component(
                            name=component.name,
                            hours_3d_layout=best_match['avg_hours_3d_layout'],
                            hours_3d_detail=best_match['avg_hours_3d_detail'],
                            hours_2d=best_match['avg_hours_2d'],
                            confidence=adjusted_confidence,
                            confidence_reason=(
                                f"Similar to '{best_match['name']}' "
                                f"(similarity: {similarity_score:.0%}, n={best_match['occurrences']})"
                            ),
                            category=component.category,
                            comment=component.comment,
                            subcomponents=component.subcomponents,
                            metadata={
                                "pattern_key": best_match['pattern_key'],
                                "match_type": "vector_similar",
                                "similarity": similarity_score,
                                "similar_to": best_match['name']
                            },
                        )
                        enriched.append(enriched_component)
                        logger.info(
                            f"🔍 Vector match: {component.name} → {best_match['name']} "
                            f"(sim: {similarity_score:.0%})"
                        )
                    else:
                        # Similar pattern but low occurrences - blend
                        blended = self._blend_similar_pattern_with_ai(
                            component, best_match
                        )
                        enriched.append(blended)
                        logger.debug(
                            f"⚖️ Blended similar: {component.name} → {best_match['name']}"
                        )
                else:
                    # Strategy 3: No match found - keep AI estimate
                    enriched.append(component)
                    logger.debug(f"🤖 AI estimate kept: {component.name}")

        return enriched

    def _blend_pattern_with_ai(
        self,
        component: Component,
        pattern: ComponentPattern,
        blend_strategy: str = "low_occurrence"
    ) -> Component:
        """
        Blend pattern data with AI estimate based on pattern occurrence count.

        Args:
            component: AI-generated component
            pattern: Matched pattern with low occurrences
            blend_strategy: Blending strategy

        Returns:
            Blended component
        """
        n = pattern.occurrences

        # Determine blend weights based on occurrences
        if n == 1:
            # Very low confidence - 40% pattern, 60% AI
            pattern_weight = 0.4
        elif n == 2:
            # Low confidence - 70% pattern, 30% AI
            pattern_weight = 0.7
        else:
            # Should not reach here, but default to pattern
            pattern_weight = 1.0

        ai_weight = 1.0 - pattern_weight

        # Blend hours
        blended_layout = (
            pattern.avg_hours_layout * pattern_weight +
            component.hours_3d_layout * ai_weight
        )
        blended_detail = (
            pattern.avg_hours_detail * pattern_weight +
            component.hours_3d_detail * ai_weight
        )
        blended_doc = (
            pattern.avg_hours_doc * pattern_weight +
            component.hours_2d * ai_weight
        )

        # Blend confidence
        blended_confidence = pattern.confidence * pattern_weight + component.confidence * ai_weight

        return Component(
            name=component.name,
            hours_3d_layout=blended_layout,
            hours_3d_detail=blended_detail,
            hours_2d=blended_doc,
            confidence=blended_confidence,
            confidence_reason=(
                f"Blended: {int(pattern_weight*100)}% pattern (n={n}) "
                f"+ {int(ai_weight*100)}% AI"
            ),
            category=component.category,
            comment=component.comment,
            subcomponents=component.subcomponents,
            metadata={
                "pattern_key": pattern.pattern_key,
                "match_type": "blended",
                "pattern_weight": pattern_weight,
                "ai_weight": ai_weight
            },
        )

    def _blend_similar_pattern_with_ai(
        self,
        component: Component,
        similar_pattern: dict
    ) -> Component:
        """
        Blend similar pattern (from vector search) with AI estimate.

        Args:
            component: AI-generated component
            similar_pattern: Similar pattern dict from vector search

        Returns:
            Blended component
        """
        similarity = similar_pattern.get('similarity', 0.0)
        occurrences = similar_pattern.get('occurrences', 0)

        # Weight based on similarity and occurrences
        if occurrences >= 2:
            pattern_weight = similarity * 0.6  # Max 60% for similar with n>=2
        else:
            pattern_weight = similarity * 0.4  # Max 40% for similar with n=1

        ai_weight = 1.0 - pattern_weight

        # Blend hours
        blended_layout = (
            similar_pattern['avg_hours_3d_layout'] * pattern_weight +
            component.hours_3d_layout * ai_weight
        )
        blended_detail = (
            similar_pattern['avg_hours_3d_detail'] * pattern_weight +
            component.hours_3d_detail * ai_weight
        )
        blended_doc = (
            similar_pattern['avg_hours_2d'] * pattern_weight +
            component.hours_2d * ai_weight
        )

        # Blend confidence
        pattern_confidence = similar_pattern.get('confidence', 0.5) * similarity
        blended_confidence = pattern_confidence * pattern_weight + component.confidence * ai_weight

        return Component(
            name=component.name,
            hours_3d_layout=blended_layout,
            hours_3d_detail=blended_detail,
            hours_2d=blended_doc,
            confidence=blended_confidence,
            confidence_reason=(
                f"Blended with similar '{similar_pattern['name']}': "
                f"{int(pattern_weight*100)}% pattern + {int(ai_weight*100)}% AI "
                f"(similarity: {similarity:.0%}, n={occurrences})"
            ),
            category=component.category,
            comment=component.comment,
            subcomponents=component.subcomponents,
            metadata={
                "pattern_key": similar_pattern['pattern_key'],
                "match_type": "blended_similar",
                "pattern_weight": pattern_weight,
                "ai_weight": ai_weight,
                "similarity": similarity,
                "similar_to": similar_pattern['name']
            },
        )

    def _apply_minimum_hours_to_components(
        self, components: list[Component], department: DepartmentCode
    ) -> list[Component]:
        """
        Zapewnia, że całkowita liczba godzin nie jest nienaturalnie niska.
        Jeśli total_hours < progu minimalnego dla działu, skaluje wszystkie godziny w górę.
        """
        total_hours = sum(
            c.total_hours for c in components
            if not getattr(c, "is_summary", False)
        )

        if total_hours <= 0:
            return components

        # Minimalne progi per dział – skalibruj pod siebie
        min_total_by_dept: dict[str, float] = {
            "131": 25.0,  # Automotive
            "132": 35.0,  # Industrial Machinery
            "133": 35.0,  # Transportation
            "134": 45.0,  # Heavy Equipment
            "135": 35.0,  # Special Purpose Machinery
        }
        min_total = min_total_by_dept.get(department.value, 30.0)

        if total_hours >= min_total:
            return components

        scale = min_total / total_hours
        logger.info(
            f"⚖️ Single-model estimate too low ({total_hours:.1f}h < {min_total:.1f}h), "
            f"scaling all component hours by x{scale:.2f}"
        )

        scaled: list[Component] = []
        for c in components:
            if getattr(c, "is_summary", False):
                scaled.append(c)
                continue

            scaled.append(
                Component(
                    name=c.name,
                    hours_3d_layout=c.hours_3d_layout * scale,
                    hours_3d_detail=c.hours_3d_detail * scale,
                    hours_2d=c.hours_2d * scale,
                    confidence=c.confidence,
                    confidence_reason=(
                        (c.confidence_reason or "")
                        + f" (scaled x{scale:.2f} to min total {min_total:.1f}h)"
                    ),
                    category=c.category,
                    comment=c.comment,
                    subcomponents=c.subcomponents,
                    metadata=c.metadata,
                )
            )

        return scaled

    # ================= PROJECT BRAIN: PRE-CHECK WYMAGAŃ =================

    def precheck_requirements(
        self,
        description: str,
        department: DepartmentCode,
        pdf_files: list[BinaryIO] | None = None,
        excel_file: BinaryIO | None = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """
        Project Brain – wstępna analiza wymagań przed estymacją.

        Zwraca JSON:
        {
          "missing_info": [...],
          "clarifying_questions": [...],
          "suggested_components": [...],
          "risk_flags": [
            {"description":"...", "impact":"low/medium/high", "mitigation":""}
          ]
        }
        """
        if not description or not description.strip():
            return {
                "missing_info": ["Brak opisu projektu."],
                "clarifying_questions": [],
                "suggested_components": [],
                "risk_flags": [],
            }

        # Parse Excel (opcjonalnie)
        excel_components: list[dict] = []
        if excel_file:
            try:
                excel_data = self.excel_parser.parse(excel_file)
                excel_components = excel_data.get("components", [])
            except Exception as e:
                logger.warning(f"[Precheck] Excel parsing failed: {e}")

        # Parse PDF (opcjonalnie)
        pdf_text = ""
        if pdf_files:
            try:
                pdf_texts = [self.pdf_parser.extract_text(f) for f in pdf_files]
                pdf_text = "\n\n".join(pdf_texts)
            except Exception as e:
                logger.warning(f"[Precheck] PDF parsing failed: {e}")

        # Podobne projekty (opcjonalnie)
        similar_projects: list[dict] = []
        try:
            similar_projects = self.pgvector.find_similar_projects(
                description, department.value, limit=3
            )
        except Exception as e:
            logger.warning(f"[Precheck] Semantic search failed: {e}")

        prompt = self._build_precheck_prompt(
            description,
            department,
            excel_components,
            pdf_text,
            similar_projects,
        )

        model_to_use = model or self.config.ollama.text_model

        try:
            ai_response = self.ai.generate_text(
                prompt,
                model=model_to_use,
                json_mode=True,
                timeout=self.config.ollama.timeout_seconds,
            )
            logger.info(f"[Precheck] AI response received ({len(ai_response)} chars)")
        except Exception as e:
            logger.error(f"[Precheck] AI generation failed: {e}", exc_info=True)
            return {
                "missing_info": ["Nie udało się przeprowadzić automatycznej analizy wymagań."],
                "clarifying_questions": [],
                "suggested_components": [],
                "risk_flags": [],
            }

        import json as _json

        try:
            data = _json.loads(ai_response)
        except _json.JSONDecodeError:
            start_idx = ai_response.find("{")
            end_idx = ai_response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                try:
                    data = _json.loads(ai_response[start_idx:end_idx])
                except Exception:
                    data = {}
            else:
                data = {}

        if not isinstance(data, dict):
            data = {}

        missing_info = data.get("missing_info", [])
        clarifying_questions = data.get("clarifying_questions") or data.get("questions", [])
        suggested_components = data.get("suggested_components", [])
        risk_flags = data.get("risk_flags", [])

        return {
            "missing_info": missing_info or [],
            "clarifying_questions": clarifying_questions or [],
            "suggested_components": suggested_components or [],
            "risk_flags": risk_flags or [],
        }

    def _build_precheck_prompt(
        self,
        description: str,
        department: DepartmentCode,
        excel_components: list[dict],
        pdf_text: str,
        similar_projects: list[dict],
    ) -> str:
        """Prompt dla Project Brain pre-check – wykrywa braki w wymaganiach i pytania doprecyzowujące."""
        if excel_components:
            names = [c.get("name", "Unknown") for c in excel_components[:8]]
            excel_summary = f"{len(excel_components)} komponentów (np.: " + ", ".join(names) + ")"
        else:
            excel_summary = "brak jawnej listy komponentów"

        pdf_preview = pdf_text[:1200] if pdf_text else "brak osobnych specyfikacji PDF"

        if similar_projects:
            similar_lines = []
            for p in similar_projects[:3]:
                name = p.get("name", "N/A")
                est = p.get("estimated_hours", 0.0) or 0.0
                sim = p.get("similarity", 0.0) or 0.0
                similar_lines.append(f"- {name}: {est:.1f}h (similarity: {sim:.0%})")
            similar_block = "\n".join(similar_lines)
        else:
            similar_block = "brak podobnych projektów w bazie"

        return f"""
Jesteś doświadczonym inżynierem CAD/CAM i PM. Twoim zadaniem jest analiza, czy opis projektu ma wystarczające informacje
do wykonania REALISTYCZNEJ estymacji godzin projektowych.

PROJECT CONTEXT:
- Department code: {department.value}
- Description (from user):
{description}

EXCEL HINTS:
{excel_summary}

PDF SPECIFICATIONS (truncated):
{pdf_preview}

SIMILAR HISTORICAL PROJECTS (if any):
{similar_block}

ZADANIE:
1. Wypisz, jakich kluczowych informacji technicznych BRAKUJE w tym opisie, aby dobrze oszacować godziny CAD.
2. Sformułuj pytania doprecyzowujące do klienta / konstruktora.
3. Zaproponuj typowe komponenty / obszary, o których warto pamiętać (np. osłony, napędy, czujniki, dokumentacja).
4. Opcjonalnie wypisz 1–5 potencjalnych ryzyk wynikających z braków w wymaganiach.

ZWŁASZCZA WEŹ POD UWAGĘ:
- Wymiary, masę, obciążenia.
- Materiały / normy.
- Napędy, sterowanie, bezpieczeństwo.
- Środowisko pracy (temperatura, wilgoć, spożywka, pył).
- Liczbę modułów / osi / stacji.

FORMAT WYJŚCIA (TYLKO JSON, bez komentarzy, bez markdownu):
{{
  "missing_info": [
    "brak długości i szerokości ramy",
    "nie podano wymagań dokładności pozycjonowania",
    "brak informacji o środowisku pracy (temperatura, wilgoć, spożywka, pył)"
  ],
  "clarifying_questions": [
    "Jaka jest dokładna długość, szerokość i wysokość konstrukcji?",
    "Jaka jest masa maksymalna przenoszonego detalu / ładunku?",
    "W jakim środowisku będzie pracowała konstrukcja (temperatura, wilgoć, pył, kontakt z żywnością)?"
  ],
  "suggested_components": [
    "osłony bezpieczeństwa wokół stref ruchomych",
    "czujniki krańcowe / bezpieczeństwa",
    "elementy poziomowania i kotwienia do posadzki"
  ],
  "risk_flags": [
    {{
      "description": "Brak informacji o środowisku pracy (spożywka / pył), co wpływa na dobór materiałów i zabezpieczeń.",
      "impact": "medium",
      "mitigation": "Doprecyzować wymagania higieniczne / IP przed rozpoczęciem szczegółowego projektu."
    }}
  ]
}}

ZASADY:
- Nie wymyślaj fikcyjnych danych – wskazuj TYLKO to, czego naprawdę brakuje.
- Listy powinny być zwięzłe (3–10 pozycji), ale konkretne.
- Pisz wyłącznie po polsku.
- Zwróć TYLKO jeden obiekt JSON w powyższym formacie.
"""
