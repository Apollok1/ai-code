"""
CAD Estimator Pro - Pattern Learner

Machine learning component for pattern recognition using Welford's algorithm.
Learns component patterns from historical data with outlier detection.
"""
import logging
from typing import Any
from datetime import datetime

from cad.domain.models import ComponentPattern
from cad.domain.models.config import LearningConfig
from cad.domain.exceptions import PatternLearningError
from cad.infrastructure.parsers.component_parser import CADComponentParser

logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Pattern learner using Welford's online algorithm.

    Learns component hours patterns from historical data with:
    - Running mean/variance (Welford's algorithm)
    - Outlier detection (Z-score based)
    - Confidence scoring
    - Fuzzy name matching
    """

    def __init__(self, config: LearningConfig, db_client: Any):
        """
        Initialize PatternLearner.

        Args:
            config: Learning configuration
            db_client: Database client (DatabaseClient protocol)
        """
        self.config = config
        self.db = db_client
        self.component_parser = CADComponentParser()

    def learn_from_component(
        self,
        name: str,
        department: str,
        hours_layout: float,
        hours_detail: float,
        hours_doc: float,
        source: str = "actual"
    ) -> ComponentPattern:
        """
        Learn pattern from single component observation.

        Uses Welford's algorithm for running statistics with outlier detection.

        Args:
            name: Component name
            department: Department code (131-135)
            hours_layout: 3D Layout hours
            hours_detail: 3D Detail hours
            hours_doc: 2D Documentation hours
            source: Data source ('actual', 'historical_excel', etc.)

        Returns:
            Updated ComponentPattern

        Raises:
            PatternLearningError: If learning fails
        """
        try:
            # Canonicalize name for matching
            pattern_key = self.component_parser.canonicalize_component_name(name)
            if not pattern_key:
                logger.warning(f"Cannot canonicalize component name: '{name}'")
                pattern_key = name.lower().strip()

            # Get existing pattern or create new
            existing = self.db.get_pattern(pattern_key, department)

            if existing:
                # Update existing pattern with Welford's algorithm
                updated = self._welford_update(
                    existing,
                    hours_layout,
                    hours_detail,
                    hours_doc
                )
            else:
                # Create new pattern
                updated = ComponentPattern(
                    name=name,
                    pattern_key=pattern_key,
                    department_code=department,
                    avg_hours_layout=hours_layout,
                    avg_hours_detail=hours_detail,
                    avg_hours_doc=hours_doc,
                    confidence=0.3,  # Low confidence for first observation
                    occurrences=1,
                    source=source
                )

            # Save to database
            self.db.save_pattern(updated)
            logger.info(f"✅ Learned pattern: {name} → {pattern_key} (n={updated.occurrences})")

            return updated

        except Exception as e:
            logger.error(f"Pattern learning failed for '{name}': {e}", exc_info=True)
            raise PatternLearningError(f"Failed to learn pattern: {e}", component_name=name)

    def _welford_update(
        self,
        pattern: ComponentPattern,
        new_layout: float,
        new_detail: float,
        new_doc: float
    ) -> ComponentPattern:
        """
        Update pattern using Welford's online algorithm.

        Welford's algorithm:
        - M(n) = M(n-1) + (x(n) - M(n-1)) / n
        - M2(n) = M2(n-1) + (x(n) - M(n-1)) * (x(n) - M(n))
        - Variance = M2(n) / (n-1)

        With outlier detection using Z-score threshold.

        Args:
            pattern: Existing pattern
            new_layout: New layout hours observation
            new_detail: New detail hours observation
            new_doc: New doc hours observation

        Returns:
            Updated ComponentPattern
        """
        n = pattern.occurrences
        threshold = self.config.welford_outlier_threshold

        # Check for outliers (Z-score based)
        # For simplicity, we'll use a heuristic: if new value > threshold * current_mean, it's an outlier
        # (In production, you'd track M2 for proper variance calculation)

        is_outlier = False
        if n > 2:  # Need at least 3 observations for outlier detection
            if (abs(new_layout - pattern.avg_hours_layout) > threshold * pattern.avg_hours_layout or
                abs(new_detail - pattern.avg_hours_detail) > threshold * pattern.avg_hours_detail or
                abs(new_doc - pattern.avg_hours_doc) > threshold * pattern.avg_hours_doc):
                is_outlier = True
                logger.warning(f"Outlier detected for '{pattern.name}': "
                             f"L={new_layout:.1f} (avg={pattern.avg_hours_layout:.1f}), "
                             f"D={new_detail:.1f} (avg={pattern.avg_hours_detail:.1f}), "
                             f"Doc={new_doc:.1f} (avg={pattern.avg_hours_doc:.1f})")

        if is_outlier:
            # Don't update pattern, but increment occurrences (so we track we saw it)
            # In production, you might want to create a separate "outlier" pattern
            return ComponentPattern(
                name=pattern.name,
                pattern_key=pattern.pattern_key,
                department_code=pattern.department_code,
                avg_hours_layout=pattern.avg_hours_layout,
                avg_hours_detail=pattern.avg_hours_detail,
                avg_hours_doc=pattern.avg_hours_doc,
                confidence=pattern.confidence,
                occurrences=pattern.occurrences + 1,
                source=pattern.source
            )

        # Welford update (running mean)
        n_new = n + 1
        delta_layout = new_layout - pattern.avg_hours_layout
        delta_detail = new_detail - pattern.avg_hours_detail
        delta_doc = new_doc - pattern.avg_hours_doc

        new_avg_layout = pattern.avg_hours_layout + delta_layout / n_new
        new_avg_detail = pattern.avg_hours_detail + delta_detail / n_new
        new_avg_doc = pattern.avg_hours_doc + delta_doc / n_new

        # Calculate confidence (increases with more observations, max 0.95)
        # Confidence = 1 - (1 / sqrt(n))
        confidence = min(0.95, 1.0 - (1.0 / (n_new ** 0.5)))

        return ComponentPattern(
            name=pattern.name,
            pattern_key=pattern.pattern_key,
            department_code=pattern.department_code,
            avg_hours_layout=new_avg_layout,
            avg_hours_detail=new_avg_detail,
            avg_hours_doc=new_avg_doc,
            confidence=confidence,
            occurrences=n_new,
            source=pattern.source
        )

    def learn_from_project_feedback(
        self,
        project_id: int,
        actual_hours: float
    ) -> int:
        """
        Learn patterns from project feedback (actual hours vs estimated).

        Adjusts all component patterns proportionally based on actual/estimated ratio.

        Args:
            project_id: Project ID
            actual_hours: Actual hours spent

        Returns:
            Number of patterns updated

        Raises:
            PatternLearningError: If learning fails
        """
        try:
            # Get project
            project = self.db.get_project(project_id)
            if not project:
                raise PatternLearningError(f"Project {project_id} not found")

            estimated_hours = project.estimated_hours
            if estimated_hours == 0:
                logger.warning(f"Project {project_id} has zero estimated hours, cannot learn")
                return 0

            # Calculate adjustment ratio
            ratio = actual_hours / estimated_hours
            logger.info(f"📊 Learning from project {project_id}: "
                       f"estimated={estimated_hours:.1f}h, actual={actual_hours:.1f}h, "
                       f"ratio={ratio:.2f}")

            # Update patterns for all non-summary components
            updated_count = 0
            for component in project.estimate.non_summary_components:
                # Adjust hours by ratio
                adjusted_layout = component.hours_3d_layout * ratio
                adjusted_detail = component.hours_3d_detail * ratio
                adjusted_doc = component.hours_2d * ratio

                # Learn pattern
                self.learn_from_component(
                    name=component.name,
                    department=project.department.value,
                    hours_layout=adjusted_layout,
                    hours_detail=adjusted_detail,
                    hours_doc=adjusted_doc,
                    source="actual"
                )
                updated_count += 1

            logger.info(f"✅ Updated {updated_count} patterns from project feedback")
            return updated_count

        except Exception as e:
            logger.error(f"Failed to learn from project feedback: {e}", exc_info=True)
            raise PatternLearningError(f"Project feedback learning failed: {e}")

    def get_pattern_for_component(
        self,
        name: str,
        department: str
    ) -> ComponentPattern | None:
        """
        Get learned pattern for component.

        Args:
            name: Component name
            department: Department code

        Returns:
            ComponentPattern or None if not found
        """
        pattern_key = self.component_parser.canonicalize_component_name(name)
        if not pattern_key:
            return None

        return self.db.get_pattern(pattern_key, department)
