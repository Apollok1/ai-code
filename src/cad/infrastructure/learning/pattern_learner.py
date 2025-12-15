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
                # Create new pattern (first observation, M2 = 0)
                updated = ComponentPattern(
                    name=name,
                    pattern_key=pattern_key,
                    department_code=department,
                    avg_hours_layout=hours_layout,
                    avg_hours_detail=hours_detail,
                    avg_hours_doc=hours_doc,
                    m2_hours_layout=0.0,  # First observation, no variance yet
                    m2_hours_detail=0.0,
                    m2_hours_doc=0.0,
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

    def _is_outlier_zscore(
        self,
        new: float,
        mean: float,
        m2: float,
        n: int
    ) -> bool:
        """
        Pełny outlier check na bazie Z-score:
        z = |x - mean| / std, gdzie std = sqrt(variance),
        variance = M2 / (n - 1).

        Zwraca True, jeśli z > threshold.

        Args:
            new: New observation
            mean: Current mean
            m2: Current M2 (sum of squared deviations)
            n: Current number of observations

        Returns:
            True if outlier detected
        """
        if n < max(2, self.config.welford_min_n):
            # Za mało danych na sensowną wariancję
            return False

        variance = m2 / (n - 1) if n > 1 else 0.0
        if variance <= 0:
            return False

        std = variance ** 0.5
        if std == 0:
            return False

        z = abs(new - mean) / std
        return z > self.config.welford_outlier_threshold

    def _welford_update(
        self,
        pattern: ComponentPattern,
        new_layout: float,
        new_detail: float,
        new_doc: float
    ) -> ComponentPattern:
        """
        Pełny Welford dla trzech wymiarów (layout/detail/doc) + Z-score outlier detection.

        Welford (dla pojedynczej serii x_n):
        - M_n = M_{n-1} + (x_n - M_{n-1}) / n
        - M2_n = M2_{n-1} + (x_n - M_{n-1}) * (x_n - M_n)
        - variance = M2_n / (n-1)

        Args:
            pattern: Existing pattern
            new_layout: New layout hours observation
            new_detail: New detail hours observation
            new_doc: New doc hours observation

        Returns:
            Updated ComponentPattern with M2 tracking
        """
        n = pattern.occurrences

        # --- OUTLIER DETECTION NA BAZIE Z-SCORE ---
        is_outlier = False
        if (
            self._is_outlier_zscore(
                new_layout,
                pattern.avg_hours_layout,
                getattr(pattern, "m2_hours_layout", 0.0),
                n,
            )
            or self._is_outlier_zscore(
                new_detail,
                pattern.avg_hours_detail,
                getattr(pattern, "m2_hours_detail", 0.0),
                n,
            )
            or self._is_outlier_zscore(
                new_doc,
                pattern.avg_hours_doc,
                getattr(pattern, "m2_hours_doc", 0.0),
                n,
            )
        ):
            is_outlier = True
            logger.warning(
                f"Outlier (Z-score) for '{pattern.name}': "
                f"L={new_layout:.1f} (avg={pattern.avg_hours_layout:.1f}), "
                f"D={new_detail:.1f} (avg={pattern.avg_hours_detail:.1f}), "
                f"Doc={new_doc:.1f} (avg={pattern.avg_hours_doc:.1f})"
            )

        if is_outlier:
            # Ignorujemy outlier: NIE zmieniamy średnich, M2 ani liczby obserwacji
            return pattern

        # --- WELFORD UPDATE DLA KAŻDEJ SKŁADOWEJ ---

        n_new = n + 1

        # Layout
        old_mean_L = pattern.avg_hours_layout
        old_m2_L = getattr(pattern, "m2_hours_layout", 0.0)
        delta_L = new_layout - old_mean_L
        mean_L = old_mean_L + delta_L / n_new
        delta2_L = new_layout - mean_L
        m2_L = old_m2_L + delta_L * delta2_L

        # Detail
        old_mean_D = pattern.avg_hours_detail
        old_m2_D = getattr(pattern, "m2_hours_detail", 0.0)
        delta_D = new_detail - old_mean_D
        mean_D = old_mean_D + delta_D / n_new
        delta2_D = new_detail - mean_D
        m2_D = old_m2_D + delta_D * delta2_D

        # Doc
        old_mean_doc = pattern.avg_hours_doc
        old_m2_doc = getattr(pattern, "m2_hours_doc", 0.0)
        delta_doc = new_doc - old_mean_doc
        mean_doc = old_mean_doc + delta_doc / n_new
        delta2_doc = new_doc - mean_doc
        m2_doc = old_m2_doc + delta_doc * delta2_doc

        # Confidence rośnie z sqrt(n), max 0.95
        confidence = min(0.95, 1.0 - (1.0 / (n_new ** 0.5)))

        return ComponentPattern(
            name=pattern.name,
            pattern_key=pattern.pattern_key,
            department_code=pattern.department_code,
            avg_hours_layout=mean_L,
            avg_hours_detail=mean_D,
            avg_hours_doc=mean_doc,
            m2_hours_layout=m2_L,
            m2_hours_detail=m2_D,
            m2_hours_doc=m2_doc,
            confidence=confidence,
            occurrences=n_new,
            source=pattern.source,
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
