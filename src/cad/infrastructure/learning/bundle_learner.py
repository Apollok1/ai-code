"""
CAD Estimator Pro - Bundle Learner

Learns typical component bundles (parent→sub relationships) from historical data.
"""
import logging
from typing import Any

from cad.domain.models.config import LearningConfig
from cad.domain.exceptions import PatternLearningError
from cad.infrastructure.parsers.component_parser import CADComponentParser

logger = logging.getLogger(__name__)


class BundleLearner:
    """
    Bundle learner for component relationships.

    Learns which sub-components typically appear with parent components.
    Example: "Wspornik" often appears with "Śruba M12" (2-4 qty)
    """

    def __init__(self, config: LearningConfig, db_client: Any):
        """
        Initialize BundleLearner.

        Args:
            config: Learning configuration
            db_client: Database client (DatabaseClient protocol)
        """
        self.config = config
        self.db = db_client
        self.component_parser = CADComponentParser()

    def learn_bundle(
        self,
        parent_name: str,
        sub_name: str,
        department: str,
        quantity: int = 1
    ) -> bool:
        """
        Learn bundle relationship (parent→sub).

        Args:
            parent_name: Parent component name
            sub_name: Sub-component name
            department: Department code
            quantity: Quantity of sub-component

        Returns:
            True if successful

        Raises:
            PatternLearningError: If learning fails
        """
        try:
            # Canonicalize names
            parent_key = self.component_parser.canonicalize_component_name(parent_name)
            sub_key = self.component_parser.canonicalize_component_name(sub_name)

            if not parent_key or not sub_key:
                logger.warning(f"Cannot canonicalize bundle: '{parent_name}' → '{sub_name}'")
                return False

            # Get existing bundle or create new
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if bundle exists
                    cur.execute("""
                        SELECT occurrences, total_qty FROM component_bundles
                        WHERE department = %s AND parent_key = %s AND sub_key = %s
                    """, (department, parent_key, sub_key))
                    row = cur.fetchone()

                    if row:
                        # Update existing
                        occurrences, total_qty = row
                        new_occurrences = occurrences + 1
                        new_total_qty = total_qty + quantity
                        confidence = min(0.95, 1.0 - (1.0 / (new_occurrences ** 0.5)))

                        cur.execute("""
                            UPDATE component_bundles
                            SET occurrences = %s, total_qty = %s, confidence = %s
                            WHERE department = %s AND parent_key = %s AND sub_key = %s
                        """, (new_occurrences, new_total_qty, confidence, department, parent_key, sub_key))

                        logger.debug(f"Updated bundle: {parent_name} → {sub_name} (n={new_occurrences})")
                    else:
                        # Create new
                        cur.execute("""
                            INSERT INTO component_bundles (
                                department, parent_key, parent_name, sub_key, sub_name,
                                occurrences, total_qty, confidence
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (department, parent_key, parent_name, sub_key, sub_name, 1, quantity, 0.5))

                        logger.debug(f"Created bundle: {parent_name} → {sub_name}")

                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Bundle learning failed: {e}", exc_info=True)
            raise PatternLearningError(f"Failed to learn bundle: {e}")

    def learn_from_component_with_subs(
        self,
        parent_name: str,
        subcomponents: list[dict],
        department: str
    ) -> int:
        """
        Learn bundles from component with sub-components.

        Args:
            parent_name: Parent component name
            subcomponents: List of sub-components [{'name': ..., 'quantity': ...}]
            department: Department code

        Returns:
            Number of bundles learned
        """
        learned = 0
        for sub in subcomponents:
            sub_name = sub.get('name', '').strip()
            quantity = int(sub.get('quantity', 1))

            if not sub_name or quantity < 1:
                continue

            try:
                self.learn_bundle(parent_name, sub_name, department, quantity)
                learned += 1
            except Exception as e:
                logger.warning(f"Failed to learn bundle {parent_name}→{sub_name}: {e}")
                continue

        if learned > 0:
            logger.info(f"✅ Learned {learned} bundles for '{parent_name}'")

        return learned

    def get_typical_bundles(
        self,
        parent_name: str,
        department: str,
        min_occurrences: int | None = None
    ) -> list[dict]:
        """
        Get typical bundles for parent component.

        Args:
            parent_name: Parent component name
            department: Department code
            min_occurrences: Min occurrences filter (None = use config default)

        Returns:
            List of bundles:
            [
                {
                    'sub_name': 'Śruba M12',
                    'avg_quantity': 3.5,
                    'occurrences': 10,
                    'confidence': 0.85
                },
                ...
            ]
        """
        min_occ = self.config.min_bundle_occurrences if min_occurrences is None else min_occurrences
        parent_key = self.component_parser.canonicalize_component_name(parent_name)

        if not parent_key:
            return []

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT sub_name, total_qty, occurrences, confidence
                        FROM component_bundles
                        WHERE department = %s AND parent_key = %s AND occurrences >= %s
                        ORDER BY occurrences DESC
                    """, (department, parent_key, min_occ))
                    rows = cur.fetchall()

                    bundles = []
                    for row in rows:
                        sub_name, total_qty, occurrences, confidence = row
                        avg_quantity = total_qty / occurrences if occurrences > 0 else 0

                        bundles.append({
                            'sub_name': sub_name,
                            'avg_quantity': avg_quantity,
                            'occurrences': occurrences,
                            'confidence': confidence
                        })

                    return bundles

        except Exception as e:
            logger.error(f"Failed to get bundles for '{parent_name}': {e}", exc_info=True)
            return []
