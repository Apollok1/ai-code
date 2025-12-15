"""
CAD Estimator Pro - Batch Importer

Parallel batch import of Excel files with pattern learning.
"""
import logging
from typing import Callable
from typing import BinaryIO, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from cad.domain.models import DepartmentCode, Component, Estimate
from cad.domain.models.config import AppConfig
from cad.infrastructure.parsers.excel_parser import CADExcelParser
from cad.infrastructure.learning.pattern_learner import PatternLearner
from cad.infrastructure.learning.bundle_learner import BundleLearner

logger = logging.getLogger(__name__)


class BatchImporter:
    """
    Batch importer for historical Excel files.

    Features:
    - Parallel processing (ThreadPoolExecutor)
    - Pattern learning from imported data
    - Bundle learning from sub-components
    - Progress tracking
    """

    def __init__(
        self,
        config: AppConfig,
        db_client: Any,
        excel_parser: CADExcelParser,
        pattern_learner: PatternLearner,
        bundle_learner: BundleLearner
    ):
        """
        Initialize BatchImporter.

        Args:
            config: Application configuration
            db_client: Database client
            excel_parser: Excel parser
            pattern_learner: Pattern learner
            bundle_learner: Bundle learner
        """
        self.config = config
        self.db = db_client
        self.excel_parser = excel_parser
        self.pattern_learner = pattern_learner
        self.bundle_learner = bundle_learner
        self.max_workers = config.ui.max_workers

    def import_batch(
        self,
        files: list[tuple[str, BinaryIO]],
        department: DepartmentCode,
        learn_patterns: bool = True,
        progress_callback: Callable | None = None
    ) -> list[dict]:
        """
        Import batch of Excel files in parallel.

        Args:
            files: List of (filename, file_stream) tuples
            department: Department code
            learn_patterns: Enable pattern learning
            progress_callback: Optional callback(current, total, filename)

        Returns:
            List of results:
            [
                {'filename': '...', 'status': 'success/error', 'project_id': 123, 'error': '...'},
                ...
            ]
        """
        logger.info(f"📦 Starting batch import: {len(files)} files, department={department.value}")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._import_single_file,
                    filename,
                    file_stream,
                    department,
                    learn_patterns
                ): filename
                for filename, file_stream in files
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file), 1):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    if progress_callback:
                        progress_callback(i, len(files), filename)

                except Exception as e:
                    logger.error(f"Import failed for '{filename}': {e}", exc_info=True)
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'error': str(e)
                    })

        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"✅ Batch import complete: {success_count}/{len(files)} successful")

        return results

    def _import_single_file(
        self,
        filename: str,
        file_stream: BinaryIO,
        department: DepartmentCode,
        learn_patterns: bool
    ) -> dict:
        """Import single Excel file."""
        try:
            # Parse Excel
            excel_data = self.excel_parser.parse(file_stream)
            components_data = excel_data['components']

            # Extract description from A1
            file_stream.seek(0)
            description = self.excel_parser.extract_description_from_a1(file_stream)
            if not description:
                description = f"Projekt historyczny: {filename}"

            # Convert to Component objects
            components = []
            for comp_data in components_data:
                if comp_data.get('is_summary'):
                    continue

                component = Component(
                    name=comp_data.get('name', 'Unknown'),
                    hours_3d_layout=float(comp_data.get('hours_3d_layout', 0)),
                    hours_3d_detail=float(comp_data.get('hours_3d_detail', 0)),
                    hours_2d=float(comp_data.get('hours_2d', 0)),
                    confidence=0.7,  # Historical data has decent confidence
                    confidence_reason="Historical import",
                    comment=comp_data.get('comment', '')
                )
                components.append(component)

            # Create estimate
            estimate = Estimate.from_components(components)

            # Save project
            from cad.domain.models import Project
            project = Project(
                id=None,
                name=filename.replace('.xlsx', '').replace('.xls', ''),
                department=department,
                estimate=estimate,
                description=description,
                is_historical=True
            )

            project_id = self.db.save_project(project)

            # Learn patterns
            if learn_patterns:
                learned_count = 0
                for component in components:
                    try:
                        self.pattern_learner.learn_from_component(
                            name=component.name,
                            department=department.value,
                            hours_layout=component.hours_3d_layout,
                            hours_detail=component.hours_3d_detail,
                            hours_doc=component.hours_2d,
                            source='historical_excel'
                        )
                        learned_count += 1

                        # Learn bundles from sub-components
                        if component.subcomponents:
                            subs = [
                                {'name': sub.name, 'quantity': sub.quantity}
                                for sub in component.subcomponents
                            ]
                            self.bundle_learner.learn_from_component_with_subs(
                                component.name,
                                subs,
                                department.value
                            )

                    except Exception as e:
                        logger.warning(f"Pattern learning failed for '{component.name}': {e}")
                        continue

                logger.info(f"📚 Learned {learned_count} patterns from '{filename}'")

            return {
                'filename': filename,
                'status': 'success',
                'project_id': project_id,
                'components_count': len(components),
                'total_hours': estimate.total_hours
            }

        except Exception as e:
            logger.error(f"Import failed for '{filename}': {e}", exc_info=True)
            return {
                'filename': filename,
                'status': 'error',
                'error': str(e)
            }
