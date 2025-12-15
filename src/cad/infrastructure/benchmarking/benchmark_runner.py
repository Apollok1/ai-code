"""
CAD Estimator Pro - Benchmark Runner

Validates pipeline performance against historical project data.
"""
import logging
from dataclasses import dataclass
from typing import Any
from datetime import datetime

from cad.domain.models import Estimate, DepartmentCode
from cad.domain.models.multi_model import StageContext
from cad.infrastructure.multi_model import MultiModelOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of single benchmark test."""

    project_id: str
    description: str
    department: DepartmentCode

    # Ground truth (actual values from completed project)
    actual_hours: float
    actual_component_count: int

    # Predicted values
    predicted_hours: float
    predicted_component_count: int

    # Error metrics
    hours_error_absolute: float
    hours_error_percentage: float
    component_count_error: int

    # Stage timings
    stage1_duration_sec: float
    stage2_duration_sec: float
    stage3_duration_sec: float
    stage4_duration_sec: float
    total_duration_sec: float

    # Model configuration used
    models_used: dict[str, str]

    # Success/failure
    success: bool
    error_message: str | None = None

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BenchmarkRunner:
    """
    Runs benchmarks on historical project data.

    Usage:
        runner = BenchmarkRunner(orchestrator, db_client)
        results = runner.run_benchmark_suite(project_ids)
        metrics = runner.calculate_aggregate_metrics(results)
    """

    def __init__(
        self,
        orchestrator: MultiModelOrchestrator,
        db_client: Any,  # DatabaseClient
    ):
        """
        Initialize benchmark runner.

        Args:
            orchestrator: Multi-model pipeline orchestrator
            db_client: Database client for loading historical data
        """
        self.orchestrator = orchestrator
        self.db_client = db_client

    def run_single_benchmark(
        self,
        project_id: str,
        description: str,
        department: DepartmentCode,
        actual_hours: float,
        actual_component_count: int,
        stage1_model: str | None = None,
        stage2_model: str | None = None,
        stage3_model: str | None = None,
        stage4_model: str | None = None,
    ) -> BenchmarkResult:
        """
        Run benchmark on single historical project.

        Args:
            project_id: Project ID for reference
            description: Project description
            department: Department code
            actual_hours: Ground truth hours (from completed project)
            actual_component_count: Ground truth component count
            stage1_model: Optional model override for Stage 1
            stage2_model: Optional model override for Stage 2
            stage3_model: Optional model override for Stage 3
            stage4_model: Optional model override for Stage 4

        Returns:
            BenchmarkResult with metrics
        """
        import time

        logger.info(f"Running benchmark: {project_id}")

        models_used = {
            'stage1': stage1_model or self.orchestrator.config.stage1_model,
            'stage2': stage2_model or self.orchestrator.config.stage2_model,
            'stage3': stage3_model or self.orchestrator.config.stage3_model,
            'stage4': stage4_model or self.orchestrator.config.stage4_model,
        }

        try:
            # Create context
            context = StageContext(
                description=description,
                department_code=department,
                pdf_texts=[],
                image_analyses=[],
                excel_data=None
            )

            # Time each stage
            start_time = time.time()

            stage1_start = time.time()
            tech_analysis = self.orchestrator.stage1.analyze(context, model=stage1_model)
            context = context.with_technical_analysis(tech_analysis)
            stage1_duration = time.time() - stage1_start

            stage2_start = time.time()
            structure = self.orchestrator.stage2.decompose(context, model=stage2_model)
            context = context.with_structural_decomposition(structure)
            stage2_duration = time.time() - stage2_start

            stage3_start = time.time()
            context = self.orchestrator.stage3.estimate_hours(context, model=stage3_model)
            stage3_duration = time.time() - stage3_start

            stage4_start = time.time()
            risks, suggestions, assumptions, warnings = self.orchestrator.stage4.analyze_risks(
                context, model=stage4_model
            )
            stage4_duration = time.time() - stage4_start

            total_duration = time.time() - start_time

            # Extract predictions
            predicted_hours = sum(c.total_hours for c in context.estimated_components)
            predicted_component_count = len(context.estimated_components)

            # Calculate errors
            hours_error_abs = abs(predicted_hours - actual_hours)
            hours_error_pct = (hours_error_abs / actual_hours * 100) if actual_hours > 0 else 0
            component_error = predicted_component_count - actual_component_count

            return BenchmarkResult(
                project_id=project_id,
                description=description,
                department=department,
                actual_hours=actual_hours,
                actual_component_count=actual_component_count,
                predicted_hours=predicted_hours,
                predicted_component_count=predicted_component_count,
                hours_error_absolute=hours_error_abs,
                hours_error_percentage=hours_error_pct,
                component_count_error=component_error,
                stage1_duration_sec=stage1_duration,
                stage2_duration_sec=stage2_duration,
                stage3_duration_sec=stage3_duration,
                stage4_duration_sec=stage4_duration,
                total_duration_sec=total_duration,
                models_used=models_used,
                success=True
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {project_id}: {e}", exc_info=True)
            return BenchmarkResult(
                project_id=project_id,
                description=description,
                department=department,
                actual_hours=actual_hours,
                actual_component_count=actual_component_count,
                predicted_hours=0.0,
                predicted_component_count=0,
                hours_error_absolute=actual_hours,
                hours_error_percentage=100.0,
                component_count_error=-actual_component_count,
                stage1_duration_sec=0.0,
                stage2_duration_sec=0.0,
                stage3_duration_sec=0.0,
                stage4_duration_sec=0.0,
                total_duration_sec=0.0,
                models_used=models_used,
                success=False,
                error_message=str(e)
            )

    def run_benchmark_suite(
        self,
        benchmarks: list[dict],
        stage1_model: str | None = None,
        stage2_model: str | None = None,
        stage3_model: str | None = None,
        stage4_model: str | None = None,
    ) -> list[BenchmarkResult]:
        """
        Run benchmarks on multiple projects.

        Args:
            benchmarks: List of dicts with keys:
                - project_id
                - description
                - department
                - actual_hours
                - actual_component_count
            stage1_model: Optional model override for all benchmarks
            stage2_model: Optional model override
            stage3_model: Optional model override
            stage4_model: Optional model override

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        logger.info(f"Starting benchmark suite with {len(benchmarks)} projects")

        for i, benchmark in enumerate(benchmarks, 1):
            logger.info(f"Benchmark {i}/{len(benchmarks)}: {benchmark['project_id']}")

            result = self.run_single_benchmark(
                project_id=benchmark['project_id'],
                description=benchmark['description'],
                department=benchmark['department'],
                actual_hours=benchmark['actual_hours'],
                actual_component_count=benchmark['actual_component_count'],
                stage1_model=stage1_model,
                stage2_model=stage2_model,
                stage3_model=stage3_model,
                stage4_model=stage4_model,
            )

            results.append(result)

            logger.info(f"Result: {result.predicted_hours:.1f}h (actual: {result.actual_hours:.1f}h), "
                       f"error: {result.hours_error_percentage:.1f}%")

        logger.info(f"Benchmark suite complete: {len(results)} results")
        return results

    @staticmethod
    def calculate_aggregate_metrics(results: list[BenchmarkResult]) -> dict[str, Any]:
        """
        Calculate aggregate metrics across all benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Dict with aggregate metrics:
            - success_rate
            - mean_hours_error_pct
            - median_hours_error_pct
            - max_hours_error_pct
            - mean_component_error
            - mean_duration_sec
            - total_tests
        """
        if not results:
            return {}

        successful = [r for r in results if r.success]

        if not successful:
            return {
                'total_tests': len(results),
                'success_rate': 0.0,
                'failed_tests': len(results)
            }

        hours_errors_pct = [r.hours_error_percentage for r in successful]
        component_errors = [abs(r.component_count_error) for r in successful]
        durations = [r.total_duration_sec for r in successful]

        return {
            'total_tests': len(results),
            'successful_tests': len(successful),
            'failed_tests': len(results) - len(successful),
            'success_rate': len(successful) / len(results) * 100,

            # Hours estimation accuracy
            'mean_hours_error_pct': sum(hours_errors_pct) / len(hours_errors_pct),
            'median_hours_error_pct': sorted(hours_errors_pct)[len(hours_errors_pct) // 2],
            'max_hours_error_pct': max(hours_errors_pct),
            'min_hours_error_pct': min(hours_errors_pct),

            # Component count accuracy
            'mean_component_error': sum(component_errors) / len(component_errors),
            'median_component_error': sorted(component_errors)[len(component_errors) // 2],

            # Performance
            'mean_duration_sec': sum(durations) / len(durations),
            'total_duration_sec': sum(durations),
        }

    @staticmethod
    def print_report(results: list[BenchmarkResult]) -> None:
        """Print human-readable benchmark report."""
        metrics = BenchmarkRunner.calculate_aggregate_metrics(results)

        print("\n" + "=" * 80)
        print("BENCHMARK REPORT")
        print("=" * 80)
        print(f"\nTotal tests: {metrics['total_tests']}")
        print(f"Successful: {metrics['successful_tests']} ({metrics['success_rate']:.1f}%)")
        print(f"Failed: {metrics['failed_tests']}")

        if metrics['successful_tests'] > 0:
            print(f"\nHours Estimation Accuracy:")
            print(f"  Mean error: {metrics['mean_hours_error_pct']:.1f}%")
            print(f"  Median error: {metrics['median_hours_error_pct']:.1f}%")
            print(f"  Max error: {metrics['max_hours_error_pct']:.1f}%")
            print(f"  Min error: {metrics['min_hours_error_pct']:.1f}%")

            print(f"\nComponent Count Accuracy:")
            print(f"  Mean error: {metrics['mean_component_error']:.1f} components")
            print(f"  Median error: {metrics['median_component_error']:.0f} components")

            print(f"\nPerformance:")
            print(f"  Mean duration: {metrics['mean_duration_sec']:.1f}s per project")
            print(f"  Total duration: {metrics['total_duration_sec']:.1f}s")

        print("\n" + "=" * 80)

        # Print individual results
        print("\nIndividual Results:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            status = "✓" if result.success else "✗"
            print(f"{status} {i}. {result.project_id}: "
                  f"predicted={result.predicted_hours:.1f}h, "
                  f"actual={result.actual_hours:.1f}h, "
                  f"error={result.hours_error_percentage:.1f}%")
