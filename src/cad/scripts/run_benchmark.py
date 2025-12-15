#!/usr/bin/env python3
"""
CAD Estimator Pro - Benchmark Script

Example script showing how to run offline benchmarks against historical data.

Usage:
    python -m cad.scripts.run_benchmark

    or with custom models:

    python -m cad.scripts.run_benchmark --stage1 qwen2.5:32b --stage2 qwen2.5:14b
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad.domain.models import DepartmentCode
from cad.domain.models.config import AppConfig
from cad.infrastructure.factory import create_ai_client, create_database_client
from cad.infrastructure.multi_model import MultiModelOrchestrator
from cad.infrastructure.benchmarking import BenchmarkRunner


# EXAMPLE BENCHMARK DATA
# In production, load this from database or CSV file
EXAMPLE_BENCHMARKS = [
    {
        'project_id': 'TEST-001',
        'description': 'Rama stalowa pod przenośnik taśmowy 5m, ciężar 500kg',
        'department': DepartmentCode.MECHANIKA,
        'actual_hours': 45.0,
        'actual_component_count': 8
    },
    {
        'project_id': 'TEST-002',
        'description': 'Obudowa elektryczna IP65, wymiary 800x600x300mm, stalowa ocynkowana',
        'department': DepartmentCode.ELEKTRYKA,
        'actual_hours': 28.5,
        'actual_component_count': 6
    },
    {
        'project_id': 'TEST-003',
        'description': 'Układ rurociągów technologicznych DN50, długość 25m, stal nierdzewna',
        'department': DepartmentCode.TECHNOLOGIA,
        'actual_hours': 62.0,
        'actual_component_count': 12
    },
]


def main():
    parser = argparse.ArgumentParser(description='Run CAD Estimator benchmarks')
    parser.add_argument('--stage1', type=str, help='Model for Stage 1')
    parser.add_argument('--stage2', type=str, help='Model for Stage 2')
    parser.add_argument('--stage3', type=str, help='Model for Stage 3')
    parser.add_argument('--stage4', type=str, help='Model for Stage 4')
    parser.add_argument('--data', type=str, help='Path to benchmark data CSV (optional)')

    args = parser.parse_args()

    print("=" * 80)
    print("CAD ESTIMATOR PRO - BENCHMARK RUNNER")
    print("=" * 80)
    print()

    # Load configuration
    print("Loading configuration...")
    config = AppConfig.from_env()

    # Initialize clients
    print("Initializing AI client...")
    ai_client = create_ai_client(config)

    print("Initializing database client...")
    db_client = create_database_client(config)

    # Create orchestrator
    print("Creating multi-model orchestrator...")
    orchestrator = MultiModelOrchestrator(
        ai_client=ai_client,
        db_client=db_client,
        config=config.multi_model
    )

    # Create benchmark runner
    print("Initializing benchmark runner...")
    runner = BenchmarkRunner(orchestrator, db_client)

    # Load benchmark data
    if args.data:
        print(f"Loading benchmark data from {args.data}...")
        # TODO: Implement CSV/JSON loading
        benchmarks = EXAMPLE_BENCHMARKS
    else:
        print("Using example benchmark data...")
        benchmarks = EXAMPLE_BENCHMARKS

    print(f"Loaded {len(benchmarks)} benchmark projects")
    print()

    # Display model configuration
    print("Model Configuration:")
    print(f"  Stage 1: {args.stage1 or config.multi_model.stage1_model}")
    print(f"  Stage 2: {args.stage2 or config.multi_model.stage2_model}")
    print(f"  Stage 3: {args.stage3 or config.multi_model.stage3_model}")
    print(f"  Stage 4: {args.stage4 or config.multi_model.stage4_model}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 80)

    results = runner.run_benchmark_suite(
        benchmarks=benchmarks,
        stage1_model=args.stage1,
        stage2_model=args.stage2,
        stage3_model=args.stage3,
        stage4_model=args.stage4,
    )

    # Print report
    runner.print_report(results)

    # Optionally save results to file
    # import json
    # with open('benchmark_results.json', 'w') as f:
    #     json.dump([vars(r) for r in results], f, indent=2, default=str)


if __name__ == '__main__':
    main()
