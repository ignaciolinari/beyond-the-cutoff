#!/usr/bin/env python3
"""Validate experiment results completeness and quality.

This script validates:
1. All expected experimental conditions are present
2. Dataset version consistency across runs
3. Dataset quality (required fields, citation coverage, task type distribution)

Usage examples:

    # Validate experiment completeness
    python scripts/validate_experiment_results.py \
        --results-dir evaluation/results

    # Validate with expected dataset hash
    python scripts/validate_experiment_results.py \
        --results-dir evaluation/results \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --check-quality
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from beyond_the_cutoff.evaluation.dataset_quality import (
    analyze_citation_coverage,
    analyze_task_type_distribution,
    generate_quality_report,
    validate_required_fields,
    validate_task_type_distribution,
)
from beyond_the_cutoff.evaluation.experiment_validation import (
    validate_experiment_results,
)
from beyond_the_cutoff.utils.experiment_logging import compute_file_sha256
from beyond_the_cutoff.utils.validation import print_validation_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate experiment results completeness and quality"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to offline dataset JSONL (for quality checks and hash validation)",
    )
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Run dataset quality checks",
    )
    parser.add_argument(
        "--expected-dataset-hash",
        type=str,
        default=None,
        help="Expected dataset SHA256 hash for validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    all_passed = True

    # Validate experiment completeness and dataset consistency
    print("[info] Validating experiment completeness and dataset consistency...", file=sys.stderr)
    expected_hash = args.expected_dataset_hash
    if args.dataset and args.dataset.exists() and not expected_hash:
        # Compute hash from dataset file
        try:
            expected_hash = compute_file_sha256(args.dataset)
            print(f"[info] Computed dataset hash: {expected_hash[:16]}...", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] Failed to compute dataset hash: {exc}", file=sys.stderr)

    experiment_result = validate_experiment_results(
        results_dir, expected_dataset_hash=expected_hash
    )
    print_validation_result(experiment_result)
    if not experiment_result.passed:
        all_passed = False

    # Dataset quality checks
    if args.check_quality and args.dataset:
        print("\n[info] Running dataset quality checks...", file=sys.stderr)
        dataset_path = args.dataset.resolve()

        # Required fields validation
        print("\n[info] Checking required fields...", file=sys.stderr)
        fields_result = validate_required_fields(dataset_path)
        print_validation_result(fields_result)
        if not fields_result.passed:
            all_passed = False

        # Task type distribution
        print("\n[info] Analyzing task type distribution...", file=sys.stderr)
        task_dist = analyze_task_type_distribution(dataset_path)
        if task_dist:
            print("[info] Task type distribution:", file=sys.stderr)
            for task_type, count in sorted(task_dist.items()):
                print(f"  {task_type}: {count}", file=sys.stderr)

        task_dist_result = validate_task_type_distribution(dataset_path)
        print_validation_result(task_dist_result)
        if not task_dist_result.passed:
            all_passed = False

        # Citation coverage
        print("\n[info] Analyzing citation coverage...", file=sys.stderr)
        citation_stats = analyze_citation_coverage(dataset_path)
        if citation_stats:
            print("[info] Citation coverage statistics:", file=sys.stderr)
            for key, value in citation_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}", file=sys.stderr)
                else:
                    print(f"  {key}: {value}", file=sys.stderr)

        # Generate comprehensive quality report
        print("\n[info] Generating quality report...", file=sys.stderr)
        quality_report = generate_quality_report(dataset_path)
        print(f"[info] Total examples: {quality_report.total_examples}", file=sys.stderr)
        print(
            f"[info] Required fields present: {quality_report.required_fields_present}",
            file=sys.stderr,
        )
        if quality_report.issues:
            print(f"[info] Found {len(quality_report.issues)} quality issue(s)", file=sys.stderr)
            for issue in quality_report.issues[:5]:  # Show first 5
                print(f"  [{issue.severity.upper()}] {issue.message}", file=sys.stderr)

    if all_passed:
        print("\n[✓] All validations passed!", file=sys.stderr)
        return 0
    else:
        print("\n[✗] Some validations failed. See details above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
