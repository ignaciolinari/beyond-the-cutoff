#!/usr/bin/env python3
"""Validate experiment configuration, reproducibility, and results.

This script provides a convenient way to validate:
- Configuration files for correctness and conflicts
- Dataset versioning across runs
- Experiment reproducibility (metadata completeness)
- Evaluation sanity (error rates, missing predictions, etc.)

Usage examples:

    # Validate a single evaluation run
    python scripts/validate_experiment.py \
        --config configs/default.yaml \
        --metadata evaluation/results/rag_baseline_0p5b/metadata.jsonl

    # Validate dataset versioning across multiple runs
    python scripts/validate_experiment.py \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --dataset evaluation/results/rag_baseline_0p5b/details.jsonl

    # Validate configuration before running evaluation
    python scripts/validate_experiment.py \
        --config configs/default.yaml \
        --model-config configs/models/base_ollama.yaml \
        --judge-config configs/judges/rag.yaml \
        --prompt-mode rag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from beyond_the_cutoff.utils.validation import (
    print_validation_result,
    validate_configuration,
    validate_dataset_versioning,
    validate_evaluation_sanity,
    validate_experiment_reproducibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate experiment configuration, reproducibility, and results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Project config file path",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Model inference config file path",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        help="Judge prompt config file path",
    )
    parser.add_argument(
        "--judge-inference",
        type=Path,
        help="Judge inference config file path",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["rag", "instruction"],
        help="Prompt mode (rag or instruction)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        dest="datasets",
        help="Dataset file path (can be specified multiple times for versioning check)",
    )
    parser.add_argument(
        "--expected-dataset-hash",
        help="Expected SHA256 hash for dataset validation",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata JSONL file path for reproducibility validation",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Metrics JSON file path for sanity check",
    )
    parser.add_argument(
        "--details",
        type=Path,
        help="Details JSONL file path for sanity check",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.1,
        help="Maximum acceptable error rate for sanity check (default: 0.1)",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1,
        help="Minimum number of examples required (default: 1)",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with non-zero code if any warnings are found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = args.config
    if config_path is None:
        # Try default config
        default_config = Path("configs/default.yaml")
        if default_config.exists():
            config_path = default_config
        else:
            print(
                "[error] No config file specified and default config not found.",
                file=sys.stderr,
            )
            print(
                "  Use --config to specify a config file.",
                file=sys.stderr,
            )
            return 1

    all_passed = True
    any_warnings = False

    # Validate configuration
    if config_path:
        config_validation = validate_configuration(
            config_path,
            model_config_path=args.model_config,
            judge_config_path=args.judge_config,
            judge_inference_path=args.judge_inference,
            prompt_mode=args.prompt_mode,
        )
        print_validation_result(config_validation)
        if not config_validation.passed:
            all_passed = False
        if any(i.severity == "warning" for i in config_validation.issues):
            any_warnings = True

    # Validate dataset versioning
    if args.datasets:
        dataset_validation = validate_dataset_versioning(
            args.datasets,
            expected_hash=args.expected_dataset_hash,
        )
        print_validation_result(dataset_validation)
        if not dataset_validation.passed:
            all_passed = False
        if any(i.severity == "warning" for i in dataset_validation.issues):
            any_warnings = True

    # Validate experiment reproducibility
    if args.metadata:
        reproducibility_validation = validate_experiment_reproducibility(args.metadata)
        print_validation_result(reproducibility_validation)
        if not reproducibility_validation.passed:
            all_passed = False
        if any(i.severity == "warning" for i in reproducibility_validation.issues):
            any_warnings = True

    # Validate evaluation sanity
    if args.metrics or args.details:
        sanity_validation = validate_evaluation_sanity(
            metrics_path=args.metrics,
            details_path=args.details,
            max_error_rate=args.max_error_rate,
            min_examples=args.min_examples,
        )
        print_validation_result(sanity_validation)
        if not sanity_validation.passed:
            all_passed = False
        if any(i.severity == "warning" for i in sanity_validation.issues):
            any_warnings = True

    # Exit with appropriate code
    if not all_passed:
        return 1
    if args.fail_on_warnings and any_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
