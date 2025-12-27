#!/usr/bin/env python3
"""Validate and run scientifically valid comparisons for the 6-condition experiment.

This script:
1. Validates that all required result files exist
2. Extracts raw metrics from each condition
3. Computes within-group and cross-group comparisons
4. Generates comparison reports with appropriate warnings

Usage:
    # Validate setup (dry run)
    python scripts/validate_experiment_comparisons.py --config configs/evaluation/six_condition_comparisons.yaml --dry-run

    # Generate comparison report
    python scripts/validate_experiment_comparisons.py --config configs/evaluation/six_condition_comparisons.yaml --output evaluation/results/comparison_report.json

    # Run specific comparison type
    python scripts/validate_experiment_comparisons.py --config configs/evaluation/six_condition_comparisons.yaml --comparison-type within-group
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelCondition:
    """Represents one experimental condition."""

    name: str
    result_dir: Path
    eval_mode: Literal["instruction", "rag"]
    training_mode: Literal["none", "instruction", "rag"]
    description: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two conditions."""

    name: str
    model_a: str
    model_b: str
    research_question: str
    comparison_type: Literal["within-group", "cross-group"]
    valid_metrics: list[str]
    invalid_metrics: list[str]
    metric_differences: dict[str, float]  # model_b - model_a
    winner: str | None
    warning: str | None
    expected: str | None


@dataclass
class ExperimentValidation:
    """Validation results for the experiment setup."""

    all_results_exist: bool
    missing_results: list[str]
    conditions_loaded: int
    warnings: list[str]


# =============================================================================
# Metrics Extraction
# =============================================================================


def extract_metrics(result_dir: Path, eval_mode: str) -> dict[str, float]:
    """Extract metrics from a condition's result directory."""
    metrics_file = result_dir / "metrics.json"

    if not metrics_file.exists():
        return {}

    with open(metrics_file) as f:
        data = json.load(f)

    # Extract relevant metrics based on eval mode
    metrics = {}

    # Common metrics (always extract)
    for key in ["factuality", "completeness", "communication"]:
        if key in data:
            metrics[key] = data[key]
        elif "dimensions" in data and key in data["dimensions"]:
            metrics[key] = data["dimensions"][key]

    # RAG-specific metrics
    if eval_mode == "rag":
        for key in ["grounding", "citation_quality"]:
            if key in data:
                metrics[key] = data[key]
            elif "dimensions" in data and key in data["dimensions"]:
                metrics[key] = data["dimensions"][key]

    # Overall/weighted total
    if "overall" in data:
        metrics["weighted_total"] = data["overall"]
    elif "weighted_score" in data:
        metrics["weighted_total"] = data["weighted_score"]

    return metrics


# =============================================================================
# Comparison Logic
# =============================================================================


def compute_comparison(
    model_a: ModelCondition,
    model_b: ModelCondition,
    comparison_config: dict[str, Any],
    comparison_type: Literal["within-group", "cross-group"],
) -> ComparisonResult:
    """Compute a single comparison between two conditions."""

    # Determine valid metrics based on comparison type
    if comparison_type == "within-group":
        # All metrics valid within same evaluation mode
        valid_metrics = ["factuality", "completeness", "communication", "weighted_total"]
        if model_a.eval_mode == "rag":
            valid_metrics.append("grounding")
        invalid_metrics = []
    else:
        # Cross-group: only raw dimension scores
        valid_metrics = comparison_config.get(
            "metrics_to_compare", ["factuality", "completeness", "communication"]
        )
        invalid_metrics = ["weighted_total", "grounding"]

    # Compute differences (model_b - model_a)
    metric_diffs = {}
    for metric in valid_metrics:
        val_a = model_a.metrics.get(metric)
        val_b = model_b.metrics.get(metric)
        if val_a is not None and val_b is not None:
            metric_diffs[metric] = val_b - val_a

    # Determine winner based on factuality (primary metric)
    winner = None
    if "factuality" in metric_diffs:
        diff = metric_diffs["factuality"]
        if diff > 0.1:  # Threshold for meaningful difference
            winner = model_b.name
        elif diff < -0.1:
            winner = model_a.name
        else:
            winner = "tie"

    return ComparisonResult(
        name=comparison_config.get("name", f"{model_a.name}_vs_{model_b.name}"),
        model_a=model_a.name,
        model_b=model_b.name,
        research_question=comparison_config.get("research_question", ""),
        comparison_type=comparison_type,
        valid_metrics=valid_metrics,
        invalid_metrics=invalid_metrics,
        metric_differences=metric_diffs,
        winner=winner,
        warning=comparison_config.get("warning"),
        expected=comparison_config.get("expected"),
    )


# =============================================================================
# Validation
# =============================================================================


def validate_experiment(
    config: dict[str, Any],
    base_dir: Path,
) -> tuple[ExperimentValidation, dict[str, ModelCondition]]:
    """Validate that all experiment results exist and load conditions."""

    conditions: dict[str, ModelCondition] = {}
    missing = []
    warnings = []

    models_config = config.get("models", {})

    for name, model_info in models_config.items():
        result_dir = base_dir / model_info["result_dir"]
        metrics_file = result_dir / "metrics.json"

        if not result_dir.exists():
            missing.append(f"{name}: {result_dir} does not exist")
            continue

        if not metrics_file.exists():
            missing.append(f"{name}: {metrics_file} does not exist")
            continue

        # Load condition
        metrics = extract_metrics(result_dir, model_info["eval_mode"])

        if not metrics:
            warnings.append(f"{name}: No metrics could be extracted from {metrics_file}")

        conditions[name] = ModelCondition(
            name=name,
            result_dir=result_dir,
            eval_mode=model_info["eval_mode"],
            training_mode=model_info["training_mode"],
            description=model_info["description"],
            metrics=metrics,
        )

    validation = ExperimentValidation(
        all_results_exist=len(missing) == 0,
        missing_results=missing,
        conditions_loaded=len(conditions),
        warnings=warnings,
    )

    return validation, conditions


# =============================================================================
# Report Generation
# =============================================================================


def generate_within_group_table(
    group_name: str,
    conditions: list[ModelCondition],
    metrics: list[str],
) -> str:
    """Generate a markdown table for within-group comparison."""

    lines = [
        f"### {group_name}",
        "",
        "| Condition | " + " | ".join(metrics) + " |",
        "|-----------|" + "|".join(["--------" for _ in metrics]) + "|",
    ]

    for cond in conditions:
        values = [
            f"{cond.metrics.get(m, 'N/A'):.2f}"
            if isinstance(cond.metrics.get(m), int | float)
            else "N/A"
            for m in metrics
        ]
        lines.append(f"| {cond.name} | " + " | ".join(values) + " |")

    return "\n".join(lines)


def generate_comparison_summary(results: list[ComparisonResult]) -> str:
    """Generate a summary of all comparisons."""

    lines = ["## Comparison Summary", ""]

    for result in results:
        lines.append(f"### {result.name}")
        lines.append(f"**Research Question**: {result.research_question}")
        lines.append(f"**Type**: {result.comparison_type}")
        lines.append(f"**Models**: {result.model_a} vs {result.model_b}")

        if result.warning:
            lines.append(f"WARNING:  **Warning**: {result.warning}")

        lines.append("")
        lines.append("**Metric Differences** (model_b - model_a):")
        for metric, diff in result.metric_differences.items():
            sign = "+" if diff > 0 else ""
            lines.append(f"- {metric}: {sign}{diff:.3f}")

        if result.winner:
            lines.append(f"\n**Winner**: {result.winner}")

        if result.expected:
            lines.append(f"**Expected**: {result.expected}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_full_report(
    config: dict[str, Any],
    conditions: dict[str, ModelCondition],
    comparison_results: list[ComparisonResult],
) -> dict[str, Any]:
    """Generate a full JSON report of all comparisons."""

    return {
        "experiment": "six_condition_comparison",
        "conditions": {
            name: {
                "eval_mode": cond.eval_mode,
                "training_mode": cond.training_mode,
                "description": cond.description,
                "metrics": cond.metrics,
            }
            for name, cond in conditions.items()
        },
        "comparisons": [
            {
                "name": r.name,
                "model_a": r.model_a,
                "model_b": r.model_b,
                "research_question": r.research_question,
                "comparison_type": r.comparison_type,
                "valid_metrics": r.valid_metrics,
                "invalid_metrics": r.invalid_metrics,
                "metric_differences": r.metric_differences,
                "winner": r.winner,
                "warning": r.warning,
                "expected": r.expected,
            }
            for r in comparison_results
        ],
        "summary": {
            "within_group_comparisons": len(
                [r for r in comparison_results if r.comparison_type == "within-group"]
            ),
            "cross_group_comparisons": len(
                [r for r in comparison_results if r.comparison_type == "cross-group"]
            ),
            "total_comparisons": len(comparison_results),
        },
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and run scientifically valid comparisons for 6-condition experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/six_condition_comparisons.yaml"),
        help="Path to comparison configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for comparison report JSON",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Output path for comparison report Markdown",
    )
    parser.add_argument(
        "--comparison-type",
        choices=["all", "within-group", "cross-group"],
        default="all",
        help="Type of comparisons to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running comparisons",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load configuration
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_dir = args.config.parent.parent.parent  # Go up to repo root

    print("=" * 70)
    print("Six-Condition Experiment Comparison Validator")
    print("=" * 70)
    print()

    # Validate experiment setup
    print("Validating experiment setup...")
    validation, conditions = validate_experiment(config, base_dir)

    print(f"  Conditions loaded: {validation.conditions_loaded}/6")

    if validation.missing_results:
        print("\nWARNING:   Missing results:")
        for missing in validation.missing_results:
            print(f"    - {missing}")

    if validation.warnings:
        print("\nWARNING:   Warnings:")
        for warning in validation.warnings:
            print(f"    - {warning}")

    if args.dry_run:
        print("\n[Dry run mode - not running comparisons]")

        # Show what comparisons would be run
        print("\nChecklist Planned comparisons:")

        within_config = config.get("within_group_comparisons", {})
        cross_config = config.get("cross_group_comparisons", {})

        print("\n  Within-group (instruction mode):")
        for comp in within_config.get("instruction_group", {}).get("comparisons", []):
            print(f"    - {comp['name']}: {comp['model_a']} vs {comp['model_b']}")

        print("\n  Within-group (RAG mode):")
        for comp in within_config.get("rag_group", {}).get("comparisons", []):
            print(f"    - {comp['name']}: {comp['model_a']} vs {comp['model_b']}")

        print("\n  Cross-group:")
        for comp in cross_config.get("comparisons", []):
            print(f"    - {comp['name']}: {comp['model_a']} vs {comp['model_b']}")
            if comp.get("warning"):
                print(f"      WARNING:   {comp['warning']}")

        return 0 if validation.all_results_exist else 1

    # Check if we have enough data to proceed
    if not validation.all_results_exist:
        print("\n✗ Cannot run comparisons - missing result files")
        print("   Run evaluations first, then re-run this script")
        return 1

    # Run comparisons
    print("\nRunning comparisons...")
    comparison_results: list[ComparisonResult] = []

    within_config = config.get("within_group_comparisons", {})
    cross_config = config.get("cross_group_comparisons", {})

    # Within-group comparisons
    if args.comparison_type in ["all", "within-group"]:
        print("\n  Within-group comparisons:")

        # Instruction group
        for comp in within_config.get("instruction_group", {}).get("comparisons", []):
            model_a = conditions.get(comp["model_a"])
            model_b = conditions.get(comp["model_b"])
            if model_a and model_b:
                result = compute_comparison(model_a, model_b, comp, "within-group")
                comparison_results.append(result)
                print(f"    ✓ {result.name}")

        # RAG group
        for comp in within_config.get("rag_group", {}).get("comparisons", []):
            model_a = conditions.get(comp["model_a"])
            model_b = conditions.get(comp["model_b"])
            if model_a and model_b:
                result = compute_comparison(model_a, model_b, comp, "within-group")
                comparison_results.append(result)
                print(f"    ✓ {result.name}")

    # Cross-group comparisons
    if args.comparison_type in ["all", "cross-group"]:
        print("\n  Cross-group comparisons:")

        for comp in cross_config.get("comparisons", []):
            model_a = conditions.get(comp["model_a"])
            model_b = conditions.get(comp["model_b"])
            if model_a and model_b:
                result = compute_comparison(model_a, model_b, comp, "cross-group")
                comparison_results.append(result)
                print(f"    ✓ {result.name}")
                if result.warning:
                    print(f"      WARNING:   {result.warning}")

    # Generate report
    report = generate_full_report(config, conditions, comparison_results)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ JSON report saved to: {args.output}")

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown = generate_comparison_summary(comparison_results)

        # Add within-group tables
        if args.comparison_type in ["all", "within-group"]:
            instruction_conds = [
                conditions[c]
                for c in ["base_baseline", "ft_only_instruction", "ft_only_rag_trained"]
                if c in conditions
            ]
            rag_conds = [
                conditions[c]
                for c in ["rag_baseline", "ft_rag_instruction", "ft_rag_trained"]
                if c in conditions
            ]

            if instruction_conds:
                markdown += "\n\n" + generate_within_group_table(
                    "Instruction Group (No RAG)",
                    instruction_conds,
                    ["factuality", "completeness", "communication", "weighted_total"],
                )

            if rag_conds:
                markdown += "\n\n" + generate_within_group_table(
                    "RAG Group",
                    rag_conds,
                    ["factuality", "completeness", "communication", "grounding", "weighted_total"],
                )

        with open(args.output_markdown, "w") as f:
            f.write(markdown)
        print(f"✓ Markdown report saved to: {args.output_markdown}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Within-group comparisons: {report['summary']['within_group_comparisons']}")
    print(f"  Cross-group comparisons:  {report['summary']['cross_group_comparisons']}")
    print(f"  Total comparisons:        {report['summary']['total_comparisons']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
