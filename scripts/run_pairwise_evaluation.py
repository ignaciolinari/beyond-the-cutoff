#!/usr/bin/env python3
"""Run automated pairwise evaluation using judge models to compute ELO rankings.

This script orchestrates head-to-head comparisons between model outputs using
one or more judge models, then computes ELO rankings with confidence intervals.

Usage:
    # Compare models using result directories
    python scripts/run_pairwise_evaluation.py \
        --results base_baseline=evaluation/results/base_baseline_0p5b \
        --results rag_baseline=evaluation/results/rag_baseline_0p5b \
        --results ft_only=evaluation/results/lora_science_0p5b_ft_only \
        --judge configs/judges/pairwise_qwen7b.yaml \
        --judge configs/judges/pairwise_qwen3b.yaml \
        --output evaluation/results/elo_rankings

    # Use a comparison plan YAML
    python scripts/run_pairwise_evaluation.py \
        --plan configs/evaluation/pairwise_evaluation_plan.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beyond_the_cutoff.evaluation.elo_ranking import head_to_head_matrix
from beyond_the_cutoff.evaluation.pairwise_judge import (
    PairwiseEvaluationConfig,
    load_predictions_from_results,
    run_pairwise_evaluation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated pairwise evaluation for ELO rankings"
    )

    # Input options
    parser.add_argument(
        "--results",
        action="append",
        metavar="NAME=PATH",
        help="Model results in format 'model_name=path/to/results/'. Can specify multiple.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="YAML file with evaluation plan (alternative to --results)",
    )

    # Judge configuration
    parser.add_argument(
        "--judge",
        action="append",
        type=Path,
        dest="judges",
        help="Path to judge config YAML. Can specify multiple for consensus.",
    )

    # Evaluation parameters
    parser.add_argument(
        "--comparisons-per-pair",
        type=int,
        default=50,
        help="Number of comparisons per model pair (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="ELO K-factor (default: 32)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples for confidence intervals (default: 1000)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Don't save intermediate comparison files",
    )

    return parser.parse_args()


def load_plan(plan_path: Path) -> dict[str, Any]:
    """Load evaluation plan from YAML."""
    import yaml

    with open(plan_path, encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def parse_results_arg(results_list: list[str] | None) -> dict[str, Path]:
    """Parse --results arguments into dict."""
    if not results_list:
        return {}

    result_dirs = {}
    for item in results_list:
        if "=" not in item:
            raise ValueError(f"Invalid --results format: {item}. Expected 'name=path'")
        name, path = item.split("=", 1)
        result_dirs[name.strip()] = Path(path.strip())

    return result_dirs


def main() -> None:
    args = parse_args()

    # Load evaluation configuration
    if args.plan:
        logger.info(f"Loading evaluation plan from {args.plan}")
        plan = load_plan(args.plan)

        result_dirs = {name: Path(path) for name, path in plan.get("models", {}).items()}
        judge_configs = [Path(p) for p in plan.get("judges", [])]
        comparisons_per_pair = plan.get("comparisons_per_pair", args.comparisons_per_pair)
        seed = plan.get("seed", args.seed)
        k_factor = plan.get("k_factor", args.k_factor)
        bootstrap_samples = plan.get("bootstrap_samples", args.bootstrap_samples)
    else:
        result_dirs = parse_results_arg(args.results)
        judge_configs = args.judges or []
        comparisons_per_pair = args.comparisons_per_pair
        seed = args.seed
        k_factor = args.k_factor
        bootstrap_samples = args.bootstrap_samples

    # Validate inputs
    if not result_dirs:
        logger.error("No model results specified. Use --results or --plan.")
        sys.exit(1)

    if not judge_configs:
        # Default to Qwen 7B judge
        default_judge = Path("configs/judges/pairwise_qwen7b.yaml")
        if default_judge.exists():
            judge_configs = [default_judge]
            logger.info(f"Using default judge: {default_judge}")
        else:
            logger.error("No judge configs specified. Use --judge or --plan.")
            sys.exit(1)

    # Validate paths
    for _name, path in result_dirs.items():
        if not path.exists():
            logger.error(f"Result directory not found: {path}")
            sys.exit(1)
        if not (path / "details.jsonl").exists():
            logger.error(f"Missing details.jsonl in {path}")
            sys.exit(1)

    for judge_path in judge_configs:
        if not judge_path.exists():
            logger.error(f"Judge config not found: {judge_path}")
            sys.exit(1)

    logger.info(f"Models to compare: {list(result_dirs.keys())}")
    logger.info(f"Judges: {[p.stem for p in judge_configs]}")
    logger.info(f"Comparisons per pair: {comparisons_per_pair}")

    # Load predictions
    logger.info("Loading model predictions...")
    predictions = load_predictions_from_results(result_dirs)

    if len(predictions) < 2:
        logger.error("Need at least 2 models for pairwise comparison")
        sys.exit(1)

    # Create evaluation config
    config = PairwiseEvaluationConfig(
        judge_configs=judge_configs,
        output_dir=args.output,
        comparisons_per_pair=comparisons_per_pair,
        randomize_order=True,
        seed=seed,
        save_intermediate=not args.no_intermediate,
        k_factor=k_factor,
        bootstrap_samples=bootstrap_samples,
    )

    # Run evaluation
    logger.info("Starting pairwise evaluation...")
    comparisons, metadata = run_pairwise_evaluation(predictions, config)

    # Print summary
    print("\n" + "=" * 60)
    print("PAIRWISE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal comparisons: {len(comparisons)}")
    print(f"Models evaluated: {metadata['n_models']}")
    print(f"Model pairs: {metadata['n_pairs']}")
    print(f"Judges used: {', '.join(metadata['judges'])}")

    # Print leaderboard
    print("\n" + "-" * 60)
    print("ELO LEADERBOARD")
    print("-" * 60)

    leaderboard_file = args.output / "leaderboard.json"
    with open(leaderboard_file, encoding="utf-8") as f:
        leaderboard_data = json.load(f)

    for i, entry in enumerate(leaderboard_data["leaderboard"]):
        ci = ""
        if entry.get("confidence_lower") and entry.get("confidence_upper"):
            ci = f" [{entry['confidence_lower']:.0f}-{entry['confidence_upper']:.0f}]"

        print(
            f"{i+1}. {entry['model']:30s} "
            f"Rating: {entry['rating']:6.0f}{ci:16s} "
            f"W:{entry['wins']:3d} L:{entry['losses']:3d} T:{entry['ties']:3d}"
        )

    # Print head-to-head summary
    print("\n" + "-" * 60)
    print("HEAD-TO-HEAD WIN RATES")
    print("-" * 60)

    h2h = head_to_head_matrix(comparisons)
    models = sorted(h2h.keys())

    # Header
    header = f"{'':20s}"
    for m in models:
        header += f"{m[:12]:>14s}"
    print(header)

    # Rows
    for m1 in models:
        row = f"{m1[:20]:20s}"
        for m2 in models:
            if m1 == m2:
                row += f"{'---':>14s}"
            elif m2 in h2h.get(m1, {}):
                stats = h2h[m1][m2]
                total = stats["wins"] + stats["losses"] + stats["ties"]
                if total > 0:
                    win_rate = (stats["wins"] + 0.5 * stats["ties"]) / total
                    row += f"{win_rate:>13.1%} "
                else:
                    row += f"{'---':>14s}"
            else:
                row += f"{'---':>14s}"
        print(row)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
