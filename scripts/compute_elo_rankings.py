#!/usr/bin/env python3
"""Compute ELO rankings from pairwise comparison data.

This script supports:
- Loading comparisons from JSONL files (judge evaluations or human annotations)
- Computing ELO ratings with bootstrap confidence intervals
- Generating head-to-head matrices
- Exporting leaderboards to JSON

Usage:
    python scripts/compute_elo_rankings.py --comparisons results/comparisons.jsonl --output results/leaderboard.json
    python scripts/compute_elo_rankings.py --annotations-dir evaluation/human_annotations --output results/human_leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beyond_the_cutoff.evaluation.elo_ranking import (
    Outcome,
    PairwiseComparison,
    compute_elo_rankings,
    head_to_head_matrix,
    load_comparisons_from_jsonl,
    save_leaderboard,
)
from beyond_the_cutoff.evaluation.human_evaluation import (
    export_annotations_for_elo,
    load_annotation_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ELO rankings from pairwise comparisons")

    # Input sources (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--comparisons",
        type=Path,
        help="Path to JSONL file with pairwise comparisons",
    )
    input_group.add_argument(
        "--annotations-dir",
        type=Path,
        help="Directory containing human annotation batch JSON files",
    )
    input_group.add_argument(
        "--eval-results",
        type=Path,
        nargs="+",
        help="Evaluation result JSON files to extract comparisons from",
    )

    # ELO parameters
    parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="ELO K-factor (higher = more volatile ratings). Default: 32",
    )
    parser.add_argument(
        "--initial-rating",
        type=float,
        default=1500.0,
        help="Starting rating for new models. Default: 1500",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals. Default: 1000",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals. Default: 0.95",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for leaderboard JSON",
    )
    parser.add_argument(
        "--head-to-head",
        type=Path,
        help="Optional: Output path for head-to-head matrix JSON",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "csv"],
        default="json",
        help="Output format. Default: json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    return parser.parse_args()


def load_comparisons_from_annotations(annotations_dir: Path) -> list[PairwiseComparison]:
    """Load comparisons from human annotation batches."""
    batches = []
    task_lookup = {}

    for path in annotations_dir.glob("*.json"):
        try:
            batch = load_annotation_batch(path)
            batches.append(batch)
            for task in batch.tasks:
                task_lookup[task.task_id] = task
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)

    return export_annotations_for_elo(batches, task_lookup)


def load_comparisons_from_eval_results(result_files: list[Path]) -> list[PairwiseComparison]:
    """Extract pairwise comparisons from evaluation result files.

    This assumes evaluation results have a format where we can extract
    model performance on the same tasks and create synthetic comparisons.
    """
    # Load all results
    results_by_model: dict[str, dict[str, dict[str, Any]]] = {}

    for path in result_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model", path.stem)
        details = data.get("details", [])

        results_by_model[model_name] = {d["task_id"]: d for d in details if "task_id" in d}

    # Create pairwise comparisons based on scores
    comparisons = []
    models = list(results_by_model.keys())

    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            results_a = results_by_model[model_a]
            results_b = results_by_model[model_b]

            common_tasks = set(results_a.keys()) & set(results_b.keys())

            for task_id in common_tasks:
                score_a = results_a[task_id].get(
                    "score", results_a[task_id].get("judge_score", 0.5)
                )
                score_b = results_b[task_id].get(
                    "score", results_b[task_id].get("judge_score", 0.5)
                )

                # Determine outcome
                outcome: Outcome
                if score_a > score_b + 0.1:
                    outcome = "win_a"
                elif score_b > score_a + 0.1:
                    outcome = "win_b"
                else:
                    outcome = "tie"

                comparisons.append(
                    PairwiseComparison(
                        model_a=model_a,
                        model_b=model_b,
                        outcome=outcome,
                        task_id=task_id,
                        annotator="judge",
                        annotation_source="judge",
                        metadata={
                            "score_a": score_a,
                            "score_b": score_b,
                        },
                    )
                )

    return comparisons


def format_leaderboard_markdown(leaderboard: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    """Format leaderboard as markdown table."""
    lines = [
        "# Model Leaderboard",
        "",
        f"Based on {metadata['n_comparisons']} pairwise comparisons.",
        "",
        "| Rank | Model | Rating | 95% CI | W/L/T | Win Rate |",
        "|------|-------|--------|--------|-------|----------|",
    ]

    for i, r in enumerate(leaderboard):
        ci = (
            f"{r['confidence_lower']:.0f}-{r['confidence_upper']:.0f}"
            if r["confidence_lower"]
            else "-"
        )
        lines.append(
            f"| {i+1} | {r['model']} | {r['rating']:.0f} | {ci} | "
            f"{r['wins']}/{r['losses']}/{r['ties']} | {r['win_rate']:.1%} |"
        )

    return "\n".join(lines)


def format_leaderboard_csv(leaderboard: list[dict[str, Any]]) -> str:
    """Format leaderboard as CSV."""
    lines = ["rank,model,rating,confidence_lower,confidence_upper,wins,losses,ties,win_rate"]

    for i, r in enumerate(leaderboard):
        lines.append(
            f"{i+1},{r['model']},{r['rating']:.1f},{r['confidence_lower'] or ''},"
            f"{r['confidence_upper'] or ''},{r['wins']},{r['losses']},{r['ties']},{r['win_rate']:.4f}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Load comparisons based on input source
    if args.comparisons:
        if args.verbose:
            print(f"Loading comparisons from {args.comparisons}")
        comparisons = load_comparisons_from_jsonl(args.comparisons)

    elif args.annotations_dir:
        if args.verbose:
            print(f"Loading annotations from {args.annotations_dir}")
        comparisons = load_comparisons_from_annotations(args.annotations_dir)

    elif args.eval_results:
        if args.verbose:
            print(f"Extracting comparisons from {len(args.eval_results)} result files")
        comparisons = load_comparisons_from_eval_results(args.eval_results)

    else:
        raise ValueError("No input source specified")

    if not comparisons:
        print("Error: No comparisons found", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(comparisons)} comparisons")
        models = {c.model_a for c in comparisons} | {c.model_b for c in comparisons}
        print(f"Models: {', '.join(sorted(models))}")

    # Compute rankings
    if args.verbose:
        print(f"Computing ELO rankings (K={args.k_factor}, bootstrap={args.bootstrap_samples})")

    leaderboard, metadata = compute_elo_rankings(
        comparisons,
        k_factor=args.k_factor,
        initial_rating=args.initial_rating,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )

    # Output
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard_dicts = [r.as_dict() for r in leaderboard]

    if args.format == "json":
        save_leaderboard(leaderboard, metadata, output_path)
    elif args.format == "markdown":
        md_content = format_leaderboard_markdown(leaderboard_dicts, metadata)
        output_path.write_text(md_content, encoding="utf-8")
    elif args.format == "csv":
        csv_content = format_leaderboard_csv(leaderboard_dicts)
        output_path.write_text(csv_content, encoding="utf-8")

    print(f"Saved leaderboard to {output_path}")

    # Head-to-head matrix
    if args.head_to_head:
        h2h = head_to_head_matrix(comparisons)
        args.head_to_head.parent.mkdir(parents=True, exist_ok=True)
        with open(args.head_to_head, "w", encoding="utf-8") as f:
            json.dump(h2h, f, indent=2)
        print(f"Saved head-to-head matrix to {args.head_to_head}")

    # Print summary
    print("\n" + "=" * 50)
    print("LEADERBOARD")
    print("=" * 50)
    for i, r in enumerate(leaderboard):
        ci = ""
        if r.confidence_lower and r.confidence_upper:
            ci = f" [{r.confidence_lower:.0f}-{r.confidence_upper:.0f}]"
        print(f"{i+1}. {r.model}: {r.rating:.0f}{ci} (W:{r.wins} L:{r.losses} T:{r.ties})")


if __name__ == "__main__":
    main()
