#!/usr/bin/env python3
"""Generate visualizations for model comparison results.

This script creates visualizations from evaluation results, including:
- Bar charts for metrics across conditions
- Error rate comparisons
- Citation metric comparisons
- Task-type breakdowns
- Timing comparisons

Usage examples:

    # Visualize from comparison report JSON
    python scripts/visualize_comparison.py \
        --report evaluation/results/comparison_report.json \
        --output evaluation/results/visualizations/

    # Visualize from individual metrics files
    python scripts/visualize_comparison.py \
        --metrics evaluation/results/rag_baseline_0p5b/metrics.json \
        --metrics evaluation/results/lora_science_0p5b_ft_only/metrics.json \
        --output evaluation/results/visualizations/

    # Generate specific visualizations only
    python scripts/visualize_comparison.py \
        --report evaluation/results/comparison_report.json \
        --output evaluation/results/visualizations/ \
        --only metrics error-rates citations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
except ImportError:
    print(
        "Error: matplotlib and seaborn are required for visualization.",
        "Install with: pip install matplotlib seaborn",
        file=sys.stderr,
    )
    sys.exit(1)


def load_comparison_report(report_path: Path) -> dict[str, Any]:
    """Load comparison report JSON."""
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def load_metrics_file(metrics_path: Path) -> dict[str, Any]:
    """Load metrics JSON file."""
    with metrics_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
        # Handle both formats: {"summary": {...}} and direct summary
        if "summary" in data:
            return data["summary"]  # type: ignore[no-any-return]
        return data


def extract_judge_scores(summary: dict[str, Any]) -> dict[str, float]:
    """Extract judge scores from summary."""
    scores: dict[str, float] = {}
    for key in ["factuality", "grounding", "completeness", "communication"]:
        if key in summary:
            scores[key] = float(summary[key])
    return scores


def extract_citation_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Extract citation metrics from summary."""
    metrics: dict[str, float] = {}
    if "citation_mean_coverage" in summary:
        metrics["coverage"] = float(summary["citation_mean_coverage"])
    # Additional citation metrics might be in citation_metrics dict
    citation_metrics = summary.get("citation_metrics", {})
    if isinstance(citation_metrics, dict):
        for key in ["precision", "recall", "mean_coverage"]:
            if key in citation_metrics:
                metrics[key.replace("mean_", "")] = float(citation_metrics[key])
    return metrics


def plot_metrics_comparison(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (12, 6)
) -> None:
    """Create bar chart comparing judge scores across models."""
    labels = list(data.keys())
    metrics = ["factuality", "grounding", "completeness", "communication"]

    # Filter to only metrics that exist in at least one model
    available_metrics = [
        m for m in metrics if any(m in extract_judge_scores(data[label]) for label in labels)
    ]

    if not available_metrics:
        print("[warn] No judge scores found for metrics comparison", file=sys.stderr)
        return

    x = np.arange(len(labels))
    width = 0.8 / len(available_metrics)

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(available_metrics))
    bars = []

    for idx, metric in enumerate(available_metrics):
        values = [extract_judge_scores(data[label]).get(metric, 0.0) for label in labels]
        offset = (idx - len(available_metrics) / 2 + 0.5) * width
        bar = ax.bar(
            x + offset, values, width, label=metric.replace("_", " ").title(), color=colors[idx]
        )
        bars.append(bar)

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Judge Scores Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = output_dir / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved metrics comparison to {output_path}")


def plot_error_rates(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (10, 6)
) -> None:
    """Create bar chart comparing error rates across models."""
    labels = list(data.keys())
    error_rates = [float(data[label].get("error_rate", 0.0)) for label in labels]
    examples_with_errors = [int(data[label].get("examples_with_errors", 0)) for label in labels]
    total_examples = [int(data[label].get("examples_evaluated", 0)) for label in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Error rate bar chart
    colors = ["#d62728" if rate > 0.1 else "#2ca02c" for rate in error_rates]
    bars1 = ax1.bar(labels, error_rates, color=colors, alpha=0.7)
    ax1.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="10% threshold")
    ax1.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Error Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Error Rate Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(max(error_rates) * 1.2, 0.15))

    # Add value labels on bars
    for bar, rate in zip(bars1, error_rates, strict=True):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Examples with errors bar chart
    bars2 = ax2.bar(labels, examples_with_errors, color="#d62728", alpha=0.7)
    ax2.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_title("Examples with Errors", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels and total examples
    for bar, errors, total in zip(bars2, examples_with_errors, total_examples, strict=True):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{errors}/{total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = output_dir / "error_rates.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved error rates comparison to {output_path}")


def plot_citation_metrics(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (12, 6)
) -> None:
    """Create bar chart comparing citation metrics across models."""
    labels = list(data.keys())

    # Filter to only RAG models (instruction-only models don't have citation metrics)
    rag_labels = []
    citation_data: dict[str, dict[str, float]] = {}

    for label in labels:
        summary = data[label]
        prompt_mode = summary.get("prompt_mode", "rag")
        if prompt_mode == "rag":
            citation_metrics = extract_citation_metrics(summary)
            if citation_metrics:
                rag_labels.append(label)
                citation_data[label] = citation_metrics

    if not rag_labels:
        print("[warn] No RAG models found for citation metrics comparison", file=sys.stderr)
        return

    metrics = ["coverage", "precision", "recall"]
    available_metrics = [
        m for m in metrics if any(m in citation_data[label] for label in rag_labels)
    ]

    if not available_metrics:
        print("[warn] No citation metrics found", file=sys.stderr)
        return

    x = np.arange(len(rag_labels))
    width = 0.8 / len(available_metrics)

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("Set2", len(available_metrics))
    bars = []

    for idx, metric in enumerate(available_metrics):
        values = [citation_data[label].get(metric, 0.0) for label in rag_labels]
        offset = (idx - len(available_metrics) / 2 + 0.5) * width
        bar = ax.bar(
            x + offset, values, width, label=metric.replace("_", " ").title(), color=colors[idx]
        )
        bars.append(bar)

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Citation Metrics Comparison (RAG Models Only)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(rag_labels, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = output_dir / "citation_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved citation metrics comparison to {output_path}")


def plot_task_type_breakdown(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (14, 8)
) -> None:
    """Create stacked bar chart showing task type distribution."""
    # This requires details.jsonl files, so we'll create a placeholder
    # In practice, this would read from details files
    print(
        "[info] Task type breakdown requires details.jsonl files (not yet implemented)",
        file=sys.stderr,
    )

    # Placeholder: show example count per model
    labels = list(data.keys())
    example_counts = [int(data[label].get("examples_evaluated", 0)) for label in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, example_counts, color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Examples", fontsize=12, fontweight="bold")
    ax.set_title("Examples Evaluated per Model", fontsize=14, fontweight="bold")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, count in zip(bars, example_counts, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = output_dir / "task_type_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved task type breakdown to {output_path}")


def plot_timing_comparison(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (12, 6)
) -> None:
    """Create bar chart comparing timing metrics across models."""
    labels = list(data.keys())

    generation_times = []
    judge_times = []
    total_times = []

    for label in labels:
        timing = data[label].get("timing", {})
        if isinstance(timing, dict):
            gen_stats = timing.get("generation", {})
            judge_stats = timing.get("judge", {})
            total_stats = timing.get("total", {})

            generation_times.append(
                gen_stats.get("mean_seconds", 0.0) if isinstance(gen_stats, dict) else 0.0
            )
            judge_times.append(
                judge_stats.get("mean_seconds", 0.0) if isinstance(judge_stats, dict) else 0.0
            )
            total_times.append(
                total_stats.get("mean_seconds", 0.0) if isinstance(total_stats, dict) else 0.0
            )
        else:
            generation_times.append(0.0)
            judge_times.append(0.0)
            total_times.append(0.0)

    if not any(generation_times) and not any(judge_times):
        print("[warn] No timing data found", file=sys.stderr)
        return

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - width, generation_times, width, label="Generation", color="#1f77b4", alpha=0.7)
    ax.bar(x, judge_times, width, label="Judge", color="#ff7f0e", alpha=0.7)
    ax.bar(x + width, total_times, width, label="Total", color="#2ca02c", alpha=0.7)

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Timing Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "timing_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved timing comparison to {output_path}")


def plot_prompt_mode_comparison(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (10, 6)
) -> None:
    """Create grouped bar chart comparing RAG vs instruction-only modes."""
    # Group models by prompt mode
    rag_models: dict[str, dict[str, Any]] = {}
    instruction_models: dict[str, dict[str, Any]] = {}

    for label, summary in data.items():
        prompt_mode = summary.get("prompt_mode", "rag")
        if prompt_mode == "rag":
            rag_models[label] = summary
        else:
            instruction_models[label] = summary

    if not rag_models or not instruction_models:
        print("[warn] Need both RAG and instruction-only models for comparison", file=sys.stderr)
        return

    # Compare factuality scores
    rag_factuality = [extract_judge_scores(s).get("factuality", 0.0) for s in rag_models.values()]
    instruction_factuality = [
        extract_judge_scores(s).get("factuality", 0.0) for s in instruction_models.values()
    ]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(2)
    width = 0.6

    ax.bar(
        x[0] - width / 2,
        np.mean(rag_factuality) if rag_factuality else 0.0,
        width,
        label="RAG Mode",
        color="#2ca02c",
        alpha=0.7,
    )
    ax.bar(
        x[1] - width / 2,
        np.mean(instruction_factuality) if instruction_factuality else 0.0,
        width,
        label="Instruction Mode",
        color="#1f77b4",
        alpha=0.7,
    )

    # Add error bars showing std dev
    if rag_factuality:
        ax.errorbar(
            x[0],
            np.mean(rag_factuality),
            yerr=np.std(rag_factuality),
            fmt="none",
            color="black",
            capsize=5,
        )
    if instruction_factuality:
        ax.errorbar(
            x[1],
            np.mean(instruction_factuality),
            yerr=np.std(instruction_factuality),
            fmt="none",
            color="black",
            capsize=5,
        )

    ax.set_xlabel("Prompt Mode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Factuality Score", fontsize=12, fontweight="bold")
    ax.set_title("RAG vs Instruction-Only Mode Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["RAG", "Instruction"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = output_dir / "prompt_mode_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved prompt mode comparison to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for model comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Comparison report JSON file (from compare_models.py)",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        type=Path,
        dest="metrics_files",
        help="Individual metrics JSON files (can be specified multiple times)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["metrics", "error-rates", "citations", "timing", "prompt-mode", "task-type"],
        help="Generate only specific visualizations (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for images (default: png)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load data
    data: dict[str, dict[str, Any]] = {}

    if args.report:
        report = load_comparison_report(args.report)
        for run in report.get("runs", []):
            if run.get("skipped"):
                continue
            label = run.get("label", "unknown")
            summary = run.get("summary", {})
            if summary:
                data[label] = summary

    if args.metrics_files:
        for metrics_path in args.metrics_files:
            if not metrics_path.exists():
                print(f"[warn] Metrics file not found: {metrics_path}", file=sys.stderr)
                continue
            try:
                summary = load_metrics_file(metrics_path)
                # Try to extract label from path or use filename
                label = summary.get("model_label") or metrics_path.stem
                data[label] = summary
            except Exception as exc:
                print(f"[error] Failed to load {metrics_path}: {exc}", file=sys.stderr)
                return 1

    if not data:
        print("[error] No data loaded. Provide --report or --metrics files.", file=sys.stderr)
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine which visualizations to generate
    visualizations = (
        args.only
        if args.only
        else ["metrics", "error-rates", "citations", "timing", "prompt-mode", "task-type"]
    )

    # Generate visualizations
    if "metrics" in visualizations:
        plot_metrics_comparison(data, args.output)

    if "error-rates" in visualizations:
        plot_error_rates(data, args.output)

    if "citations" in visualizations:
        plot_citation_metrics(data, args.output)

    if "timing" in visualizations:
        plot_timing_comparison(data, args.output)

    if "prompt-mode" in visualizations:
        plot_prompt_mode_comparison(data, args.output)

    if "task-type" in visualizations:
        plot_task_type_breakdown(data, args.output)

    print(f"\n[info] Visualizations saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
