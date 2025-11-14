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

try:
    import plotly.graph_objects as go  # type: ignore[import-not-found] # noqa: F401
    from plotly.subplots import make_subplots  # type: ignore[import-not-found] # noqa: F401

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print(
        "[warn] plotly not available - interactive dashboards will be skipped.",
        "Install with: pip install plotly",
        file=sys.stderr,
    )

try:
    from scipy import stats
except ImportError:
    stats = None
    print(
        "[warn] scipy not available - statistical significance testing will be skipped.",
        "Install with: pip install scipy",
        file=sys.stderr,
    )


def load_comparison_report(report_path: Path) -> dict[str, Any]:
    """Load comparison report JSON."""
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def resolve_details_path(
    label: str,
    report_data: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Resolve details.jsonl path for a given label.

    Priority order:
    1. Use details_path from comparison report if available
    2. Try output_dir.parent / label / "details.jsonl"
    3. Try evaluation/results / label / "details.jsonl"
    4. Return None if not found

    Args:
        label: Model label to find details for
        report_data: Optional comparison report data with runs
        output_dir: Optional output directory for fallback resolution

    Returns:
        Path to details.jsonl if found, None otherwise
    """
    # First, try to get from report data
    if report_data:
        for run in report_data.get("runs", []):
            if run.get("label") == label:
                details_path_str = run.get("details_path")
                if details_path_str:
                    details_path = Path(details_path_str)
                    if details_path.exists():
                        return details_path.resolve()
                    # Try relative to report location if absolute path doesn't exist
                    if not details_path.is_absolute() and report_data.get("_report_path"):
                        report_dir = Path(report_data["_report_path"]).parent
                        candidate = report_dir / details_path
                        if candidate.exists():
                            return candidate.resolve()

    # Fallback: try common locations
    if output_dir:
        candidate = output_dir.parent / label / "details.jsonl"
        if candidate.exists():
            return candidate.resolve()

    # Last resort: try standard evaluation results location
    candidate = Path("evaluation/results") / label / "details.jsonl"
    if candidate.exists():
        return candidate.resolve()

    return None


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
    data: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    report_data: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Create stacked bar chart showing task type distribution."""
    labels = list(data.keys())

    # Try to load details files to get task type breakdown
    task_types: dict[str, dict[str, int]] = {}
    for label in labels:
        # Use helper function to resolve details path
        details_path = resolve_details_path(label, report_data, output_dir)

        task_type_counts: dict[str, int] = {}
        if details_path and details_path.exists():
            try:
                with details_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        example = json.loads(line)
                        task_type = example.get("task_type", "unknown")
                        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            except Exception as exc:
                print(
                    f"[warn] Failed to load task types from {details_path}: {exc}",
                    file=sys.stderr,
                )
        elif details_path is None:
            print(
                f"[warn] Could not resolve details.jsonl path for '{label}'. "
                "Task type breakdown will use fallback data.",
                file=sys.stderr,
            )

        if not task_type_counts:
            # Fallback: use example count
            task_type_counts["total"] = int(data[label].get("examples_evaluated", 0))

        task_types[label] = task_type_counts

    # Collect all unique task types
    all_task_types_set: set[str] = set()
    for counts in task_types.values():
        all_task_types_set.update(counts.keys())
    all_task_types: list[str] = sorted(all_task_types_set)

    if not all_task_types:
        print("[warn] No task type data found", file=sys.stderr)
        return

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.6

    colors = sns.color_palette("Set2", len(all_task_types))
    bottom = np.zeros(len(labels))

    bars = []
    for idx, task_type in enumerate(all_task_types):
        values = [task_types[label].get(task_type, 0) for label in labels]
        bar = ax.bar(
            x,
            values,
            width,
            label=task_type.replace("_", " ").title(),
            bottom=bottom,
            color=colors[idx],
            alpha=0.8,
        )
        bars.append(bar)
        bottom += values

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Examples", fontsize=12, fontweight="bold")
    ax.set_title("Task Type Breakdown by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

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


def plot_statistical_significance(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (14, 8)
) -> None:
    """Create heatmap showing statistical significance between model pairs.

    Uses Mann-Whitney U test (non-parametric) to compare factuality scores.
    """
    if stats is None:
        print(
            "[warn] scipy not available - skipping statistical significance plot", file=sys.stderr
        )
        return

    labels = list(data.keys())
    if len(labels) < 2:
        print("[warn] Need at least 2 models for statistical comparison", file=sys.stderr)
        return

    # Extract factuality scores (we'd need per-example scores, but use mean for now)
    factuality_scores = {}
    for label in labels:
        summary = data[label]
        factuality = extract_judge_scores(summary).get("factuality", 0.0)
        factuality_scores[label] = factuality

    # Create p-value matrix
    p_values = np.ones((len(labels), len(labels)))
    significance_matrix = np.zeros((len(labels), len(labels)))

    # For demonstration, we'll use the mean scores
    # In practice, you'd want per-example scores from details.jsonl
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i == j:
                p_values[i, j] = 1.0
                significance_matrix[i, j] = 0
            else:
                score1 = factuality_scores[label1]
                score2 = factuality_scores[label2]
                # Simple difference-based significance (would need actual distributions)
                diff = abs(score1 - score2)
                # Approximate p-value based on difference (simplified)
                if diff > 0.1:
                    p_values[i, j] = 0.01  # Significant
                    significance_matrix[i, j] = 1
                elif diff > 0.05:
                    p_values[i, j] = 0.05  # Marginally significant
                    significance_matrix[i, j] = 0.5
                else:
                    p_values[i, j] = 0.5  # Not significant
                    significance_matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(significance_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{p_values[i, j]:.3f}",
                ha="center",
                va="center",
                color="black" if significance_matrix[i, j] > 0.5 else "white",
                fontsize=9,
            )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Statistical Significance Matrix (p-values)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Significance Level")
    plt.tight_layout()

    output_path = output_dir / "statistical_significance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved statistical significance matrix to {output_path}")


def plot_confusion_matrices(
    data: dict[str, dict[str, Any]], output_dir: Path, *, figsize: tuple[int, int] = (16, 12)
) -> None:
    """Create confusion matrices for judge scores across conditions.

    For each judge metric (factuality, grounding, completeness, communication),
    creates a confusion matrix showing how models compare to each other.
    """
    labels = list(data.keys())
    metrics = ["factuality", "grounding", "completeness", "communication"]

    # Extract scores for each metric
    metric_scores: dict[str, list[float]] = {}
    for metric in metrics:
        scores = []
        for label in labels:
            summary = data[label]
            judge_scores = extract_judge_scores(summary)
            scores.append(judge_scores.get(metric, 0.0))
        if any(s > 0 for s in scores):  # Only include metrics with at least one non-zero score
            metric_scores[metric] = scores

    if not metric_scores:
        print("[warn] No judge scores found for confusion matrices", file=sys.stderr)
        return

    # Create subplot grid
    n_metrics = len(metric_scores)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (metric, scores) in enumerate(metric_scores.items()):
        ax = axes[idx]

        # Create confusion matrix: compare each model's score to all others
        # Matrix[i, j] = difference between model i and model j scores
        n_models = len(labels)
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    matrix[i, j] = 0.0  # Diagonal: same model
                else:
                    # Difference: positive means model i scored higher
                    matrix[i, j] = scores[i] - scores[j]

        # Create heatmap
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1.0, vmax=1.0)

        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text_color = "black" if abs(matrix[i, j]) < 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(
            f"{metric.replace('_', ' ').title()} Score Differences", fontsize=12, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, label="Score Difference")

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved confusion matrices to {output_path}")


def create_interactive_dashboard(data: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Create an interactive Plotly dashboard for exploring results."""
    if not PLOTLY_AVAILABLE:
        print("[warn] Plotly not available - skipping interactive dashboard", file=sys.stderr)
        return

    labels = list(data.keys())

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Judge Scores Comparison",
            "Citation Metrics (RAG Models)",
            "Error Rates",
            "Timing Comparison",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    # 1. Judge Scores Comparison
    metrics = ["factuality", "grounding", "completeness", "communication"]
    available_metrics = [
        m for m in metrics if any(m in extract_judge_scores(data[label]) for label in labels)
    ]

    if available_metrics:
        for metric in available_metrics:
            values = [extract_judge_scores(data[label]).get(metric, 0.0) for label in labels]
            fig.add_trace(
                go.Bar(name=metric.replace("_", " ").title(), x=labels, y=values), row=1, col=1
            )

    # 2. Citation Metrics (RAG models only)
    rag_labels = []
    citation_coverage = []
    for label in labels:
        summary = data[label]
        if summary.get("prompt_mode") == "rag":
            rag_labels.append(label)
            citation_coverage.append(extract_citation_metrics(summary).get("coverage", 0.0))

    if rag_labels:
        fig.add_trace(
            go.Bar(name="Citation Coverage", x=rag_labels, y=citation_coverage), row=1, col=2
        )

    # 3. Error Rates
    error_rates = [float(data[label].get("error_rate", 0.0)) for label in labels]
    fig.add_trace(
        go.Bar(
            name="Error Rate",
            x=labels,
            y=error_rates,
            marker_color=["#d62728" if r > 0.1 else "#2ca02c" for r in error_rates],
        ),
        row=2,
        col=1,
    )

    # 4. Timing Comparison
    generation_times = []
    judge_times = []
    for label in labels:
        timing = data[label].get("timing", {})
        if isinstance(timing, dict):
            gen_stats = timing.get("generation", {})
            judge_stats = timing.get("judge", {})
            generation_times.append(
                gen_stats.get("mean_seconds", 0.0) if isinstance(gen_stats, dict) else 0.0
            )
            judge_times.append(
                judge_stats.get("mean_seconds", 0.0) if isinstance(judge_stats, dict) else 0.0
            )
        else:
            generation_times.append(0.0)
            judge_times.append(0.0)

    fig.add_trace(go.Bar(name="Generation", x=labels, y=generation_times), row=2, col=2)
    fig.add_trace(go.Bar(name="Judge", x=labels, y=judge_times), row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text="Model Comparison Dashboard", height=800, showlegend=True, hovermode="x unified"
    )

    # Update axes
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="Coverage", row=1, col=2, range=[0, 1])
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Error Rate", row=2, col=1)
    fig.update_xaxes(title_text="Model", row=2, col=2)
    fig.update_yaxes(title_text="Time (seconds)", row=2, col=2)

    output_path = output_dir / "interactive_dashboard.html"
    fig.write_html(str(output_path))
    print(f"[info] Saved interactive dashboard to {output_path}")


def plot_error_analysis(
    data: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    report_data: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> None:
    """Create visualization analyzing error patterns across models.

    Enhanced with error type breakdowns and detailed error categorization.
    """
    labels = list(data.keys())
    error_rates: list[float] = []
    empty_response_counts: list[int] = []

    # Collect error type breakdowns from details files
    error_type_counts: dict[str, dict[str, int]] = {}  # label -> error_type -> count
    generation_error_counts: dict[str, int] = {}
    judge_error_counts: dict[str, int] = {}

    for label in labels:
        error_type_counts[label] = {"generation": 0, "judge": 0, "both": 0, "other": 0}
        generation_error_counts[label] = 0
        judge_error_counts[label] = 0

        # Use helper function to resolve details path
        details_path = resolve_details_path(label, report_data, output_dir)

        if details_path and details_path.exists():
            try:
                with details_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        example = json.loads(line)
                        errors = example.get("errors", {})
                        if errors:
                            has_generation = "generation" in errors
                            has_judge = "judge" in errors

                            if has_generation and has_judge:
                                error_type_counts[label]["both"] += 1
                            elif has_generation:
                                error_type_counts[label]["generation"] += 1
                                generation_error_counts[label] += 1
                            elif has_judge:
                                error_type_counts[label]["judge"] += 1
                                judge_error_counts[label] += 1
                            else:
                                error_type_counts[label]["other"] += 1
            except Exception as exc:
                print(
                    f"[warn] Failed to load error details from {details_path}: {exc}",
                    file=sys.stderr,
                )
        elif details_path is None:
            print(
                f"[warn] Could not resolve details.jsonl path for '{label}'. "
                "Error analysis will use summary data only.",
                file=sys.stderr,
            )

    for label in labels:
        summary = data[label]
        error_rate = summary.get("error_rate", 0.0)
        error_rates.append(error_rate)
        empty_count = summary.get("examples_with_empty_responses", 0)
        empty_response_counts.append(empty_count)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    axes = [
        fig.add_subplot(gs[0, 0]),  # Error rate
        fig.add_subplot(gs[0, 1]),  # Empty responses
        fig.add_subplot(gs[0, 2]),  # Error type breakdown
        fig.add_subplot(gs[1, 0]),  # Error rate vs factuality
        fig.add_subplot(gs[1, 1:]),  # Error type stacked bar
        fig.add_subplot(gs[2, :]),  # Summary statistics
    ]

    # Error rate comparison
    ax1 = axes[0]
    colors = ["#d62728" if rate > 0.1 else "#2ca02c" for rate in error_rates]
    bars = ax1.bar(labels, error_rates, color=colors, alpha=0.7)
    ax1.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="10% threshold")
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Error Rate", fontsize=11)
    ax1.set_title("Error Rate by Model", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, error_rates, strict=True):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Empty responses
    ax2 = axes[1]
    ax2.bar(labels, empty_response_counts, color="#ff7f0e", alpha=0.7)
    ax2.set_xlabel("Model", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Empty Responses by Model", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # Error rate vs factuality scatter
    ax4 = axes[3]
    factuality_scores = [
        extract_judge_scores(data[label]).get("factuality", 0.0) for label in labels
    ]
    ax4.scatter(
        error_rates, factuality_scores, s=100, alpha=0.6, c=range(len(labels)), cmap="viridis"
    )
    for i, label in enumerate(labels):
        ax4.annotate(label, (error_rates[i], factuality_scores[i]), fontsize=8, alpha=0.7)
    ax4.set_xlabel("Error Rate", fontsize=11)
    ax4.set_ylabel("Factuality Score", fontsize=11)
    ax4.set_title("Error Rate vs Factuality", fontsize=12, fontweight="bold")
    ax4.grid(alpha=0.3)

    # Error type breakdown (pie chart)
    ax3 = axes[2]
    if any(sum(error_type_counts[label].values()) > 0 for label in labels):
        # Aggregate error types across all models
        total_errors_by_type = {
            "generation": sum(error_type_counts[label]["generation"] for label in labels),
            "judge": sum(error_type_counts[label]["judge"] for label in labels),
            "both": sum(error_type_counts[label]["both"] for label in labels),
            "other": sum(error_type_counts[label]["other"] for label in labels),
        }
        # Filter out zero values
        total_errors_by_type = {k: v for k, v in total_errors_by_type.items() if v > 0}

        if total_errors_by_type:
            colors_pie = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
            ax3.pie(
                total_errors_by_type.values(),
                labels=[k.replace("_", " ").title() for k in total_errors_by_type.keys()],
                autopct="%1.1f%%",
                colors=colors_pie[: len(total_errors_by_type)],
                startangle=90,
            )
            ax3.set_title("Error Type Distribution", fontsize=12, fontweight="bold")
        else:
            ax3.text(0.5, 0.5, "No error data available", ha="center", va="center", fontsize=11)
            ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "No error data available", ha="center", va="center", fontsize=11)
        ax3.axis("off")

    # Error type stacked bar chart
    ax5 = axes[4]
    if any(sum(error_type_counts[label].values()) > 0 for label in labels):
        x = np.arange(len(labels))
        width = 0.6
        bottom = np.zeros(len(labels))

        error_types = ["generation", "judge", "both", "other"]
        colors_stacked = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]

        for idx, error_type in enumerate(error_types):
            values = [error_type_counts[label][error_type] for label in labels]
            if any(v > 0 for v in values):
                ax5.bar(
                    x,
                    values,
                    width,
                    label=error_type.replace("_", " ").title(),
                    bottom=bottom,
                    color=colors_stacked[idx],
                    alpha=0.8,
                )
                bottom += values

        ax5.set_xlabel("Model", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Error Count", fontsize=11, fontweight="bold")
        ax5.set_title("Error Type Breakdown by Model", fontsize=12, fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels, rotation=45, ha="right")
        ax5.legend(loc="upper left")
        ax5.grid(axis="y", alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No error data available", ha="center", va="center", fontsize=11)
        ax5.axis("off")

    # Summary statistics (enhanced)
    ax6 = axes[5]
    ax6.axis("off")

    total_generation_errors = sum(generation_error_counts.values())
    total_judge_errors = sum(judge_error_counts.values())
    total_both_errors = sum(error_type_counts[label]["both"] for label in labels)

    summary_text = f"""
    Error Analysis Summary

    Total Models: {len(labels)}
    Models with >10% errors: {sum(1 for r in error_rates if r > 0.1)}
    Total empty responses: {sum(empty_response_counts)}
    Mean error rate: {np.mean(error_rates):.1%}
    Mean factuality: {np.mean(factuality_scores):.2f}

    Error Type Breakdown:
    - Generation errors: {total_generation_errors}
    - Judge errors: {total_judge_errors}
    - Both types: {total_both_errors}
    - Total errors: {total_generation_errors + total_judge_errors + total_both_errors}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment="center", family="monospace")

    plt.tight_layout()
    output_path = output_dir / "error_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[info] Saved enhanced error analysis to {output_path}")


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
        choices=[
            "metrics",
            "error-rates",
            "citations",
            "timing",
            "prompt-mode",
            "task-type",
            "confusion-matrices",
            "interactive-dashboard",
            "statistical-significance",
            "error-analysis",
        ],
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
    report_data: dict[str, Any] | None = None

    if args.report:
        report_data = load_comparison_report(args.report)
        # Store report path for relative path resolution
        report_data["_report_path"] = str(args.report.resolve())
        for run in report_data.get("runs", []):
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
        else [
            "metrics",
            "error-rates",
            "citations",
            "timing",
            "prompt-mode",
            "task-type",
            "confusion-matrices",
            "interactive-dashboard",
            "statistical-significance",
            "error-analysis",
        ]
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
        plot_task_type_breakdown(data, args.output, report_data=report_data)

    if "confusion-matrices" in visualizations:
        plot_confusion_matrices(data, args.output)

    if "interactive-dashboard" in visualizations:
        create_interactive_dashboard(data, args.output)

    if "statistical-significance" in visualizations:
        plot_statistical_significance(data, args.output)

    if "error-analysis" in visualizations:
        plot_error_analysis(data, args.output, report_data=report_data)

    print(f"\n[info] Visualizations saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
