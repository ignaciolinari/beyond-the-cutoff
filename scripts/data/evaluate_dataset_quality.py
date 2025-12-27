#!/usr/bin/env python3
"""Evaluate dataset quality using an LLM judge.

This script assesses the semantic quality of generated training examples
to catch issues like:
- Unanswerable questions (context doesn't support the answer)
- Incorrect gold answers
- Unclear instructions
- Mismatched instruction/response pairs

Usage:
    # Evaluate a sample of 50 examples using the default 7B judge
    python scripts/evaluate_dataset_quality.py \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --sample-size 50

    # Evaluate all examples with a specific judge config
    python scripts/evaluate_dataset_quality.py \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --judge-inference configs/judges/ollama_qwen7b.yaml

    # Filter to specific task types
    python scripts/evaluate_dataset_quality.py \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --task-type qa \
        --task-type citations \
        --sample-size 30

    # Output detailed results to JSON
    python scripts/evaluate_dataset_quality.py \
        --dataset evaluation/datasets/offline_dataset.jsonl \
        --output evaluation/quality_report.json \
        --include-verdicts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from beyond_the_cutoff.config import InferenceConfig
from beyond_the_cutoff.evaluation.dataset_judge import (
    DatasetQualityJudge,
    DatasetQualityResult,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dataset quality using an LLM judge")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to offline dataset JSONL file",
    )
    parser.add_argument(
        "--judge-inference",
        type=Path,
        default=None,
        help="Path to judge inference config YAML (default: uses Qwen 7B via Ollama)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        action="append",
        dest="task_types",
        default=None,
        help="Filter to specific task types (repeatable, e.g., --task-type qa --task-type citations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON results (default: print to stdout)",
    )
    parser.add_argument(
        "--include-verdicts",
        action="store_true",
        help="Include individual verdicts in output (verbose)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.6,
        help="Minimum score threshold for each criterion (default: 0.6)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def load_inference_config(path: Path | None) -> InferenceConfig:
    """Load inference config from YAML or use default."""
    if path is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return InferenceConfig.model_validate(data)

    # Default: Qwen 3 8B via Ollama with thinking mode enabled
    # IMPORTANT: Use a DIFFERENT model than the generator (Qwen 2.5 7B) to avoid
    # self-preference bias where a model rates its own outputs more favorably.
    #
    # Qwen3 Thinking Mode:
    # - Enabled when temperature > 0 (we use 0.6)
    # - Model outputs <think>...</think> reasoning before the answer
    # - Improves judgment quality through chain-of-thought reasoning
    return InferenceConfig(
        provider="ollama",
        model="qwen3:8b",  # Different from generator (qwen2.5:7b-instruct-q4_K_M)
        host="http://localhost",
        port=11434,
        timeout=180.0,
        max_new_tokens=1024,  # Increased for thinking + response
        temperature=0.6,  # Enable thinking mode (requires temp > 0)
        stop_sequences=["<|im_start|>", "<|im_end|>"],  # Qwen3 ChatML stop tokens
    )


def print_summary(result: DatasetQualityResult) -> None:
    """Print human-readable summary to stderr."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("DATASET QUALITY ASSESSMENT REPORT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print(f"\nExamples evaluated: {result.total_evaluated}", file=sys.stderr)
    print(f"Passed: {result.passed_count} ({result.pass_rate:.1%})", file=sys.stderr)
    print(f"Failed: {result.failed_count}", file=sys.stderr)

    print("\n--- Mean Scores ---", file=sys.stderr)
    for key, value in sorted(result.mean_scores.items()):
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        status = "✓" if value >= 0.6 else "✗"
        print(f"  {key:15} {bar} {value:.2f} {status}", file=sys.stderr)

    print("\n--- Score Distributions ---", file=sys.stderr)
    for key, buckets in result.score_distributions.items():
        print(f"  {key}:", file=sys.stderr)
        for bucket, count in buckets.items():
            pct = count / result.total_evaluated * 100 if result.total_evaluated else 0
            bar = "▓" * int(pct / 5)
            print(f"    {bucket}: {count:3d} ({pct:5.1f}%) {bar}", file=sys.stderr)

    if result.common_issues:
        print("\n--- Common Issues ---", file=sys.stderr)
        for issue, count in list(result.common_issues.items())[:10]:
            print(f"  [{count:3d}] {issue[:80]}", file=sys.stderr)

    if result.failed_examples:
        print(
            f"\n--- Failed Examples (showing first 5 of {len(result.failed_examples)}) ---",
            file=sys.stderr,
        )
        for ex in result.failed_examples[:5]:
            print(f"\n  Task: {ex.task_id} ({ex.task_type})", file=sys.stderr)
            print(f"  Scores: {ex.scores}", file=sys.stderr)
            print(f"  Issues: {ex.issues}", file=sys.stderr)
            print(f"  Reasoning: {ex.reasoning[:150]}...", file=sys.stderr)

    # Overall verdict
    print("\n" + "=" * 60, file=sys.stderr)
    if result.pass_rate >= 0.9:
        print("✓ EXCELLENT: Dataset quality is very high (≥90% pass rate)", file=sys.stderr)
    elif result.pass_rate >= 0.75:
        print("✓ GOOD: Dataset quality is acceptable (≥75% pass rate)", file=sys.stderr)
    elif result.pass_rate >= 0.6:
        print("WARNING:  FAIR: Dataset has quality issues (60-75% pass rate)", file=sys.stderr)
    else:
        print("✗ POOR: Dataset needs significant cleanup (<60% pass rate)", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    # Load judge config
    inference_config = load_inference_config(args.judge_inference)

    if not args.quiet:
        print(f"[info] Using judge model: {inference_config.model}", file=sys.stderr)
        print(f"[info] Dataset: {args.dataset}", file=sys.stderr)
        if args.sample_size:
            print(f"[info] Sample size: {args.sample_size}", file=sys.stderr)
        if args.task_types:
            print(f"[info] Task types: {', '.join(args.task_types)}", file=sys.stderr)

    # Initialize judge
    judge = DatasetQualityJudge(
        inference_config,
        pass_threshold=args.pass_threshold,
    )

    # Run evaluation
    task_types_set = set(args.task_types) if args.task_types else None
    result = judge.evaluate_dataset(
        args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        task_types=task_types_set,
    )

    # Print summary
    if not args.quiet:
        print_summary(result)

    # Output results
    output_data = result.to_dict()
    if args.include_verdicts:
        output_data["verdicts"] = [
            {
                "task_id": v.task_id,
                "task_type": v.task_type,
                "passed": v.passed,
                "scores": v.scores,
                "issues": v.issues,
                "reasoning": v.reasoning,
            }
            for v in result.verdicts
        ]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[info] Results written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    # Exit code based on pass rate
    if result.pass_rate >= 0.75:
        return 0
    elif result.pass_rate >= 0.5:
        return 1  # Warning
    else:
        return 2  # Failure


if __name__ == "__main__":
    sys.exit(main())
