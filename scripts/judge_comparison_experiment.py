#!/usr/bin/env python3
"""
Judge Comparison Experiment

Runs a small sample (20 questions) through all 6 conditions with 2 different judges
(Qwen3 8B and Llama 3.1 8B) to:
1. Verify the evaluation pipeline is working correctly with the fixed judge prompts
2. Compare judge agreement/speed to decide which to use for full evaluation

Usage:
    python scripts/judge_comparison_experiment.py --sample-size 20
"""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Save items to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def create_sample_dataset(
    full_dataset_path: Path,
    sample_path: Path,
    sample_size: int,
    seed: int = 42,
) -> list[str]:
    """Create a random sample of the dataset and return task IDs."""
    random.seed(seed)
    items = load_jsonl(full_dataset_path)

    if len(items) <= sample_size:
        sample = items
    else:
        sample = random.sample(items, sample_size)

    save_jsonl(sample, sample_path)
    print(f"Created sample dataset with {len(sample)} examples at {sample_path}")

    return [item["task_id"] for item in sample]


def create_judge_plan(
    original_plan_path: Path,
    output_plan_path: Path,
    sample_dataset_path: str,
    judge_inference_path: str,
    output_dir: str,
) -> None:
    """Create a modified plan with specific judge and output dir."""
    with open(original_plan_path) as f:
        content = f.read()

    # Replace dataset path
    content = content.replace(
        "dataset: ../../evaluation/datasets/eval_dataset.jsonl", f"dataset: {sample_dataset_path}"
    )

    # Replace judge inference config
    content = content.replace(
        "judge_inference: ../judges/dataset_quality_judge.yaml",
        f"judge_inference: {judge_inference_path}",
    )

    # Replace output dir
    content = content.replace("output_dir: ../../evaluation/results", f"output_dir: {output_dir}")

    # Disable skip_if_exists to force re-evaluation
    content = content.replace("skip_if_exists: true", "skip_if_exists: false")

    with open(output_plan_path, "w") as f:
        f.write(content)

    print(f"Created judge plan at {output_plan_path}")


def run_evaluation(
    plan_path: Path,
    responses_dir: Path,
    output_path: Path,
    label: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run evaluation with a specific judge and return timing info."""

    cmd = [
        "python",
        "scripts/compare_models.py",
        "--plan",
        str(plan_path),
        "--responses-dir",
        str(responses_dir),
        "--output",
        str(output_path),
        "--force",  # Force re-evaluation
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\n{'='*60}")
    print(f"Running evaluation with {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    return {
        "judge": label,
        "elapsed_seconds": elapsed,
        "exit_code": result.returncode,
    }


def collect_results(results_dir: Path, conditions: list[str]) -> dict[str, dict[str, Any] | None]:
    """Collect evaluation results for all conditions."""
    collected: dict[str, dict[str, Any] | None] = {}

    for condition in conditions:
        metrics_path = results_dir / condition / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                collected[condition] = json.load(f)
        else:
            print(f"[warn] Missing metrics for {condition}")
            collected[condition] = None

    return collected


def compare_judges(
    qwen_results: dict[str, dict[str, Any] | None],
    llama_results: dict[str, dict[str, Any] | None],
    conditions: list[str],
) -> None:
    """Compare results between two judges."""

    print("\n" + "=" * 80)
    print("JUDGE COMPARISON RESULTS")
    print("=" * 80)

    metrics_to_compare = ["factuality", "completeness", "communication", "grounding"]

    # Header
    print(f"\n{'Condition':<45} {'Metric':<15} {'Qwen3':>8} {'Llama':>8} {'Diff':>8}")
    print("-" * 84)

    total_diffs: dict[str, list[float]] = {m: [] for m in metrics_to_compare}

    for condition in conditions:
        qwen = qwen_results.get(condition)
        llama = llama_results.get(condition)

        if qwen is None or llama is None:
            print(f"{condition:<45} {'N/A':<15} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue

        for metric in metrics_to_compare:
            qwen_val = qwen.get(metric, 0)
            llama_val = llama.get(metric, 0)
            diff = abs(llama_val - qwen_val)
            total_diffs[metric].append(diff)

            diff_str = f"{llama_val - qwen_val:+.3f}"
            print(f"{condition:<45} {metric:<15} {qwen_val:>8.3f} {llama_val:>8.3f} {diff_str:>8}")
        print("-" * 84)

    # Summary
    print(f"\n{'AVERAGE ABSOLUTE DIFFERENCE':<45}")
    print("-" * 84)
    for metric in metrics_to_compare:
        if total_diffs[metric]:
            avg_diff = sum(total_diffs[metric]) / len(total_diffs[metric])
            print(f"  {metric:<43} {avg_diff:>8.3f}")

    # Overall agreement assessment
    all_diffs = [d for diffs in total_diffs.values() for d in diffs]
    if all_diffs:
        avg_all = sum(all_diffs) / len(all_diffs)
        print(f"\n  {'Overall average difference:':<43} {avg_all:>8.3f}")

        if avg_all < 0.1:
            print("\n  ✅ Judges show HIGH agreement (diff < 0.1)")
        elif avg_all < 0.2:
            print("\n  ⚠️  Judges show MODERATE agreement (0.1 < diff < 0.2)")
        else:
            print("\n  ❌ Judges show LOW agreement (diff > 0.2)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge comparison experiment")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of examples to sample (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip response generation (use existing)",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip Qwen3 evaluation (if already done)",
    )
    parser.add_argument(
        "--skip-llama",
        action="store_true",
        help="Skip Llama evaluation (if already done)",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    full_dataset = project_root / "evaluation/datasets/eval_dataset.jsonl"
    sample_dataset = project_root / "evaluation/datasets/eval_dataset_sample.jsonl"
    responses_dir = project_root / "evaluation/responses"

    # Plan paths
    original_plan = project_root / "configs/evaluation/compare_0p5b_experiments.yaml"
    sample_plan = project_root / "configs/evaluation/compare_0p5b_sample.yaml"
    qwen_plan = project_root / "configs/evaluation/compare_0p5b_sample_qwen.yaml"
    llama_plan = project_root / "configs/evaluation/compare_0p5b_sample_llama.yaml"

    # Output directories
    qwen_output_dir = project_root / "evaluation/results_qwen_sample"
    llama_output_dir = project_root / "evaluation/results_llama_sample"

    # Conditions to evaluate
    conditions = [
        "base_baseline_0p5b",
        "rag_baseline_0p5b",
        "lora_science_0p5b_ft_only",
        "hybrid_science_0p5b_instruction_only",
        "lora_science_0p5b_rag_trained_ft_only",
        "hybrid_science_0p5b_rag_trained",
    ]

    # Step 1: Create sample dataset
    print("\n" + "=" * 80)
    print("STEP 1: Creating sample dataset")
    print("=" * 80)
    create_sample_dataset(full_dataset, sample_dataset, args.sample_size, args.seed)

    # Step 2: Create base sample plan (for generation)
    print("\n" + "=" * 80)
    print("STEP 2: Creating experiment plans")
    print("=" * 80)

    # Create base sample plan
    with open(original_plan) as f:
        content = f.read()
    content = content.replace(
        "dataset: ../../evaluation/datasets/eval_dataset.jsonl",
        "dataset: ../../evaluation/datasets/eval_dataset_sample.jsonl",
    )
    with open(sample_plan, "w") as f:
        f.write(content)
    print(f"Created base sample plan at {sample_plan}")

    # Create Qwen plan
    create_judge_plan(
        original_plan,
        qwen_plan,
        "../../evaluation/datasets/eval_dataset_sample.jsonl",
        "../judges/dataset_quality_judge.yaml",  # Qwen3 8B
        "../../evaluation/results_qwen_sample",
    )

    # Create Llama plan
    create_judge_plan(
        original_plan,
        llama_plan,
        "../../evaluation/datasets/eval_dataset_sample.jsonl",
        "../judges/llama3_judge.yaml",  # Llama 3.1 8B
        "../../evaluation/results_llama_sample",
    )

    # Step 3: Generate responses for sample (if not already done)
    if not args.skip_generation:
        print("\n" + "=" * 80)
        print("STEP 3: Generating responses for sample")
        print("=" * 80)

        gen_cmd = [
            "python",
            "scripts/generate_responses.py",
            "--plan",
            str(sample_plan),
            "--output-dir",
            str(responses_dir),
        ]
        print(f"Command: {' '.join(gen_cmd)}")
        subprocess.run(gen_cmd)
    else:
        print("\n" + "=" * 80)
        print("STEP 3: Skipping response generation (--skip-generation)")
        print("=" * 80)

    timing_results = []

    # Step 4a: Run Qwen3 evaluation
    if not args.skip_qwen:
        print("\n" + "=" * 80)
        print("STEP 4a: Running Qwen3 8B evaluation")
        print("=" * 80)

        qwen_timing = run_evaluation(
            plan_path=qwen_plan,
            responses_dir=responses_dir,
            output_path=qwen_output_dir / "comparison_report.json",
            label="qwen3_8b",
            limit=args.sample_size,  # Only evaluate sample_size examples per condition
        )
        timing_results.append(qwen_timing)
        print(f"\nQwen3 evaluation completed in {qwen_timing['elapsed_seconds']:.1f}s")
    else:
        print("\n[info] Skipping Qwen3 evaluation (--skip-qwen)")

    # Step 4b: Run Llama evaluation
    if not args.skip_llama:
        print("\n" + "=" * 80)
        print("STEP 4b: Running Llama 3.1 8B evaluation")
        print("=" * 80)

        llama_timing = run_evaluation(
            plan_path=llama_plan,
            responses_dir=responses_dir,
            output_path=llama_output_dir / "comparison_report.json",
            label="llama3.1_8b",
            limit=args.sample_size,  # Only evaluate sample_size examples per condition
        )
        timing_results.append(llama_timing)
        print(f"\nLlama evaluation completed in {llama_timing['elapsed_seconds']:.1f}s")
    else:
        print("\n[info] Skipping Llama evaluation (--skip-llama)")

    # Step 5: Compare results
    print("\n" + "=" * 80)
    print("STEP 5: Comparing results")
    print("=" * 80)

    qwen_results = collect_results(qwen_output_dir, conditions)
    llama_results = collect_results(llama_output_dir, conditions)

    if any(qwen_results.values()) and any(llama_results.values()):
        compare_judges(qwen_results, llama_results, conditions)
    else:
        print("\n[warn] Not enough results to compare judges")
        if qwen_results:
            print("\nQwen results available:")
            for cond, metrics in qwen_results.items():
                if metrics:
                    print(f"  {cond}: factuality={metrics.get('factuality', 'N/A'):.3f}")
        if llama_results:
            print("\nLlama results available:")
            for cond, metrics in llama_results.items():
                if metrics:
                    print(f"  {cond}: factuality={metrics.get('factuality', 'N/A'):.3f}")

    # Print timing summary
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    for result in timing_results:
        per_example = result["elapsed_seconds"] / (args.sample_size * 6) if args.sample_size else 0
        print(
            f"{result['judge']}: {result['elapsed_seconds']:.1f}s total ({per_example:.1f}s per example)"
        )

    if len(timing_results) == 2:
        speedup = timing_results[0]["elapsed_seconds"] / timing_results[1]["elapsed_seconds"]
        faster = timing_results[1]["judge"] if speedup > 1 else timing_results[0]["judge"]
        print(f"\n{faster} is {abs(1/speedup - 1)*100 + 100:.0f}% the speed of the other")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print("\nSample results saved to:")
    print(f"  - Qwen3: {qwen_output_dir}")
    print(f"  - Llama: {llama_output_dir}")
    print("\nTo continue with full evaluation using chosen judge:")
    print("  1. Delete sample results if desired")
    print("  2. Run the full evaluation with the appropriate plan")


if __name__ == "__main__":
    main()
