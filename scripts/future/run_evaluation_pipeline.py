#!/usr/bin/env python3
"""Unified script to run complete evaluation workflows.

This script orchestrates the full evaluation pipeline including:
1. Response generation (Phase 1)
2. Pairwise evaluation/judging (Phase 2)
3. ELO ranking computation
4. Results visualization

Usage:
    # Run full 6-condition evaluation
    python scripts/run_evaluation_pipeline.py full-comparison \
        --plan configs/evaluation/six_condition_plan.yaml \
        --output-dir evaluation/results/six_condition/

    # Run quantization comparison
    python scripts/run_evaluation_pipeline.py quantization \
        --plan configs/evaluation/quantization_comparison.yaml \
        --output-dir evaluation/results/quantization/

    # Run retrieval ablation with ELO
    python scripts/run_evaluation_pipeline.py retrieval-ablation \
        --plan configs/evaluation/retrieval_ablation.yaml \
        --output-dir evaluation/results/retrieval_ablation/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[step] {description}", file=sys.stderr)
    print(f"[cmd]  {' '.join(cmd)}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"[done] {description} completed in {elapsed:.1f}s", file=sys.stderr)
        return True
    else:
        print(f"[error] {description} failed (exit code {result.returncode})", file=sys.stderr)
        return False


def run_full_comparison(
    plan_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
) -> bool:
    """Run full comparison workflow (generation + evaluation)."""

    responses_dir = output_dir / "responses"
    results_dir = output_dir / "results"

    # Phase 1: Generate responses
    if not skip_generation:
        cmd = [
            "python",
            "scripts/generate_responses.py",
            "--plan",
            str(plan_path),
            "--output-dir",
            str(responses_dir),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])

        if not run_command(cmd, "Phase 1: Response Generation"):
            return False

    # Phase 2: Evaluate responses
    if not skip_evaluation:
        cmd = [
            "python",
            "scripts/compare_models.py",
            "--plan",
            str(plan_path),
            "--responses-dir",
            str(responses_dir),
            "--output",
            str(results_dir / "comparison_results.jsonl"),
        ]

        if not run_command(cmd, "Phase 2: Pairwise Evaluation"):
            return False

    # Visualize results
    results_file = results_dir / "comparison_results.jsonl"
    if results_file.exists():
        cmd = [
            "python",
            "scripts/visualize_comparison.py",
            "--results",
            str(results_file),
        ]
        run_command(cmd, "Visualize Results")

    return True


def run_quantization_comparison(
    plan_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    register_f16: bool = False,
) -> bool:
    """Run quantization comparison (Q4_K_M vs F16)."""

    # Optionally register F16 model first
    if register_f16:
        modelfile_path = Path("ollama/Modelfile.rag_trained_f16")
        if modelfile_path.exists():
            cmd = ["ollama", "create", "lora_science_0p5_f16", "-f", str(modelfile_path)]
            if not run_command(cmd, "Register F16 model with Ollama"):
                print("[warning] F16 model registration failed, continuing...", file=sys.stderr)

    # Run full comparison workflow
    return run_full_comparison(plan_path, output_dir, limit=limit)


def run_retrieval_ablation(
    plan_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    skip_generation: bool = False,
    skip_pairwise: bool = False,
    skip_elo: bool = False,
) -> bool:
    """Run retrieval ablation with ELO ranking."""

    responses_dir = output_dir / "responses"
    pairwise_dir = output_dir / "pairwise"
    elo_dir = output_dir / "elo"

    # Phase 1: Generate responses with different retrieval configs
    if not skip_generation:
        cmd = [
            "python",
            "scripts/run_retrieval_ablation.py",
            "--plan",
            str(plan_path),
            "--output-dir",
            str(responses_dir),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])

        if not run_command(cmd, "Phase 1: Retrieval Ablation Generation"):
            return False

    # Phase 2: Run pairwise comparisons
    if not skip_pairwise:
        # Find all response files
        response_files = list(responses_dir.glob("*.jsonl"))
        response_files = [f for f in response_files if f.name != "ablation_metadata.json"]

        if len(response_files) < 2:
            print("[error] Need at least 2 response files for pairwise comparison", file=sys.stderr)
            return False

        cmd = [
            "python",
            "scripts/run_pairwise_evaluation.py",
            "--results",
            *[str(f) for f in response_files],
            "--output",
            str(pairwise_dir / "pairwise_results.jsonl"),
        ]

        if not run_command(cmd, "Phase 2: Pairwise Comparisons"):
            return False

    # Phase 3: Compute ELO rankings
    if not skip_elo:
        pairwise_file = pairwise_dir / "pairwise_results.jsonl"
        if pairwise_file.exists():
            cmd = [
                "python",
                "scripts/compute_elo_rankings.py",
                "--comparisons",
                str(pairwise_file),
                "--output",
                str(elo_dir / "elo_rankings.json"),
            ]

            if not run_command(cmd, "Phase 3: ELO Ranking"):
                return False

    # Print final rankings
    elo_file = elo_dir / "elo_rankings.json"
    if elo_file.exists():
        print("\n" + "=" * 80, file=sys.stderr)
        print("[info] Final ELO Rankings:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        with open(elo_file) as f:
            rankings = json.load(f)
        for rank, item in enumerate(rankings.get("rankings", []), start=1):
            print(
                f"  {rank}. {item['model']} - ELO: {item['elo']:.1f} "
                f"(W: {item.get('wins', 0)}, L: {item.get('losses', 0)})",
                file=sys.stderr,
            )

    return True


def run_end_to_end_validation(
    plan_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> bool:
    """Run end-to-end validation with live retrieval."""

    cmd = [
        "python",
        "scripts/evaluate_end_to_end.py",
        "--plan",
        str(plan_path),
        "--output-dir",
        str(output_dir),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    return run_command(cmd, "End-to-End Validation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete evaluation workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    # Full comparison workflow
    p_full = subparsers.add_parser(
        "full-comparison",
        help="Run full 6-condition comparison",
    )
    p_full.add_argument("--plan", type=Path, required=True)
    p_full.add_argument("--output-dir", type=Path, required=True)
    p_full.add_argument("--limit", type=int)
    p_full.add_argument("--skip-generation", action="store_true")
    p_full.add_argument("--skip-evaluation", action="store_true")

    # Quantization comparison
    p_quant = subparsers.add_parser(
        "quantization",
        help="Run quantization comparison (Q4_K_M vs F16)",
    )
    p_quant.add_argument("--plan", type=Path, required=True)
    p_quant.add_argument("--output-dir", type=Path, required=True)
    p_quant.add_argument("--limit", type=int)
    p_quant.add_argument("--register-f16", action="store_true", help="Register F16 model first")

    # Retrieval ablation
    p_retrieval = subparsers.add_parser(
        "retrieval-ablation",
        help="Run retrieval ablation with ELO ranking",
    )
    p_retrieval.add_argument("--plan", type=Path, required=True)
    p_retrieval.add_argument("--output-dir", type=Path, required=True)
    p_retrieval.add_argument("--limit", type=int)
    p_retrieval.add_argument("--skip-generation", action="store_true")
    p_retrieval.add_argument("--skip-pairwise", action="store_true")
    p_retrieval.add_argument("--skip-elo", action="store_true")

    # End-to-end validation
    p_e2e = subparsers.add_parser(
        "end-to-end",
        help="Run end-to-end validation with live retrieval",
    )
    p_e2e.add_argument("--plan", type=Path, required=True)
    p_e2e.add_argument("--output-dir", type=Path, required=True)
    p_e2e.add_argument("--limit", type=int)

    args = parser.parse_args()

    print("=" * 80, file=sys.stderr)
    print(f"[info] Evaluation Pipeline: {args.workflow}", file=sys.stderr)
    print(f"[info] Started: {datetime.now().isoformat()}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    start_time = time.time()
    success = False

    if args.workflow == "full-comparison":
        success = run_full_comparison(
            args.plan,
            args.output_dir,
            limit=args.limit,
            skip_generation=args.skip_generation,
            skip_evaluation=args.skip_evaluation,
        )
    elif args.workflow == "quantization":
        success = run_quantization_comparison(
            args.plan,
            args.output_dir,
            limit=args.limit,
            register_f16=args.register_f16,
        )
    elif args.workflow == "retrieval-ablation":
        success = run_retrieval_ablation(
            args.plan,
            args.output_dir,
            limit=args.limit,
            skip_generation=args.skip_generation,
            skip_pairwise=args.skip_pairwise,
            skip_elo=args.skip_elo,
        )
    elif args.workflow == "end-to-end":
        success = run_end_to_end_validation(
            args.plan,
            args.output_dir,
            limit=args.limit,
        )

    total_time = time.time() - start_time

    print("\n" + "=" * 80, file=sys.stderr)
    if success:
        print(f"[success] Pipeline completed in {total_time:.1f}s", file=sys.stderr)
    else:
        print(f"[failed] Pipeline failed after {total_time:.1f}s", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
