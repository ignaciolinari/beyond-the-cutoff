#!/usr/bin/env python3
"""Run comparative evaluation sweeps across multiple model configurations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.evaluation.comparison import (
    build_comparison_report,
    describe_plan,
    execute_comparison_plan,
    load_comparison_plan,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple evaluation runs via a plan YAML")
    parser.add_argument("--config", default="configs/default.yaml", help="Project config path")
    parser.add_argument("--plan", required=True, help="Comparison plan YAML path")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional aggregated report JSON path (stdout always prints summary)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override the example limit for all runs in the plan",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override retry budget for all runs",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Override retry delay for all runs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even if metrics artifacts already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned runs and exit without executing",
    )
    parser.add_argument(
        "--no-validate-examples",
        action="store_true",
        help="Skip validation that all experiments use the same examples",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()

    print("=" * 80, file=sys.stderr)
    print("[info] Starting comparative evaluation", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    config_load_start = time.time()
    config_path = Path(args.config).resolve()
    project_cfg: ProjectConfig = load_config(config_path)
    config_load_time = time.time() - config_load_start
    print(f"[info] Loaded config from {args.config} ({config_load_time:.2f}s)", file=sys.stderr)

    plan_load_start = time.time()
    plan_path = Path(args.plan).resolve()
    plan = load_comparison_plan(plan_path)
    plan_load_time = time.time() - plan_load_start
    print(
        f"[info] Loaded comparison plan from {args.plan} ({plan_load_time:.2f}s)", file=sys.stderr
    )
    print(f"[info] Found {len(plan.runs)} run(s) to execute", file=sys.stderr)

    if args.dry_run:
        print("[info] DRY RUN MODE - showing plan without executing", file=sys.stderr)
        rows = describe_plan(plan, project_cfg)
        print(json.dumps({"runs": rows}, indent=2))
        return

    if args.force:
        print("[info] Force mode: will re-run evaluations even if results exist", file=sys.stderr)
    if args.limit:
        print(f"[info] Limiting all runs to {args.limit} examples", file=sys.stderr)
    if args.max_retries is not None:
        print(f"[info] Overriding max retries to {args.max_retries}", file=sys.stderr)
    if args.retry_delay is not None:
        print(f"[info] Overriding retry delay to {args.retry_delay}s", file=sys.stderr)

    print("-" * 80, file=sys.stderr)
    print("[info] Executing comparison plan...", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    execution_start = time.time()
    results = execute_comparison_plan(
        plan,
        project_config=project_cfg,
        config_path=config_path,
        limit_override=args.limit,
        max_retries_override=args.max_retries,
        retry_delay_override=args.retry_delay,
        force=args.force,
        validate_same_examples=not args.no_validate_examples,
    )
    execution_time = time.time() - execution_start

    print("-" * 80, file=sys.stderr)
    print(
        f"[info] Comparison plan execution completed ({execution_time:.2f}s, {execution_time/60:.1f} minutes)",
        file=sys.stderr,
    )

    skipped_count = sum(1 for r in results if r.skipped)
    completed_count = len(results) - skipped_count
    print(f"[info] Completed: {completed_count}/{len(results)} runs", file=sys.stderr)
    if skipped_count > 0:
        print(f"[info] Skipped: {skipped_count} runs (already exist)", file=sys.stderr)

    print("[info] Building comparison report...", file=sys.stderr)
    report = build_comparison_report(results)

    total_time = time.time() - start_time
    print("=" * 80, file=sys.stderr)
    print(f"[info] Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    print(json.dumps(report.as_dict(), indent=2))

    if args.output:
        output_path = Path(args.output)
        write_report(report, output_path)
        print(f"[info] Report written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
