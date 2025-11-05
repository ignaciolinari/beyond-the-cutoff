#!/usr/bin/env python3
"""Run comparative evaluation sweeps across multiple model configurations."""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    project_cfg: ProjectConfig = load_config(config_path)

    plan_path = Path(args.plan).resolve()
    plan = load_comparison_plan(plan_path)

    if args.dry_run:
        rows = describe_plan(plan, project_cfg)
        print(json.dumps({"runs": rows}, indent=2))
        return

    results = execute_comparison_plan(
        plan,
        project_config=project_cfg,
        config_path=config_path,
        limit_override=args.limit,
        max_retries_override=args.max_retries,
        retry_delay_override=args.retry_delay,
        force=args.force,
    )

    report = build_comparison_report(results)
    print(json.dumps(report.as_dict(), indent=2))

    if args.output:
        write_report(report, Path(args.output))


if __name__ == "__main__":
    main()
