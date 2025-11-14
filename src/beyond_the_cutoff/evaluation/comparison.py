"""Helpers for executing and aggregating comparative evaluation runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import ProjectConfig
from beyond_the_cutoff.evaluation.runner import (
    EvaluationResult,
    load_inference_from_yaml,
    run_evaluation,
)


@dataclass
class PlanDefaults:
    dataset: Path | None
    judge_config: Path | None
    judge_inference: Path | None
    output_dir: Path | None
    metrics_filename: str
    details_filename: str
    metadata_filename: str
    limit: int | None
    max_retries: int
    retry_delay: float
    skip_if_exists: bool
    prompt_mode: str


@dataclass
class ComparisonRunSpec:
    label: str
    model_config: Path | None
    judge_inference: Path | None
    dataset: Path | None
    judge_config: Path | None
    limit: int | None
    output_dir: Path | None
    metrics_path: Path | None
    details_path: Path | None
    metadata_path: Path | None
    max_retries: int | None
    retry_delay: float | None
    skip_if_exists: bool
    prompt_mode: str | None


@dataclass
class ComparisonPlan:
    defaults: PlanDefaults
    runs: list[ComparisonRunSpec]
    source_path: Path


@dataclass
class ComparisonRunResult:
    label: str
    summary: dict[str, Any]
    metrics_path: Path | None
    details_path: Path | None
    metadata_path: Path
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class ComparisonReport:
    runs: list[ComparisonRunResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": [
                {
                    "label": run.label,
                    "summary": run.summary or None,
                    "metrics_path": str(run.metrics_path) if run.metrics_path else None,
                    "details_path": str(run.details_path) if run.details_path else None,
                    "metadata_path": str(run.metadata_path),
                    "skipped": run.skipped,
                    "skip_reason": run.skip_reason,
                }
                for run in self.runs
            ],
        }


def load_comparison_plan(path: Path) -> ComparisonPlan:
    import yaml  # imported lazily to avoid mandatory dependency on load

    resolved_path = path.resolve()
    data = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Comparison plan must be a mapping")

    defaults_payload = data.get("defaults", {})
    runs_payload = data.get("runs")
    if not isinstance(runs_payload, list):
        raise TypeError("Comparison plan is missing 'runs' list")

    base_dir = resolved_path.parent

    defaults = PlanDefaults(
        dataset=_resolve_optional_path(base_dir, defaults_payload.get("dataset")),
        judge_config=_resolve_optional_path(base_dir, defaults_payload.get("judge_config")),
        judge_inference=_resolve_optional_path(base_dir, defaults_payload.get("judge_inference")),
        output_dir=_resolve_optional_path(base_dir, defaults_payload.get("output_dir")),
        metrics_filename=str(defaults_payload.get("metrics_filename", "metrics.json")),
        details_filename=str(defaults_payload.get("details_filename", "details.jsonl")),
        metadata_filename=str(defaults_payload.get("metadata_filename", "metadata.jsonl")),
        limit=_ensure_optional_int(defaults_payload.get("limit")),
        max_retries=int(defaults_payload.get("max_retries", 2)),
        retry_delay=float(defaults_payload.get("retry_delay", 15.0)),
        skip_if_exists=bool(defaults_payload.get("skip_if_exists", True)),
        prompt_mode=str(defaults_payload.get("prompt_mode", "rag")),
    )

    run_specs: list[ComparisonRunSpec] = []
    for raw in runs_payload:
        if not isinstance(raw, dict):
            raise TypeError("Each run entry must be a mapping")
        label = str(raw.get("label"))
        if not label or label == "None":
            raise ValueError("Each run entry requires a non-empty 'label'")
        prompt_mode_raw = raw.get("prompt_mode")
        prompt_mode_value: str | None = None
        if prompt_mode_raw is not None:
            prompt_mode_value = str(prompt_mode_raw)
        elif defaults.prompt_mode:
            prompt_mode_value = defaults.prompt_mode

        run_specs.append(
            ComparisonRunSpec(
                label=label,
                model_config=_resolve_optional_path(base_dir, raw.get("model_config")),
                judge_inference=_resolve_optional_path(base_dir, raw.get("judge_inference")),
                dataset=_resolve_optional_path(base_dir, raw.get("dataset")),
                judge_config=_resolve_optional_path(base_dir, raw.get("judge_config")),
                limit=_ensure_optional_int(raw.get("limit")),
                output_dir=_resolve_optional_path(base_dir, raw.get("output_dir")),
                metrics_path=_resolve_optional_path(base_dir, raw.get("metrics_path")),
                details_path=_resolve_optional_path(base_dir, raw.get("details_path")),
                metadata_path=_resolve_optional_path(base_dir, raw.get("metadata_path")),
                max_retries=_ensure_optional_int(raw.get("max_retries")),
                retry_delay=_ensure_optional_float(raw.get("retry_delay")),
                skip_if_exists=bool(raw.get("skip_if_exists", defaults.skip_if_exists)),
                prompt_mode=prompt_mode_value,
            )
        )

    return ComparisonPlan(defaults=defaults, runs=run_specs, source_path=resolved_path)


def execute_comparison_plan(
    plan: ComparisonPlan,
    *,
    project_config: ProjectConfig,
    config_path: Path,
    limit_override: int | None = None,
    max_retries_override: int | None = None,
    retry_delay_override: float | None = None,
    force: bool = False,
    validate_same_examples: bool = True,
) -> list[ComparisonRunResult]:
    results: list[ComparisonRunResult] = []
    for spec in plan.runs:
        resolved_paths = _compute_artifact_paths(plan.defaults, spec)
        metrics_path, details_path, metadata_path = resolved_paths

        if spec.skip_if_exists and not force and metrics_path.exists():
            results.append(
                ComparisonRunResult(
                    label=spec.label,
                    summary={},
                    metrics_path=metrics_path,
                    details_path=details_path if (details_path and details_path.exists()) else None,
                    metadata_path=metadata_path,
                    skipped=True,
                    skip_reason="metrics artifact already exists",
                )
            )
            continue

        dataset_source = (
            spec.dataset or plan.defaults.dataset or project_config.evaluation.offline_dataset_path
        )
        dataset_path = Path(dataset_source).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset for run '{spec.label}' not found: {dataset_path}\n"
                f"  Checked paths:\n"
                f"    - Run-specific: {spec.dataset}\n"
                f"    - Plan default: {plan.defaults.dataset}\n"
                f"    - Config default: {project_config.evaluation.offline_dataset_path}\n"
                f"  Ensure the dataset file exists or update the comparison plan configuration."
            )
        judge_prompt_path = spec.judge_config or plan.defaults.judge_config
        if judge_prompt_path is None:
            raise ValueError(
                f"Run '{spec.label}' is missing judge_config and no default was provided.\n"
                f"  Either specify judge_config in the run specification or set a default in the plan.\n"
                f"  Judge configs are typically in configs/judges/ directory."
            )
        judge_prompt_path = judge_prompt_path.resolve()
        if not judge_prompt_path.exists():
            raise FileNotFoundError(
                f"Judge config for run '{spec.label}' not found: {judge_prompt_path}\n"
                f"  Checked paths:\n"
                f"    - Run-specific: {spec.judge_config}\n"
                f"    - Plan default: {plan.defaults.judge_config}\n"
                f"  Ensure the judge config file exists or update the comparison plan configuration."
            )

        model_cfg = (
            load_inference_from_yaml(spec.model_config)
            if spec.model_config
            else project_config.inference
        )

        judge_inference_path = spec.judge_inference or plan.defaults.judge_inference
        if judge_inference_path:
            judge_inference_cfg = load_inference_from_yaml(judge_inference_path)
        else:
            judge_inference_cfg = model_cfg

        limit = (
            limit_override
            if limit_override is not None
            else spec.limit
            if spec.limit is not None
            else plan.defaults.limit
        )
        max_retries = (
            max_retries_override
            if max_retries_override is not None
            else spec.max_retries
            if spec.max_retries is not None
            else plan.defaults.max_retries
        )
        retry_delay = (
            retry_delay_override
            if retry_delay_override is not None
            else spec.retry_delay
            if spec.retry_delay is not None
            else plan.defaults.retry_delay
        )

        prompt_mode = spec.prompt_mode or plan.defaults.prompt_mode or "rag"

        result: EvaluationResult = run_evaluation(
            project_config=project_config,
            dataset_path=dataset_path,
            model_cfg=model_cfg,
            judge_prompt_path=judge_prompt_path,
            judge_inference_cfg=judge_inference_cfg,
            model_label=spec.label,
            config_path=config_path,
            model_config_path=spec.model_config,
            judge_inference_path=judge_inference_path,
            limit=limit,
            output_path=metrics_path,
            details_output_path=details_path,
            metadata_output_path=metadata_path,
            max_retries=max_retries,
            retry_delay=retry_delay,
            prompt_mode=prompt_mode,
        )

        results.append(
            ComparisonRunResult(
                label=spec.label,
                summary=result.summary,
                metrics_path=result.metrics_path,
                details_path=result.details_path,
                metadata_path=result.metadata_path,
            )
        )

    # Validate that all runs evaluated the same examples
    if validate_same_examples and len(results) > 1:
        task_id_sets: list[tuple[str, set[str]]] = []  # (label, task_ids)
        for run_result in results:
            if run_result.skipped:
                continue
            summary = run_result.summary
            evaluated_ids = summary.get("evaluated_task_ids", [])
            if evaluated_ids:
                task_id_sets.append((run_result.label, set(evaluated_ids)))

        if len(task_id_sets) > 1:
            first_label, first_set = task_id_sets[0]
            for _idx, (label, task_set) in enumerate(task_id_sets[1:], start=1):
                if task_set != first_set:
                    missing = first_set - task_set
                    extra = task_set - first_set

                    # Build detailed error message
                    error_parts = [
                        "Experiments evaluated different examples.",
                        f"  Baseline run '{first_label}' evaluated {len(first_set)} examples.",
                        f"  Run '{label}' evaluated {len(task_set)} examples.",
                    ]

                    if missing:
                        missing_preview = sorted(missing)[:10]
                        missing_msg = f"  Missing in '{label}': {missing_preview}"
                        if len(missing) > 10:
                            missing_msg += f" ... and {len(missing) - 10} more"
                        error_parts.append(missing_msg)

                    if extra:
                        extra_preview = sorted(extra)[:10]
                        extra_msg = f"  Extra in '{label}': {extra_preview}"
                        if len(extra) > 10:
                            extra_msg += f" ... and {len(extra) - 10} more"
                        error_parts.append(extra_msg)

                    error_parts.append(
                        "  This may indicate dataset filtering differences, limit overrides, or evaluation errors."
                    )
                    error_parts.append(
                        "  Check that all runs use the same dataset and limit settings."
                    )

                    raise ValueError("\n".join(error_parts))

    return results


def build_comparison_report(results: Sequence[ComparisonRunResult]) -> ComparisonReport:
    """Build comparison report with enhanced metrics separation by prompt mode."""
    # Add aggregated metrics separated by prompt mode
    rag_summaries: list[dict[str, Any]] = []
    instruction_summaries: list[dict[str, Any]] = []

    for result in results:
        if result.skipped or not result.summary:
            continue
        prompt_mode = result.summary.get("prompt_mode", "rag")
        if prompt_mode == "instruction":
            instruction_summaries.append(result.summary)
        else:
            rag_summaries.append(result.summary)

    # Note: The report structure remains the same, but we could add aggregated stats here
    # For now, the separation is visible in individual run summaries via prompt_mode field
    return ComparisonReport(runs=list(results))


def describe_plan(plan: ComparisonPlan, project_config: ProjectConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in plan.runs:
        metrics_path, details_path, metadata_path = _compute_artifact_paths(
            plan.defaults, spec, ensure_dirs=False
        )
        dataset_source = (
            spec.dataset or plan.defaults.dataset or project_config.evaluation.offline_dataset_path
        )
        judge_prompt_path = spec.judge_config or plan.defaults.judge_config
        judge_inference_path = spec.judge_inference or plan.defaults.judge_inference
        rows.append(
            {
                "label": spec.label,
                "dataset": str(dataset_source),
                "model_config": str(spec.model_config) if spec.model_config else None,
                "judge_config": str(judge_prompt_path) if judge_prompt_path else None,
                "judge_inference": str(judge_inference_path) if judge_inference_path else None,
                "limit": spec.limit if spec.limit is not None else plan.defaults.limit,
                "metrics_path": str(metrics_path),
                "details_path": str(details_path) if details_path else None,
                "metadata_path": str(metadata_path),
                "skip_if_exists": spec.skip_if_exists,
                "prompt_mode": spec.prompt_mode or plan.defaults.prompt_mode or "rag",
            }
        )
    return rows


def write_report(report: ComparisonReport, path: Path) -> None:
    payload = report.as_dict()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_optional_path(base: Path, raw: Any) -> Path | None:
    if raw in (None, "", False):
        return None
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _ensure_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _ensure_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _compute_artifact_paths(
    defaults: PlanDefaults,
    spec: ComparisonRunSpec,
    *,
    ensure_dirs: bool = True,
) -> tuple[Path, Path | None, Path]:
    output_dir = spec.output_dir or defaults.output_dir
    if output_dir is None:
        output_dir = Path("evaluation/results") / spec.label
    else:
        output_dir = output_dir / spec.label
    if ensure_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = spec.metrics_path or output_dir / defaults.metrics_filename
    details_path = spec.details_path or output_dir / defaults.details_filename
    metadata_path = spec.metadata_path or output_dir / defaults.metadata_filename

    return metrics_path, details_path, metadata_path
