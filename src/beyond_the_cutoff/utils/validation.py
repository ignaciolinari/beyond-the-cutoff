"""Validation utilities for ensuring experiment reproducibility and correctness."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from beyond_the_cutoff.config import InferenceConfig, load_config
from beyond_the_cutoff.utils.experiment_logging import compute_file_sha256


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    severity: str  # "error", "warning", "info"
    message: str
    context: dict[str, Any] | None = None


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    issues: list[ValidationIssue]
    summary: str


def validate_dataset_versioning(
    dataset_paths: Sequence[Path], *, expected_hash: str | None = None
) -> ValidationResult:
    """Validate that datasets match expected version or are consistent across runs.

    Args:
        dataset_paths: List of dataset paths to check
        expected_hash: Optional expected SHA256 hash to validate against

    Returns:
        ValidationResult with issues if datasets don't match
    """
    issues: list[ValidationIssue] = []

    if not dataset_paths:
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message="No dataset paths provided for validation",
                )
            ],
            summary="No datasets to validate",
        )

    # Compute hashes for all datasets
    hashes: dict[str, str] = {}
    for path in dataset_paths:
        if not path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Dataset file not found: {path}",
                    context={"path": str(path)},
                )
            )
            continue
        try:
            file_hash = compute_file_sha256(path)
            hashes[str(path)] = file_hash
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Failed to compute hash for {path}: {exc}",
                    context={"path": str(path), "error": str(exc)},
                )
            )

    # Check if all datasets have the same hash
    if len(hashes) > 1:
        unique_hashes = set(hashes.values())
        if len(unique_hashes) > 1:
            hash_groups: dict[str, list[str]] = {}
            for path_str, file_hash in hashes.items():
                hash_groups.setdefault(file_hash, []).append(path_str)

            issue_paths = []
            for file_hash, paths in hash_groups.items():
                issue_paths.append(f"  Hash {file_hash[:16]}...: {', '.join(paths)}")

            issues.append(
                ValidationIssue(
                    severity="error",
                    message="Datasets have different hashes - they may be different versions",
                    context={
                        "unique_hashes": len(unique_hashes),
                        "hash_groups": hash_groups,
                    },
                )
            )

    # Check against expected hash if provided
    if expected_hash and hashes:
        first_hash = next(iter(hashes.values()))
        if first_hash != expected_hash:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Dataset hash ({first_hash[:16]}...) does not match expected hash ({expected_hash[:16]}...)",
                    context={
                        "computed_hash": first_hash,
                        "expected_hash": expected_hash,
                    },
                )
            )

    passed = len([i for i in issues if i.severity == "error"]) == 0
    summary = (
        f"Dataset versioning validation: {'PASSED' if passed else 'FAILED'}"
        f" ({len(hashes)} dataset(s) checked)"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def validate_configuration(
    config_path: Path,
    *,
    model_config_path: Path | None = None,
    judge_config_path: Path | None = None,
    judge_inference_path: Path | None = None,
    prompt_mode: str | None = None,
) -> ValidationResult:
    """Validate configuration files and check for conflicts.

    Args:
        config_path: Path to project config file
        model_config_path: Optional path to model inference config
        judge_config_path: Optional path to judge prompt config
        judge_inference_path: Optional path to judge inference config
        prompt_mode: Optional prompt mode ("rag" or "instruction")

    Returns:
        ValidationResult with issues if configs are invalid or conflict
    """
    issues: list[ValidationIssue] = []

    # Validate project config exists and is loadable
    if not config_path.exists():
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Project config not found: {config_path}",
                    context={"path": str(config_path)},
                )
            ],
            summary="Configuration validation failed: project config missing",
        )

    try:
        load_config(config_path)
    except Exception as exc:
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Failed to load project config: {exc}",
                    context={"path": str(config_path), "error": str(exc)},
                )
            ],
            summary="Configuration validation failed: project config invalid",
        )

    # Validate model config if provided
    if model_config_path:
        if not model_config_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Model config not found: {model_config_path}",
                    context={"path": str(model_config_path)},
                )
            )
        else:
            try:
                InferenceConfig.model_validate(
                    yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Model config is invalid: {exc}",
                        context={"path": str(model_config_path), "error": str(exc)},
                    )
                )

    # Validate judge config if provided
    if judge_config_path:
        if not judge_config_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Judge config not found: {judge_config_path}",
                    context={"path": str(judge_config_path)},
                )
            )
        else:
            # Check if judge config name matches prompt mode
            try:
                judge_data = yaml.safe_load(judge_config_path.read_text(encoding="utf-8"))
                judge_name = str(judge_data.get("name", "")).lower()
                is_instruction_judge = "instruction" in judge_name and "rag" not in judge_name
                is_rag_judge = "rag" in judge_name and "instruction" not in judge_name

                if prompt_mode:
                    if prompt_mode == "instruction" and is_rag_judge:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                message=(
                                    f"Prompt mode is 'instruction' but judge config '{judge_name}' "
                                    "appears to be for RAG evaluation. Consider using an instruction-only judge config."
                                ),
                                context={
                                    "prompt_mode": prompt_mode,
                                    "judge_name": judge_name,
                                },
                            )
                        )
                    elif prompt_mode == "rag" and is_instruction_judge:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                message=(
                                    f"Prompt mode is 'rag' but judge config '{judge_name}' "
                                    "appears to be for instruction-only evaluation. Consider using a RAG judge config."
                                ),
                                context={
                                    "prompt_mode": prompt_mode,
                                    "judge_name": judge_name,
                                },
                            )
                        )
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Could not validate judge config compatibility: {exc}",
                        context={"path": str(judge_config_path), "error": str(exc)},
                    )
                )

    # Validate judge inference config if provided
    if judge_inference_path:
        if not judge_inference_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Judge inference config not found: {judge_inference_path}",
                    context={"path": str(judge_inference_path)},
                )
            )
        else:
            try:
                InferenceConfig.model_validate(
                    yaml.safe_load(judge_inference_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Judge inference config is invalid: {exc}",
                        context={"path": str(judge_inference_path), "error": str(exc)},
                    )
                )

    # Validate prompt mode if provided
    if prompt_mode and prompt_mode not in ("rag", "instruction"):
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Invalid prompt_mode: {prompt_mode!r}. Must be 'rag' or 'instruction'",
                context={"prompt_mode": prompt_mode},
            )
        )

    passed = len([i for i in issues if i.severity == "error"]) == 0
    warning_count = len([i for i in issues if i.severity == "warning"])
    summary = (
        f"Configuration validation: {'PASSED' if passed else 'FAILED'}"
        f" ({warning_count} warning(s))"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def validate_experiment_reproducibility(
    metadata_path: Path, *, required_fields: Sequence[str] | None = None
) -> ValidationResult:
    """Validate that experiment metadata includes all required fields for reproducibility.

    Args:
        metadata_path: Path to metadata.jsonl file
        required_fields: Optional list of required top-level fields (defaults to common fields)

    Returns:
        ValidationResult with issues if metadata is incomplete
    """
    issues: list[ValidationIssue] = []

    if required_fields is None:
        required_fields = [
            "timestamp",
            "model_label",
            "dataset",
            "project_config",
            "inference",
            "judge",
        ]

    if not metadata_path.exists():
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Metadata file not found: {metadata_path}",
                    context={"path": str(metadata_path)},
                )
            ],
            summary="Reproducibility validation failed: metadata file missing",
        )

    # Read all records
    records: list[dict[str, Any]] = []
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as exc:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"Invalid JSON in metadata file at line {line_num}: {exc}",
                            context={"line": line_num, "error": str(exc)},
                        )
                    )
    except Exception as exc:
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Failed to read metadata file: {exc}",
                    context={"path": str(metadata_path), "error": str(exc)},
                )
            ],
            summary="Reproducibility validation failed: cannot read metadata",
        )

    if not records:
        issues.append(
            ValidationIssue(
                severity="warning",
                message="Metadata file is empty - no experiment records found",
                context={"path": str(metadata_path)},
            )
        )

    # Validate each record
    for idx, record in enumerate(records, start=1):
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Record {idx} missing required fields: {', '.join(missing_fields)}",
                    context={
                        "record_index": idx,
                        "missing_fields": missing_fields,
                        "available_fields": list(record.keys()),
                    },
                )
            )

        # Validate nested structure for dataset
        if "dataset" in record:
            dataset_info = record["dataset"]
            if not isinstance(dataset_info, dict):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Record {idx} has invalid dataset field (not a dict)",
                        context={"record_index": idx},
                    )
                )
            else:
                if "sha256" not in dataset_info:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            message=f"Record {idx} dataset missing SHA256 hash",
                            context={"record_index": idx},
                        )
                    )
                if "path" not in dataset_info:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"Record {idx} dataset missing path",
                            context={"record_index": idx},
                        )
                    )

        # Validate nested structure for configs
        for config_type in ["project_config", "inference", "judge"]:
            if config_type in record:
                config_info = record[config_type]
                if not isinstance(config_info, dict):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"Record {idx} has invalid {config_type} field (not a dict)",
                            context={"record_index": idx, "config_type": config_type},
                        )
                    )

    passed = len([i for i in issues if i.severity == "error"]) == 0
    summary = (
        f"Reproducibility validation: {'PASSED' if passed else 'FAILED'}"
        f" ({len(records)} record(s) checked)"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def validate_evaluation_sanity(
    metrics_path: Path | None = None,
    details_path: Path | None = None,
    *,
    max_error_rate: float = 0.1,
    min_examples: int = 1,
) -> ValidationResult:
    """Validate evaluation results for common issues.

    Args:
        metrics_path: Optional path to metrics JSON file
        details_path: Optional path to details JSONL file
        max_error_rate: Maximum acceptable error rate (default: 0.1 = 10%)
        min_examples: Minimum number of examples required (default: 1)

    Returns:
        ValidationResult with issues if evaluation has problems
    """
    issues: list[ValidationIssue] = []

    # Validate metrics file if provided
    metrics_data: dict[str, Any] | None = None
    if metrics_path:
        if not metrics_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Metrics file not found: {metrics_path}",
                    context={"path": str(metrics_path)},
                )
            )
        else:
            try:
                metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not isinstance(metrics_data, dict):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message="Metrics file does not contain a JSON object",
                            context={"path": str(metrics_path)},
                        )
                    )
                    metrics_data = None
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Failed to parse metrics file: {exc}",
                        context={"path": str(metrics_path), "error": str(exc)},
                    )
                )

    # Validate details file if provided
    details_records: list[dict[str, Any]] = []
    if details_path:
        if not details_path.exists():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Details file not found: {details_path}",
                    context={"path": str(details_path)},
                )
            )
        else:
            try:
                with details_path.open("r", encoding="utf-8") as handle:
                    for line_num, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            details_records.append(record)
                        except json.JSONDecodeError as exc:
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    message=f"Invalid JSON in details file at line {line_num}: {exc}",
                                    context={"line": line_num, "error": str(exc)},
                                )
                            )
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Failed to read details file: {exc}",
                        context={"path": str(details_path), "error": str(exc)},
                    )
                )

    # Check example count
    if metrics_data:
        examples_evaluated = metrics_data.get("examples_evaluated", 0)
        if examples_evaluated < min_examples:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Too few examples evaluated: {examples_evaluated} < {min_examples}",
                    context={
                        "examples_evaluated": examples_evaluated,
                        "min_examples": min_examples,
                    },
                )
            )

    # Check error rate
    if metrics_data:
        error_rate = metrics_data.get("error_rate", 0.0)
        if error_rate > max_error_rate:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"High error rate detected: {error_rate:.1%} > {max_error_rate:.1%}. "
                        "Consider reviewing error logs."
                    ),
                    context={"error_rate": error_rate, "max_error_rate": max_error_rate},
                )
            )

    # Check for empty responses
    if details_records:
        empty_responses = [
            r
            for r in details_records
            if not str(r.get("model_answer", "")).strip()
            and (not r.get("errors") or "generation" not in r.get("errors", {}))
        ]
        if empty_responses:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Found {len(empty_responses)} example(s) with empty responses",
                    context={"empty_count": len(empty_responses)},
                )
            )

    # Check for missing predictions
    if details_records:
        missing_predictions = [
            r for r in details_records if "model_answer" not in r or not r.get("model_answer")
        ]
        if missing_predictions:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Found {len(missing_predictions)} example(s) with missing predictions",
                    context={"missing_count": len(missing_predictions)},
                )
            )

    # Check citation metrics consistency
    if details_records:
        rag_count = 0
        instruction_count = 0
        mixed_count = 0
        for record in details_records:
            citation_metrics = record.get("citation_metrics", {})
            mode = citation_metrics.get("mode")
            if mode == "rag":
                rag_count += 1
            elif mode == "instruction_only":
                instruction_count += 1
            elif citation_metrics:
                mixed_count += 1

        if mixed_count > 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Found {mixed_count} example(s) with citation metrics missing mode specification",
                    context={"mixed_count": mixed_count},
                )
            )

    passed = len([i for i in issues if i.severity == "error"]) == 0
    summary = (
        f"Evaluation sanity check: {'PASSED' if passed else 'FAILED'}"
        f" ({len(details_records)} detail record(s) checked)"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def print_validation_result(result: ValidationResult, *, file: Any = sys.stderr) -> None:
    """Print validation result in a human-readable format."""
    print(f"[{'✓' if result.passed else '✗'}] {result.summary}", file=file)
    if result.issues:
        for issue in result.issues:
            icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "•")
            print(f"  {icon} [{issue.severity.upper()}] {issue.message}", file=file)
            if issue.context:
                for key, value in issue.context.items():
                    if isinstance(value, str | int | float | bool | type(None)):
                        print(f"      {key}: {value}", file=file)
                    elif isinstance(value, list | dict):
                        print(f"      {key}: {json.dumps(value, indent=8)[:200]}...", file=file)


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "validate_dataset_versioning",
    "validate_configuration",
    "validate_experiment_reproducibility",
    "validate_evaluation_sanity",
    "print_validation_result",
]
