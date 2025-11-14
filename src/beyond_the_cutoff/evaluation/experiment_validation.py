"""Validation utilities for experiment completeness and consistency."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from beyond_the_cutoff.utils.validation import ValidationIssue, ValidationResult

# Expected labels for the 6-condition experiment
EXPECTED_CONDITIONS = {
    "base_baseline_0p5b",
    "rag_baseline_0p5b",
    "lora_science_0p5b_ft_only",
    "hybrid_science_0p5b_instruction_only",
    "lora_science_0p5b_rag_trained_ft_only",
    "hybrid_science_0p5b_rag_trained",
}


@dataclass
class ExperimentRunInfo:
    """Information about a single experiment run."""

    label: str
    metadata_path: Path | None = None
    metrics_path: Path | None = None
    details_path: Path | None = None
    dataset_hash: str | None = None


def validate_experiment_completeness(
    results_dir: Path, *, expected_labels: set[str] | None = None
) -> ValidationResult:
    """Validate that all expected experimental conditions have been evaluated.

    Args:
        results_dir: Directory containing evaluation results
        expected_labels: Set of expected condition labels. If None, uses default 6-condition set.

    Returns:
        ValidationResult indicating which conditions are missing
    """
    if expected_labels is None:
        expected_labels = EXPECTED_CONDITIONS

    issues: list[ValidationIssue] = []
    found_labels: set[str] = set()

    if not results_dir.exists():
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Results directory does not exist: {results_dir}",
                )
            ],
            summary="Results directory not found",
        )

    # Scan for metadata files
    for label in expected_labels:
        label_dir = results_dir / label
        metadata_path = label_dir / "metadata.jsonl"
        metrics_path = label_dir / "metrics.json"

        if not label_dir.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Missing condition directory: {label}",
                    context={"expected_path": str(label_dir)},
                )
            )
            continue

        if not metadata_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Missing metadata file for condition: {label}",
                    context={"expected_path": str(metadata_path)},
                )
            )
            continue

        if not metrics_path.exists():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Missing metrics file for condition: {label}",
                    context={"expected_path": str(metrics_path)},
                )
            )
        else:
            found_labels.add(label)

    missing_labels = expected_labels - found_labels
    if missing_labels:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Missing {len(missing_labels)} condition(s): {', '.join(sorted(missing_labels))}",
                context={"missing_labels": list(missing_labels)},
            )
        )

    passed = len(found_labels) == len(expected_labels)
    summary = (
        f"Found {len(found_labels)}/{len(expected_labels)} conditions"
        if passed
        else f"Missing {len(missing_labels)} condition(s)"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def validate_dataset_consistency(
    metadata_paths: Sequence[Path], *, expected_hash: str | None = None
) -> ValidationResult:
    """Validate that all experiment runs used the same dataset version.

    Args:
        metadata_paths: List of metadata.jsonl files from different runs
        expected_hash: Optional expected dataset hash to validate against

    Returns:
        ValidationResult indicating dataset version consistency
    """
    issues: list[ValidationIssue] = []
    dataset_hashes: dict[str, list[str]] = {}  # hash -> list of run labels

    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Metadata file not found: {metadata_path}",
                    context={"path": str(metadata_path)},
                )
            )
            continue

        try:
            # Read last line (most recent run) from metadata.jsonl
            with metadata_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
                if not lines:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"Metadata file is empty: {metadata_path}",
                            context={"path": str(metadata_path)},
                        )
                    )
                    continue

                last_record = json.loads(lines[-1])
                dataset_info = last_record.get("dataset", {})
                dataset_hash = dataset_info.get("sha256")

                if not dataset_hash:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            message=f"Metadata file missing dataset hash: {metadata_path}",
                            context={"path": str(metadata_path)},
                        )
                    )
                    continue

                run_label = last_record.get("model_label", metadata_path.parent.name)
                dataset_hashes.setdefault(dataset_hash, []).append(run_label)

        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Failed to read metadata file {metadata_path}: {exc}",
                    context={"path": str(metadata_path), "error": str(exc)},
                )
            )

    # Check consistency
    if len(dataset_hashes) > 1:
        hash_summary = []
        for dataset_hash, run_labels in dataset_hashes.items():
            hash_summary.append(f"  Hash {dataset_hash[:16]}...: {', '.join(run_labels)}")

        issues.append(
            ValidationIssue(
                severity="error",
                message="Experiments used different dataset versions",
                context={
                    "hash_groups": hash_summary,
                    "unique_hashes": len(dataset_hashes),
                },
            )
        )

    # Check against expected hash if provided
    if expected_hash and dataset_hashes:
        found_hashes = set(dataset_hashes.keys())
        if expected_hash not in found_hashes:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Dataset hash mismatch: expected {expected_hash[:16]}..., "
                    f"found {len(found_hashes)} different hash(es)",
                    context={"expected_hash": expected_hash, "found_hashes": list(found_hashes)},
                )
            )

    passed = len(dataset_hashes) <= 1 and (expected_hash is None or expected_hash in dataset_hashes)
    summary = (
        "All experiments used the same dataset version"
        if passed
        else f"Found {len(dataset_hashes)} different dataset version(s)"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def validate_experiment_results(
    results_dir: Path,
    *,
    expected_labels: set[str] | None = None,
    expected_dataset_hash: str | None = None,
) -> ValidationResult:
    """Comprehensive validation of experiment results.

    Checks:
    1. All expected conditions are present
    2. All runs used the same dataset version
    3. Metadata files are present and valid

    Args:
        results_dir: Directory containing evaluation results
        expected_labels: Set of expected condition labels
        expected_dataset_hash: Optional expected dataset hash

    Returns:
        Combined ValidationResult
    """
    issues: list[ValidationIssue] = []
    all_passed = True

    # Check completeness
    completeness_result = validate_experiment_completeness(
        results_dir, expected_labels=expected_labels
    )
    issues.extend(completeness_result.issues)
    if not completeness_result.passed:
        all_passed = False

    # Check dataset consistency
    if expected_labels is None:
        expected_labels = EXPECTED_CONDITIONS

    metadata_paths: list[Path] = []
    for label in expected_labels:
        metadata_path = results_dir / label / "metadata.jsonl"
        if metadata_path.exists():
            metadata_paths.append(metadata_path)

    if metadata_paths:
        consistency_result = validate_dataset_consistency(
            metadata_paths, expected_hash=expected_dataset_hash
        )
        issues.extend(consistency_result.issues)
        if not consistency_result.passed:
            all_passed = False

    summary_parts = [
        completeness_result.summary,
    ]
    if metadata_paths:
        summary_parts.append(consistency_result.summary)

    return ValidationResult(
        passed=all_passed,
        issues=issues,
        summary=" | ".join(summary_parts),
    )


__all__ = [
    "validate_experiment_completeness",
    "validate_dataset_consistency",
    "validate_experiment_results",
    "EXPECTED_CONDITIONS",
]
