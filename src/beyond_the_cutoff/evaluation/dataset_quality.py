"""Dataset quality validation and analysis utilities."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from beyond_the_cutoff.utils.validation import ValidationIssue, ValidationResult


@dataclass
class DatasetQualityReport:
    """Report on dataset quality metrics."""

    total_examples: int
    task_type_distribution: dict[str, int]
    citation_coverage_stats: dict[str, float]
    required_fields_present: bool
    issues: list[ValidationIssue]


def validate_required_fields(dataset_path: Path) -> ValidationResult:
    """Validate that dataset has all required fields.

    Args:
        dataset_path: Path to offline dataset JSONL file

    Returns:
        ValidationResult with issues for missing required fields
    """
    issues: list[ValidationIssue] = []
    required_fields = {"task_id", "instruction", "rag"}

    if not dataset_path.exists():
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Dataset file not found: {dataset_path}",
                )
            ],
            summary="Dataset file not found",
        )

    example_count = 0
    missing_field_counts: dict[str, int] = defaultdict(int)

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                    example_count += 1
                    task_id = example.get("task_id", f"line_{idx}")

                    # Check required fields
                    for field in required_fields:
                        if field not in example:
                            missing_field_counts[field] += 1
                            if missing_field_counts[field] <= 5:  # Limit to first 5 examples
                                issues.append(
                                    ValidationIssue(
                                        severity="error",
                                        message=f"Example {task_id} missing required field: {field}",
                                        context={"task_id": task_id, "field": field, "line": idx},
                                    )
                                )

                    # Check nested rag field structure
                    if "rag" in example:
                        rag = example["rag"]
                        if not isinstance(rag, dict):
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    message=f"Example {task_id} has invalid 'rag' field (must be dict)",
                                    context={"task_id": task_id, "line": idx},
                                )
                            )

                except json.JSONDecodeError as exc:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f"Invalid JSON on line {idx}: {exc}",
                            context={"line": idx, "error": str(exc)},
                        )
                    )

    except Exception as exc:
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message=f"Failed to read dataset file: {exc}",
                    context={"path": str(dataset_path), "error": str(exc)},
                )
            ],
            summary=f"Failed to read dataset: {exc}",
        )

    # Add summary issues for fields missing in many examples
    for field, count in missing_field_counts.items():
        if count > 5:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Field '{field}' missing in {count} example(s)",
                    context={"field": field, "missing_count": count},
                )
            )

    passed = len(missing_field_counts) == 0
    summary = (
        f"All {example_count} examples have required fields"
        if passed
        else f"Found {len(missing_field_counts)} missing field(s) across {example_count} examples"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def analyze_citation_coverage(dataset_path: Path) -> dict[str, float]:
    """Analyze citation coverage statistics in the dataset.

    Args:
        dataset_path: Path to offline dataset JSONL file

    Returns:
        Dictionary with citation coverage statistics
    """
    stats: dict[str, list[float]] = defaultdict(list)

    if not dataset_path.exists():
        return {}

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                    rag = example.get("rag", {})
                    contexts = rag.get("contexts") or example.get("contexts", [])

                    if not contexts:
                        continue

                    # Check reference answer for citations
                    reference_answer = example.get("reference_answer") or example.get(
                        "fine_tune_output", ""
                    )
                    if not reference_answer:
                        continue

                    # Count citation markers in reference answer
                    citation_markers = [
                        f"[{i+1}]" for i in range(len(contexts))
                    ]  # Expected citations
                    found_citations = sum(
                        1 for marker in citation_markers if marker in reference_answer
                    )
                    coverage = found_citations / len(contexts) if contexts else 0.0

                    stats["coverage"].append(coverage)
                    stats["num_contexts"].append(float(len(contexts)))
                    stats["found_citations"].append(float(found_citations))

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    except Exception:
        return {}

    # Compute summary statistics
    result: dict[str, float] = {}
    if stats["coverage"]:
        result["mean_coverage"] = sum(stats["coverage"]) / len(stats["coverage"])
        result["min_coverage"] = min(stats["coverage"])
        result["max_coverage"] = max(stats["coverage"])
        result["examples_with_citations"] = sum(1 for c in stats["coverage"] if c > 0)
        result["total_examples_analyzed"] = float(len(stats["coverage"]))
        result["mean_contexts_per_example"] = (
            sum(stats["num_contexts"]) / len(stats["num_contexts"])
            if stats["num_contexts"]
            else 0.0
        )

    return result


def analyze_task_type_distribution(dataset_path: Path) -> dict[str, int]:
    """Analyze distribution of task types in the dataset.

    Args:
        dataset_path: Path to offline dataset JSONL file

    Returns:
        Dictionary mapping task_type to count
    """
    task_types: list[str] = []

    if not dataset_path.exists():
        return {}

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                    task_type = example.get("task_type", "unknown")
                    task_types.append(str(task_type))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    except Exception:
        return {}

    return dict(Counter(task_types))


def validate_task_type_distribution(
    dataset_path: Path,
    *,
    min_examples_per_type: int = 5,
    expected_types: set[str] | None = None,
) -> ValidationResult:
    """Validate task type distribution meets expectations.

    Args:
        dataset_path: Path to offline dataset JSONL file
        min_examples_per_type: Minimum examples required per task type
        expected_types: Optional set of expected task types

    Returns:
        ValidationResult with distribution issues
    """
    issues: list[ValidationIssue] = []
    distribution = analyze_task_type_distribution(dataset_path)

    if not distribution:
        return ValidationResult(
            passed=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    message="Could not analyze task type distribution",
                )
            ],
            summary="Failed to analyze task types",
        )

    # Check minimum examples per type
    for task_type, count in distribution.items():
        if count < min_examples_per_type:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Task type '{task_type}' has only {count} example(s), "
                    f"below minimum of {min_examples_per_type}",
                    context={
                        "task_type": task_type,
                        "count": count,
                        "minimum": min_examples_per_type,
                    },
                )
            )

    # Check expected types
    if expected_types:
        found_types = set(distribution.keys())
        missing_types = expected_types - found_types
        extra_types = found_types - expected_types

        for missing_type in missing_types:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Expected task type '{missing_type}' not found in dataset",
                    context={"expected_type": missing_type},
                )
            )

        if extra_types:
            issues.append(
                ValidationIssue(
                    severity="info",
                    message=f"Found unexpected task type(s): {', '.join(sorted(extra_types))}",
                    context={"extra_types": list(extra_types)},
                )
            )

    passed = len([i for i in issues if i.severity == "error"]) == 0
    summary = (
        f"Found {len(distribution)} task type(s) with {sum(distribution.values())} total examples"
    )

    return ValidationResult(passed=passed, issues=issues, summary=summary)


def generate_quality_report(dataset_path: Path) -> DatasetQualityReport:
    """Generate comprehensive quality report for a dataset.

    Args:
        dataset_path: Path to offline dataset JSONL file

    Returns:
        DatasetQualityReport with quality metrics and issues
    """
    issues: list[ValidationIssue] = []

    # Validate required fields
    field_validation = validate_required_fields(dataset_path)
    issues.extend(field_validation.issues)
    required_fields_present = field_validation.passed

    # Analyze citation coverage
    citation_stats = analyze_citation_coverage(dataset_path)
    if citation_stats:
        mean_coverage = citation_stats.get("mean_coverage", 0.0)
        if mean_coverage < 0.2:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Low citation coverage: {mean_coverage:.1%}",
                    context={"mean_coverage": mean_coverage},
                )
            )

    # Analyze task type distribution
    task_distribution = analyze_task_type_distribution(dataset_path)
    if not task_distribution:
        issues.append(
            ValidationIssue(
                severity="warning",
                message="Could not analyze task type distribution",
            )
        )

    # Count total examples
    total_examples = 0
    if dataset_path.exists():
        try:
            with dataset_path.open("r", encoding="utf-8") as handle:
                total_examples = sum(1 for line in handle if line.strip())
        except Exception:
            pass

    return DatasetQualityReport(
        total_examples=total_examples,
        task_type_distribution=task_distribution,
        citation_coverage_stats=citation_stats,
        required_fields_present=required_fields_present,
        issues=issues,
    )


__all__ = [
    "validate_required_fields",
    "analyze_citation_coverage",
    "analyze_task_type_distribution",
    "validate_task_type_distribution",
    "generate_quality_report",
    "DatasetQualityReport",
]
