"""Utilities for recording experiment metadata and run summaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import InferenceConfig, ProjectConfig


def compute_file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hash for *path* (hex digest)."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _serialise_inference_config(config: InferenceConfig) -> dict[str, Any]:
    payload = config.model_dump()
    # Avoid leaking credential-like data if present
    payload.pop("host", None)
    payload.pop("port", None)
    return payload


def _serialise_project_config(config: ProjectConfig) -> dict[str, Any]:
    return {
        "project": config.project.model_dump(mode="json"),
        "paths": config.paths.model_dump(mode="json"),
        "retrieval": config.retrieval.model_dump(mode="json"),
        "fine_tuning": config.fine_tuning.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "dataset_generation": config.dataset_generation.model_dump(mode="json"),
        "inference": _serialise_inference_config(config.inference),
    }


def _extract_dataset_metadata(dataset_path: Path) -> dict[str, Any]:
    """Extract metadata from dataset file if available.

    Looks for metadata in first line or dataset header.
    Returns enhanced metadata dict.
    """
    metadata: dict[str, Any] = {}

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
            if first_line.strip():
                try:
                    first_record = json.loads(first_line)
                    # Check for common metadata fields
                    if "dataset_version" in first_record:
                        metadata["version"] = first_record["dataset_version"]
                    if "generated_at" in first_record:
                        metadata["generated_at"] = first_record["generated_at"]
                    if "generator_model" in first_record:
                        metadata["generator_model"] = first_record["generator_model"]
                    if "source_offline_dataset" in first_record:
                        metadata["source_dataset"] = first_record["source_offline_dataset"]
                except json.JSONDecodeError:
                    pass

        # Get file modification time as fallback generation timestamp
        if "generated_at" not in metadata:
            mtime = dataset_path.stat().st_mtime
            metadata["file_mtime"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        # If we can't read metadata, continue without it
        pass

    return metadata


def append_experiment_record(
    metadata_path: Path,
    *,
    project_config: ProjectConfig,
    dataset_path: Path,
    model_config: InferenceConfig,
    judge_config: InferenceConfig,
    metrics: Mapping[str, Any],
    score_rows: Sequence[Mapping[str, Any]],
    model_label: str,
    config_path: Path,
    model_config_path: Path | None,
    judge_prompt_path: Path,
    judge_inference_path: Path | None,
    details_path: Path | None = None,
    metrics_path: Path | None = None,
) -> None:
    """Append a JSON record summarising the evaluation run."""

    # Extract enhanced dataset metadata
    dataset_metadata = _extract_dataset_metadata(dataset_path)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_label": model_label,
        "metrics": dict(metrics),
        "dataset": {
            "path": str(dataset_path),
            "sha256": compute_file_sha256(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "example_count": len(score_rows),
            **dataset_metadata,  # Include extracted metadata (version, generated_at, etc.)
        },
        "project_config": {
            "path": str(config_path),
            "sha256": compute_file_sha256(config_path) if config_path.exists() else None,
            "snapshot": _serialise_project_config(project_config),
        },
        "inference": {
            "config_path": str(model_config_path) if model_config_path else None,
            "config_sha256": compute_file_sha256(model_config_path)
            if model_config_path and model_config_path.exists()
            else None,
            "parameters": _serialise_inference_config(model_config),
        },
        "judge": {
            "prompt_path": str(judge_prompt_path),
            "prompt_sha256": compute_file_sha256(judge_prompt_path),
            "inference_path": str(judge_inference_path) if judge_inference_path else None,
            "inference_sha256": compute_file_sha256(judge_inference_path)
            if judge_inference_path and judge_inference_path.exists()
            else None,
            "parameters": _serialise_inference_config(judge_config),
        },
    }

    if metrics_path is not None:
        record["metrics_artifact"] = str(metrics_path)
    if details_path is not None:
        record["details_artifact"] = str(details_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = [
    "append_experiment_record",
    "compute_file_sha256",
]
