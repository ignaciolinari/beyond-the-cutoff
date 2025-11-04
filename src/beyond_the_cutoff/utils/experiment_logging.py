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
        "project": config.project.model_dump(),
        "paths": config.paths.model_dump(),
        "retrieval": config.retrieval.model_dump(),
        "fine_tuning": config.fine_tuning.model_dump(),
        "evaluation": config.evaluation.model_dump(),
        "dataset_generation": config.dataset_generation.model_dump(),
        "inference": _serialise_inference_config(config.inference),
    }


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

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_label": model_label,
        "metrics": dict(metrics),
        "dataset": {
            "path": str(dataset_path),
            "sha256": compute_file_sha256(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "example_count": len(score_rows),
        },
        "project_config": {
            "path": str(config_path),
            "snapshot": _serialise_project_config(project_config),
        },
        "inference": {
            "config_path": str(model_config_path) if model_config_path else None,
            "parameters": _serialise_inference_config(model_config),
        },
        "judge": {
            "prompt_path": str(judge_prompt_path),
            "inference_path": str(judge_inference_path) if judge_inference_path else None,
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
