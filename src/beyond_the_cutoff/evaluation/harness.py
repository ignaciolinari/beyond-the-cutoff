"""Shared helpers for computing the consolidated evaluation harness metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import ProjectConfig
from beyond_the_cutoff.evaluation.retrieval_metrics import evaluate_retrieval
from beyond_the_cutoff.evaluation.scoring import score_predictions
from beyond_the_cutoff.retrieval.query import RAGPipeline


def compute_harness(
    *,
    dataset_records: Sequence[Mapping[str, Any]],
    project_config: ProjectConfig,
    predictions: Mapping[str, str],
    bert_lang: str = "en",
    topk_values: Sequence[int] = (1, 3, 5, 10),
    index_override: Path | None = None,
    mapping_override: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return consolidated summary + per-example rows for the harness."""

    generation_summary, generation_examples = score_predictions(
        dataset_records,
        predictions,
        bert_lang=bert_lang,
    )

    index_path, mapping_path = _resolve_index_artifacts(
        project_config,
        index_override=index_override,
        mapping_override=mapping_override,
    )
    pipeline = RAGPipeline(project_config, index_path=index_path, mapping_path=mapping_path)

    retrieval_summary, retrieval_examples = evaluate_retrieval(
        dataset_records,
        pipeline,
        topk_values=topk_values,
    )

    summary = {
        "generation": generation_summary,
        "retrieval": retrieval_summary,
    }

    merged_details = _merge_example_metrics(generation_examples, retrieval_examples)
    return summary, merged_details


def _merge_example_metrics(
    generation_examples: Sequence[Mapping[str, Any]],
    retrieval_examples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    retrieval_by_id = {
        row.get("task_id"): dict(row) for row in retrieval_examples if isinstance(row, Mapping)
    }

    merged: list[dict[str, Any]] = []
    for gen_row in generation_examples:
        task_id = gen_row.get("task_id")
        merged.append(
            {
                "task_id": task_id,
                "generation": dict(gen_row),
                "retrieval": retrieval_by_id.pop(task_id, None),
            }
        )

    for task_id, retrieval_row in retrieval_by_id.items():
        merged.append(
            {
                "task_id": task_id,
                "generation": None,
                "retrieval": retrieval_row,
            }
        )

    return merged


def _resolve_index_artifacts(
    project_config: ProjectConfig,
    *,
    index_override: Path | None,
    mapping_override: Path | None,
) -> tuple[Path, Path]:
    if index_override is not None:
        candidate = index_override.resolve()
    else:
        candidate = project_config.paths.external_data / "index"
    if candidate.is_dir():
        index_path = candidate / "index.faiss"
    else:
        index_path = candidate
    index_path = index_path.resolve()

    if mapping_override is not None:
        mapping_path = mapping_override.resolve()
    else:
        if index_path.suffix == ".faiss":
            mapping_path = index_path.with_name("mapping.tsv")
        else:
            mapping_path = index_path / "mapping.tsv"
    mapping_path = mapping_path.resolve()

    if not index_path.exists():  # pragma: no cover - defensive path
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not mapping_path.exists():  # pragma: no cover - defensive path
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    return index_path, mapping_path


__all__ = ["compute_harness"]
