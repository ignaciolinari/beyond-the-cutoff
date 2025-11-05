"""Helpers for evaluating retrieval performance (Hit@k, MRR)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalExampleMetrics:
    """Per-example retrieval diagnostics."""

    task_id: str
    relevant_chunk_ids: list[int]
    retrieved_chunk_ids: list[int]
    hit_at_k: dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: int | None
    skipped: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "hit_at_k": {str(k): v for k, v in self.hit_at_k.items()},
            "reciprocal_rank": self.reciprocal_rank,
        }
        if self.first_relevant_rank is not None:
            payload["first_relevant_rank"] = self.first_relevant_rank
        if self.skipped:
            payload["skipped"] = self.skipped
        if self.error:
            payload["error"] = self.error
        return payload


def evaluate_retrieval(
    dataset: Iterable[Mapping[str, Any]],
    pipeline: Any,
    *,
    topk_values: Sequence[int] = (1, 3, 5, 10),
    relevant_field: str = "selected_chunk_ids",
    fallback_relevant_fields: Sequence[str] = ("retrieved_chunk_ids",),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute retrieval metrics for ``pipeline`` over ``dataset``.

    The ``pipeline`` argument must expose a :meth:`prepare_prompt` method that accepts
    the question as its first positional argument and returns a mapping containing a
    ``retrieved`` sequence, where each entry includes an ``id`` field for the chunk id.

    The function returns a tuple with the aggregated summary dictionary and a list of
    per-example metrics (ready for JSONL serialization).
    """

    topk_sorted = sorted({k for k in topk_values if k >= 1})
    hit_totals: dict[int, float] = dict.fromkeys(topk_sorted, 0.0)
    reciprocal_ranks: list[float] = []

    evaluated = 0
    skipped_no_relevance = 0
    retrieval_failures = 0

    per_example_dicts: list[dict[str, Any]] = []

    for record in dataset:
        task_id = str(record.get("task_id") or "")
        instruction = _extract_instruction(record)
        if not instruction:
            continue

        relevant_ids = _extract_relevant_ids(
            record,
            primary=relevant_field,
            fallbacks=fallback_relevant_fields,
        )
        if not relevant_ids:
            skipped_no_relevance += 1
            per_example_dicts.append(
                RetrievalExampleMetrics(
                    task_id=task_id,
                    relevant_chunk_ids=[],
                    retrieved_chunk_ids=[],
                    hit_at_k=dict.fromkeys(topk_sorted, 0.0),
                    reciprocal_rank=0.0,
                    first_relevant_rank=None,
                    skipped=True,
                ).as_dict()
            )
            continue

        require_citations = bool(record.get("metadata", {}).get("require_citations"))

        try:
            prepared = pipeline.prepare_prompt(
                instruction,
                require_citations=require_citations,
            )
        except Exception as exc:  # pragma: no cover - runtime failure path
            retrieval_failures += 1
            per_example_dicts.append(
                RetrievalExampleMetrics(
                    task_id=task_id,
                    relevant_chunk_ids=relevant_ids,
                    retrieved_chunk_ids=[],
                    hit_at_k=dict.fromkeys(topk_sorted, 0.0),
                    reciprocal_rank=0.0,
                    first_relevant_rank=None,
                    error=f"{type(exc).__name__}: {exc}",
                ).as_dict()
            )
            continue

        retrieved_ids = _extract_retrieved_ids(prepared)
        if not retrieved_ids:
            per_example_dicts.append(
                RetrievalExampleMetrics(
                    task_id=task_id,
                    relevant_chunk_ids=relevant_ids,
                    retrieved_chunk_ids=[],
                    hit_at_k=dict.fromkeys(topk_sorted, 0.0),
                    reciprocal_rank=0.0,
                    first_relevant_rank=None,
                ).as_dict()
            )
            evaluated += 1
            continue

        hits = {k: _compute_hit_at_k(retrieved_ids, relevant_ids, k) for k in topk_sorted}
        reciprocal_rank = _compute_reciprocal_rank(retrieved_ids, relevant_ids)
        first_rank = _first_relevant_rank(retrieved_ids, relevant_ids)

        for k, value in hits.items():
            hit_totals[k] += value
        reciprocal_ranks.append(reciprocal_rank)
        evaluated += 1

        per_example_dicts.append(
            RetrievalExampleMetrics(
                task_id=task_id,
                relevant_chunk_ids=relevant_ids,
                retrieved_chunk_ids=retrieved_ids,
                hit_at_k=hits,
                reciprocal_rank=reciprocal_rank,
                first_relevant_rank=first_rank,
            ).as_dict()
        )

    summary = {
        "examples_evaluated": evaluated,
        "skipped_no_relevance": skipped_no_relevance,
        "retrieval_failures": retrieval_failures,
        "hit_at_k": {
            str(k): (hit_totals[k] / evaluated if evaluated else 0.0) for k in topk_sorted
        },
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
    }

    return summary, per_example_dicts


def _extract_instruction(record: Mapping[str, Any]) -> str:
    text = record.get("instruction")
    if isinstance(text, str) and text.strip():
        return text.strip()
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    return ""


def _extract_relevant_ids(
    record: Mapping[str, Any],
    *,
    primary: str,
    fallbacks: Sequence[str],
) -> list[int]:
    metadata = record.get("metadata")
    fields_to_check: list[tuple[str, Any]] = []
    if isinstance(metadata, Mapping):
        fields_to_check.append((primary, metadata.get(primary)))
        for field in fallbacks:
            fields_to_check.append((field, metadata.get(field)))

    rag_block = record.get("rag")
    if isinstance(rag_block, Mapping):
        fields_to_check.append((primary, rag_block.get(primary)))
        for field in fallbacks:
            fields_to_check.append((field, rag_block.get(field)))

    for _field_name, value in fields_to_check:
        ids = _coerce_int_list(value)
        if ids:
            return ids
    return []


def _coerce_int_list(value: Any) -> list[int]:
    if isinstance(value, list | tuple):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result
    return []


def _extract_retrieved_ids(prepared: Mapping[str, Any]) -> list[int]:
    retrieved = prepared.get("retrieved")
    if not isinstance(retrieved, Sequence):
        return []
    ids: list[int] = []
    for entry in retrieved:
        if not isinstance(entry, Mapping):
            continue
        chunk_id = entry.get("id")
        if chunk_id is None:
            continue
        try:
            ids.append(int(chunk_id))
        except (TypeError, ValueError):
            continue
    return ids


def _compute_hit_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Sequence[int],
    k: int,
) -> float:
    limit = min(k, len(retrieved_ids))
    if limit == 0:
        return 0.0
    retrieved_slice = retrieved_ids[:limit]
    relevant_set = set(relevant_ids)
    return 1.0 if any(chunk_id in relevant_set for chunk_id in retrieved_slice) else 0.0


def _compute_reciprocal_rank(
    retrieved_ids: Sequence[int],
    relevant_ids: Sequence[int],
) -> float:
    rank = _first_relevant_rank(retrieved_ids, relevant_ids)
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def _first_relevant_rank(
    retrieved_ids: Sequence[int],
    relevant_ids: Sequence[int],
) -> int | None:
    relevant_set = set(relevant_ids)
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return idx
    return None


__all__ = [
    "RetrievalExampleMetrics",
    "evaluate_retrieval",
]
