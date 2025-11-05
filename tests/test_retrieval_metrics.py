from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from beyond_the_cutoff.evaluation.retrieval_metrics import evaluate_retrieval


class DummyPipeline:
    def __init__(
        self,
        responses: Mapping[str, Mapping[str, Any]],
        failures: set[str] | None = None,
    ) -> None:
        self._responses = responses
        self._failures = failures or set()

    def prepare_prompt(self, question: str, *, require_citations: bool = True) -> Mapping[str, Any]:
        if question in self._failures:
            raise RuntimeError("pipeline failure")
        return self._responses[question]


def test_evaluate_retrieval_basic_metrics() -> None:
    dataset: list[Mapping[str, Any]] = [
        {
            "task_id": "a",
            "instruction": "Q1",
            "metadata": {"selected_chunk_ids": [42]},
        },
        {
            "task_id": "b",
            "instruction": "Q2",
            "metadata": {"selected_chunk_ids": [5, 7]},
        },
        {"task_id": "c", "instruction": "Q3"},
        {
            "task_id": "d",
            "instruction": "Q4",
            "metadata": {"selected_chunk_ids": [8]},
        },
    ]

    pipeline = DummyPipeline(
        {
            "Q1": {"retrieved": [{"id": 42}, {"id": 99}]},
            "Q2": {"retrieved": [{"id": 2}, {"id": 5}, {"id": 7}]},
            "Q4": {"retrieved": [{"id": 1}]},
        },
        failures={"Q4"},
    )

    summary, per_example = evaluate_retrieval(
        dataset,
        pipeline,
        topk_values=(1, 3),
    )

    assert summary["examples_evaluated"] == 2
    assert summary["skipped_no_relevance"] == 1
    assert summary["retrieval_failures"] == 1
    assert summary["hit_at_k"]["1"] == pytest.approx(0.5)
    assert summary["hit_at_k"]["3"] == pytest.approx(1.0)
    assert summary["mean_reciprocal_rank"] == pytest.approx(0.75)

    skip_row = next(row for row in per_example if row["task_id"] == "c")
    assert skip_row.get("skipped") is True

    failure_row = next(row for row in per_example if row["task_id"] == "d")
    assert isinstance(failure_row.get("error"), str)
