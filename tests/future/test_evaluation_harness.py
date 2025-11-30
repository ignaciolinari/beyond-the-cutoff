from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.evaluation import harness


def test_compute_harness_combines_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_config: ProjectConfig = load_config("configs/default.yaml")
    dataset: list[Mapping[str, Any]] = [{"task_id": "task-1"}, {"task_id": "task-2"}]
    predictions: dict[str, str] = {"task-1": "foo", "task-2": "bar"}

    expected_generation_summary = {"bleu": 0.8}
    expected_generation_examples = [
        {"task_id": "task-1", "score": 0.9},
        {"task_id": "task-2", "score": 0.7},
    ]
    expected_retrieval_summary = {"hit@1": 0.5}
    expected_retrieval_examples = [
        {"task_id": "task-1", "hit@1": 1.0},
        {"task_id": "task-3", "hit@1": 0.0},
    ]

    def fake_score(
        dataset_records: list[Mapping[str, Any]],
        mapping: Mapping[str, str],
        bert_lang: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert dataset_records == dataset
        assert mapping == predictions
        assert bert_lang == "en"
        return expected_generation_summary, expected_generation_examples

    def fake_evaluate(
        dataset_records: list[Mapping[str, Any]],
        pipeline: DummyPipeline,
        topk_values: tuple[int, ...],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert dataset_records == dataset
        assert isinstance(pipeline, DummyPipeline)
        assert tuple(topk_values) == (1, 3)
        return expected_retrieval_summary, expected_retrieval_examples

    class DummyPipeline:
        def __init__(self, config: Any, index_path: Path, mapping_path: Path) -> None:
            self.config = config
            self.index_path = index_path
            self.mapping_path = mapping_path

    monkeypatch.setattr(harness, "score_predictions", fake_score)
    monkeypatch.setattr(harness, "evaluate_retrieval", fake_evaluate)
    monkeypatch.setattr(harness, "RAGPipeline", DummyPipeline)

    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "mapping.tsv"
    index_path.write_bytes(b"")
    mapping_path.write_text("", encoding="utf-8")

    summary, details = harness.compute_harness(
        dataset_records=dataset,
        project_config=project_config,
        predictions=predictions,
        bert_lang="en",
        topk_values=(1, 3),
        index_override=index_path,
        mapping_override=mapping_path,
    )

    assert summary == {
        "generation": expected_generation_summary,
        "retrieval": expected_retrieval_summary,
    }
    assert details == [
        {
            "task_id": "task-1",
            "generation": {"task_id": "task-1", "score": 0.9},
            "retrieval": {"task_id": "task-1", "hit@1": 1.0},
        },
        {
            "task_id": "task-2",
            "generation": {"task_id": "task-2", "score": 0.7},
            "retrieval": None,
        },
        {
            "task_id": "task-3",
            "generation": None,
            "retrieval": {"task_id": "task-3", "hit@1": 0.0},
        },
    ]


def test_compute_harness_rejects_missing_index(
    tmp_path: Path,
) -> None:
    project_config: ProjectConfig = load_config("configs/default.yaml")
    dataset: list[Mapping[str, Any]] = [{"task_id": "x"}]

    with pytest.raises(FileNotFoundError):
        harness.compute_harness(
            dataset_records=dataset,
            project_config=project_config,
            predictions={},
            index_override=tmp_path / "missing.faiss",
            mapping_override=tmp_path / "missing.tsv",
        )
