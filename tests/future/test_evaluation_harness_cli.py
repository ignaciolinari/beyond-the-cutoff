from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from beyond_the_cutoff.evaluation import harness as harness_module
from scripts import evaluation_harness


def test_evaluation_harness_cli_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_records: list[Mapping[str, Any]] = [
        {"task_id": "task-1", "instruction": "Q1"},
        {"task_id": "task-2", "instruction": "Q2"},
    ]
    predictions_rows: list[Mapping[str, Any]] = [
        {"task_id": "task-1", "model_answer": "foo"},
        {"task_id": "task-2", "model_answer": "bar"},
    ]

    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in dataset_records) + "\n", encoding="utf-8"
    )
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in predictions_rows) + "\n",
        encoding="utf-8",
    )

    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "mapping.tsv"
    index_path.write_bytes(b"")
    mapping_path.write_text("", encoding="utf-8")

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
        dataset_records_arg: list[Mapping[str, Any]],
        mapping: Mapping[str, str],
        bert_lang: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert dataset_records_arg == dataset_records
        expected_mapping = {
            str(row["task_id"]): str(row["model_answer"]) for row in predictions_rows
        }
        assert mapping == expected_mapping
        assert bert_lang == "en"
        return expected_generation_summary, expected_generation_examples

    def fake_evaluate(
        dataset_records_arg: list[Mapping[str, Any]],
        pipeline: DummyPipeline,
        topk_values: tuple[int, ...],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert dataset_records_arg == dataset_records
        assert isinstance(pipeline, DummyPipeline)
        assert tuple(topk_values) == (1, 3)
        return expected_retrieval_summary, expected_retrieval_examples

    class DummyPipeline:
        def __init__(self, config: Any, index_path: Path, mapping_path: Path) -> None:
            self.config = config
            self.index_path = index_path
            self.mapping_path = mapping_path

    monkeypatch.setattr(evaluation_harness, "compute_harness", harness_module.compute_harness)
    monkeypatch.setattr(harness_module, "score_predictions", fake_score)
    monkeypatch.setattr(harness_module, "evaluate_retrieval", fake_evaluate)
    monkeypatch.setattr(harness_module, "RAGPipeline", DummyPipeline)

    summary_path = tmp_path / "summary.json"
    details_path = tmp_path / "details.jsonl"

    argv = [
        "evaluation_harness.py",
        "--config",
        "configs/default.yaml",
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(summary_path),
        "--details-output",
        str(details_path),
        "--retrieval-topk",
        "1,3",
        "--index",
        str(index_path),
        "--mapping",
        str(mapping_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    evaluation_harness.main()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload == {
        "generation": expected_generation_summary,
        "retrieval": expected_retrieval_summary,
    }

    details_lines = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines()
    ]
    assert details_lines == [
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
