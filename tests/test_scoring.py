from __future__ import annotations

from typing import Any

import pytest

from beyond_the_cutoff.evaluation import scoring


def _sample_dataset() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "1",
            "task_type": "qa",
            "expected_response": "Paris is the capital of France.",
            "rag": {
                "contexts": [
                    "Paris is the capital of France.",
                    "Berlin is the capital of Germany.",
                ]
            },
        },
        {
            "task_id": "2",
            "task_type": "summaries",
            "expected_response": "This study analyses European capitals.",
            "rag": {"contexts": ["The paper focuses on European capitals and governance."]},
        },
    ]


def _sample_predictions() -> dict[str, str]:
    return {
        "1": "France's capital is Paris. [1]",
        "2": "The paper analyses European capitals and governance. [1]",
    }


def _stub_text_metrics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bleu: float = 0.0,
    bert: dict[str, float] | None = None,
) -> None:
    bert_values = (
        dict(bert)
        if bert is not None
        else {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    )

    def fake_bleu(preds: Any, refs: Any) -> float:
        return bleu

    def fake_bertscore(preds: Any, refs: Any, lang: str) -> dict[str, float]:
        return dict(bert_values)

    monkeypatch.setattr(scoring, "compute_bleu", fake_bleu)
    monkeypatch.setattr(scoring, "compute_bertscore", fake_bertscore)


def test_score_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dataset = _sample_dataset()
    sample_predictions = _sample_predictions()
    _stub_text_metrics(
        monkeypatch,
        bleu=0.42,
        bert={"precision": 0.5, "recall": 0.6, "f1": 0.55},
    )

    summary, per_example = scoring.score_predictions(
        sample_dataset, sample_predictions, bert_lang="en"
    )

    assert summary["examples_scored"] == 2
    assert summary["metrics"]["overall"]["bleu"] == 0.42
    assert summary["metrics"]["overall"]["bertscore"]["f1"] == 0.55
    assert "qa" in summary["metrics"]["by_task_type"]
    assert "summaries" in summary["metrics"]["by_task_type"]

    example_ids = {row["task_id"] for row in per_example}
    assert example_ids == {"1", "2"}
    for row in per_example:
        assert "factuality" in row
        assert "citation_precision" in row
        assert "citation_recall" in row


def test_score_predictions_records_missing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dataset = _sample_dataset()
    sample_predictions = _sample_predictions()
    _stub_text_metrics(monkeypatch)
    partial_predictions = {"1": sample_predictions["1"]}

    summary, per_example = scoring.score_predictions(sample_dataset, partial_predictions)

    assert summary["examples_scored"] == 1
    assert summary["missing_predictions"] == ["2"]
    assert summary["metrics"]["overall"]["bleu"] == 0.0
    assert summary["metrics"]["overall"]["bertscore"]["f1"] == 0.0
    assert "qa" in summary["metrics"]["by_task_type"]
    assert "summaries" not in summary["metrics"]["by_task_type"]
    assert len(per_example) == 1


def test_score_predictions_handles_empty_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_text_metrics(monkeypatch)
    dataset = [
        {
            "task_id": "freeform-1",
            "task_type": "qa",
            "expected_response": "An answer with supporting facts.",
            "rag": {"contexts": []},
        }
    ]
    predictions = {"freeform-1": "This is a grounded answer without citations."}

    summary, per_example = scoring.score_predictions(dataset, predictions)

    assert summary["examples_scored"] == 1
    assert summary["metrics"]["overall"]["citation_precision"] == 1.0
    assert summary["metrics"]["overall"]["citation_recall"] == 1.0
    assert per_example[0]["factuality"] == 0.0
    assert per_example[0]["citation_precision"] == 1.0
    assert per_example[0]["citation_recall"] == 1.0


def test_score_predictions_flags_citation_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_text_metrics(monkeypatch)
    dataset = [
        {
            "task_id": "edge-1",
            "task_type": "qa",
            "expected_response": "Paris is the capital of France.",
            "rag": {
                "contexts": [
                    "Paris is the capital of France.",
                    "Berlin is the capital of Germany.",
                ]
            },
        }
    ]
    predictions = {"edge-1": "The capital is Paris according to [3]."}

    summary, per_example = scoring.score_predictions(dataset, predictions)

    example = per_example[0]
    assert example["citation_precision"] == 0.0
    assert example["citation_recall"] == 0.0
    assert summary["metrics"]["overall"]["citation_precision"] == 0.0
    assert summary["metrics"]["overall"]["citation_recall"] == 0.0
