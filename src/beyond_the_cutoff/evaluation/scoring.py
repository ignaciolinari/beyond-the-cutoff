"""Higher-level helpers for scoring model predictions against references."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .metrics import (
    citation_precision,
    citation_recall,
    compute_bertscore,
    compute_bleu,
    evaluate_citations,
    grounded_fraction,
    normalize_contexts,
)


@dataclass
class ExampleScore:
    """Container for per-example metrics."""

    task_id: str
    task_type: str
    reference: str
    prediction: str
    factuality: float
    citation_precision: float
    citation_recall: float
    citation_coverage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "reference": self.reference,
            "prediction": self.prediction,
            "factuality": self.factuality,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_coverage": self.citation_coverage,
        }


def score_predictions(
    dataset: Iterable[Mapping[str, Any]],
    predictions: Mapping[str, str],
    *,
    task_types: Sequence[str] | None = None,
    bert_lang: str = "en",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score predictions against references and return summary + per-example rows."""

    allowed_types = {t.strip() for t in task_types} if task_types else None

    per_example: list[ExampleScore] = []
    missing_predictions: list[str] = []
    skipped_examples = 0

    global_predictions: list[str] = []
    global_references: list[str] = []

    type_tracker: dict[str, dict[str, Any]] = {}

    for record in dataset:
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            skipped_examples += 1
            continue
        task_type = str(record.get("task_type") or "").strip() or "unknown"
        if allowed_types is not None and task_type not in allowed_types:
            continue

        prediction = predictions.get(task_id)
        if prediction is None:
            missing_predictions.append(task_id)
            continue

        reference = _extract_reference(record)
        if reference is None:
            skipped_examples += 1
            continue

        contexts_raw = _extract_contexts(record)
        contexts_norm = normalize_contexts(contexts_raw)

        citation_metrics = evaluate_citations(prediction, contexts_norm)
        precision = citation_precision(citation_metrics)
        recall = citation_recall(citation_metrics, total_contexts=len(contexts_norm))
        coverage = float(citation_metrics.get("mean_coverage", 0.0))
        factuality = grounded_fraction(prediction, contexts_norm)

        per_example.append(
            ExampleScore(
                task_id=task_id,
                task_type=task_type,
                reference=reference,
                prediction=prediction,
                factuality=factuality,
                citation_precision=precision,
                citation_recall=recall,
                citation_coverage=coverage,
            )
        )

        global_predictions.append(prediction)
        global_references.append(reference)

        tracker = type_tracker.setdefault(
            task_type,
            {
                "predictions": [],
                "references": [],
                "factuality": [],
                "citation_precision": [],
                "citation_recall": [],
                "citation_coverage": [],
            },
        )
        tracker["predictions"].append(prediction)
        tracker["references"].append(reference)
        tracker["factuality"].append(factuality)
        tracker["citation_precision"].append(precision)
        tracker["citation_recall"].append(recall)
        tracker["citation_coverage"].append(coverage)

    summary = {
        "examples_scored": len(per_example),
        "missing_predictions": missing_predictions,
        "skipped_examples": skipped_examples,
        "metrics": {
            "overall": _aggregate_metrics(
                global_predictions,
                global_references,
                per_example,
                bert_lang=bert_lang,
            ),
            "by_task_type": {
                task_type: _aggregate_metrics(
                    bucket["predictions"],
                    bucket["references"],
                    [example for example in per_example if example.task_type == task_type],
                    bert_lang=bert_lang,
                )
                for task_type, bucket in type_tracker.items()
            },
        },
    }

    return summary, [example.as_dict() for example in per_example]


def _aggregate_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    examples: Sequence[ExampleScore],
    *,
    bert_lang: str,
) -> dict[str, Any]:
    bleu_score = compute_bleu(predictions, references)
    bert_scores = compute_bertscore(predictions, references, lang=bert_lang)
    factuality_values = [example.factuality for example in examples]
    citation_precision_values = [example.citation_precision for example in examples]
    citation_recall_values = [example.citation_recall for example in examples]
    citation_coverage_values = [example.citation_coverage for example in examples]

    return {
        "bleu": bleu_score,
        "bertscore": bert_scores,
        "factuality_grounded_fraction": _safe_mean(factuality_values),
        "citation_precision": _safe_mean(citation_precision_values),
        "citation_recall": _safe_mean(citation_recall_values),
        "citation_coverage": _safe_mean(citation_coverage_values),
    }


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_reference(record: Mapping[str, Any]) -> str | None:
    for key in (
        "reference_answer",
        "reference_summary",
        "expected_response",
        "answer",
        "response",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_contexts(record: Mapping[str, Any]) -> Sequence[str]:
    rag_block = record.get("rag")
    if isinstance(rag_block, Mapping):
        contexts = _coerce_context_list(rag_block.get("contexts"))
        if contexts:
            return contexts
    contexts = _coerce_context_list(record.get("contexts"))
    if contexts:
        return contexts
    return []


def _coerce_context_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [item for item in value if isinstance(item, str)]
    return []
