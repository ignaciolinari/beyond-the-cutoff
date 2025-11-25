"""Human evaluation protocol for validating model outputs and judge reliability.

This module provides:
- Sampling strategies for selecting evaluation examples
- Annotation data structures
- Inter-annotator agreement metrics (Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha)
- Human-judge correlation analysis
- Export/import utilities for annotation data
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from .elo_ranking import Outcome, PairwiseComparison


class AnnotationStatus(str, Enum):
    """Status of an annotation task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class SamplingStrategy(str, Enum):
    """Strategy for sampling evaluation examples."""

    RANDOM = "random"
    STRATIFIED = "stratified"  # By task type
    DISAGREEMENT = "disagreement"  # Focus on judge disagreements
    UNIFORM_MODEL = "uniform_model"  # Equal samples per model


@dataclass
class AnnotationTask:
    """A single task for human annotation."""

    task_id: str
    question: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    reference: str | None = None
    contexts: list[str] = field(default_factory=list)
    task_type: str = "unknown"
    judge_verdict: Outcome | None = None  # Pre-existing judge verdict for comparison
    judge_rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "reference": self.reference,
            "contexts": self.contexts,
            "task_type": self.task_type,
            "judge_verdict": self.judge_verdict,
            "judge_rationale": self.judge_rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationTask:
        return cls(
            task_id=data["task_id"],
            question=data["question"],
            model_a=data["model_a"],
            model_b=data["model_b"],
            response_a=data["response_a"],
            response_b=data["response_b"],
            reference=data.get("reference"),
            contexts=data.get("contexts", []),
            task_type=data.get("task_type", "unknown"),
            judge_verdict=data.get("judge_verdict"),
            judge_rationale=data.get("judge_rationale"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HumanAnnotation:
    """A human annotation for a task."""

    task_id: str
    annotator_id: str
    verdict: Outcome
    confidence: Literal["low", "medium", "high"] = "medium"
    rationale: str | None = None
    duration_seconds: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    flags: list[str] = field(default_factory=list)  # e.g., "unclear_question", "both_wrong"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "annotator_id": self.annotator_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanAnnotation:
        return cls(
            task_id=data["task_id"],
            annotator_id=data["annotator_id"],
            verdict=data["verdict"],
            confidence=data.get("confidence", "medium"),
            rationale=data.get("rationale"),
            duration_seconds=data.get("duration_seconds"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            flags=data.get("flags", []),
        )

    def to_comparison(
        self,
        task: AnnotationTask,
    ) -> PairwiseComparison:
        """Convert this annotation to a PairwiseComparison."""
        return PairwiseComparison(
            model_a=task.model_a,
            model_b=task.model_b,
            outcome=self.verdict,
            task_id=self.task_id,
            question=task.question,
            response_a=task.response_a,
            response_b=task.response_b,
            annotator=self.annotator_id,
            annotation_source="human",
            timestamp=self.timestamp,
            metadata={
                "confidence": self.confidence,
                "rationale": self.rationale,
                "flags": self.flags,
            },
        )


@dataclass
class AnnotationBatch:
    """A batch of annotation tasks for a human evaluator."""

    batch_id: str
    annotator_id: str
    tasks: list[AnnotationTask]
    annotations: list[HumanAnnotation] = field(default_factory=list)
    status: AnnotationStatus = AnnotationStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        return len(self.annotations) / len(self.tasks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "annotator_id": self.annotator_id,
            "tasks": [t.as_dict() for t in self.tasks],
            "annotations": [a.as_dict() for a in self.annotations],
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationBatch:
        return cls(
            batch_id=data["batch_id"],
            annotator_id=data["annotator_id"],
            tasks=[AnnotationTask.from_dict(t) for t in data["tasks"]],
            annotations=[HumanAnnotation.from_dict(a) for a in data.get("annotations", [])],
            status=AnnotationStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
        )


# =============================================================================
# Sampling Functions
# =============================================================================


def sample_for_annotation(
    predictions: list[dict[str, Any]],
    n_samples: int,
    strategy: SamplingStrategy = SamplingStrategy.RANDOM,
    seed: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Sample examples for human annotation.

    Args:
        predictions: List of prediction records with model outputs.
        n_samples: Number of samples to select.
        strategy: Sampling strategy to use.
        seed: Random seed for reproducibility.
        **kwargs: Strategy-specific arguments.

    Returns:
        Selected samples for annotation.
    """
    if seed is not None:
        random.seed(seed)

    if not predictions:
        return []

    n_samples = min(n_samples, len(predictions))

    if strategy == SamplingStrategy.RANDOM:
        return random.sample(predictions, n_samples)

    elif strategy == SamplingStrategy.STRATIFIED:
        return _stratified_sample(predictions, n_samples, key="task_type")

    elif strategy == SamplingStrategy.DISAGREEMENT:
        # Prioritize examples where judge scores differ significantly
        score_field = kwargs.get("score_field", "judge_score")
        threshold = kwargs.get("threshold", 0.3)
        disagreements = [p for p in predictions if abs(p.get(score_field, 0.5) - 0.5) < threshold]
        if len(disagreements) >= n_samples:
            return random.sample(disagreements, n_samples)
        # Fall back to random for remaining
        remaining = n_samples - len(disagreements)
        others = [p for p in predictions if p not in disagreements]
        return disagreements + random.sample(others, min(remaining, len(others)))

    elif strategy == SamplingStrategy.UNIFORM_MODEL:
        return _stratified_sample(predictions, n_samples, key="model")

    else:
        return random.sample(predictions, n_samples)


def _stratified_sample(
    items: list[dict[str, Any]],
    n_samples: int,
    key: str,
) -> list[dict[str, Any]]:
    """Sample uniformly across strata defined by a key."""
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        stratum = str(item.get(key, "unknown"))
        by_stratum[stratum].append(item)

    strata = list(by_stratum.keys())
    per_stratum = max(1, n_samples // len(strata))

    sampled = []
    for stratum in strata:
        pool = by_stratum[stratum]
        sampled.extend(random.sample(pool, min(per_stratum, len(pool))))

    # Fill remaining quota randomly
    if len(sampled) < n_samples:
        remaining = [i for i in items if i not in sampled]
        sampled.extend(random.sample(remaining, min(n_samples - len(sampled), len(remaining))))

    return sampled[:n_samples]


def create_pairwise_tasks(
    model_predictions: dict[str, list[dict[str, Any]]],
    model_pairs: list[tuple[str, str]] | None = None,
    n_per_pair: int = 50,
    seed: int | None = None,
) -> list[AnnotationTask]:
    """Create pairwise comparison tasks from model predictions.

    Args:
        model_predictions: Dict mapping model name to list of predictions.
        model_pairs: Specific pairs to compare, or None for all pairs.
        n_per_pair: Number of comparisons per model pair.
        seed: Random seed.

    Returns:
        List of annotation tasks.
    """
    if seed is not None:
        random.seed(seed)

    models = list(model_predictions.keys())
    if model_pairs is None:
        # All unique pairs
        model_pairs = [
            (models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))
        ]

    tasks = []
    for model_a, model_b in model_pairs:
        preds_a = {p["task_id"]: p for p in model_predictions.get(model_a, [])}
        preds_b = {p["task_id"]: p for p in model_predictions.get(model_b, [])}

        common_tasks = set(preds_a.keys()) & set(preds_b.keys())
        selected = random.sample(list(common_tasks), min(n_per_pair, len(common_tasks)))

        for task_id in selected:
            pred_a = preds_a[task_id]
            pred_b = preds_b[task_id]

            # Randomize presentation order
            if random.random() < 0.5:
                model_a_shown, model_b_shown = model_a, model_b
                resp_a, resp_b = pred_a.get("model_answer", ""), pred_b.get("model_answer", "")
            else:
                model_a_shown, model_b_shown = model_b, model_a
                resp_a, resp_b = pred_b.get("model_answer", ""), pred_a.get("model_answer", "")

            tasks.append(
                AnnotationTask(
                    task_id=f"{task_id}_{model_a}_{model_b}",
                    question=pred_a.get("question", pred_a.get("prompt", "")),
                    model_a=model_a_shown,
                    model_b=model_b_shown,
                    response_a=resp_a,
                    response_b=resp_b,
                    reference=pred_a.get("reference", pred_a.get("expected_answer")),
                    contexts=pred_a.get("contexts", []),
                    task_type=pred_a.get("task_type", "unknown"),
                    metadata={
                        "original_task_id": task_id,
                        "presentation_order": "original" if model_a_shown == model_a else "swapped",
                    },
                )
            )

    return tasks


# =============================================================================
# Inter-Annotator Agreement Metrics
# =============================================================================


def cohens_kappa(
    annotations_1: Sequence[str],
    annotations_2: Sequence[str],
) -> float:
    """Compute Cohen's Kappa for two annotators.

    Args:
        annotations_1: List of labels from annotator 1.
        annotations_2: List of labels from annotator 2.

    Returns:
        Cohen's Kappa coefficient (-1 to 1, where 1 is perfect agreement).
    """
    if len(annotations_1) != len(annotations_2):
        raise ValueError("Annotation lists must have the same length")

    n = len(annotations_1)
    if n == 0:
        return 0.0

    # Count agreements
    agreements = sum(1 for a, b in zip(annotations_1, annotations_2, strict=True) if a == b)
    p_o = agreements / n  # Observed agreement

    # Count category frequencies
    freq_1 = Counter(annotations_1)
    freq_2 = Counter(annotations_2)
    all_categories = set(freq_1.keys()) | set(freq_2.keys())

    # Expected agreement by chance
    p_e = sum((freq_1.get(cat, 0) / n) * (freq_2.get(cat, 0) / n) for cat in all_categories)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(
    annotations: Sequence[Sequence[str]],
) -> float:
    """Compute Fleiss' Kappa for multiple annotators.

    Args:
        annotations: List of lists, where each inner list contains
                    the annotations for one item from all annotators.

    Returns:
        Fleiss' Kappa coefficient.
    """
    if not annotations:
        return 0.0

    n_items = len(annotations)
    n_raters = len(annotations[0])

    if n_raters < 2:
        return 1.0

    # Get all categories
    all_categories = sorted({cat for item_annots in annotations for cat in item_annots})
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    n_categories = len(all_categories)

    # Build count matrix: n_ij = number of raters who assigned category j to item i
    counts = []
    for item_annots in annotations:
        row = [0] * n_categories
        for annot in item_annots:
            row[cat_to_idx[annot]] += 1
        counts.append(row)

    # Compute P_i (agreement for each item)
    p_items = []
    for row in counts:
        sum_sq = sum(n * n for n in row)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 1.0
        p_items.append(p_i)

    p_bar = sum(p_items) / n_items  # Mean observed agreement

    # Compute P_j (proportion of all assignments to category j)
    total_assignments = n_items * n_raters
    p_j = [
        sum(counts[i][j] for i in range(n_items)) / total_assignments for j in range(n_categories)
    ]

    p_e = sum(p * p for p in p_j)  # Expected agreement by chance

    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0

    return (p_bar - p_e) / (1.0 - p_e)


def compute_agreement_stats(
    annotations: dict[str, list[HumanAnnotation]],
) -> dict[str, Any]:
    """Compute inter-annotator agreement statistics.

    Args:
        annotations: Dict mapping task_id to list of annotations for that task.

    Returns:
        Dict with agreement metrics.
    """
    # Find tasks with multiple annotations
    multi_annotated = {
        task_id: annots for task_id, annots in annotations.items() if len(annots) >= 2
    }

    if not multi_annotated:
        return {
            "n_multi_annotated": 0,
            "cohens_kappa": None,
            "fleiss_kappa": None,
            "raw_agreement": None,
        }

    # For Fleiss' Kappa: need same number of annotators per item
    # Group by number of annotators
    verdicts_by_task = {
        task_id: [a.verdict for a in annots] for task_id, annots in multi_annotated.items()
    }

    # Raw agreement (percentage of items where all annotators agree)
    unanimous = sum(1 for verdicts in verdicts_by_task.values() if len(set(verdicts)) == 1)
    raw_agreement = unanimous / len(multi_annotated)

    # Fleiss' Kappa for all multi-annotated items
    fleiss_input = list(verdicts_by_task.values())
    fk = fleiss_kappa(fleiss_input) if fleiss_input else None

    # Cohen's Kappa for pairs of annotators
    annotator_pairs: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for _task_id, annots in multi_annotated.items():
        for i, a1 in enumerate(annots):
            for a2 in annots[i + 1 :]:
                pair: tuple[str, str] = (
                    sorted([a1.annotator_id, a2.annotator_id])[0],
                    sorted([a1.annotator_id, a2.annotator_id])[1],
                )
                if pair not in annotator_pairs:
                    annotator_pairs[pair] = ([], [])
                annotator_pairs[pair][0].append(a1.verdict)
                annotator_pairs[pair][1].append(a2.verdict)

    kappas = {}
    for pair, (list1, list2) in annotator_pairs.items():
        if len(list1) >= 5:  # Require minimum samples
            kappas[f"{pair[0]}_vs_{pair[1]}"] = cohens_kappa(list1, list2)

    return {
        "n_multi_annotated": len(multi_annotated),
        "cohens_kappa": kappas if kappas else None,
        "fleiss_kappa": fk,
        "raw_agreement": raw_agreement,
    }


def human_judge_correlation(
    human_annotations: list[HumanAnnotation],
    judge_verdicts: dict[str, Outcome],
) -> dict[str, Any]:
    """Compute correlation between human annotations and judge verdicts.

    Args:
        human_annotations: List of human annotations.
        judge_verdicts: Dict mapping task_id to judge verdict.

    Returns:
        Correlation metrics.
    """
    matched = []
    for annot in human_annotations:
        if annot.task_id in judge_verdicts:
            matched.append((annot.verdict, judge_verdicts[annot.task_id]))

    if not matched:
        return {
            "n_matched": 0,
            "agreement_rate": None,
            "cohens_kappa": None,
        }

    human_verdicts = [m[0] for m in matched]
    judge_verdicts_list = [m[1] for m in matched]

    agreement = sum(1 for h, j in matched if h == j) / len(matched)
    kappa = cohens_kappa(human_verdicts, judge_verdicts_list)

    # Breakdown by verdict type
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for h, j in matched:
        confusion[h][j] += 1

    return {
        "n_matched": len(matched),
        "agreement_rate": agreement,
        "cohens_kappa": kappa,
        "confusion_matrix": dict(confusion),
    }


# =============================================================================
# I/O Utilities
# =============================================================================


def save_annotation_batch(batch: AnnotationBatch, path: Path) -> None:
    """Save an annotation batch to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(batch.as_dict(), f, indent=2, ensure_ascii=False)


def load_annotation_batch(path: Path) -> AnnotationBatch:
    """Load an annotation batch from JSON."""
    with open(path, encoding="utf-8") as f:
        return AnnotationBatch.from_dict(json.load(f))


def export_annotations_for_elo(
    batches: list[AnnotationBatch],
    task_lookup: dict[str, AnnotationTask] | None = None,
) -> list[PairwiseComparison]:
    """Export human annotations as PairwiseComparisons for ELO calculation.

    Args:
        batches: List of completed annotation batches.
        task_lookup: Optional dict mapping task_id to AnnotationTask.

    Returns:
        List of PairwiseComparison objects.
    """
    # Build task lookup if not provided
    if task_lookup is None:
        task_lookup = {}
        for batch in batches:
            for task in batch.tasks:
                task_lookup[task.task_id] = task

    comparisons = []
    for batch in batches:
        for annotation in batch.annotations:
            task_or_none = task_lookup.get(annotation.task_id)
            if task_or_none is not None:
                comparisons.append(annotation.to_comparison(task_or_none))

    return comparisons


__all__ = [
    "AnnotationStatus",
    "SamplingStrategy",
    "AnnotationTask",
    "HumanAnnotation",
    "AnnotationBatch",
    "sample_for_annotation",
    "create_pairwise_tasks",
    "cohens_kappa",
    "fleiss_kappa",
    "compute_agreement_stats",
    "human_judge_correlation",
    "save_annotation_batch",
    "load_annotation_batch",
    "export_annotations_for_elo",
]
