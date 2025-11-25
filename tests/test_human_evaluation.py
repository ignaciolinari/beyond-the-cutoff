"""Tests for the human evaluation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from beyond_the_cutoff.evaluation.human_evaluation import (
    AnnotationBatch,
    AnnotationStatus,
    AnnotationTask,
    HumanAnnotation,
    SamplingStrategy,
    cohens_kappa,
    compute_agreement_stats,
    create_pairwise_tasks,
    export_annotations_for_elo,
    fleiss_kappa,
    human_judge_correlation,
    load_annotation_batch,
    sample_for_annotation,
    save_annotation_batch,
)


class TestAnnotationTask:
    """Tests for AnnotationTask dataclass."""

    def test_serialization_roundtrip(self) -> None:
        task = AnnotationTask(
            task_id="task_001",
            question="What is the capital of France?",
            model_a="gpt-4",
            model_b="claude-3",
            response_a="Paris is the capital of France.",
            response_b="The capital of France is Paris.",
            reference="Paris",
            contexts=["France is a country in Europe.", "Paris is a city."],
            task_type="factual_qa",
            judge_verdict="tie",
            metadata={"source": "test"},
        )

        data = task.as_dict()
        restored = AnnotationTask.from_dict(data)

        assert restored.task_id == task.task_id
        assert restored.question == task.question
        assert restored.model_a == task.model_a
        assert restored.response_a == task.response_a
        assert restored.contexts == task.contexts
        assert restored.judge_verdict == task.judge_verdict


class TestHumanAnnotation:
    """Tests for HumanAnnotation dataclass."""

    def test_serialization_roundtrip(self) -> None:
        annotation = HumanAnnotation(
            task_id="task_001",
            annotator_id="annotator_1",
            verdict="win_a",
            confidence="high",
            rationale="Response A was more accurate.",
            duration_seconds=45.5,
            flags=["both_responses_correct"],
        )

        data = annotation.as_dict()
        restored = HumanAnnotation.from_dict(data)

        assert restored.task_id == annotation.task_id
        assert restored.verdict == annotation.verdict
        assert restored.confidence == annotation.confidence
        assert restored.duration_seconds == annotation.duration_seconds
        assert restored.flags == annotation.flags

    def test_to_comparison(self) -> None:
        task = AnnotationTask(
            task_id="task_001",
            question="Test question",
            model_a="model_1",
            model_b="model_2",
            response_a="Answer A",
            response_b="Answer B",
        )

        annotation = HumanAnnotation(
            task_id="task_001",
            annotator_id="human_1",
            verdict="win_b",
            confidence="medium",
        )

        comparison = annotation.to_comparison(task)

        assert comparison.model_a == "model_1"
        assert comparison.model_b == "model_2"
        assert comparison.outcome == "win_b"
        assert comparison.annotator == "human_1"
        assert comparison.annotation_source == "human"


class TestAnnotationBatch:
    """Tests for AnnotationBatch dataclass."""

    def test_progress_calculation(self) -> None:
        tasks = [
            AnnotationTask(
                task_id=f"t{i}",
                question="Q",
                model_a="A",
                model_b="B",
                response_a="RA",
                response_b="RB",
            )
            for i in range(10)
        ]

        annotations = [
            HumanAnnotation(task_id=f"t{i}", annotator_id="ann", verdict="tie") for i in range(4)
        ]

        batch = AnnotationBatch(
            batch_id="batch_1",
            annotator_id="annotator_1",
            tasks=tasks,
            annotations=annotations,
        )

        assert batch.progress == 0.4

    def test_empty_batch_progress(self) -> None:
        batch = AnnotationBatch(
            batch_id="batch_1",
            annotator_id="annotator_1",
            tasks=[],
            annotations=[],
        )

        assert batch.progress == 1.0

    def test_serialization_roundtrip(self) -> None:
        tasks = [
            AnnotationTask(
                task_id="t1",
                question="Q",
                model_a="A",
                model_b="B",
                response_a="RA",
                response_b="RB",
            )
        ]
        annotations = [HumanAnnotation(task_id="t1", annotator_id="ann", verdict="win_a")]

        batch = AnnotationBatch(
            batch_id="batch_1",
            annotator_id="annotator_1",
            tasks=tasks,
            annotations=annotations,
            status=AnnotationStatus.IN_PROGRESS,
        )

        data = batch.as_dict()
        restored = AnnotationBatch.from_dict(data)

        assert restored.batch_id == batch.batch_id
        assert len(restored.tasks) == 1
        assert len(restored.annotations) == 1
        assert restored.status == AnnotationStatus.IN_PROGRESS


class TestSampling:
    """Tests for sampling functions."""

    def test_random_sampling(self) -> None:
        predictions = [{"task_id": f"t{i}", "model": "A"} for i in range(100)]

        sampled = sample_for_annotation(
            predictions, n_samples=10, strategy=SamplingStrategy.RANDOM, seed=42
        )

        assert len(sampled) == 10
        assert all("task_id" in s for s in sampled)

    def test_stratified_sampling(self) -> None:
        predictions = [{"task_id": f"t{i}", "task_type": "qa"} for i in range(50)] + [
            {"task_id": f"s{i}", "task_type": "summary"} for i in range(50)
        ]

        sampled = sample_for_annotation(
            predictions, n_samples=20, strategy=SamplingStrategy.STRATIFIED, seed=42
        )

        assert len(sampled) == 20
        # Should have roughly equal representation
        qa_count = sum(1 for s in sampled if s["task_type"] == "qa")
        summary_count = sum(1 for s in sampled if s["task_type"] == "summary")
        assert abs(qa_count - summary_count) <= 4

    def test_sample_more_than_available(self) -> None:
        predictions = [{"task_id": f"t{i}"} for i in range(5)]

        sampled = sample_for_annotation(predictions, n_samples=10, seed=42)

        assert len(sampled) == 5


class TestPairwiseTasks:
    """Tests for pairwise task creation."""

    def test_create_pairwise_tasks(self) -> None:
        model_predictions = {
            "model_A": [
                {"task_id": "t1", "model_answer": "Answer A1", "question": "Q1"},
                {"task_id": "t2", "model_answer": "Answer A2", "question": "Q2"},
            ],
            "model_B": [
                {"task_id": "t1", "model_answer": "Answer B1", "question": "Q1"},
                {"task_id": "t2", "model_answer": "Answer B2", "question": "Q2"},
            ],
        }

        tasks = create_pairwise_tasks(model_predictions, n_per_pair=2, seed=42)

        assert len(tasks) == 2
        for task in tasks:
            assert task.question in ["Q1", "Q2"]
            assert task.model_a in ["model_A", "model_B"]
            assert task.model_b in ["model_A", "model_B"]
            assert task.model_a != task.model_b

    def test_handles_non_overlapping_tasks(self) -> None:
        model_predictions = {
            "model_A": [{"task_id": "t1", "model_answer": "A1", "question": "Q1"}],
            "model_B": [{"task_id": "t2", "model_answer": "B2", "question": "Q2"}],
        }

        tasks = create_pairwise_tasks(model_predictions, n_per_pair=10, seed=42)

        assert len(tasks) == 0  # No common tasks


class TestInterAnnotatorAgreement:
    """Tests for inter-annotator agreement metrics."""

    def test_cohens_kappa_perfect_agreement(self) -> None:
        annotations_1 = ["win_a", "win_b", "tie", "win_a"]
        annotations_2 = ["win_a", "win_b", "tie", "win_a"]

        kappa = cohens_kappa(annotations_1, annotations_2)

        assert kappa == pytest.approx(1.0, abs=0.001)

    def test_cohens_kappa_no_agreement(self) -> None:
        # Systematically opposite
        annotations_1 = ["win_a", "win_a", "win_a", "win_a"]
        annotations_2 = ["win_b", "win_b", "win_b", "win_b"]

        kappa = cohens_kappa(annotations_1, annotations_2)

        # Should be negative or near zero
        assert kappa < 0.1

    def test_cohens_kappa_partial_agreement(self) -> None:
        annotations_1 = ["win_a", "win_b", "win_a", "tie"]
        annotations_2 = ["win_a", "win_a", "win_a", "win_b"]

        kappa = cohens_kappa(annotations_1, annotations_2)

        # Should be between -1 and 1
        assert -1 <= kappa <= 1

    def test_cohens_kappa_different_lengths(self) -> None:
        with pytest.raises(ValueError):
            cohens_kappa(["a", "b"], ["a"])

    def test_fleiss_kappa_perfect_agreement(self) -> None:
        # 3 annotators, all agree on each item
        annotations = [
            ["win_a", "win_a", "win_a"],
            ["win_b", "win_b", "win_b"],
            ["tie", "tie", "tie"],
        ]

        kappa = fleiss_kappa(annotations)

        assert kappa == pytest.approx(1.0, abs=0.001)

    def test_fleiss_kappa_moderate_agreement(self) -> None:
        # Some disagreement
        annotations = [
            ["win_a", "win_a", "win_b"],
            ["win_b", "win_b", "win_b"],
            ["tie", "win_a", "tie"],
        ]

        kappa = fleiss_kappa(annotations)

        assert -1 <= kappa <= 1

    def test_compute_agreement_stats_multi_annotated(self) -> None:
        annotations = {
            "t1": [
                HumanAnnotation(task_id="t1", annotator_id="a1", verdict="win_a"),
                HumanAnnotation(task_id="t1", annotator_id="a2", verdict="win_a"),
            ],
            "t2": [
                HumanAnnotation(task_id="t2", annotator_id="a1", verdict="win_b"),
                HumanAnnotation(task_id="t2", annotator_id="a2", verdict="tie"),
            ],
        }

        stats = compute_agreement_stats(annotations)

        assert stats["n_multi_annotated"] == 2
        assert stats["raw_agreement"] == 0.5  # 1 of 2 unanimous

    def test_compute_agreement_stats_single_annotated(self) -> None:
        annotations = {
            "t1": [HumanAnnotation(task_id="t1", annotator_id="a1", verdict="win_a")],
        }

        stats = compute_agreement_stats(annotations)

        assert stats["n_multi_annotated"] == 0
        assert stats["fleiss_kappa"] is None


class TestHumanJudgeCorrelation:
    """Tests for human-judge correlation analysis."""

    def test_perfect_correlation(self) -> None:
        from beyond_the_cutoff.evaluation.elo_ranking import Outcome

        human_annotations = [
            HumanAnnotation(task_id="t1", annotator_id="h1", verdict="win_a"),
            HumanAnnotation(task_id="t2", annotator_id="h1", verdict="win_b"),
            HumanAnnotation(task_id="t3", annotator_id="h1", verdict="tie"),
        ]

        judge_verdicts: dict[str, Outcome] = {"t1": "win_a", "t2": "win_b", "t3": "tie"}

        stats = human_judge_correlation(human_annotations, judge_verdicts)

        assert stats["n_matched"] == 3
        assert stats["agreement_rate"] == 1.0
        assert stats["cohens_kappa"] == pytest.approx(1.0, abs=0.001)

    def test_no_correlation(self) -> None:
        from beyond_the_cutoff.evaluation.elo_ranking import Outcome

        human_annotations = [
            HumanAnnotation(task_id="t1", annotator_id="h1", verdict="win_a"),
            HumanAnnotation(task_id="t2", annotator_id="h1", verdict="win_a"),
        ]

        judge_verdicts: dict[str, Outcome] = {"t1": "win_b", "t2": "win_b"}

        stats = human_judge_correlation(human_annotations, judge_verdicts)

        assert stats["agreement_rate"] == 0.0

    def test_partial_overlap(self) -> None:
        from beyond_the_cutoff.evaluation.elo_ranking import Outcome

        human_annotations = [
            HumanAnnotation(task_id="t1", annotator_id="h1", verdict="win_a"),
            HumanAnnotation(task_id="t2", annotator_id="h1", verdict="win_b"),
        ]

        # Only t1 has a judge verdict
        judge_verdicts: dict[str, Outcome] = {"t1": "win_a", "t3": "tie"}

        stats = human_judge_correlation(human_annotations, judge_verdicts)

        assert stats["n_matched"] == 1


class TestFileIO:
    """Tests for file I/O utilities."""

    def test_batch_save_and_load(self) -> None:
        tasks = [
            AnnotationTask(
                task_id="t1",
                question="Q",
                model_a="A",
                model_b="B",
                response_a="RA",
                response_b="RB",
            )
        ]

        batch = AnnotationBatch(
            batch_id="test_batch",
            annotator_id="test_annotator",
            tasks=tasks,
            status=AnnotationStatus.PENDING,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_annotation_batch(batch, path)
            loaded = load_annotation_batch(path)

            assert loaded.batch_id == batch.batch_id
            assert loaded.annotator_id == batch.annotator_id
            assert len(loaded.tasks) == 1
            assert loaded.status == AnnotationStatus.PENDING
        finally:
            path.unlink()


class TestExportAnnotationsForELO:
    """Tests for exporting annotations to ELO format."""

    def test_export_basic(self) -> None:
        tasks = [
            AnnotationTask(
                task_id="t1",
                question="Q1",
                model_a="A",
                model_b="B",
                response_a="RA",
                response_b="RB",
            ),
        ]

        batch = AnnotationBatch(
            batch_id="b1",
            annotator_id="ann1",
            tasks=tasks,
            annotations=[
                HumanAnnotation(task_id="t1", annotator_id="ann1", verdict="win_a"),
            ],
        )

        comparisons = export_annotations_for_elo([batch])

        assert len(comparisons) == 1
        assert comparisons[0].model_a == "A"
        assert comparisons[0].outcome == "win_a"
        assert comparisons[0].annotation_source == "human"

    def test_export_multiple_batches(self) -> None:
        tasks = [
            AnnotationTask(
                task_id="t1",
                question="Q",
                model_a="A",
                model_b="B",
                response_a="RA",
                response_b="RB",
            ),
        ]

        batch1 = AnnotationBatch(
            batch_id="b1",
            annotator_id="ann1",
            tasks=tasks,
            annotations=[
                HumanAnnotation(task_id="t1", annotator_id="ann1", verdict="win_a"),
            ],
        )

        batch2 = AnnotationBatch(
            batch_id="b2",
            annotator_id="ann2",
            tasks=tasks,
            annotations=[
                HumanAnnotation(task_id="t1", annotator_id="ann2", verdict="win_b"),
            ],
        )

        comparisons = export_annotations_for_elo([batch1, batch2])

        assert len(comparisons) == 2
