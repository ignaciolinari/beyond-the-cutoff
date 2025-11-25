"""Tests for dataset splitting functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from scripts.split_dataset import (
    SplitStats,
    extract_paper_id,
    load_dataset,
    save_dataset,
    split_dataset,
)


def _make_example(
    task_id: str,
    task_type: str = "qa",
    paper_path: str | None = None,
    instruction: str = "Test question?",
    expected_response: str = "Test answer.",
) -> dict[str, Any]:
    """Create a test example with optional paper path in RAG metadata."""
    example: dict[str, Any] = {
        "task_id": task_id,
        "task_type": task_type,
        "instruction": instruction,
        "expected_response": expected_response,
        "rag": {
            "query": instruction,
            "contexts": ["Context chunk 1", "Context chunk 2"],
            "retrieved": [],
        },
    }
    if paper_path:
        example["rag"]["retrieved"] = [
            {"source_path": paper_path, "chunk_idx": 0, "text": "Chunk text"}
        ]
    return example


class TestExtractPaperId:
    """Tests for paper ID extraction."""

    def test_extract_from_rag_metadata(self) -> None:
        """Should extract paper ID from RAG source_path."""
        example = _make_example(
            "test_qa_0",
            paper_path="/data/papers/arxiv_2501.12345.pdf",
        )
        assert extract_paper_id(example) == "arxiv_2501.12345"

    def test_extract_from_task_id_fallback(self) -> None:
        """Should fall back to task_id prefix when no RAG metadata."""
        example = _make_example("paper_name_qa_0", paper_path=None)
        # Clear RAG retrieved to force fallback
        example["rag"]["retrieved"] = []
        paper_id = extract_paper_id(example)
        assert paper_id == "paper_name"

    def test_extract_unknown_fallback(self) -> None:
        """Should return 'unknown' when no identifiable paper info."""
        example: dict[str, Any] = {"task_id": "", "rag": {}}
        assert extract_paper_id(example) == "unknown"


class TestSplitDataset:
    """Tests for the split_dataset function."""

    def test_basic_split_ratio(self) -> None:
        """Should split examples roughly according to eval_ratio."""
        examples = [
            _make_example(f"paper_a_qa_{i}", paper_path="/papers/paper_a.pdf") for i in range(10)
        ]
        train, eval_, stats = split_dataset(examples, eval_ratio=0.3, seed=42)

        assert len(train) + len(eval_) == 10
        assert len(eval_) >= 1  # At least min_eval_per_paper
        assert stats.total_examples == 10
        assert stats.train_examples == len(train)
        assert stats.eval_examples == len(eval_)

    def test_multiple_papers_split(self) -> None:
        """Should split each paper's examples proportionally."""
        examples = []
        # Paper A: 10 examples
        for i in range(10):
            examples.append(_make_example(f"paper_a_qa_{i}", paper_path="/papers/paper_a.pdf"))
        # Paper B: 10 examples
        for i in range(10):
            examples.append(_make_example(f"paper_b_qa_{i}", paper_path="/papers/paper_b.pdf"))

        train, eval_, stats = split_dataset(examples, eval_ratio=0.3, seed=42)

        assert len(train) + len(eval_) == 20
        assert stats.papers_count == 2
        assert stats.papers_with_eval == 2  # Both papers should have eval examples

    def test_min_eval_per_paper(self) -> None:
        """Should ensure minimum eval examples per paper."""
        examples = [
            _make_example(f"paper_a_qa_{i}", paper_path="/papers/paper_a.pdf") for i in range(5)
        ]
        train, eval_, stats = split_dataset(examples, eval_ratio=0.1, min_eval_per_paper=2, seed=42)

        # Even with 10% ratio (0.5 examples), should have at least 2 in eval
        assert len(eval_) >= 2
        assert len(train) <= 3  # Rest go to train

    def test_small_paper_handling(self) -> None:
        """Should handle papers with very few examples."""
        # Paper with only 2 examples
        examples = [
            _make_example("paper_a_qa_0", paper_path="/papers/paper_a.pdf"),
            _make_example("paper_a_qa_1", paper_path="/papers/paper_a.pdf"),
        ]
        train, eval_, stats = split_dataset(examples, eval_ratio=0.3, seed=42)

        # Should have at least 1 in train (never put ALL in eval)
        assert len(train) >= 1
        assert len(train) + len(eval_) == 2

    def test_single_example_paper(self) -> None:
        """Papers with only 1 example should go to train."""
        examples = [
            _make_example("paper_a_qa_0", paper_path="/papers/paper_a.pdf"),
        ]
        train, eval_, stats = split_dataset(examples, eval_ratio=0.5, seed=42)

        # Single example should go to train, not eval
        assert len(train) == 1
        assert len(eval_) == 0

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same split."""
        examples = [
            _make_example(f"paper_a_qa_{i}", paper_path="/papers/paper_a.pdf") for i in range(20)
        ]

        train1, eval1, _ = split_dataset(examples, eval_ratio=0.3, seed=42)
        train2, eval2, _ = split_dataset(examples, eval_ratio=0.3, seed=42)

        assert [e["task_id"] for e in train1] == [e["task_id"] for e in train2]
        assert [e["task_id"] for e in eval1] == [e["task_id"] for e in eval2]

    def test_different_seed_different_split(self) -> None:
        """Different seeds should produce different splits."""
        examples = [
            _make_example(f"paper_a_qa_{i}", paper_path="/papers/paper_a.pdf") for i in range(20)
        ]

        train1, eval1, _ = split_dataset(examples, eval_ratio=0.3, seed=42)
        train2, eval2, _ = split_dataset(examples, eval_ratio=0.3, seed=123)

        # With enough examples, different seeds should give different orderings
        train1_ids = {e["task_id"] for e in train1}
        train2_ids = {e["task_id"] for e in train2}
        # The sets might differ (different examples selected for eval)
        # or at minimum the order should differ
        assert train1_ids != train2_ids or [e["task_id"] for e in train1] != [
            e["task_id"] for e in train2
        ]

    def test_task_type_tracking(self) -> None:
        """Should track task type distribution in stats."""
        examples = [
            _make_example("p_qa_0", task_type="qa", paper_path="/papers/p.pdf"),
            _make_example("p_qa_1", task_type="qa", paper_path="/papers/p.pdf"),
            _make_example("p_sum_0", task_type="summaries", paper_path="/papers/p.pdf"),
            _make_example("p_sum_1", task_type="summaries", paper_path="/papers/p.pdf"),
        ]

        _, _, stats = split_dataset(examples, eval_ratio=0.3, seed=42)

        assert "qa" in stats.task_type_distribution
        assert "summaries" in stats.task_type_distribution
        qa_dist = stats.task_type_distribution["qa"]
        assert qa_dist["train"] + qa_dist["eval"] == 2


class TestLoadSaveDataset:
    """Tests for loading and saving dataset files."""

    def test_roundtrip(self) -> None:
        """Should preserve examples through save/load cycle."""
        examples = [
            _make_example("test_qa_0", paper_path="/papers/test.pdf"),
            _make_example("test_qa_1", task_type="summaries"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_dataset(examples, path)
            loaded = load_dataset(path)

        assert len(loaded) == 2
        assert loaded[0]["task_id"] == "test_qa_0"
        assert loaded[1]["task_type"] == "summaries"

    def test_load_empty_lines(self) -> None:
        """Should skip empty lines when loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            path.write_text('{"task_id": "1"}\n\n{"task_id": "2"}\n   \n{"task_id": "3"}\n')
            loaded = load_dataset(path)

        assert len(loaded) == 3

    def test_creates_parent_dirs(self) -> None:
        """Should create parent directories when saving."""
        examples = [_make_example("test_qa_0")]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test.jsonl"
            save_dataset(examples, path)
            assert path.exists()


class TestSplitStats:
    """Tests for SplitStats summary formatting."""

    def test_summary_format(self) -> None:
        """Summary should include all key statistics."""
        stats = SplitStats(
            total_examples=100,
            train_examples=70,
            eval_examples=30,
            papers_count=10,
            papers_with_eval=10,
            task_type_distribution={
                "qa": {"train": 40, "eval": 20},
                "summaries": {"train": 30, "eval": 10},
            },
        )

        summary = stats.summary()

        assert "100" in summary  # total
        assert "70" in summary  # train
        assert "30" in summary  # eval
        assert "10" in summary  # papers
        assert "qa" in summary
        assert "summaries" in summary
