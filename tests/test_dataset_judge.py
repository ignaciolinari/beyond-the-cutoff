"""Tests for the dataset quality judge."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from beyond_the_cutoff.config import InferenceConfig
from beyond_the_cutoff.evaluation.dataset_judge import (
    DatasetQualityJudge,
    DatasetQualityResult,
    ExampleVerdict,
)


@pytest.fixture
def mock_inference_config() -> InferenceConfig:
    """Create a mock inference config."""
    return InferenceConfig(
        provider="ollama",
        model="test-model",
        host="http://localhost",
        port=11434,
        timeout=60.0,
        max_new_tokens=256,
        temperature=0.0,
    )


@pytest.fixture
def sample_example() -> dict[str, Any]:
    """Create a sample dataset example."""
    return {
        "task_id": "test-001",
        "task_type": "qa",
        "instruction": "What is the main finding of the study?",
        "expected_response": "The study found that X improves Y by 20% [1].",
        "contexts": [
            "The experimental results show that X improves Y by 20% across all conditions.",
            "Prior work had suggested only a 10% improvement was possible.",
        ],
        "rag": {
            "contexts": [
                "The experimental results show that X improves Y by 20% across all conditions.",
                "Prior work had suggested only a 10% improvement was possible.",
            ]
        },
    }


class TestExampleVerdict:
    """Tests for ExampleVerdict dataclass."""

    def test_verdict_creation(self) -> None:
        verdict = ExampleVerdict(
            task_id="test-001",
            task_type="qa",
            passed=True,
            scores={"answerability": 0.9, "correctness": 0.8, "clarity": 0.9, "coherence": 0.85},
            issues=[],
            reasoning="All criteria met.",
        )
        assert verdict.passed is True
        assert verdict.task_id == "test-001"
        assert len(verdict.scores) == 4

    def test_verdict_with_issues(self) -> None:
        verdict = ExampleVerdict(
            task_id="test-002",
            task_type="citations",
            passed=False,
            scores={"answerability": 0.3, "correctness": 0.5, "clarity": 0.8, "coherence": 0.7},
            issues=["Context doesn't contain information needed to answer"],
            reasoning="Low answerability - context lacks key information.",
        )
        assert verdict.passed is False
        assert len(verdict.issues) == 1


class TestDatasetQualityJudge:
    """Tests for DatasetQualityJudge."""

    def test_extract_contexts_from_rag(
        self, mock_inference_config: InferenceConfig, sample_example: dict[str, Any]
    ) -> None:
        """Test context extraction from RAG field."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        contexts = judge._extract_contexts(sample_example)
        assert len(contexts) == 2
        assert "20%" in contexts[0]

    def test_extract_contexts_from_top_level(self, mock_inference_config: InferenceConfig) -> None:
        """Test context extraction from top-level contexts field."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        example = {
            "task_id": "test",
            "contexts": ["Context A", "Context B"],
        }
        contexts = judge._extract_contexts(example)
        assert contexts == ["Context A", "Context B"]

    def test_format_contexts(self, mock_inference_config: InferenceConfig) -> None:
        """Test context formatting for judge prompt."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        contexts = ["First context.", "Second context."]
        formatted = judge._format_contexts(contexts)
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "First context." in formatted

    def test_format_contexts_empty(self, mock_inference_config: InferenceConfig) -> None:
        """Test formatting with no contexts."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        formatted = judge._format_contexts([])
        assert "No contexts provided" in formatted

    def test_parse_verdict_valid_json(self, mock_inference_config: InferenceConfig) -> None:
        """Test parsing a valid judge response."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        raw_response = """
        {
            "scores": {
                "answerability": 0.9,
                "correctness": 0.85,
                "clarity": 0.95,
                "coherence": 0.9
            },
            "passed": true,
            "issues": [],
            "reasoning": "All criteria are met. The answer is well-supported by the context."
        }
        """
        verdict = judge._parse_verdict(raw_response, "test-001", "qa")
        assert verdict.passed is True
        assert verdict.scores["answerability"] == 0.9
        assert verdict.scores["correctness"] == 0.85
        assert len(verdict.issues) == 0

    def test_parse_verdict_with_issues(self, mock_inference_config: InferenceConfig) -> None:
        """Test parsing a verdict with issues."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        raw_response = """
        {
            "scores": {
                "answerability": 0.4,
                "correctness": 0.5,
                "clarity": 0.8,
                "coherence": 0.7
            },
            "passed": false,
            "issues": ["Context missing key information", "Answer contains unsupported claim"],
            "reasoning": "The context doesn't contain enough information to fully answer the question."
        }
        """
        verdict = judge._parse_verdict(raw_response, "test-002", "qa")
        assert verdict.passed is False
        assert verdict.scores["answerability"] == 0.4
        assert len(verdict.issues) == 2

    def test_parse_verdict_invalid_json(self, mock_inference_config: InferenceConfig) -> None:
        """Test parsing an invalid response."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        raw_response = "This is not valid JSON at all."
        verdict = judge._parse_verdict(raw_response, "test-003", "qa")
        assert verdict.passed is False
        assert "not parseable" in verdict.issues[0]

    def test_aggregate_results(self, mock_inference_config: InferenceConfig) -> None:
        """Test result aggregation."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        verdicts = [
            ExampleVerdict(
                task_id="test-001",
                task_type="qa",
                passed=True,
                scores={"answerability": 0.9, "correctness": 0.9, "clarity": 0.9, "coherence": 0.9},
                issues=[],
                reasoning="Good",
            ),
            ExampleVerdict(
                task_id="test-002",
                task_type="qa",
                passed=False,
                scores={"answerability": 0.4, "correctness": 0.5, "clarity": 0.8, "coherence": 0.7},
                issues=["Low answerability"],
                reasoning="Bad",
            ),
        ]
        result = judge._aggregate_results(verdicts)
        assert result.total_evaluated == 2
        assert result.passed_count == 1
        assert result.failed_count == 1
        assert result.pass_rate == 0.5
        assert result.mean_scores["answerability"] == 0.65  # (0.9 + 0.4) / 2
        assert len(result.failed_examples) == 1

    def test_aggregate_results_empty(self, mock_inference_config: InferenceConfig) -> None:
        """Test aggregation with no verdicts."""
        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            judge = DatasetQualityJudge(mock_inference_config)

        result = judge._aggregate_results([])
        assert result.total_evaluated == 0
        assert result.pass_rate == 0.0

    def test_to_dict(self, mock_inference_config: InferenceConfig) -> None:
        """Test DatasetQualityResult serialization."""
        result = DatasetQualityResult(
            total_evaluated=10,
            passed_count=8,
            failed_count=2,
            pass_rate=0.8,
            mean_scores={"answerability": 0.75, "correctness": 0.8},
            score_distributions={
                "answerability": {
                    "0.0-0.2": 0,
                    "0.2-0.4": 1,
                    "0.4-0.6": 1,
                    "0.6-0.8": 3,
                    "0.8-1.0": 5,
                }
            },
            common_issues={"low answerability": 2},
            verdicts=[],
            failed_examples=[],
        )
        data = result.to_dict()
        assert data["total_evaluated"] == 10
        assert data["pass_rate"] == 0.8
        assert "answerability" in data["mean_scores"]


class TestDatasetQualityJudgeIntegration:
    """Integration tests that require mocking the full LLM client."""

    def test_evaluate_example_success(
        self, mock_inference_config: InferenceConfig, sample_example: dict[str, Any]
    ) -> None:
        """Test evaluating a single example successfully."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": """
        {
            "scores": {
                "answerability": 0.9,
                "correctness": 0.85,
                "clarity": 0.9,
                "coherence": 0.9
            },
            "passed": true,
            "issues": [],
            "reasoning": "Well-formed example with good context support."
        }
        """
        }

        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = mock_client
            judge = DatasetQualityJudge(mock_inference_config)
            verdict = judge.evaluate_example(sample_example)

        assert verdict.passed is True
        assert verdict.task_id == "test-001"
        mock_client.generate.assert_called_once()

    def test_evaluate_example_failure(
        self, mock_inference_config: InferenceConfig, sample_example: dict[str, Any]
    ) -> None:
        """Test evaluating an example that fails quality checks."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": """
        {
            "scores": {
                "answerability": 0.3,
                "correctness": 0.4,
                "clarity": 0.8,
                "coherence": 0.7
            },
            "passed": false,
            "issues": ["Context doesn't fully support the answer", "Missing key details"],
            "reasoning": "The expected response makes claims not found in the provided contexts."
        }
        """
        }

        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = mock_client
            judge = DatasetQualityJudge(mock_inference_config)
            verdict = judge.evaluate_example(sample_example)

        assert verdict.passed is False
        assert len(verdict.issues) == 2

    def test_evaluate_example_client_error(
        self, mock_inference_config: InferenceConfig, sample_example: dict[str, Any]
    ) -> None:
        """Test handling of LLM client errors."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Connection failed")

        with patch(
            "beyond_the_cutoff.evaluation.dataset_judge.build_generation_client"
        ) as mock_build:
            mock_build.return_value = mock_client
            judge = DatasetQualityJudge(mock_inference_config)
            verdict = judge.evaluate_example(sample_example)

        assert verdict.passed is False
        assert "Judge evaluation failed" in verdict.issues[0]
