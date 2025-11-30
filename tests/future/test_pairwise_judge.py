"""Tests for the pairwise judge module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beyond_the_cutoff.config import InferenceConfig
from beyond_the_cutoff.evaluation.pairwise_judge import (
    MultiJudgeEvaluator,
    PairwiseJudge,
    PairwiseJudgeConfig,
    PairwiseJudgment,
    compute_consensus,
    load_predictions_from_results,
)


@pytest.fixture
def mock_inference_config() -> InferenceConfig:
    """Create a mock inference config for testing."""
    return InferenceConfig(
        provider="ollama",
        model="test-model",
    )


@pytest.fixture
def mock_judge_config(mock_inference_config: InferenceConfig) -> PairwiseJudgeConfig:
    """Create a mock judge config for testing."""
    return PairwiseJudgeConfig(
        name="test_judge",
        inference=mock_inference_config,
        max_retries=1,
    )


class TestPairwiseJudgeConfig:
    """Tests for PairwiseJudgeConfig."""

    def test_from_yaml(self) -> None:
        yaml_content = """
name: test_judge
inference:
  provider: ollama
  model: qwen2.5:7b-instruct-q4_K_M
  base_url: http://localhost:11434
max_retries: 3
retry_delay: 2.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = Path(f.name)

        try:
            config = PairwiseJudgeConfig.from_yaml(path)
            assert config.name == "test_judge"
            assert config.inference.model == "qwen2.5:7b-instruct-q4_K_M"
            assert config.max_retries == 3
        finally:
            path.unlink()


class TestPairwiseJudge:
    """Tests for PairwiseJudge."""

    def test_parse_valid_response(self, mock_judge_config: PairwiseJudgeConfig) -> None:
        judge = PairwiseJudge(mock_judge_config)

        response = json.dumps(
            {
                "verdict": "A",
                "reasoning": "Response A is more accurate.",
                "confidence": "high",
                "scores": {
                    "response_a": {"factuality": 0.9, "completeness": 0.8, "clarity": 0.9},
                    "response_b": {"factuality": 0.6, "completeness": 0.7, "clarity": 0.8},
                },
            }
        )

        verdict, reasoning, confidence, scores_a, scores_b = judge._parse_response(response)

        assert verdict == "win_a"
        assert reasoning == "Response A is more accurate."
        assert confidence == "high"
        assert scores_a["factuality"] == 0.9
        assert scores_b["factuality"] == 0.6

    def test_parse_verdict_b(self, mock_judge_config: PairwiseJudgeConfig) -> None:
        judge = PairwiseJudge(mock_judge_config)

        response = json.dumps({"verdict": "B", "reasoning": "B is better", "confidence": "medium"})
        verdict, _, _, _, _ = judge._parse_response(response)

        assert verdict == "win_b"

    def test_parse_tie_verdict(self, mock_judge_config: PairwiseJudgeConfig) -> None:
        judge = PairwiseJudge(mock_judge_config)

        response = json.dumps({"verdict": "tie", "reasoning": "Equal quality", "confidence": "low"})
        verdict, _, confidence, _, _ = judge._parse_response(response)

        assert verdict == "tie"
        assert confidence == "low"

    def test_parse_malformed_json(self, mock_judge_config: PairwiseJudgeConfig) -> None:
        judge = PairwiseJudge(mock_judge_config)

        # Malformed JSON should default to tie since keyword detection is limited
        response = "This is not valid JSON at all."
        verdict, _, _, _, _ = judge._parse_response(response)

        # Default fallback is tie when parsing fails
        assert verdict == "tie"

    def test_compare_with_mock_client(self, mock_judge_config: PairwiseJudgeConfig) -> None:
        with patch(
            "beyond_the_cutoff.evaluation.pairwise_judge.build_generation_client"
        ) as mock_build:
            mock_client = MagicMock()
            mock_client.generate.return_value = {
                "response": json.dumps(
                    {
                        "verdict": "A",
                        "reasoning": "A is better",
                        "confidence": "high",
                        "scores": {"response_a": {}, "response_b": {}},
                    }
                )
            }
            mock_build.return_value = mock_client

            judge = PairwiseJudge(mock_judge_config)
            judgment = judge.compare(
                question="What is X?",
                response_a="Answer A",
                response_b="Answer B",
            )

            assert judgment.verdict == "win_a"
            assert judgment.judge_name == "test_judge"
            mock_client.generate.assert_called_once()


class TestComputeConsensus:
    """Tests for consensus computation."""

    def test_unanimous_verdict(self) -> None:
        judgments = [
            PairwiseJudgment(
                judge_name="j1",
                verdict="win_a",
                reasoning="",
                confidence="high",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
            PairwiseJudgment(
                judge_name="j2",
                verdict="win_a",
                reasoning="",
                confidence="high",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
        ]

        verdict, agreement = compute_consensus(judgments)

        assert verdict == "win_a"
        assert agreement == 1.0

    def test_majority_verdict(self) -> None:
        judgments = [
            PairwiseJudgment(
                judge_name="j1",
                verdict="win_a",
                reasoning="",
                confidence="high",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
            PairwiseJudgment(
                judge_name="j2",
                verdict="win_a",
                reasoning="",
                confidence="medium",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
            PairwiseJudgment(
                judge_name="j3",
                verdict="win_b",
                reasoning="",
                confidence="low",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
        ]

        verdict, agreement = compute_consensus(judgments)

        assert verdict == "win_a"
        # win_a: 3 + 2 = 5, win_b: 1, total: 6, agreement: 5/6 ≈ 0.833
        assert agreement == pytest.approx(5 / 6, abs=0.01)

    def test_weighted_by_confidence(self) -> None:
        # Two low-confidence A votes vs one high-confidence B vote
        judgments = [
            PairwiseJudgment(
                judge_name="j1",
                verdict="win_a",
                reasoning="",
                confidence="low",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
            PairwiseJudgment(
                judge_name="j2",
                verdict="win_a",
                reasoning="",
                confidence="low",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
            PairwiseJudgment(
                judge_name="j3",
                verdict="win_b",
                reasoning="",
                confidence="high",
                scores_a={},
                scores_b={},
                raw_response="",
                latency_seconds=1.0,
            ),
        ]

        verdict, _ = compute_consensus(judgments)

        # A: 1+1=2, B: 3, so B wins
        assert verdict == "win_b"

    def test_empty_judgments(self) -> None:
        verdict, agreement = compute_consensus([])

        assert verdict == "tie"
        assert agreement == 0.0


class TestMultiJudgeEvaluator:
    """Tests for MultiJudgeEvaluator."""

    def test_evaluate_pair_with_mock_judges(self, mock_judge_config: PairwiseJudgeConfig) -> None:
        with patch(
            "beyond_the_cutoff.evaluation.pairwise_judge.build_generation_client"
        ) as mock_build:
            mock_client = MagicMock()
            mock_client.generate.return_value = {
                "response": json.dumps(
                    {
                        "verdict": "A",
                        "reasoning": "A is better",
                        "confidence": "high",
                        "scores": {"response_a": {}, "response_b": {}},
                    }
                )
            }
            mock_build.return_value = mock_client

            evaluator = MultiJudgeEvaluator.from_configs([mock_judge_config, mock_judge_config])
            result = evaluator.evaluate_pair(
                task_id="t1",
                model_a="model_1",
                model_b="model_2",
                question="What is X?",
                response_a="Answer A",
                response_b="Answer B",
            )

            assert result.consensus_verdict == "win_a"
            assert result.agreement_rate == 1.0
            assert len(result.judgments) == 2


class TestLoadPredictions:
    """Tests for loading predictions from result directories."""

    def test_load_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "model_a"
            result_dir.mkdir()

            details_file = result_dir / "details.jsonl"
            with open(details_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"task_id": "t1", "model_answer": "Answer 1"}) + "\n")
                f.write(json.dumps({"task_id": "t2", "model_answer": "Answer 2"}) + "\n")

            predictions = load_predictions_from_results({"model_a": result_dir})

            assert "model_a" in predictions
            assert len(predictions["model_a"]) == 2
            assert predictions["model_a"][0]["task_id"] == "t1"

    def test_missing_directory_warning(self) -> None:
        predictions = load_predictions_from_results({"missing": Path("/nonexistent/path")})
        assert "missing" not in predictions
