"""Tests for the ELO ranking system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from beyond_the_cutoff.evaluation.elo_ranking import (
    ELOCalculator,
    ELORating,
    PairwiseComparison,
    bootstrap_elo_confidence,
    compute_elo_rankings,
    head_to_head_matrix,
    load_comparisons_from_jsonl,
    save_comparisons_to_jsonl,
    save_leaderboard,
)


class TestPairwiseComparison:
    """Tests for the PairwiseComparison dataclass."""

    def test_serialization_roundtrip(self) -> None:
        comp = PairwiseComparison(
            model_a="model_1",
            model_b="model_2",
            outcome="win_a",
            task_id="task_001",
            question="What is X?",
            response_a="Answer A",
            response_b="Answer B",
            annotator="judge_1",
            annotation_source="judge",
            metadata={"score_a": 0.8, "score_b": 0.3},
        )

        data = comp.as_dict()
        restored = PairwiseComparison.from_dict(data)

        assert restored.model_a == comp.model_a
        assert restored.model_b == comp.model_b
        assert restored.outcome == comp.outcome
        assert restored.task_id == comp.task_id
        assert restored.annotator == comp.annotator
        assert restored.metadata == comp.metadata


class TestELOCalculator:
    """Tests for the ELOCalculator class."""

    def test_initial_rating(self) -> None:
        calc = ELOCalculator(initial_rating=1500)
        assert calc.get_rating("new_model") == 1500

    def test_expected_score(self) -> None:
        calc = ELOCalculator()

        # Equal ratings should give 0.5 expected score
        assert calc.expected_score(1500, 1500) == pytest.approx(0.5, abs=0.001)

        # Higher rated player should have higher expected score
        assert calc.expected_score(1600, 1400) > 0.5
        assert calc.expected_score(1400, 1600) < 0.5

        # 400 point difference should give ~90% expected
        assert calc.expected_score(1900, 1500) == pytest.approx(0.909, abs=0.01)

    def test_win_updates_ratings(self) -> None:
        calc = ELOCalculator(k_factor=32, initial_rating=1500)

        comp = PairwiseComparison(
            model_a="model_1",
            model_b="model_2",
            outcome="win_a",
            task_id="t1",
        )

        new_a, new_b = calc.update_ratings(comp)

        # Winner should gain, loser should lose
        assert new_a > 1500
        assert new_b < 1500

        # Changes should be symmetric
        assert (new_a - 1500) == pytest.approx(1500 - new_b, abs=0.001)

    def test_tie_splits_rating_change(self) -> None:
        calc = ELOCalculator(k_factor=32, initial_rating=1500)

        comp = PairwiseComparison(
            model_a="model_1",
            model_b="model_2",
            outcome="tie",
            task_id="t1",
        )

        new_a, new_b = calc.update_ratings(comp)

        # Tie between equal players should not change ratings significantly
        assert new_a == pytest.approx(1500, abs=0.1)
        assert new_b == pytest.approx(1500, abs=0.1)

    def test_stats_tracking(self) -> None:
        calc = ELOCalculator()

        comparisons = [
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t1"),
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t2"),
            PairwiseComparison(model_a="A", model_b="B", outcome="win_b", task_id="t3"),
            PairwiseComparison(model_a="A", model_b="B", outcome="tie", task_id="t4"),
        ]

        for comp in comparisons:
            calc.update_ratings(comp)

        leaderboard = calc.get_leaderboard()

        # Model A should be ranked higher
        assert leaderboard[0].model == "A"
        assert leaderboard[0].wins == 2
        assert leaderboard[0].losses == 1
        assert leaderboard[0].ties == 1

        assert leaderboard[1].model == "B"
        assert leaderboard[1].wins == 1
        assert leaderboard[1].losses == 2

    def test_leaderboard_sorting(self) -> None:
        calc = ELOCalculator()

        # Model C beats A, A beats B repeatedly
        for _ in range(5):
            calc.update_ratings(
                PairwiseComparison(model_a="C", model_b="A", outcome="win_a", task_id="t1")
            )
            calc.update_ratings(
                PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t2")
            )

        leaderboard = calc.get_leaderboard()

        assert leaderboard[0].model == "C"
        assert leaderboard[1].model == "A"
        assert leaderboard[2].model == "B"

    def test_state_serialization(self) -> None:
        calc = ELOCalculator(k_factor=24, initial_rating=1200)

        calc.update_ratings(
            PairwiseComparison(model_a="X", model_b="Y", outcome="win_a", task_id="t1")
        )

        state = calc.get_state()

        new_calc = ELOCalculator()
        new_calc.load_state(state)

        assert new_calc.k_factor == 24
        assert new_calc.initial_rating == 1200
        assert new_calc.get_rating("X") == calc.get_rating("X")
        assert new_calc.get_rating("Y") == calc.get_rating("Y")


class TestBootstrapConfidence:
    """Tests for bootstrap confidence interval calculation."""

    def test_returns_intervals_for_all_models(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id=f"t{i}")
            for i in range(20)
        ]

        results = bootstrap_elo_confidence(comparisons, n_bootstrap=50, seed=42)

        assert "A" in results
        assert "B" in results

        # Check tuple structure
        for _model, (lower, mean, upper) in results.items():
            assert lower <= mean <= upper

    def test_winner_has_higher_rating(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="winner", model_b="loser", outcome="win_a", task_id=f"t{i}")
            for i in range(30)
        ]

        results = bootstrap_elo_confidence(comparisons, n_bootstrap=100, seed=42)

        winner_lower, winner_mean, winner_upper = results["winner"]
        loser_lower, loser_mean, loser_upper = results["loser"]

        assert winner_mean > loser_mean

    def test_confidence_level_affects_width(self) -> None:
        # Mix outcomes to introduce variance in bootstrap samples
        comparisons = [
            PairwiseComparison(
                model_a="A",
                model_b="B",
                outcome="win_a" if i % 3 != 0 else "win_b",
                task_id=f"t{i}",
            )
            for i in range(50)
        ]

        narrow = bootstrap_elo_confidence(comparisons, confidence=0.5, n_bootstrap=500, seed=42)
        wide = bootstrap_elo_confidence(comparisons, confidence=0.99, n_bootstrap=500, seed=42)

        narrow_width = narrow["A"][2] - narrow["A"][0]
        wide_width = wide["A"][2] - wide["A"][0]

        # With mixed outcomes, we should see wider CI for higher confidence
        assert wide_width >= narrow_width


class TestComputeELORankings:
    """Tests for the main ranking computation function."""

    def test_returns_leaderboard_and_metadata(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t1"),
            PairwiseComparison(model_a="B", model_b="C", outcome="win_b", task_id="t2"),
        ]

        leaderboard, metadata = compute_elo_rankings(comparisons, bootstrap_samples=50, seed=42)

        assert len(leaderboard) == 3
        assert metadata["n_comparisons"] == 2
        assert metadata["n_models"] == 3
        assert "timestamp" in metadata

    def test_confidence_intervals_attached(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id=f"t{i}")
            for i in range(20)
        ]

        leaderboard, _ = compute_elo_rankings(comparisons, bootstrap_samples=50, seed=42)

        for rating in leaderboard:
            assert rating.confidence_lower is not None
            assert rating.confidence_upper is not None


class TestHeadToHeadMatrix:
    """Tests for head-to-head matrix computation."""

    def test_basic_matrix(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t1"),
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t2"),
            PairwiseComparison(model_a="A", model_b="B", outcome="win_b", task_id="t3"),
        ]

        matrix = head_to_head_matrix(comparisons)

        assert matrix["A"]["B"]["wins"] == 2
        assert matrix["A"]["B"]["losses"] == 1
        assert matrix["B"]["A"]["wins"] == 1
        assert matrix["B"]["A"]["losses"] == 2

    def test_ties_tracked(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="X", model_b="Y", outcome="tie", task_id="t1"),
            PairwiseComparison(model_a="X", model_b="Y", outcome="tie", task_id="t2"),
        ]

        matrix = head_to_head_matrix(comparisons)

        assert matrix["X"]["Y"]["ties"] == 2
        assert matrix["Y"]["X"]["ties"] == 2


class TestFileIO:
    """Tests for file I/O utilities."""

    def test_jsonl_roundtrip(self) -> None:
        comparisons = [
            PairwiseComparison(model_a="A", model_b="B", outcome="win_a", task_id="t1"),
            PairwiseComparison(model_a="B", model_b="C", outcome="tie", task_id="t2"),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            save_comparisons_to_jsonl(comparisons, path)
            loaded = load_comparisons_from_jsonl(path)

            assert len(loaded) == 2
            assert loaded[0].model_a == "A"
            assert loaded[1].outcome == "tie"
        finally:
            path.unlink()

    def test_save_leaderboard(self) -> None:
        leaderboard = [
            ELORating(
                model="top_model",
                rating=1600,
                games_played=10,
                wins=7,
                losses=2,
                ties=1,
                confidence_lower=1550,
                confidence_upper=1650,
            ),
        ]
        metadata = {"n_comparisons": 10, "timestamp": "2024-01-01T00:00:00"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_leaderboard(leaderboard, metadata, path)

            with open(path) as f:
                data = json.load(f)

            assert len(data["leaderboard"]) == 1
            assert data["leaderboard"][0]["model"] == "top_model"
            assert data["metadata"]["n_comparisons"] == 10
        finally:
            path.unlink()
