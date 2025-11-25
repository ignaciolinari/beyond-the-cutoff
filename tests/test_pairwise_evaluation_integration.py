"""Integration tests for pairwise evaluation system.

Tests the complete pairwise evaluation pipeline for the 6-condition experiment,
including ELO ranking computation, tournament comparison, and result aggregation.
"""

from __future__ import annotations

from typing import Literal

import pytest

from beyond_the_cutoff.evaluation.elo_ranking import (
    ELORating,
    PairwiseComparison,
    compute_elo_rankings,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def six_condition_models() -> list[str]:
    """Return the 6 condition model identifiers."""
    return [
        "base_baseline",
        "rag_baseline",
        "ft_only_instruction",
        "ft_rag_instruction",
        "ft_only_rag_trained",
        "ft_rag_trained",
    ]


@pytest.fixture
def simulated_tournament_results() -> list[PairwiseComparison]:
    """Create simulated tournament results for all model pairs."""
    # Simulate a tournament where RAG models generally beat non-RAG
    # and fine-tuned models beat baseline
    models = [
        "base_baseline",
        "rag_baseline",
        "ft_only_instruction",
        "ft_rag_instruction",
        "ft_only_rag_trained",
        "ft_rag_trained",
    ]

    # Approximate "strength" ranking for simulation
    strength: dict[str, int] = {
        "base_baseline": 1,
        "ft_only_instruction": 2,
        "ft_only_rag_trained": 3,
        "rag_baseline": 4,
        "ft_rag_instruction": 5,
        "ft_rag_trained": 6,
    }

    comparisons: list[PairwiseComparison] = []
    task_ids = ["q1", "q2", "q3"]

    for task_id in task_ids:
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                # Determine outcome based on strength
                if strength[model_a] > strength[model_b]:
                    outcome: Literal["win_a", "win_b", "tie"] = "win_a"
                elif strength[model_b] > strength[model_a]:
                    outcome = "win_b"
                else:
                    outcome = "tie"

                comparisons.append(
                    PairwiseComparison(
                        model_a=model_a,
                        model_b=model_b,
                        outcome=outcome,
                        task_id=task_id,
                        annotation_source="judge",
                    )
                )

    return comparisons


# =============================================================================
# ELO Ranking Tests
# =============================================================================


class TestELORanking:
    """Tests for ELO ranking computation."""

    def test_elo_rankings_return_all_models(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test that ELO rankings include all models."""
        leaderboard, metadata = compute_elo_rankings(simulated_tournament_results)

        expected_models = {
            "base_baseline",
            "rag_baseline",
            "ft_only_instruction",
            "ft_rag_instruction",
            "ft_only_rag_trained",
            "ft_rag_trained",
        }
        actual_models = {r.model for r in leaderboard}
        assert actual_models == expected_models

    def test_elo_rankings_are_numeric(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test that all ELO ratings are numeric."""
        leaderboard, _ = compute_elo_rankings(simulated_tournament_results)

        for rating in leaderboard:
            assert isinstance(rating, ELORating)
            assert isinstance(rating.rating, int | float)

    def test_winner_has_higher_elo_than_loser(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test that consistent winners have higher ELO than consistent losers."""
        leaderboard, _ = compute_elo_rankings(simulated_tournament_results)

        ratings_by_model = {r.model: r.rating for r in leaderboard}

        # ft_rag_trained should beat base_baseline consistently
        # so should have higher ELO
        assert ratings_by_model["ft_rag_trained"] > ratings_by_model["base_baseline"]

    def test_elo_rankings_ordering(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test that ELO rankings roughly match expected strength ordering."""
        leaderboard, _ = compute_elo_rankings(simulated_tournament_results)

        # Leaderboard is sorted by rating descending
        model_order = [r.model for r in leaderboard]

        # base_baseline should be near bottom, ft_rag_trained near top
        assert model_order.index("base_baseline") > model_order.index("ft_rag_trained")

    def test_elo_with_ties(self) -> None:
        """Test ELO computation handles ties correctly."""
        comparisons = [
            PairwiseComparison(
                model_a="model_a",
                model_b="model_b",
                outcome="tie",
                task_id="q1",
                annotation_source="judge",
            ),
            PairwiseComparison(
                model_a="model_a",
                model_b="model_b",
                outcome="tie",
                task_id="q2",
                annotation_source="judge",
            ),
        ]

        leaderboard, _ = compute_elo_rankings(comparisons)
        ratings_by_model = {r.model: r.rating for r in leaderboard}

        # With all ties, ratings should be very close
        diff = abs(ratings_by_model["model_a"] - ratings_by_model["model_b"])
        assert diff < 50  # Allow some numerical variation


# =============================================================================
# Tournament Structure Tests
# =============================================================================


class TestTournamentStructure:
    """Tests for tournament structure and completeness."""

    def test_all_pairs_compared(
        self,
        six_condition_models: list[str],
        simulated_tournament_results: list[PairwiseComparison],
    ) -> None:
        """Test that all model pairs are compared at least once."""
        compared_pairs: set[frozenset[str]] = set()

        for comp in simulated_tournament_results:
            pair = frozenset([comp.model_a, comp.model_b])
            compared_pairs.add(pair)

        # Calculate expected pairs
        expected_pairs = set()
        for i, model_a in enumerate(six_condition_models):
            for model_b in six_condition_models[i + 1 :]:
                expected_pairs.add(frozenset([model_a, model_b]))

        assert compared_pairs == expected_pairs

    def test_comparison_count_per_pair(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test comparison count per model pair."""
        pair_counts: dict[frozenset[str], int] = {}

        for comp in simulated_tournament_results:
            pair = frozenset([comp.model_a, comp.model_b])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Each pair should have 3 comparisons (one per question)
        for count in pair_counts.values():
            assert count == 3


# =============================================================================
# Result Aggregation Tests
# =============================================================================


class TestResultAggregation:
    """Tests for aggregating pairwise results."""

    def test_leaderboard_from_comparisons(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test creating a leaderboard from comparisons."""
        leaderboard, metadata = compute_elo_rankings(simulated_tournament_results)

        assert len(leaderboard) == 6
        assert metadata["n_comparisons"] == len(simulated_tournament_results)
        assert metadata["n_models"] == 6

    def test_win_rate_calculation(self) -> None:
        """Test calculating win rates from comparisons."""
        comparisons = [
            PairwiseComparison(
                model_a="model_a",
                model_b="model_b",
                outcome="win_a" if i < 7 else "win_b",
                task_id=f"q{i}",
                annotation_source="judge",
            )
            for i in range(10)
        ]

        wins_a = sum(1 for c in comparisons if c.outcome == "win_a")
        wins_b = sum(1 for c in comparisons if c.outcome == "win_b")

        assert wins_a == 7
        assert wins_b == 3

    def test_head_to_head_extraction(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test extracting head-to-head results for a specific pair."""
        # Extract comparisons between base_baseline and ft_rag_trained
        h2h = [
            c
            for c in simulated_tournament_results
            if {c.model_a, c.model_b} == {"base_baseline", "ft_rag_trained"}
        ]

        assert len(h2h) == 3  # One per question
        # ft_rag_trained should win all
        for comp in h2h:
            if comp.model_a == "ft_rag_trained":
                assert comp.outcome == "win_a"
            else:
                assert comp.outcome == "win_b"


# =============================================================================
# Within-Group Comparison Tests
# =============================================================================


class TestWithinGroupComparisons:
    """Tests for within-group comparisons (instruction-only or RAG-augmented)."""

    def test_instruction_only_group(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test filtering to instruction-only group comparisons."""
        instruction_models = {
            "base_baseline",
            "ft_only_instruction",
            "ft_only_rag_trained",
        }

        within_group = [
            c
            for c in simulated_tournament_results
            if c.model_a in instruction_models and c.model_b in instruction_models
        ]

        # 3 pairs × 3 questions = 9 comparisons
        assert len(within_group) == 9

    def test_rag_group(self, simulated_tournament_results: list[PairwiseComparison]) -> None:
        """Test filtering to RAG-augmented group comparisons."""
        rag_models = {"rag_baseline", "ft_rag_instruction", "ft_rag_trained"}

        within_group = [
            c
            for c in simulated_tournament_results
            if c.model_a in rag_models and c.model_b in rag_models
        ]

        # 3 pairs × 3 questions = 9 comparisons
        assert len(within_group) == 9

    def test_within_group_elo(self, simulated_tournament_results: list[PairwiseComparison]) -> None:
        """Test computing ELO within a group only."""
        rag_models = {"rag_baseline", "ft_rag_instruction", "ft_rag_trained"}

        within_group = [
            c
            for c in simulated_tournament_results
            if c.model_a in rag_models and c.model_b in rag_models
        ]

        leaderboard, _ = compute_elo_rankings(within_group)
        models_in_leaderboard = {r.model for r in leaderboard}

        assert models_in_leaderboard == rag_models
        # ft_rag_trained should still be highest
        ratings_by_model = {r.model: r.rating for r in leaderboard}
        assert ratings_by_model["ft_rag_trained"] > ratings_by_model["rag_baseline"]


# =============================================================================
# Cross-Group Comparison Tests
# =============================================================================


class TestCrossGroupComparisons:
    """Tests for cross-group comparisons (instruction vs RAG)."""

    def test_cross_group_extraction(
        self, simulated_tournament_results: list[PairwiseComparison]
    ) -> None:
        """Test extracting cross-group comparisons."""
        instruction_models = {
            "base_baseline",
            "ft_only_instruction",
            "ft_only_rag_trained",
        }
        rag_models = {"rag_baseline", "ft_rag_instruction", "ft_rag_trained"}

        cross_group = [
            c
            for c in simulated_tournament_results
            if (c.model_a in instruction_models and c.model_b in rag_models)
            or (c.model_a in rag_models and c.model_b in instruction_models)
        ]

        # 9 cross-group pairs × 3 questions = 27 comparisons
        assert len(cross_group) == 27


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in pairwise evaluation."""

    def test_empty_comparisons(self) -> None:
        """Test handling empty comparison list."""
        leaderboard, metadata = compute_elo_rankings([])
        assert leaderboard == []
        assert metadata["n_comparisons"] == 0

    def test_single_comparison(self) -> None:
        """Test with just one comparison."""
        comparisons = [
            PairwiseComparison(
                model_a="model_a",
                model_b="model_b",
                outcome="win_a",
                task_id="q1",
                annotation_source="judge",
            )
        ]

        leaderboard, _ = compute_elo_rankings(comparisons)
        models_in_leaderboard = {r.model for r in leaderboard}

        assert "model_a" in models_in_leaderboard
        assert "model_b" in models_in_leaderboard

        ratings_by_model = {r.model: r.rating for r in leaderboard}
        assert ratings_by_model["model_a"] > ratings_by_model["model_b"]

    def test_all_ties(self) -> None:
        """Test tournament where all comparisons are ties."""
        comparisons = [
            PairwiseComparison(
                model_a="model_a",
                model_b="model_b",
                outcome="tie",
                task_id=f"q{i}",
                annotation_source="judge",
            )
            for i in range(5)
        ]

        leaderboard, _ = compute_elo_rankings(comparisons)
        ratings_by_model = {r.model: r.rating for r in leaderboard}

        # Both should have similar ratings
        diff = abs(ratings_by_model["model_a"] - ratings_by_model["model_b"])
        assert diff < 50
