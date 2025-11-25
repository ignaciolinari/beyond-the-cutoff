"""ELO-based ranking system for pairwise model comparisons.

This module implements an ELO rating system for comparing language model outputs.
It supports:
- Pairwise comparisons (head-to-head matchups)
- Multiple comparison sources (human annotators, judge models)
- Bootstrap confidence intervals
- Rating history tracking
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Type alias for comparison outcomes
Outcome = Literal["win_a", "win_b", "tie"]


@dataclass
class PairwiseComparison:
    """A single pairwise comparison between two models."""

    model_a: str
    model_b: str
    outcome: Outcome  # "win_a", "win_b", or "tie"
    task_id: str
    question: str | None = None
    response_a: str | None = None
    response_b: str | None = None
    annotator: str = "unknown"  # human annotator ID or judge model name
    annotation_source: Literal["human", "judge"] = "judge"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "outcome": self.outcome,
            "task_id": self.task_id,
            "question": self.question,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "annotator": self.annotator,
            "annotation_source": self.annotation_source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairwiseComparison:
        return cls(
            model_a=data["model_a"],
            model_b=data["model_b"],
            outcome=data["outcome"],
            task_id=data["task_id"],
            question=data.get("question"),
            response_a=data.get("response_a"),
            response_b=data.get("response_b"),
            annotator=data.get("annotator", "unknown"),
            annotation_source=data.get("annotation_source", "judge"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ELORating:
    """ELO rating for a single model."""

    model: str
    rating: float
    games_played: int
    wins: int
    losses: int
    ties: int
    confidence_lower: float | None = None
    confidence_upper: float | None = None

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.games_played

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "rating": round(self.rating, 1),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": round(self.win_rate, 3),
            "confidence_lower": round(self.confidence_lower, 1) if self.confidence_lower else None,
            "confidence_upper": round(self.confidence_upper, 1) if self.confidence_upper else None,
        }


class ELOCalculator:
    """ELO rating calculator with configurable K-factor and starting rating.

    The ELO system works by:
    1. Computing expected win probability based on rating difference
    2. Updating ratings based on actual vs expected outcome
    3. K-factor controls how much ratings change per game

    For ties, we use 0.5 as the score (between 0 for loss and 1 for win).
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
        tie_value: float = 0.5,
    ):
        """Initialize the ELO calculator.

        Args:
            k_factor: How much ratings change per game. Higher = more volatile.
                      32 is standard for new players, 16 for established.
            initial_rating: Starting rating for new models.
            tie_value: Score value for ties (0.5 = split the difference).
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.tie_value = tie_value
        self._ratings: dict[str, float] = {}
        self._stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"games": 0, "wins": 0, "losses": 0, "ties": 0}
        )
        self._history: list[dict[str, Any]] = []

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B.

        Uses the standard ELO formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def get_rating(self, model: str) -> float:
        """Get current rating for a model, initializing if needed."""
        if model not in self._ratings:
            self._ratings[model] = self.initial_rating
        return self._ratings[model]

    def update_ratings(self, comparison: PairwiseComparison) -> tuple[float, float]:
        """Update ratings based on a comparison result.

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        model_a = comparison.model_a
        model_b = comparison.model_b

        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        # Determine actual scores
        if comparison.outcome == "win_a":
            score_a, score_b = 1.0, 0.0
            self._stats[model_a]["wins"] += 1
            self._stats[model_b]["losses"] += 1
        elif comparison.outcome == "win_b":
            score_a, score_b = 0.0, 1.0
            self._stats[model_a]["losses"] += 1
            self._stats[model_b]["wins"] += 1
        else:  # tie
            score_a, score_b = self.tie_value, self.tie_value
            self._stats[model_a]["ties"] += 1
            self._stats[model_b]["ties"] += 1

        self._stats[model_a]["games"] += 1
        self._stats[model_b]["games"] += 1

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self._ratings[model_a] = new_rating_a
        self._ratings[model_b] = new_rating_b

        # Record history
        self._history.append(
            {
                "comparison": comparison.as_dict(),
                "rating_before": {model_a: rating_a, model_b: rating_b},
                "rating_after": {model_a: new_rating_a, model_b: new_rating_b},
            }
        )

        return new_rating_a, new_rating_b

    def process_comparisons(
        self,
        comparisons: list[PairwiseComparison],
        shuffle: bool = True,
    ) -> None:
        """Process a batch of comparisons.

        Args:
            comparisons: List of pairwise comparisons to process.
            shuffle: Whether to randomize order (recommended for fairness).
        """
        to_process = list(comparisons)
        if shuffle:
            random.shuffle(to_process)

        for comparison in to_process:
            self.update_ratings(comparison)

    def get_leaderboard(self) -> list[ELORating]:
        """Get sorted leaderboard of all models."""
        leaderboard = []
        for model, rating in self._ratings.items():
            stats = self._stats[model]
            leaderboard.append(
                ELORating(
                    model=model,
                    rating=rating,
                    games_played=stats["games"],
                    wins=stats["wins"],
                    losses=stats["losses"],
                    ties=stats["ties"],
                )
            )
        return sorted(leaderboard, key=lambda x: x.rating, reverse=True)

    def reset(self) -> None:
        """Reset all ratings and statistics."""
        self._ratings.clear()
        self._stats.clear()
        self._history.clear()

    def get_state(self) -> dict[str, Any]:
        """Get the full state for serialization."""
        return {
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating,
            "tie_value": self.tie_value,
            "ratings": dict(self._ratings),
            "stats": {k: dict(v) for k, v in self._stats.items()},
            "history": self._history,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from a serialized dict."""
        self.k_factor = state.get("k_factor", self.k_factor)
        self.initial_rating = state.get("initial_rating", self.initial_rating)
        self.tie_value = state.get("tie_value", self.tie_value)
        self._ratings = state.get("ratings", {})
        self._stats = defaultdict(
            lambda: {"games": 0, "wins": 0, "losses": 0, "ties": 0},
            {k: dict(v) for k, v in state.get("stats", {}).items()},
        )
        self._history = state.get("history", [])


def bootstrap_elo_confidence(
    comparisons: list[PairwiseComparison],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
    seed: int | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for ELO ratings.

    Args:
        comparisons: List of pairwise comparisons.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        k_factor: ELO K-factor.
        initial_rating: Starting rating.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping model name to (lower, mean, upper) rating tuple.
    """
    if seed is not None:
        random.seed(seed)

    n_comparisons = len(comparisons)
    if n_comparisons == 0:
        return {}

    # Collect bootstrap samples
    model_ratings: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = [comparisons[random.randint(0, n_comparisons - 1)] for _ in range(n_comparisons)]

        calculator = ELOCalculator(
            k_factor=k_factor,
            initial_rating=initial_rating,
        )
        calculator.process_comparisons(sample, shuffle=True)

        for model, rating in calculator._ratings.items():
            model_ratings[model].append(rating)

    # Compute confidence intervals
    alpha = 1.0 - confidence
    results: dict[str, tuple[float, float, float]] = {}

    for model, ratings in model_ratings.items():
        sorted_ratings = sorted(ratings)
        n = len(sorted_ratings)
        lower_idx = int(alpha / 2 * n)
        upper_idx = int((1 - alpha / 2) * n) - 1

        lower = sorted_ratings[lower_idx]
        upper = sorted_ratings[upper_idx]
        mean = sum(ratings) / n

        results[model] = (lower, mean, upper)

    return results


def compute_elo_rankings(
    comparisons: list[PairwiseComparison],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> tuple[list[ELORating], dict[str, Any]]:
    """Compute ELO rankings with confidence intervals.

    Args:
        comparisons: List of pairwise comparisons.
        k_factor: ELO K-factor.
        initial_rating: Starting rating.
        bootstrap_samples: Number of bootstrap samples for CI.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (leaderboard, metadata dict).
    """
    if seed is not None:
        random.seed(seed)

    # Compute main ratings
    calculator = ELOCalculator(k_factor=k_factor, initial_rating=initial_rating)
    calculator.process_comparisons(comparisons, shuffle=True)
    leaderboard = calculator.get_leaderboard()

    # Compute bootstrap confidence intervals
    ci_results = bootstrap_elo_confidence(
        comparisons,
        n_bootstrap=bootstrap_samples,
        confidence=confidence,
        k_factor=k_factor,
        initial_rating=initial_rating,
        seed=seed,
    )

    # Merge CI into leaderboard
    for rating in leaderboard:
        if rating.model in ci_results:
            lower, _, upper = ci_results[rating.model]
            rating.confidence_lower = lower
            rating.confidence_upper = upper

    metadata = {
        "n_comparisons": len(comparisons),
        "n_models": len(leaderboard),
        "k_factor": k_factor,
        "initial_rating": initial_rating,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence,
        "timestamp": datetime.now().isoformat(),
    }

    return leaderboard, metadata


def head_to_head_matrix(
    comparisons: list[PairwiseComparison],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute head-to-head win/loss/tie matrix.

    Returns:
        Nested dict: matrix[model_a][model_b] = {"wins": N, "losses": N, "ties": N}
    """
    matrix: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    )

    for comp in comparisons:
        if comp.outcome == "win_a":
            matrix[comp.model_a][comp.model_b]["wins"] += 1
            matrix[comp.model_b][comp.model_a]["losses"] += 1
        elif comp.outcome == "win_b":
            matrix[comp.model_a][comp.model_b]["losses"] += 1
            matrix[comp.model_b][comp.model_a]["wins"] += 1
        else:
            matrix[comp.model_a][comp.model_b]["ties"] += 1
            matrix[comp.model_b][comp.model_a]["ties"] += 1

    return dict(matrix)


def load_comparisons_from_jsonl(path: Path) -> list[PairwiseComparison]:
    """Load comparisons from a JSONL file."""
    comparisons = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                comparisons.append(PairwiseComparison.from_dict(data))
    return comparisons


def save_comparisons_to_jsonl(
    comparisons: list[PairwiseComparison],
    path: Path,
) -> None:
    """Save comparisons to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for comp in comparisons:
            f.write(json.dumps(comp.as_dict(), ensure_ascii=False) + "\n")


def save_leaderboard(
    leaderboard: list[ELORating],
    metadata: dict[str, Any],
    path: Path,
) -> None:
    """Save leaderboard to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "leaderboard": [r.as_dict() for r in leaderboard],
        "metadata": metadata,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


__all__ = [
    "PairwiseComparison",
    "ELORating",
    "ELOCalculator",
    "bootstrap_elo_confidence",
    "compute_elo_rankings",
    "head_to_head_matrix",
    "load_comparisons_from_jsonl",
    "save_comparisons_to_jsonl",
    "save_leaderboard",
]
