"""Automated pairwise comparison using judge models.

This module enables automated ELO ranking by having judge models compare
pairs of model outputs head-to-head. Supports multiple judges for consensus.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from beyond_the_cutoff.config import InferenceConfig
from beyond_the_cutoff.evaluation.elo_ranking import (
    Outcome,
    PairwiseComparison,
    compute_elo_rankings,
    save_comparisons_to_jsonl,
    save_leaderboard,
)
from beyond_the_cutoff.models import build_generation_client

logger = logging.getLogger(__name__)


# =============================================================================
# Pairwise Judge Configuration
# =============================================================================

PAIRWISE_JUDGE_PROMPT = """You are an expert evaluator comparing two AI assistant responses to a scientific question.

QUESTION:
{question}

{context_section}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

{reference_section}

EVALUATION CRITERIA:
1. **Factual Accuracy**: Which response contains more accurate information supported by evidence?
2. **Completeness**: Which response more fully addresses the question?
3. **Citation Quality**: Which response better uses citations to support claims? (if contexts provided)
4. **Clarity**: Which response is clearer and better organized?

INSTRUCTIONS:
- Compare the two responses carefully against the question and any provided context.
- Consider all criteria but prioritize factual accuracy.
- If one response is clearly better, choose it.
- If responses are roughly equal in quality, declare a tie.
- Respond ONLY with valid JSON matching the format below.

OUTPUT FORMAT:
{{
    "verdict": "A" | "B" | "tie",
    "reasoning": "Brief explanation of your choice (2-3 sentences)",
    "confidence": "high" | "medium" | "low",
    "scores": {{
        "response_a": {{
            "factuality": <0-1>,
            "completeness": <0-1>,
            "clarity": <0-1>
        }},
        "response_b": {{
            "factuality": <0-1>,
            "completeness": <0-1>,
            "clarity": <0-1>
        }}
    }}
}}

Respond with JSON only, no additional text."""


@dataclass
class PairwiseJudgeConfig:
    """Configuration for a pairwise judge."""

    name: str
    inference: InferenceConfig
    prompt_template: str = PAIRWISE_JUDGE_PROMPT
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: Path) -> PairwiseJudgeConfig:
        """Load judge config from YAML file."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        inference_data = data.get("inference", {})
        if not inference_data:
            raise ValueError(f"Missing 'inference' section in {path}")

        return cls(
            name=data.get("name", path.stem),
            inference=InferenceConfig.model_validate(inference_data),
            prompt_template=data.get("prompt_template", PAIRWISE_JUDGE_PROMPT),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            timeout=data.get("timeout", 120.0),
        )


@dataclass
class PairwiseJudgment:
    """Result of a single pairwise comparison by a judge."""

    judge_name: str
    verdict: Outcome
    reasoning: str
    confidence: Literal["high", "medium", "low"]
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    raw_response: str
    latency_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "raw_response": self.raw_response,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }


# =============================================================================
# Pairwise Judge Implementation
# =============================================================================


class PairwiseJudge:
    """A judge that compares pairs of model responses."""

    def __init__(self, config: PairwiseJudgeConfig):
        self.config = config
        self.client = build_generation_client(config.inference)
        self._call_count = 0

    def compare(
        self,
        question: str,
        response_a: str,
        response_b: str,
        contexts: list[str] | None = None,
        reference: str | None = None,
    ) -> PairwiseJudgment:
        """Compare two responses and return a judgment.

        Args:
            question: The question being answered.
            response_a: First model's response.
            response_b: Second model's response.
            contexts: Optional retrieved contexts (for RAG evaluation).
            reference: Optional reference/gold answer.

        Returns:
            PairwiseJudgment with verdict and analysis.
        """
        # Build prompt
        context_section = ""
        if contexts:
            formatted_contexts = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
            context_section = f"RETRIEVED CONTEXTS:\n{formatted_contexts}\n"

        reference_section = ""
        if reference:
            reference_section = f"REFERENCE ANSWER:\n{reference}\n"

        prompt = self.config.prompt_template.format(
            question=question,
            response_a=response_a,
            response_b=response_b,
            context_section=context_section,
            reference_section=reference_section,
        )

        # Call judge with retries
        start_time = time.time()
        raw_response = ""
        error = None

        for attempt in range(self.config.max_retries):
            try:
                result = self.client.generate(prompt)
                raw_response = result.get("response", "")
                self._call_count += 1
                break
            except Exception as e:
                error = str(e)
                logger.warning(f"Judge {self.config.name} attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        latency = time.time() - start_time

        # Parse response
        verdict, reasoning, confidence, scores_a, scores_b = self._parse_response(raw_response)

        return PairwiseJudgment(
            judge_name=self.config.name,
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            scores_a=scores_a,
            scores_b=scores_b,
            raw_response=raw_response,
            latency_seconds=latency,
            error=error,
        )

    def _parse_response(
        self, response: str
    ) -> tuple[Outcome, str, Literal["high", "medium", "low"], dict[str, Any], dict[str, Any]]:
        """Parse judge response into structured data."""
        verdict: Outcome = "tie"
        reasoning = ""
        confidence: Literal["high", "medium", "low"] = "medium"
        scores_a: dict[str, float] = {}
        scores_b: dict[str, float] = {}

        # Try to extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())

                raw_verdict = str(data.get("verdict", "tie")).upper()
                if raw_verdict == "A":
                    verdict = "win_a"
                elif raw_verdict == "B":
                    verdict = "win_b"
                else:
                    verdict = "tie"

                reasoning = str(data.get("reasoning", ""))

                raw_confidence = str(data.get("confidence", "medium")).lower()
                if raw_confidence in ("high", "medium", "low"):
                    confidence = raw_confidence  # type: ignore[assignment]

                scores = data.get("scores", {})
                scores_a = scores.get("response_a", {})
                scores_b = scores.get("response_b", {})

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse judge response: {e}")
            # Fall back to keyword detection
            response_lower = response.lower()
            if "response a" in response_lower and "better" in response_lower:
                verdict = "win_a"
            elif "response b" in response_lower and "better" in response_lower:
                verdict = "win_b"

        return verdict, reasoning, confidence, scores_a, scores_b


# =============================================================================
# Multi-Judge Orchestration
# =============================================================================


@dataclass
class MultiJudgeResult:
    """Result from multiple judges evaluating the same pair."""

    task_id: str
    model_a: str
    model_b: str
    judgments: list[PairwiseJudgment]
    consensus_verdict: Outcome
    agreement_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "judgments": [j.as_dict() for j in self.judgments],
            "consensus_verdict": self.consensus_verdict,
            "agreement_rate": self.agreement_rate,
        }


def compute_consensus(judgments: list[PairwiseJudgment]) -> tuple[Outcome, float]:
    """Compute consensus verdict from multiple judges.

    Uses weighted voting based on confidence levels.
    """
    if not judgments:
        return "tie", 0.0

    confidence_weights = {"high": 3, "medium": 2, "low": 1}

    votes: dict[Outcome, float] = defaultdict(float)
    for j in judgments:
        weight = confidence_weights.get(j.confidence, 2)
        votes[j.verdict] += weight

    total_weight = sum(votes.values())
    consensus = max(votes.keys(), key=lambda k: votes[k])
    agreement = votes[consensus] / total_weight if total_weight > 0 else 0.0

    return consensus, agreement


class MultiJudgeEvaluator:
    """Orchestrates multiple judges for pairwise comparison."""

    def __init__(self, judges: list[PairwiseJudge]):
        self.judges = judges

    @classmethod
    def from_configs(cls, configs: list[PairwiseJudgeConfig]) -> MultiJudgeEvaluator:
        """Create evaluator from list of judge configs."""
        judges = [PairwiseJudge(cfg) for cfg in configs]
        return cls(judges)

    @classmethod
    def from_yaml_files(cls, paths: list[Path]) -> MultiJudgeEvaluator:
        """Create evaluator from YAML config files."""
        configs = [PairwiseJudgeConfig.from_yaml(p) for p in paths]
        return cls.from_configs(configs)

    def evaluate_pair(
        self,
        task_id: str,
        model_a: str,
        model_b: str,
        question: str,
        response_a: str,
        response_b: str,
        contexts: list[str] | None = None,
        reference: str | None = None,
    ) -> MultiJudgeResult:
        """Have all judges evaluate a single pair."""
        judgments = []

        for judge in self.judges:
            judgment = judge.compare(
                question=question,
                response_a=response_a,
                response_b=response_b,
                contexts=contexts,
                reference=reference,
            )
            judgments.append(judgment)

        consensus, agreement = compute_consensus(judgments)

        return MultiJudgeResult(
            task_id=task_id,
            model_a=model_a,
            model_b=model_b,
            judgments=judgments,
            consensus_verdict=consensus,
            agreement_rate=agreement,
        )


# =============================================================================
# Batch Processing
# =============================================================================


@dataclass
class PairwiseEvaluationConfig:
    """Configuration for batch pairwise evaluation."""

    judge_configs: list[Path]
    output_dir: Path
    comparisons_per_pair: int = 50
    randomize_order: bool = True
    seed: int | None = 42
    save_intermediate: bool = True
    k_factor: float = 32.0
    bootstrap_samples: int = 1000


def run_pairwise_evaluation(
    model_predictions: dict[str, list[dict[str, Any]]],
    config: PairwiseEvaluationConfig,
    model_pairs: list[tuple[str, str]] | None = None,
) -> tuple[list[PairwiseComparison], dict[str, Any]]:
    """Run batch pairwise evaluation across models.

    Args:
        model_predictions: Dict mapping model name to list of predictions.
                          Each prediction should have: task_id, question, model_answer,
                          and optionally: contexts, reference.
        config: Evaluation configuration.
        model_pairs: Specific pairs to compare, or None for all pairs.

    Returns:
        Tuple of (list of comparisons, metadata dict).
    """
    if config.seed is not None:
        random.seed(config.seed)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize judges
    logger.info(f"Loading {len(config.judge_configs)} judges...")
    evaluator = MultiJudgeEvaluator.from_yaml_files(config.judge_configs)

    # Get model pairs
    models = list(model_predictions.keys())
    if model_pairs is None:
        model_pairs = [
            (models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))
        ]

    logger.info(f"Evaluating {len(model_pairs)} model pairs...")

    # Find common tasks
    all_comparisons: list[PairwiseComparison] = []
    results_by_pair: dict[tuple[str, str], list[MultiJudgeResult]] = {}

    for model_a, model_b in model_pairs:
        preds_a = {p["task_id"]: p for p in model_predictions.get(model_a, [])}
        preds_b = {p["task_id"]: p for p in model_predictions.get(model_b, [])}

        common_tasks = list(set(preds_a.keys()) & set(preds_b.keys()))

        if not common_tasks:
            logger.warning(f"No common tasks between {model_a} and {model_b}")
            continue

        # Sample tasks
        n_tasks = min(config.comparisons_per_pair, len(common_tasks))
        selected_tasks = random.sample(common_tasks, n_tasks)

        logger.info(f"Comparing {model_a} vs {model_b} on {n_tasks} tasks...")

        pair_results = []
        for i, task_id in enumerate(selected_tasks):
            pred_a = preds_a[task_id]
            pred_b = preds_b[task_id]

            # Randomize presentation order
            if config.randomize_order and random.random() < 0.5:
                # Swap order
                shown_a, shown_b = model_b, model_a
                resp_a = pred_b.get("model_answer", "")
                resp_b = pred_a.get("model_answer", "")
                swapped = True
            else:
                shown_a, shown_b = model_a, model_b
                resp_a = pred_a.get("model_answer", "")
                resp_b = pred_b.get("model_answer", "")
                swapped = False

            # Get question and context
            question = pred_a.get("question", pred_a.get("prompt", ""))
            contexts = pred_a.get("contexts", [])
            reference = pred_a.get("reference", pred_a.get("expected_answer"))

            # Evaluate
            result = evaluator.evaluate_pair(
                task_id=task_id,
                model_a=shown_a,
                model_b=shown_b,
                question=question,
                response_a=resp_a,
                response_b=resp_b,
                contexts=contexts if contexts else None,
                reference=reference,
            )

            pair_results.append(result)

            # Create comparison (unswap verdict if needed)
            verdict = result.consensus_verdict
            if swapped:
                if verdict == "win_a":
                    verdict = "win_b"
                elif verdict == "win_b":
                    verdict = "win_a"

            comparison = PairwiseComparison(
                model_a=model_a,
                model_b=model_b,
                outcome=verdict,
                task_id=task_id,
                question=question,
                response_a=pred_a.get("model_answer", ""),
                response_b=pred_b.get("model_answer", ""),
                annotator=",".join(j.judge_name for j in result.judgments),
                annotation_source="judge",
                metadata={
                    "judgments": [j.as_dict() for j in result.judgments],
                    "agreement_rate": result.agreement_rate,
                    "swapped": swapped,
                },
            )
            all_comparisons.append(comparison)

            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{n_tasks} comparisons")

        results_by_pair[(model_a, model_b)] = pair_results

        # Save intermediate results
        if config.save_intermediate:
            pair_file = config.output_dir / f"comparisons_{model_a}_vs_{model_b}.jsonl"
            pair_comparisons = [
                c for c in all_comparisons if (c.model_a, c.model_b) == (model_a, model_b)
            ]
            save_comparisons_to_jsonl(pair_comparisons, pair_file)

    # Save all comparisons
    all_comparisons_file = config.output_dir / "all_comparisons.jsonl"
    save_comparisons_to_jsonl(all_comparisons, all_comparisons_file)
    logger.info(f"Saved {len(all_comparisons)} comparisons to {all_comparisons_file}")

    # Compute ELO rankings
    logger.info("Computing ELO rankings...")
    leaderboard, elo_metadata = compute_elo_rankings(
        all_comparisons,
        k_factor=config.k_factor,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )

    # Save leaderboard
    leaderboard_file = config.output_dir / "leaderboard.json"
    save_leaderboard(leaderboard, elo_metadata, leaderboard_file)
    logger.info(f"Saved leaderboard to {leaderboard_file}")

    # Build metadata
    metadata = {
        "n_models": len(models),
        "n_pairs": len(model_pairs),
        "n_comparisons": len(all_comparisons),
        "judges": [cfg.stem for cfg in config.judge_configs],
        "comparisons_per_pair": config.comparisons_per_pair,
        "elo_metadata": elo_metadata,
        "timestamp": datetime.now().isoformat(),
    }

    # Save metadata
    metadata_file = config.output_dir / "evaluation_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return all_comparisons, metadata


def load_predictions_from_results(
    result_dirs: dict[str, Path],
) -> dict[str, list[dict[str, Any]]]:
    """Load predictions from evaluation result directories.

    Args:
        result_dirs: Dict mapping model name to result directory path.
                    Each directory should contain a details.jsonl file.

    Returns:
        Dict mapping model name to list of predictions.
    """
    predictions = {}

    for model_name, result_dir in result_dirs.items():
        details_file = result_dir / "details.jsonl"
        if not details_file.exists():
            logger.warning(f"Details file not found: {details_file}")
            continue

        model_preds = []
        with open(details_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    model_preds.append(json.loads(line))

        predictions[model_name] = model_preds
        logger.info(f"Loaded {len(model_preds)} predictions for {model_name}")

    return predictions


__all__ = [
    "PAIRWISE_JUDGE_PROMPT",
    "PairwiseJudgeConfig",
    "PairwiseJudgment",
    "PairwiseJudge",
    "MultiJudgeResult",
    "MultiJudgeEvaluator",
    "PairwiseEvaluationConfig",
    "run_pairwise_evaluation",
    "load_predictions_from_results",
    "compute_consensus",
]
