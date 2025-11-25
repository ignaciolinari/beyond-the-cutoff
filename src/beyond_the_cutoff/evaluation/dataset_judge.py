"""LLM-based quality assessment for generated dataset examples.

This module provides semantic validation of dataset examples using an LLM judge
to catch issues that structural validation cannot detect:
- Unanswerable questions (context doesn't support the answer)
- Incorrect gold answers (factual errors in expected_response)
- Unclear or ambiguous instructions
- Mismatched instruction/response pairs

IMPORTANT: Self-Evaluation Bias
-------------------------------
The judge model should be DIFFERENT from the generator model to avoid
"self-preference bias" where a model rates its own outputs more favorably.

Recommended setup:
- Generator: Qwen 2.5 7B (qwen2.5:7b-instruct-q4_K_M)
- Judge: Qwen 3 8B (qwen3:8b) or Llama 3.1 8B (llama3.1:8b)

Using a different model family provides more independent assessment.
For highest quality, consider using a cloud model (Claude, GPT-4) for judging.

Usage:
    from beyond_the_cutoff.evaluation.dataset_judge import DatasetQualityJudge

    judge = DatasetQualityJudge(inference_config)
    results = judge.evaluate_dataset(dataset_path, sample_size=50)

    # Or evaluate a single example
    verdict = judge.evaluate_example(example)
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import InferenceConfig
from ..models import LLMClient, build_generation_client

logger = logging.getLogger(__name__)


@dataclass
class ExampleVerdict:
    """Quality verdict for a single dataset example."""

    task_id: str
    task_type: str
    passed: bool
    scores: dict[str, float]  # answerability, correctness, clarity, coherence
    issues: list[str]
    reasoning: str
    raw_judge_response: str | None = None


@dataclass
class DatasetQualityResult:
    """Aggregated quality assessment for a dataset."""

    total_evaluated: int
    passed_count: int
    failed_count: int
    pass_rate: float
    mean_scores: dict[str, float]
    score_distributions: dict[str, dict[str, int]]  # score -> bucket counts
    common_issues: dict[str, int]  # issue type -> count
    verdicts: list[ExampleVerdict] = field(default_factory=list)
    failed_examples: list[ExampleVerdict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": self.pass_rate,
            "mean_scores": self.mean_scores,
            "score_distributions": self.score_distributions,
            "common_issues": self.common_issues,
            "failed_examples": [
                {
                    "task_id": v.task_id,
                    "task_type": v.task_type,
                    "issues": v.issues,
                    "reasoning": v.reasoning,
                    "scores": v.scores,
                }
                for v in self.failed_examples
            ],
        }


# Judge prompt for evaluating dataset quality
DATASET_QUALITY_JUDGE_PROMPT = """You are evaluating the quality of a training example for a retrieval-augmented QA system.

TASK TYPE: {task_type}

INSTRUCTION (the question/prompt that will be given to the model):
{instruction}

CONTEXT (retrieved passages the model will see):
{contexts}

EXPECTED RESPONSE (the gold answer the model should produce):
{expected_response}

Evaluate this training example on these criteria:

1. **Answerability** (0.0-1.0): Can the instruction be fully answered using ONLY the provided contexts?
   - 1.0: All information needed is clearly present in the contexts
   - 0.5: Partial information available, some inference required
   - 0.0: Context doesn't contain the necessary information

2. **Correctness** (0.0-1.0): Is the expected response factually accurate given the contexts?
   - 1.0: Every claim is directly supported by the contexts
   - 0.5: Mostly correct but contains minor unsupported claims
   - 0.0: Contains factual errors or unsupported major claims

3. **Clarity** (0.0-1.0): Is the instruction clear and unambiguous?
   - 1.0: Clear, specific, well-formed question/instruction
   - 0.5: Understandable but could be more precise
   - 0.0: Vague, confusing, or poorly worded

4. **Coherence** (0.0-1.0): Does the expected response appropriately address the instruction?
   - 1.0: Directly and completely addresses the instruction
   - 0.5: Addresses the instruction but with gaps or tangents
   - 0.0: Off-topic or doesn't match the instruction

PASS CRITERIA: All scores >= 0.6 AND answerability + correctness >= 1.4

Respond with ONLY valid JSON in this format:
{{
  "scores": {{
    "answerability": <float 0-1>,
    "correctness": <float 0-1>,
    "clarity": <float 0-1>,
    "coherence": <float 0-1>
  }},
  "passed": <boolean>,
  "issues": [<list of specific issue descriptions, empty if none>],
  "reasoning": "<brief explanation of scores>"
}}"""


class DatasetQualityJudge:
    """LLM-based judge for evaluating dataset example quality."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        *,
        pass_threshold: float = 0.6,
        critical_threshold: float = 1.4,  # answerability + correctness minimum
    ) -> None:
        self._client = build_generation_client(inference_config)
        self._pass_threshold = pass_threshold
        self._critical_threshold = critical_threshold

    @property
    def client(self) -> LLMClient:
        return self._client

    def evaluate_example(self, example: dict[str, Any]) -> ExampleVerdict:
        """Evaluate a single dataset example."""
        task_id = example.get("task_id", "unknown")
        task_type = example.get("task_type", "unknown")
        instruction = example.get("instruction", "")
        expected_response = example.get("expected_response", "")

        # Extract contexts from RAG field or contexts list
        contexts = self._extract_contexts(example)
        contexts_text = self._format_contexts(contexts)

        prompt = DATASET_QUALITY_JUDGE_PROMPT.format(
            task_type=task_type,
            instruction=instruction,
            contexts=contexts_text,
            expected_response=expected_response,
        )

        try:
            response_dict = self._client.generate(prompt)
            raw_response = str(response_dict.get("response", ""))
            verdict = self._parse_verdict(raw_response, task_id, task_type)
            verdict.raw_judge_response = raw_response
            return verdict
        except Exception as exc:
            logger.warning("Judge evaluation failed for %s: %s", task_id, exc)
            return ExampleVerdict(
                task_id=task_id,
                task_type=task_type,
                passed=False,
                scores={"answerability": 0.0, "correctness": 0.0, "clarity": 0.0, "coherence": 0.0},
                issues=[f"Judge evaluation failed: {exc}"],
                reasoning="Evaluation error",
                raw_judge_response=None,
            )

    def evaluate_dataset(
        self,
        dataset_path: Path,
        *,
        sample_size: int | None = None,
        seed: int = 42,
        task_types: set[str] | None = None,
    ) -> DatasetQualityResult:
        """Evaluate a sample of examples from a dataset.

        Args:
            dataset_path: Path to JSONL dataset file
            sample_size: Number of examples to evaluate (None = all)
            seed: Random seed for sampling
            task_types: Optional filter for specific task types

        Returns:
            DatasetQualityResult with aggregated metrics
        """
        examples = self._load_examples(dataset_path, task_types=task_types)

        if sample_size is not None and len(examples) > sample_size:
            rng = random.Random(seed)
            examples = rng.sample(examples, sample_size)

        logger.info("Evaluating %d dataset examples", len(examples))

        verdicts: list[ExampleVerdict] = []
        for idx, example in enumerate(examples, start=1):
            if idx % 10 == 0:
                logger.info("Progress: %d/%d examples evaluated", idx, len(examples))
            verdict = self.evaluate_example(example)
            verdicts.append(verdict)

        return self._aggregate_results(verdicts)

    def _extract_contexts(self, example: dict[str, Any]) -> list[str]:
        """Extract context strings from various example formats."""
        # Try RAG field first (nested structure)
        rag = example.get("rag", {})
        if isinstance(rag, dict):
            contexts = rag.get("contexts", [])
            if contexts:
                return [str(c) for c in contexts]

        # Try top-level contexts field
        contexts = example.get("contexts", [])
        if contexts:
            return [str(c) for c in contexts]

        # Try citations field
        citations = example.get("citations", [])
        if citations:
            return [
                str(c.get("rendered_context") or c.get("excerpt") or "")
                for c in citations
                if isinstance(c, dict)
            ]

        return []

    def _format_contexts(self, contexts: list[str]) -> str:
        """Format contexts for the judge prompt."""
        if not contexts:
            return "(No contexts provided)"

        formatted = []
        for idx, ctx in enumerate(contexts, start=1):
            truncated = ctx[:1500] + "..." if len(ctx) > 1500 else ctx
            formatted.append(f"[{idx}] {truncated}")
        return "\n\n".join(formatted)

    def _parse_verdict(self, raw_response: str, task_id: str, task_type: str) -> ExampleVerdict:
        """Parse judge response into ExampleVerdict."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        if not json_match:
            return ExampleVerdict(
                task_id=task_id,
                task_type=task_type,
                passed=False,
                scores={"answerability": 0.0, "correctness": 0.0, "clarity": 0.0, "coherence": 0.0},
                issues=["Judge response not parseable as JSON"],
                reasoning=raw_response[:200],
            )

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            return ExampleVerdict(
                task_id=task_id,
                task_type=task_type,
                passed=False,
                scores={"answerability": 0.0, "correctness": 0.0, "clarity": 0.0, "coherence": 0.0},
                issues=[f"JSON parse error: {exc}"],
                reasoning=raw_response[:200],
            )

        scores = parsed.get("scores", {})
        # Ensure all scores are floats
        clean_scores = {
            "answerability": float(scores.get("answerability", 0.0)),
            "correctness": float(scores.get("correctness", 0.0)),
            "clarity": float(scores.get("clarity", 0.0)),
            "coherence": float(scores.get("coherence", 0.0)),
        }

        # Determine pass/fail based on thresholds
        all_above_threshold = all(s >= self._pass_threshold for s in clean_scores.values())
        critical_sum = clean_scores["answerability"] + clean_scores["correctness"]
        passed = all_above_threshold and critical_sum >= self._critical_threshold

        # Override with judge's verdict if provided
        judge_passed = parsed.get("passed")
        if isinstance(judge_passed, bool):
            passed = judge_passed

        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []

        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return ExampleVerdict(
            task_id=task_id,
            task_type=task_type,
            passed=passed,
            scores=clean_scores,
            issues=issues,
            reasoning=reasoning,
        )

    def _load_examples(
        self, dataset_path: Path, *, task_types: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Load examples from JSONL file."""
        examples: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                    if task_types is None or example.get("task_type") in task_types:
                        examples.append(example)
                except json.JSONDecodeError:
                    continue
        return examples

    def _aggregate_results(self, verdicts: list[ExampleVerdict]) -> DatasetQualityResult:
        """Aggregate individual verdicts into summary statistics."""
        if not verdicts:
            return DatasetQualityResult(
                total_evaluated=0,
                passed_count=0,
                failed_count=0,
                pass_rate=0.0,
                mean_scores={},
                score_distributions={},
                common_issues={},
                verdicts=[],
                failed_examples=[],
            )

        passed_count = sum(1 for v in verdicts if v.passed)
        failed_count = len(verdicts) - passed_count

        # Calculate mean scores
        score_sums: dict[str, float] = {}
        score_counts: dict[str, int] = {}
        for verdict in verdicts:
            for key, value in verdict.scores.items():
                score_sums[key] = score_sums.get(key, 0.0) + value
                score_counts[key] = score_counts.get(key, 0) + 1

        mean_scores = {
            key: score_sums[key] / score_counts[key] for key in score_sums if score_counts[key] > 0
        }

        # Calculate score distributions (buckets: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        score_distributions: dict[str, dict[str, int]] = {}
        for key in ["answerability", "correctness", "clarity", "coherence"]:
            buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
            for verdict in verdicts:
                score = verdict.scores.get(key, 0.0)
                if score < 0.2:
                    buckets["0.0-0.2"] += 1
                elif score < 0.4:
                    buckets["0.2-0.4"] += 1
                elif score < 0.6:
                    buckets["0.4-0.6"] += 1
                elif score < 0.8:
                    buckets["0.6-0.8"] += 1
                else:
                    buckets["0.8-1.0"] += 1
            score_distributions[key] = buckets

        # Count common issues
        issue_counts: dict[str, int] = {}
        for verdict in verdicts:
            for issue in verdict.issues:
                # Normalize issue text for grouping
                normalized = issue.lower().strip()[:100]
                issue_counts[normalized] = issue_counts.get(normalized, 0) + 1

        # Sort by frequency, keep top 20
        common_issues = dict(sorted(issue_counts.items(), key=lambda x: -x[1])[:20])

        failed_examples = [v for v in verdicts if not v.passed]

        return DatasetQualityResult(
            total_evaluated=len(verdicts),
            passed_count=passed_count,
            failed_count=failed_count,
            pass_rate=passed_count / len(verdicts) if verdicts else 0.0,
            mean_scores=mean_scores,
            score_distributions=score_distributions,
            common_issues=common_issues,
            verdicts=verdicts,
            failed_examples=failed_examples,
        )


__all__ = [
    "DatasetQualityJudge",
    "ExampleVerdict",
    "DatasetQualityResult",
    "DATASET_QUALITY_JUDGE_PROMPT",
]
