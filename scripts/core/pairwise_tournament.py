#!/usr/bin/env python3
"""Pairwise Tournament: Head-to-head comparison between two models using LLM judge.

This script compares responses from two models on the same questions and has
an LLM judge decide which response is better (or if it's a tie).

The script then computes ELO ratings based on the pairwise outcomes.

Usage:
    python scripts/core/pairwise_tournament.py \
        --model-a rag_baseline_0p5b \
        --model-b hybrid_science_0p5b_rag_trained \
        --output evaluation/results/tournament/base_vs_ft_rag.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MatchResult:
    """Result of a single head-to-head match."""

    task_id: str
    question: str
    winner: str  # "A", "B", or "tie"
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    judge_reasoning: str
    presentation_order: str  # "A_first" or "B_first" (for bias detection)
    raw_judge_response: str


@dataclass
class TournamentResults:
    """Aggregated tournament results."""

    model_a: str
    model_b: str
    total_matches: int
    model_a_wins: int
    model_b_wins: int
    ties: int
    model_a_elo: float
    model_b_elo: float
    matches: list[dict[str, Any]]
    position_bias: dict[str, Any] | None = None
    statistical_significance: dict[str, Any] | None = None


def load_responses(results_dir: Path, condition: str) -> dict[str, dict[str, Any]]:
    """Load all responses for a condition, keyed by task_id.

    Supports two formats:
    1. Direct .jsonl file: results_dir/{condition}.jsonl
    2. Subdirectory: results_dir/{condition}/details.jsonl
    """
    responses: dict[str, dict[str, Any]] = {}

    # Try direct .jsonl file first (evaluation/responses/*.jsonl format)
    direct_file = results_dir / f"{condition}.jsonl"
    if direct_file.exists():
        with open(direct_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record.get("task_id", record.get("id", ""))
                    if task_id:
                        # Normalize field names
                        if "response" in record and "model_answer" not in record:
                            record["model_answer"] = record["response"]
                        responses[task_id] = record
        return responses

    # Fallback to subdirectory format (evaluation/results/interleaved/{condition}/details.jsonl)
    details_file = results_dir / condition / "details.jsonl"
    if details_file.exists():
        with open(details_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record.get("task_id", record.get("id", ""))
                    if task_id:
                        if "response" in record and "model_answer" not in record:
                            record["model_answer"] = record["response"]
                        responses[task_id] = record
        return responses

    raise FileNotFoundError(
        f"Could not find responses for {condition} in {results_dir}. "
        f"Tried: {direct_file} and {details_file}"
    )


def create_pairwise_prompt(
    question: str, contexts: str, expected: str, response_1: str, response_2: str
) -> str:
    """Create the prompt for pairwise comparison."""
    return f"""You are an expert scientific evaluator. Your task is to compare two responses to the same question and decide which one is better.

--- QUESTION ---
{question}

--- REFERENCE CONTEXTS (from recent papers) ---
{contexts}

--- EXPECTED ANSWER (ground truth) ---
{expected}

--- RESPONSE A ---
{response_1}

--- RESPONSE B ---
{response_2}

--- EVALUATION CRITERIA ---
Compare the two responses based on:
1. **Factual Accuracy**: Which response better matches the expected answer and avoids errors?
2. **Completeness**: Which response covers more key points from the expected answer?
3. **Grounding**: Which response better uses the provided contexts with proper citations?
4. **Clarity**: Which response is clearer and better organized?

--- INSTRUCTIONS ---
You must decide which response is better overall, or if they are roughly equal (tie).

Think step by step, then provide your verdict.

Respond with ONLY valid JSON in this exact format:
{{
  "reasoning": "Step-by-step comparison of the two responses...",
  "verdict": "A" | "B" | "tie"
}}

Do not include any text outside the JSON object."""


def call_judge(prompt: str, config: dict[str, Any], max_retries: int = 3) -> tuple[str, str, str]:
    """Call the LLM judge and return (verdict, reasoning, raw_response)."""
    import httpx

    host = config.get("host", "http://localhost")
    port = config.get("port", 11434)
    model = config.get("model", "qwen3:8b")
    timeout = config.get("timeout", 180.0)
    temperature = config.get("temperature", 0.6)

    url = f"{host}:{port}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": config.get("top_p", 0.95),
            "num_predict": config.get("max_new_tokens", 1024),
        },
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                raw_response = result.get("response", "")

                # Parse the JSON from the response
                # Handle potential <think>...</think> tags from Qwen3
                clean_response = raw_response
                if "<think>" in clean_response:
                    # Extract content after </think>
                    parts = clean_response.split("</think>")
                    if len(parts) > 1:
                        clean_response = parts[-1].strip()

                # Try to find JSON in the response
                import re

                json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', clean_response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    verdict = parsed.get("verdict", "tie").upper()
                    if verdict not in ["A", "B", "TIE"]:
                        verdict = "tie"
                    reasoning = parsed.get("reasoning", "")
                    return verdict.lower() if verdict == "TIE" else verdict, reasoning, raw_response
                else:
                    # Fallback: look for A, B, or tie in the response
                    if "verdict" in clean_response.lower():
                        if '"a"' in clean_response.lower() or "'a'" in clean_response.lower():
                            return (
                                "A",
                                "Failed to parse JSON, inferred A from response",
                                raw_response,
                            )
                        elif '"b"' in clean_response.lower() or "'b'" in clean_response.lower():
                            return (
                                "B",
                                "Failed to parse JSON, inferred B from response",
                                raw_response,
                            )
                    return "tie", "Failed to parse verdict from response", raw_response

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [retry {attempt+1}] Error: {e}")
                time.sleep(5)
            else:
                return "tie", f"Error after {max_retries} attempts: {e}", ""

    return "tie", "Max retries exceeded", ""


def compute_elo(
    wins_a: int, wins_b: int, ties: int, k: float = 32, initial: float = 1500
) -> tuple[float, float]:
    """Compute ELO ratings from match results.

    Uses standard ELO formula where ties count as 0.5 for each player.
    """
    elo_a = initial
    elo_b = initial

    # Simulate matches in random order to avoid order bias
    outcomes = ["A"] * wins_a + ["B"] * wins_b + ["tie"] * ties
    random.shuffle(outcomes)

    for outcome in outcomes:
        # Expected scores
        exp_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        exp_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        # Actual scores
        if outcome == "A":
            score_a, score_b = 1.0, 0.0
        elif outcome == "B":
            score_a, score_b = 0.0, 1.0
        else:  # tie
            score_a, score_b = 0.5, 0.5

        # Update ELO
        elo_a += k * (score_a - exp_a)
        elo_b += k * (score_b - exp_b)

    return elo_a, elo_b


def analyze_position_bias(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze if there's position bias (first response advantage)."""
    # When A presented first
    a_first = [m for m in matches if m["presentation_order"] == "A_first"]
    # When B presented first (A was second)
    b_first = [m for m in matches if m["presentation_order"] == "B_first"]

    def get_stats(match_list: list[dict[str, Any]]) -> dict[str, Any]:
        if not match_list:
            return {
                "a_wins": 0,
                "b_wins": 0,
                "ties": 0,
                "total": 0,
                "first_wins_pct": 0,
                "second_wins_pct": 0,
            }
        a_wins = sum(1 for m in match_list if m["winner"] == "A")
        b_wins = sum(1 for m in match_list if m["winner"] == "B")
        ties = sum(1 for m in match_list if m["winner"] == "tie")
        total = len(match_list)
        return {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "total": total,
            "a_wins_pct": a_wins / total * 100 if total > 0 else 0,
            "b_wins_pct": b_wins / total * 100 if total > 0 else 0,
            "ties_pct": ties / total * 100 if total > 0 else 0,
        }

    stats_a_first = get_stats(a_first)
    stats_b_first = get_stats(b_first)

    # Calculate first-position advantage
    # When A first: first position = A, second position = B
    # When B first: first position = B, second position = A
    first_pos_wins = stats_a_first["a_wins"] + stats_b_first["b_wins"]
    second_pos_wins = stats_a_first["b_wins"] + stats_b_first["a_wins"]
    total_decisive = first_pos_wins + second_pos_wins

    first_pos_rate = first_pos_wins / total_decisive * 100 if total_decisive > 0 else 50

    # Binomial test for position bias (null hypothesis: 50% each)
    from math import sqrt

    n = total_decisive
    if n > 0:
        p_observed = first_pos_wins / n
        # Standard error under null (p=0.5)
        se = sqrt(0.5 * 0.5 / n)
        z_score = (p_observed - 0.5) / se if se > 0 else 0
        # Two-tailed p-value approximation
        p_value = 2 * (
            1
            - 0.5 * (1 + (z_score / sqrt(1 + z_score**2 / 3)) * (1 + 0.044715 * z_score**2) ** 0.5)
        )
        # Simpler approximation for p-value
        import math

        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
    else:
        z_score = 0
        p_value = 1.0

    return {
        "when_a_first": stats_a_first,
        "when_b_first": stats_b_first,
        "first_position_wins": first_pos_wins,
        "second_position_wins": second_pos_wins,
        "first_position_win_rate": first_pos_rate,
        "z_score": z_score,
        "p_value": p_value,
        "significant_bias": p_value < 0.05,
    }


def compute_statistical_significance(wins_a: int, wins_b: int, ties: int) -> dict[str, Any]:
    """Compute statistical significance of the result using sign test."""
    import math

    # Sign test: only consider decisive matches (exclude ties)
    n = wins_a + wins_b
    if n == 0:
        return {"test": "sign_test", "n_decisive": 0, "p_value": 1.0, "significant": False}

    # Under null hypothesis, P(A wins) = P(B wins) = 0.5
    k = min(wins_a, wins_b)  # number of successes for minority

    # Binomial CDF approximation for two-tailed test
    # P(X <= k) where X ~ Binomial(n, 0.5)
    # Using normal approximation for large n
    if n >= 20:
        # Normal approximation
        mean = n * 0.5
        std = math.sqrt(n * 0.5 * 0.5)
        z = (k + 0.5 - mean) / std  # continuity correction
        p_value = 2 * 0.5 * (1 + math.erf(z / math.sqrt(2)))
    else:
        # Exact binomial (simplified calculation)
        from math import comb

        p_value = 0
        for i in range(k + 1):
            p_value += comb(n, i) * (0.5**n)
        p_value *= 2  # two-tailed
        p_value = min(p_value, 1.0)

    # Effect size (simple win rate difference)
    effect_size = abs(wins_a - wins_b) / n if n > 0 else 0

    # Confidence interval for win rate difference (Wilson score interval)
    p_hat = wins_a / (wins_a + wins_b) if (wins_a + wins_b) > 0 else 0.5
    z = 1.96  # 95% CI
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return {
        "test": "sign_test",
        "n_decisive": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "effect_size": effect_size,
        "win_rate_a": wins_a / n if n > 0 else 0.5,
        "ci_95_lower": max(0, center - margin),
        "ci_95_upper": min(1, center + margin),
    }


def load_completed_matches(
    output_file: Path,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load already completed matches from the incremental JSONL file."""
    matches: dict[str, dict[str, Any]] = {}
    completed_ids: set[str] = set()

    jsonl_file = output_file.with_suffix(".jsonl")
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            for line in f:
                match = json.loads(line)
                task_id = match["task_id"]
                matches[task_id] = match
                completed_ids.add(task_id)

    return matches, completed_ids


def save_match_incremental(match: dict[str, Any], output_file: Path) -> None:
    """Append a single match result to the incremental JSONL file."""
    jsonl_file = output_file.with_suffix(".jsonl")
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(match) + "\n")


def run_tournament(
    results_dir: Path,
    model_a: str,
    model_b: str,
    judge_config: dict[str, Any],
    output_file: Path | None = None,
    randomize_order: bool = True,
    limit: int | None = None,
    resume: bool = True,
) -> TournamentResults:
    """Run the pairwise tournament between two models."""

    print(f"\n{'='*70}")
    print(f"  PAIRWISE TOURNAMENT: {model_a} vs {model_b}")
    print(f"{'='*70}\n")

    # Load responses
    print(f"Loading responses from {results_dir}...")
    responses_a = load_responses(results_dir, model_a)
    responses_b = load_responses(results_dir, model_b)

    # Find common task_ids
    common_ids = set(responses_a.keys()) & set(responses_b.keys())
    print(f"Found {len(common_ids)} common examples")

    if limit:
        common_ids = set(sorted(common_ids)[:limit])
        print(f"Limited to {len(common_ids)} examples")

    # Check for existing progress
    completed_matches: dict[str, dict[str, Any]] = {}
    completed_ids: set[str] = set()
    if resume and output_file:
        completed_matches, completed_ids = load_completed_matches(output_file)
        if completed_ids:
            print(f"Resuming: found {len(completed_ids)} completed matches")

    # Run matches
    matches = list(completed_matches.values())
    wins_a = sum(1 for m in matches if m["winner"] == "A")
    wins_b = sum(1 for m in matches if m["winner"] == "B")
    ties = sum(1 for m in matches if m["winner"] == "tie")

    pending_ids = sorted(common_ids - completed_ids)
    total_to_process = len(pending_ids)

    if total_to_process == 0:
        print("All matches already completed!")
    else:
        print(f"Processing {total_to_process} remaining matches...")

    for i, task_id in enumerate(pending_ids):
        rec_a = responses_a[task_id]
        rec_b = responses_b[task_id]

        question = rec_a.get("instruction", "")
        expected = rec_a.get("expected_response", "")

        # Get contexts - they should be in the record or we reconstruct from expected
        contexts = rec_a.get("contexts", "")
        if not contexts and "[1]" in expected:
            # Contexts were used but not stored - note this
            contexts = "(Contexts not stored in record - judge should focus on response quality)"

        resp_a = rec_a.get("model_answer", "")
        resp_b = rec_b.get("model_answer", "")

        # Randomize presentation order to detect position bias
        if randomize_order and random.random() > 0.5:
            order = "B_first"
            prompt = create_pairwise_prompt(question, contexts, expected, resp_b, resp_a)
        else:
            order = "A_first"
            prompt = create_pairwise_prompt(question, contexts, expected, resp_a, resp_b)

        print(f"[{i+1}/{total_to_process}] Task {task_id[:8]}... ", end="", flush=True)

        verdict, reasoning, raw = call_judge(prompt, judge_config)

        # Flip verdict if we presented B first
        if order == "B_first":
            if verdict == "A":
                verdict = "B"
            elif verdict == "B":
                verdict = "A"

        # Count results
        if verdict == "A":
            wins_a += 1
            print(f"Winner: {model_a}")
        elif verdict == "B":
            wins_b += 1
            print(f"Winner: {model_b}")
        else:
            ties += 1
            print("Tie")

        match_result = asdict(
            MatchResult(
                task_id=task_id,
                question=question[:200] + "..." if len(question) > 200 else question,
                winner=verdict,
                model_a=model_a,
                model_b=model_b,
                response_a=resp_a[:500] + "..." if len(resp_a) > 500 else resp_a,
                response_b=resp_b[:500] + "..." if len(resp_b) > 500 else resp_b,
                judge_reasoning=reasoning[:500] + "..." if len(reasoning) > 500 else reasoning,
                presentation_order=order,
                raw_judge_response=raw[:1000] if len(raw) > 1000 else raw,
            )
        )

        matches.append(match_result)

        # Save incrementally
        if output_file:
            save_match_incremental(match_result, output_file)

    # Compute ELO
    elo_a, elo_b = compute_elo(wins_a, wins_b, ties)

    # Compute additional analyses
    position_bias = analyze_position_bias(matches)
    stat_significance = compute_statistical_significance(wins_a, wins_b, ties)

    # Print summary
    total = len(matches)
    decisive = wins_a + wins_b

    print(f"\n{'='*70}")
    print("  TOURNAMENT RESULTS")
    print(f"{'='*70}")

    print("\n  Stats MATCH OUTCOMES")
    print(f"  Total matches: {total}")
    print(f"  Decisive: {decisive} | Ties: {ties} ({ties/total*100:.1f}%)")

    print("\n  Winner WIN RATE (excluding ties)")
    if decisive > 0:
        wr_a = wins_a / decisive * 100
        wr_b = wins_b / decisive * 100
        print(f"  {model_a}: {wins_a}/{decisive} = {wr_a:.1f}%")
        print(f"  {model_b}: {wins_b}/{decisive} = {wr_b:.1f}%")
        print(f"  Difference: {abs(wr_a - wr_b):.1f} percentage points")
    else:
        print("  No decisive matches (all ties)")

    print("\n  Trend WIN RATE (including ties as 0.5)")
    wr_with_ties_a = (wins_a + ties * 0.5) / total * 100
    wr_with_ties_b = (wins_b + ties * 0.5) / total * 100
    print(f"  {model_a}: {wr_with_ties_a:.1f}%")
    print(f"  {model_b}: {wr_with_ties_b:.1f}%")

    print("\n  Goal ELO RATINGS (reference only)")
    print(f"  {model_a}: {elo_a:.0f}")
    print(f"  {model_b}: {elo_b:.0f}")
    print(f"  Difference: {abs(elo_a - elo_b):.0f} points")

    # Position bias analysis
    print("\n  Repeat POSITION BIAS ANALYSIS")
    pb = position_bias
    print(f"  When A presented first ({pb['when_a_first']['total']} matches):")
    print(
        f"    A wins: {pb['when_a_first']['a_wins_pct']:.1f}%, B wins: {pb['when_a_first']['b_wins_pct']:.1f}%, Ties: {pb['when_a_first']['ties_pct']:.1f}%"
    )
    print(f"  When B presented first ({pb['when_b_first']['total']} matches):")
    print(
        f"    A wins: {pb['when_b_first']['a_wins_pct']:.1f}%, B wins: {pb['when_b_first']['b_wins_pct']:.1f}%, Ties: {pb['when_b_first']['ties_pct']:.1f}%"
    )
    print(f"  First position win rate: {pb['first_position_win_rate']:.1f}% (expected: 50%)")
    if pb["significant_bias"]:
        print(f"  WARNING:   SIGNIFICANT POSITION BIAS DETECTED (p={pb['p_value']:.4f})")
    else:
        print(f"  ✓ No significant position bias (p={pb['p_value']:.4f})")

    # Statistical significance
    print("\n  Trend STATISTICAL SIGNIFICANCE (Sign Test)")
    ss = stat_significance
    print(f"  Decisive matches: {ss['n_decisive']} (excluding {ties} ties)")
    print(
        f"  A win rate: {ss['win_rate_a']*100:.1f}% (95% CI: {ss['ci_95_lower']*100:.1f}%-{ss['ci_95_upper']*100:.1f}%)"
    )
    print(f"  Effect size: {ss['effect_size']*100:.1f}% difference")
    print(f"  p-value: {ss['p_value']:.4f}")
    if ss["significant"]:
        print("  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p >= 0.05)")

    # Winner announcement based on win count (not ELO)
    if wins_a > wins_b:
        winner = model_a
        winner_wr = wins_a / decisive * 100 if decisive > 0 else 50
    elif wins_b > wins_a:
        winner = model_b
        winner_wr = wins_b / decisive * 100 if decisive > 0 else 50
    else:
        winner = "Tie"
        winner_wr = 50

    print(f"\n  {'='*66}")
    if ss["significant"]:
        print(f"  Winner WINNER: {winner}")
        print(f"     Win rate: {winner_wr:.1f}% (p={ss['p_value']:.4f}, statistically significant)")
    else:
        if winner != "Tie":
            print(f"  Stats LEADING: {winner} ({winner_wr:.1f}% win rate)")
            print(f"     But NOT statistically significant (p={ss['p_value']:.4f})")
        else:
            print("  Agreement RESULT: Exactly tied")
    print(f"  {'='*66}\n")

    return TournamentResults(
        model_a=model_a,
        model_b=model_b,
        total_matches=len(matches),
        model_a_wins=wins_a,
        model_b_wins=wins_b,
        ties=ties,
        model_a_elo=elo_a,
        model_b_elo=elo_b,
        matches=matches,
        position_bias=position_bias,
        statistical_significance=stat_significance,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pairwise Tournament between two models")

    # Default to responses/ if it exists (has more data), otherwise interleaved/
    default_dir = Path("evaluation/responses")
    if not default_dir.exists():
        default_dir = Path("evaluation/results/interleaved")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_dir,
        help="Directory with evaluation results (default: evaluation/responses or evaluation/results/interleaved)",
    )
    parser.add_argument(
        "--model-a", type=str, default="rag_baseline_0p5b", help="First model (condition name)"
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="hybrid_science_0p5b_rag_trained",
        help="Second model (condition name)",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=Path("configs/judges/qwen3_8b_thinking.yaml"),
        help="Judge model config",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches")
    parser.add_argument(
        "--no-randomize", action="store_true", help="Don't randomize presentation order"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from previous progress, start fresh"
    )

    args = parser.parse_args()

    # Load judge config
    with open(args.judge_config) as f:
        judge_config = yaml.safe_load(f)

    # Run tournament
    results = run_tournament(
        results_dir=args.results_dir,
        model_a=args.model_a,
        model_b=args.model_b,
        judge_config=judge_config,
        output_file=args.output,
        randomize_order=not args.no_randomize,
        limit=args.limit,
        resume=not args.no_resume,
    )

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
