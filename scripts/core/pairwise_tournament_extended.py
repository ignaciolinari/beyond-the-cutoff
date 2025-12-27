#!/usr/bin/env python3
"""Extended Pairwise Tournament: Continue from existing results with new responses.

This script extends existing tournament results by:
1. Loading completed matches from a previous tournament
2. Finding new task_ids available in the responses directory
3. Running only the new matches

Usage:
    python scripts/core/pairwise_tournament_extended.py \
        --existing evaluation/results/tournament/base_rag_vs_ft_rag.jsonl \
        --responses-dir evaluation/responses \
        --output evaluation/results/tournament/base_rag_vs_ft_rag_extended.json
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


def load_responses(results_dir: Path, condition: str) -> dict[str, dict[str, Any]]:
    """Load all responses for a condition, keyed by task_id."""
    responses: dict[str, dict[str, Any]] = {}

    # Try direct .jsonl file first
    direct_file = results_dir / f"{condition}.jsonl"
    if direct_file.exists():
        with open(direct_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record.get("task_id", record.get("id", ""))
                    if task_id:
                        if "response" in record and "model_answer" not in record:
                            record["model_answer"] = record["response"]
                        responses[task_id] = record
        return responses

    # Fallback to subdirectory format
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

    raise FileNotFoundError(f"Could not find responses for {condition} in {results_dir}")


def load_existing_matches(jsonl_file: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing matches from a JSONL file."""
    matches: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if jsonl_file.exists():
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    match = json.loads(line)
                    matches.append(match)
                    completed_ids.add(match["task_id"])

    return matches, completed_ids


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
                clean_response = raw_response
                if "<think>" in clean_response:
                    parts = clean_response.split("</think>")
                    if len(parts) > 1:
                        clean_response = parts[-1].strip()

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
                    if "verdict" in clean_response.lower():
                        if '"a"' in clean_response.lower() or "'a'" in clean_response.lower():
                            return "A", "Inferred A from response", raw_response
                        elif '"b"' in clean_response.lower() or "'b'" in clean_response.lower():
                            return "B", "Inferred B from response", raw_response
                    return "tie", "Failed to parse verdict", raw_response

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [retry {attempt+1}] Error: {e}")
                time.sleep(5)
            else:
                return "tie", f"Error after {max_retries} attempts: {e}", ""

    return "tie", "Max retries exceeded", ""


def save_match_incremental(match: dict[str, Any], output_file: Path) -> None:
    """Append a single match result to the incremental JSONL file."""
    jsonl_file = output_file.with_suffix(".jsonl")
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(match) + "\n")


def compute_statistics(matches: list[dict[str, Any]], model_a: str, model_b: str) -> dict[str, Any]:
    """Compute win rates and statistics from matches."""
    import math

    wins_a = sum(1 for m in matches if m["winner"] == "A")
    wins_b = sum(1 for m in matches if m["winner"] == "B")
    ties = sum(1 for m in matches if m["winner"] == "tie")
    total = len(matches)
    decisive = wins_a + wins_b

    # Win rates
    if decisive > 0:
        win_rate_a = wins_a / decisive
        win_rate_b = wins_b / decisive
    else:
        win_rate_a = win_rate_b = 0.5

    # Sign test for significance
    if decisive > 0:
        k = min(wins_a, wins_b)
        if decisive >= 20:
            mean = decisive * 0.5
            std = math.sqrt(decisive * 0.5 * 0.5)
            z = (k + 0.5 - mean) / std
            p_value = 2 * 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            from math import comb

            p_value = sum(comb(decisive, i) * (0.5**decisive) for i in range(k + 1)) * 2
            p_value = min(p_value, 1.0)
    else:
        p_value = 1.0

    return {
        "model_a": model_a,
        "model_b": model_b,
        "total_matches": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "decisive": decisive,
        "win_rate_a": win_rate_a,
        "win_rate_b": win_rate_b,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def print_results(stats: dict[str, Any]) -> None:
    """Print formatted results."""
    print(f"\n{'='*70}")
    print("  TOURNAMENT RESULTS")
    print(f"{'='*70}")

    print(f"\nStats MATCH OUTCOMES ({stats['total_matches']} total)")
    print(
        f"  Decisive: {stats['decisive']} | Ties: {stats['ties']} ({stats['ties']/stats['total_matches']*100:.1f}%)"
    )

    print("\nWinner WIN RATE (excluding ties)")
    if stats["decisive"] > 0:
        print(
            f"  {stats['model_a']}: {stats['wins_a']}/{stats['decisive']} = {stats['win_rate_a']*100:.1f}%"
        )
        print(
            f"  {stats['model_b']}: {stats['wins_b']}/{stats['decisive']} = {stats['win_rate_b']*100:.1f}%"
        )
        print(
            f"  Difference: {abs(stats['win_rate_a'] - stats['win_rate_b'])*100:.1f} percentage points"
        )

    print("\nTrend STATISTICAL SIGNIFICANCE (Sign Test)")
    print(f"  p-value: {stats['p_value']:.4f}")
    if stats["significant"]:
        print("  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p >= 0.05)")

    # Winner
    print(f"\n{'='*70}")
    if stats["wins_a"] > stats["wins_b"]:
        winner = stats["model_a"]
        wr = stats["win_rate_a"] * 100
    elif stats["wins_b"] > stats["wins_a"]:
        winner = stats["model_b"]
        wr = stats["win_rate_b"] * 100
    else:
        winner = None
        wr = 50

    if winner:
        if stats["significant"]:
            print(f"  Winner WINNER: {winner}")
            print(f"     Win rate: {wr:.1f}% (statistically significant)")
        else:
            print(f"  Stats LEADING: {winner} ({wr:.1f}% win rate)")
            print(f"     But NOT statistically significant (p={stats['p_value']:.4f})")
    else:
        print("  Agreement RESULT: Exactly tied")
    print(f"{'='*70}\n")


def run_extended_tournament(
    existing_file: Path,
    responses_dir: Path,
    model_a: str,
    model_b: str,
    judge_config: dict[str, Any],
    output_file: Path,
) -> dict[str, Any]:
    """Run extended tournament, continuing from existing results."""

    print(f"\n{'='*70}")
    print("  EXTENDED PAIRWISE TOURNAMENT")
    print(f"  {model_a} vs {model_b}")
    print(f"{'='*70}\n")

    # Load existing matches
    existing_matches, completed_ids = load_existing_matches(existing_file)
    print(f"Loaded {len(existing_matches)} existing matches from {existing_file}")

    # Load all responses
    print(f"Loading responses from {responses_dir}...")
    responses_a = load_responses(responses_dir, model_a)
    responses_b = load_responses(responses_dir, model_b)

    common_ids = set(responses_a.keys()) & set(responses_b.keys())
    print(f"Found {len(common_ids)} common examples")

    # Find new task_ids to process
    new_ids = sorted(common_ids - completed_ids)
    print(f"New examples to process: {len(new_ids)}")

    if not new_ids:
        print("\n✓ All examples already processed!")
        stats = compute_statistics(existing_matches, model_a, model_b)
        print_results(stats)
        return stats

    # Copy existing matches to output
    output_jsonl = output_file.with_suffix(".jsonl")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # If output is different from existing, copy existing matches
    if output_jsonl != existing_file:
        with open(output_jsonl, "w") as f:
            for match in existing_matches:
                f.write(json.dumps(match) + "\n")
        print(f"Copied existing matches to {output_jsonl}")

    all_matches = existing_matches.copy()

    # Process new matches
    print(f"\nProcessing {len(new_ids)} new matches...")

    for i, task_id in enumerate(new_ids):
        rec_a = responses_a[task_id]
        rec_b = responses_b[task_id]

        question = rec_a.get("instruction", "")
        expected = rec_a.get("expected_response", "")
        contexts = rec_a.get("contexts", "")
        if isinstance(contexts, list):
            contexts = "\n".join(contexts)

        resp_a = rec_a.get("model_answer", "")
        resp_b = rec_b.get("model_answer", "")

        # Randomize presentation order
        if random.random() > 0.5:
            order = "B_first"
            prompt = create_pairwise_prompt(question, contexts, expected, resp_b, resp_a)
        else:
            order = "A_first"
            prompt = create_pairwise_prompt(question, contexts, expected, resp_a, resp_b)

        print(f"[{i+1}/{len(new_ids)}] Task {task_id[:8]}... ", end="", flush=True)

        verdict, reasoning, raw = call_judge(prompt, judge_config)

        # Flip verdict if B was presented first
        if order == "B_first":
            if verdict == "A":
                verdict = "B"
            elif verdict == "B":
                verdict = "A"

        if verdict == "A":
            print(f"Winner: {model_a}")
        elif verdict == "B":
            print(f"Winner: {model_b}")
        else:
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

        all_matches.append(match_result)
        save_match_incremental(match_result, output_file)

    # Compute and print final statistics
    stats = compute_statistics(all_matches, model_a, model_b)
    print_results(stats)

    # Save final JSON
    with open(output_file, "w") as f:
        json.dump(
            {
                "statistics": stats,
                "matches": all_matches,
            },
            f,
            indent=2,
        )
    print(f"Save Results saved to {output_file}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Extended Pairwise Tournament")
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("evaluation/results/tournament/base_rag_vs_ft_rag.jsonl"),
        help="Existing JSONL file with completed matches",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=Path("evaluation/responses"),
        help="Directory with all responses (should have 154 examples)",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="rag_baseline_0p5b",
        help="First model name",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="hybrid_science_0p5b_rag_trained",
        help="Second model name",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=Path("configs/judges/qwen3_8b_thinking.yaml"),
        help="Judge model config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/tournament/base_rag_vs_ft_rag_extended.json"),
        help="Output file for extended results",
    )

    args = parser.parse_args()

    # Load judge config
    with open(args.judge_config) as f:
        judge_config = yaml.safe_load(f)

    run_extended_tournament(
        existing_file=args.existing,
        responses_dir=args.responses_dir,
        model_a=args.model_a,
        model_b=args.model_b,
        judge_config=judge_config,
        output_file=args.output,
    )

    return 0


if __name__ == "__main__":
    random.seed(42)
    sys.exit(main())
