#!/usr/bin/env python3
"""Analyze pairwise evaluation results from manual Claude/GPT-4 evaluation.

This script takes the JSON results you paste from Claude/GPT-4 and computes
the final scores, ELO ratings, and statistical significance.

Usage:
    python scripts/utility/analyze_pairwise_results.py

The script will prompt you to paste the results for each batch interactively,
or you can provide a pre-filled results file.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def load_mapping() -> list[dict[str, Any]]:
    """Load the batch mapping to know which model was A/B."""
    mapping_file = Path("evaluation/exports/batches/batch_mapping.json")
    if not mapping_file.exists():
        print(f"❌ Error: {mapping_file} not found")
        print("   Run generate_pairwise_batches.py first")
        sys.exit(1)
    with open(mapping_file) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from Claude/GPT response text."""
    # Try to find JSON block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            result: dict[str, Any] = json.loads(json_match.group(1))
            return result
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            pass

    return None


def compute_elo(
    wins_a: int, wins_b: int, ties: int, k: float = 32, initial: float = 1500
) -> tuple[float, float]:
    """Compute ELO ratings from match results."""
    import random

    elo_a = initial
    elo_b = initial

    outcomes = ["A"] * wins_a + ["B"] * wins_b + ["tie"] * ties
    random.seed(42)
    random.shuffle(outcomes)

    for outcome in outcomes:
        exp_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        exp_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        if outcome == "A":
            score_a, score_b = 1.0, 0.0
        elif outcome == "B":
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5

        elo_a += k * (score_a - exp_a)
        elo_b += k * (score_b - exp_b)

    return elo_a, elo_b


def compute_significance(wins_a: int, wins_b: int) -> tuple[float, bool]:
    """Compute p-value using sign test."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0, False

    k = min(wins_a, wins_b)

    if n >= 20:
        mean = n * 0.5
        std = math.sqrt(n * 0.5 * 0.5)
        z = (k + 0.5 - mean) / std
        p_value = 2 * 0.5 * (1 + math.erf(z / math.sqrt(2)))
    else:
        from math import comb

        p_value = sum(comb(n, i) * (0.5**n) for i in range(k + 1)) * 2
        p_value = min(p_value, 1.0)

    return p_value, p_value < 0.05


def interactive_mode() -> None:
    """Run in interactive mode, prompting for each batch."""
    mapping = load_mapping()

    # Group mapping by batch
    batches: dict[int, list[dict[str, Any]]] = {}
    for item in mapping:
        batch_num = int(item["batch"])
        if batch_num not in batches:
            batches[batch_num] = []
        batches[batch_num].append(item)

    # Model names
    model_baseline = "rag_baseline_0p5b"
    model_finetuned = "hybrid_science_0p5b_rag_trained"

    # Collect results (typed in raw_responses section)

    print("\n" + "=" * 70)
    print("  PAIRWISE EVALUATION RESULTS ANALYZER")
    print("=" * 70)
    print(f"\nTotal batches to process: {len(batches)}")
    print("For each batch, paste the JSON response from Claude/GPT-4")
    print("Type 'skip' to skip a batch, 'done' to finish early\n")

    # Store raw responses for later analysis
    raw_responses: dict[int, str] = {}
    all_results: list[dict[str, Any]] = []

    for batch_num in sorted(batches.keys()):
        batch_items = batches[batch_num]
        print(f"\n{'─'*70}")
        print(f"Batch {batch_num:02d} ({len(batch_items)} examples)")
        print("Paste the JSON response (then press Enter twice):")

        lines = []
        empty_count = 0
        while empty_count < 2:
            try:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)
                if line.strip().lower() == "skip":
                    break
                if line.strip().lower() == "done":
                    break
            except EOFError:
                break

        text = "\n".join(lines)

        if "skip" in text.lower():
            print(f"  Skipped batch {batch_num}")
            continue

        if "done" in text.lower():
            print("  Finishing early...")
            break

        # Save raw response
        raw_responses[batch_num] = text

        # Parse JSON
        parsed = parse_json_response(text)
        if not parsed or "evaluations" not in parsed:
            print(f"  ⚠️  Could not parse JSON for batch {batch_num}, skipping")
            continue

        evaluations = parsed["evaluations"]
        print(f"  ✓ Parsed {len(evaluations)} evaluations")

        # Map verdicts to actual models
        for eval_item in evaluations:
            example_num = eval_item["example"]
            verdict = eval_item["verdict"].upper()

            # Find the mapping for this example
            mapping_item = next((m for m in batch_items if m["example"] == example_num), None)
            if not mapping_item:
                print(f"  ⚠️  No mapping found for example {example_num}")
                continue

            # Convert verdict to actual model
            if verdict == "A":
                winner = mapping_item["presented_as_A"]
            elif verdict == "B":
                winner = mapping_item["presented_as_B"]
            else:
                winner = "tie"

            all_results.append(
                {
                    "batch": batch_num,
                    "example": example_num,
                    "task_id": mapping_item["task_id"],
                    "verdict": verdict,
                    "winner_model": winner,
                    "confidence": eval_item.get("confidence", "unknown"),
                    "reasoning": eval_item.get("reasoning", ""),
                }
            )

    # Compute final results
    print_final_results(all_results, model_baseline, model_finetuned)

    # Save processed results
    output_file = Path("evaluation/exports/batches/evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Processed results saved to {output_file}")

    # Save raw responses for reproducibility
    raw_file = Path("evaluation/exports/batches/raw_judge_responses.json")
    with open(raw_file, "w") as f:
        json.dump(
            {f"batch_{k:02d}": v for k, v in raw_responses.items()},
            f,
            indent=2,
        )
    print(f"💾 Raw judge responses saved to {raw_file}")

    # Save full audit trail
    audit_file = Path("evaluation/exports/batches/evaluation_audit.json")
    audit_data = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "total_batches_processed": len(raw_responses),
        "total_examples_evaluated": len(all_results),
        "model_a": model_baseline,
        "model_b": model_finetuned,
        "raw_responses": {f"batch_{k:02d}": v for k, v in raw_responses.items()},
        "processed_results": all_results,
        "mapping_file": "batch_mapping.json",
    }
    with open(audit_file, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"💾 Full audit trail saved to {audit_file}")


def print_final_results(
    results: list[dict[str, Any]], model_baseline: str, model_finetuned: str
) -> None:
    """Print final aggregated results."""
    if not results:
        print("\n❌ No results to analyze")
        return

    # Count wins
    baseline_wins = sum(1 for r in results if r["winner_model"] == model_baseline)
    finetuned_wins = sum(1 for r in results if r["winner_model"] == model_finetuned)
    ties = sum(1 for r in results if r["winner_model"] == "tie")
    total = len(results)

    # Compute ELO
    elo_baseline, elo_finetuned = compute_elo(baseline_wins, finetuned_wins, ties)

    # Compute significance
    p_value, is_significant = compute_significance(baseline_wins, finetuned_wins)

    # Confidence breakdown
    high_conf = sum(1 for r in results if r.get("confidence") == "high")
    med_conf = sum(1 for r in results if r.get("confidence") == "medium")
    low_conf = sum(1 for r in results if r.get("confidence") == "low")

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    print(f"\n📊 MATCH OUTCOMES ({total} total)")
    print(
        f"  Base+RAG (rag_baseline):     {baseline_wins:3d} wins ({baseline_wins/total*100:5.1f}%)"
    )
    print(
        f"  FT-RAG+RAG (hybrid_trained): {finetuned_wins:3d} wins ({finetuned_wins/total*100:5.1f}%)"
    )
    print(f"  Ties:                        {ties:3d}      ({ties/total*100:5.1f}%)")

    print("\n🎯 ELO RATINGS")
    print(f"  Base+RAG:     {elo_baseline:.0f}")
    print(f"  FT-RAG+RAG:   {elo_finetuned:.0f}")
    print(f"  Difference:   {abs(elo_baseline - elo_finetuned):.0f} points")

    print("\n📈 STATISTICAL SIGNIFICANCE (Sign Test)")
    print(f"  Decisive matches: {baseline_wins + finetuned_wins}")
    print(f"  p-value: {p_value:.4f}")
    if is_significant:
        print("  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p >= 0.05)")

    print("\n🔍 JUDGE CONFIDENCE")
    print(f"  High:   {high_conf:3d} ({high_conf/total*100:5.1f}%)")
    print(f"  Medium: {med_conf:3d} ({med_conf/total*100:5.1f}%)")
    print(f"  Low:    {low_conf:3d} ({low_conf/total*100:5.1f}%)")

    # Winner
    print("\n" + "=" * 70)
    if baseline_wins > finetuned_wins:
        winner = "Base+RAG (rag_baseline_0p5b)"
        winner_elo = elo_baseline
    elif finetuned_wins > baseline_wins:
        winner = "FT-RAG+RAG (hybrid_science_0p5b_rag_trained)"
        winner_elo = elo_finetuned
    else:
        winner = "TIE"
        winner_elo = (elo_baseline + elo_finetuned) / 2

    if is_significant:
        print(f"  🏆 WINNER: {winner}")
        print(f"     (ELO: {winner_elo:.0f}, statistically significant)")
    else:
        print("  🤝 RESULT: No clear winner")
        print("     (Difference not statistically significant)")
    print("=" * 70)


def add_specific_batches(batch_numbers: list[int]) -> None:
    """Add specific batch results to existing evaluation."""
    mapping = load_mapping()

    # Load existing results
    results_file = Path("evaluation/exports/batches/evaluation_results.json")
    raw_file = Path("evaluation/exports/batches/raw_judge_responses.json")

    existing_results: list[dict[str, Any]] = []
    raw_responses: dict[str, str] = {}

    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    if raw_file.exists():
        with open(raw_file) as f:
            raw_responses = json.load(f)

    # Remove existing entries for the batches we're replacing
    existing_results = [r for r in existing_results if r["batch"] not in batch_numbers]
    for bn in batch_numbers:
        key = f"batch_{bn:02d}"
        if key in raw_responses:
            del raw_responses[key]

    # Group mapping by batch
    batches: dict[int, list[dict[str, Any]]] = {}
    for item in mapping:
        batch_num = int(item["batch"])
        if batch_num not in batches:
            batches[batch_num] = []
        batches[batch_num].append(item)

    model_baseline = "rag_baseline_0p5b"
    model_finetuned = "hybrid_science_0p5b_rag_trained"

    print(f"\nAdding batches: {batch_numbers}")
    print("Paste the JSON response for each batch (Enter twice to confirm)\n")

    for batch_num in batch_numbers:
        if batch_num not in batches:
            print(f"⚠️  Batch {batch_num} not found in mapping, skipping")
            continue

        batch_items = batches[batch_num]
        print(f"\n{'─'*70}")
        print(f"Batch {batch_num:02d} ({len(batch_items)} examples)")
        print("Paste the JSON response:")

        lines = []
        empty_count = 0
        while empty_count < 2:
            try:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)
            except EOFError:
                break

        text = "\n".join(lines)
        raw_responses[f"batch_{batch_num:02d}"] = text

        parsed = parse_json_response(text)
        if not parsed or "evaluations" not in parsed:
            print(f"  ⚠️  Could not parse JSON for batch {batch_num}")
            continue

        evaluations = parsed["evaluations"]
        print(f"  ✓ Parsed {len(evaluations)} evaluations")

        for eval_item in evaluations:
            example_num = eval_item["example"]
            verdict = eval_item["verdict"].upper()

            mapping_item = next((m for m in batch_items if m["example"] == example_num), None)
            if not mapping_item:
                continue

            if verdict == "A":
                winner = mapping_item["presented_as_A"]
            elif verdict == "B":
                winner = mapping_item["presented_as_B"]
            else:
                winner = "tie"

            existing_results.append(
                {
                    "batch": batch_num,
                    "example": example_num,
                    "task_id": mapping_item["task_id"],
                    "verdict": verdict,
                    "winner_model": winner,
                    "confidence": eval_item.get("confidence", "unknown"),
                    "reasoning": eval_item.get("reasoning", ""),
                }
            )

    # Sort by batch and example
    existing_results.sort(key=lambda x: (x["batch"], x["example"]))

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"\n💾 Updated results saved to {results_file}")

    with open(raw_file, "w") as f:
        json.dump(raw_responses, f, indent=2)
    print(f"💾 Updated raw responses saved to {raw_file}")

    # Show updated stats
    print_final_results(existing_results, model_baseline, model_finetuned)


def file_mode(results_file: Path) -> None:
    """Analyze results from a pre-filled file."""
    # Just verify mapping exists
    load_mapping()

    with open(results_file) as f:
        all_results = json.load(f)

    model_baseline = "rag_baseline_0p5b"
    model_finetuned = "hybrid_science_0p5b_rag_trained"

    print_final_results(all_results, model_baseline, model_finetuned)


def continue_adding_batches() -> None:
    """Continue adding batches starting from the next available batch number."""
    mapping = load_mapping()

    # Load existing results
    results_file = Path("evaluation/exports/batches/evaluation_results.json")
    raw_file = Path("evaluation/exports/batches/raw_judge_responses.json")

    existing_results: list[dict[str, Any]] = []
    raw_responses: dict[str, str] = {}

    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    if raw_file.exists():
        with open(raw_file) as f:
            raw_responses = json.load(f)

    # Find which batches already have results
    evaluated_batches: set[int] = {int(r["batch"]) for r in existing_results}

    # Group mapping by batch
    batches: dict[int, list[dict[str, Any]]] = {}
    for item in mapping:
        batch_num = int(item["batch"])
        if batch_num not in batches:
            batches[batch_num] = []
        batches[batch_num].append(item)

    # Find batches that need evaluation
    all_batches = set(batches.keys())
    pending_batches = sorted(all_batches - evaluated_batches)

    if not pending_batches:
        print(f"\n✅ All {len(all_batches)} batches have been evaluated!")
        print("   Use option 3 to replace specific batches if needed.")
        return

    print("\n📊 Status:")
    print(f"   Total batches in mapping: {len(all_batches)}")
    print(f"   Already evaluated: {len(evaluated_batches)}")
    print(f"   Pending: {len(pending_batches)}")
    print(f"\nPending batches: {pending_batches}")

    model_baseline = "rag_baseline_0p5b"
    model_finetuned = "hybrid_science_0p5b_rag_trained"

    print("\nFor each batch, paste the JSON response from Claude/GPT-4/Gemini")
    print("Type 'skip' to skip, 'done' to finish early\n")

    for batch_num in pending_batches:
        batch_items = batches[batch_num]
        print(f"\n{'─'*70}")
        print(f"Batch {batch_num:02d} ({len(batch_items)} examples)")
        print("Paste the JSON response (then press Enter twice):")

        lines = []
        empty_count = 0
        while empty_count < 2:
            try:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)
                if line.strip().lower() in ("skip", "done"):
                    break
            except EOFError:
                break

        text = "\n".join(lines)

        if "skip" in text.lower():
            print(f"  Skipped batch {batch_num}")
            continue

        if "done" in text.lower():
            print("  Finishing early...")
            break

        raw_responses[f"batch_{batch_num:02d}"] = text

        parsed = parse_json_response(text)
        if not parsed or "evaluations" not in parsed:
            print(f"  ⚠️  Could not parse JSON for batch {batch_num}")
            continue

        evaluations = parsed["evaluations"]
        print(f"  ✓ Parsed {len(evaluations)} evaluations")

        for eval_item in evaluations:
            example_num = eval_item["example"]
            verdict = eval_item["verdict"].upper()

            mapping_item = next((m for m in batch_items if m["example"] == example_num), None)
            if not mapping_item:
                continue

            if verdict == "A":
                winner = mapping_item["presented_as_A"]
            elif verdict == "B":
                winner = mapping_item["presented_as_B"]
            else:
                winner = "tie"

            existing_results.append(
                {
                    "batch": batch_num,
                    "example": example_num,
                    "task_id": mapping_item["task_id"],
                    "verdict": verdict,
                    "winner_model": winner,
                    "confidence": eval_item.get("confidence", "unknown"),
                    "reasoning": eval_item.get("reasoning", ""),
                }
            )

    # Sort by batch and example
    existing_results.sort(key=lambda x: (x["batch"], x["example"]))

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"\n💾 Updated results saved to {results_file}")

    with open(raw_file, "w") as f:
        json.dump(raw_responses, f, indent=2)
    print(f"💾 Updated raw responses saved to {raw_file}")

    # Show updated stats
    print_final_results(existing_results, model_baseline, model_finetuned)


def main() -> int:
    """Main entry point."""
    results_file = Path("evaluation/exports/batches/evaluation_results.json")
    mapping_file = Path("evaluation/exports/batches/batch_mapping.json")

    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

        # Count evaluated batches
        evaluated_batches = len({r["batch"] for r in existing_results})
        total_examples = len(existing_results)

        # Count total batches in mapping
        total_batches = 0
        if mapping_file.exists():
            with open(mapping_file) as f:
                mapping = json.load(f)
            total_batches = max(item["batch"] for item in mapping) if mapping else 0

        print(f"Found existing results: {total_examples} examples from {evaluated_batches} batches")
        if total_batches > evaluated_batches:
            print(f"⚠️  {total_batches - evaluated_batches} batches pending evaluation")

        print("\nOptions:")
        print("  1. Analyze existing results")
        print("  2. Start fresh (interactive mode)")
        print("  3. Add/replace specific batches")
        print("  4. Continue adding pending batches")
        choice = input("Choice [1/2/3/4]: ").strip()

        if choice == "1":
            file_mode(results_file)
            return 0

        if choice == "3":
            batch_input = input("Enter batch numbers to add (comma-separated, e.g. 7,10): ")
            batch_numbers = [int(x.strip()) for x in batch_input.split(",")]
            add_specific_batches(batch_numbers)
            return 0

        if choice == "4":
            continue_adding_batches()
            return 0

    interactive_mode()
    return 0


if __name__ == "__main__":
    sys.exit(main())
