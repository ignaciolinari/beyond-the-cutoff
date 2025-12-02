#!/usr/bin/env python3
"""Analyze pairwise evaluation results for quantization comparison (Q4_K_M vs F16).

This script takes the JSON results you paste from Gemini and computes
the final scores, ELO ratings, and statistical significance.

Usage:
    python scripts/utility/analyze_quantization_results.py

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

# Model names for quantization comparison
MODEL_Q4 = "hybrid_science_0p5b_rag_trained"  # Quantized (Q4_K_M)
MODEL_F16 = "hybrid_science_0p5b_rag_trained_f16"  # Non-quantized (F16)

# Directory for quantization batches
BATCH_DIR = Path("evaluation/exports/quantization_batches")


def load_mapping() -> list[dict[str, Any]]:
    """Load the batch mapping to know which model was A/B."""
    mapping_file = BATCH_DIR / "batch_mapping.json"
    if not mapping_file.exists():
        print(f"❌ Error: {mapping_file} not found")
        print("   Run generate_quantization_batches.py first")
        sys.exit(1)
    with open(mapping_file) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from Gemini response text."""
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


def compute_wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 1.0

    p_hat = wins / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator

    return max(0, center - margin), min(1, center + margin)


def print_final_results(results: list[dict[str, Any]]) -> None:
    """Print final aggregated results for quantization comparison."""
    if not results:
        print("\n❌ No results to analyze")
        return

    # Count wins
    q4_wins = sum(1 for r in results if r["winner_model"] == MODEL_Q4)
    f16_wins = sum(1 for r in results if r["winner_model"] == MODEL_F16)
    ties = sum(1 for r in results if r["winner_model"] == "tie")
    total = len(results)
    decisive = q4_wins + f16_wins

    # Compute ELO
    elo_q4, elo_f16 = compute_elo(q4_wins, f16_wins, ties)

    # Compute significance
    p_value, is_significant = compute_significance(q4_wins, f16_wins)

    # Win rate with CI (excluding ties)
    if decisive > 0:
        q4_win_rate = q4_wins / decisive
        f16_win_rate = f16_wins / decisive
        q4_ci_low, q4_ci_high = compute_wilson_ci(q4_wins, decisive)
        f16_ci_low, f16_ci_high = compute_wilson_ci(f16_wins, decisive)
    else:
        q4_win_rate = f16_win_rate = 0.5
        q4_ci_low = q4_ci_high = f16_ci_low = f16_ci_high = 0.5

    # Confidence breakdown
    high_conf = sum(1 for r in results if r.get("confidence", "").lower() == "high")
    med_conf = sum(1 for r in results if r.get("confidence", "").lower() == "medium")
    low_conf = sum(1 for r in results if r.get("confidence", "").lower() == "low")

    # Position bias analysis
    q4_first_matches = [r for r in results if r.get("order") == "q4_first"]
    f16_first_matches = [r for r in results if r.get("order") == "f16_first"]

    q4_wins_when_first = sum(1 for r in q4_first_matches if r["winner_model"] == MODEL_Q4)
    q4_wins_when_second = sum(1 for r in f16_first_matches if r["winner_model"] == MODEL_Q4)

    print("\n" + "=" * 70)
    print("  QUANTIZATION COMPARISON RESULTS (Q4_K_M vs F16)")
    print("=" * 70)

    print(f"\n📊 MATCH OUTCOMES ({total} total)")
    print(f"  Q4_K_M (quantized):     {q4_wins:3d} wins ({q4_wins/total*100:5.1f}%)")
    print(f"  F16 (non-quantized):    {f16_wins:3d} wins ({f16_wins/total*100:5.1f}%)")
    print(f"  Ties:                   {ties:3d}      ({ties/total*100:5.1f}%)")

    print(f"\n📈 WIN RATE (excluding {ties} ties)")
    print(
        f"  Q4_K_M: {q4_win_rate*100:.1f}% (95% CI: {q4_ci_low*100:.1f}% - {q4_ci_high*100:.1f}%)"
    )
    print(
        f"  F16:    {f16_win_rate*100:.1f}% (95% CI: {f16_ci_low*100:.1f}% - {f16_ci_high*100:.1f}%)"
    )

    print("\n🎯 ELO RATINGS")
    print(f"  Q4_K_M: {elo_q4:.0f}")
    print(f"  F16:    {elo_f16:.0f}")
    print(f"  Difference: {abs(elo_q4 - elo_f16):.0f} points")

    print("\n📈 STATISTICAL SIGNIFICANCE (Sign Test)")
    print(f"  Decisive matches: {decisive}")
    print(f"  p-value: {p_value:.4f}")
    if is_significant:
        print("  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p >= 0.05)")

    print("\n🔄 POSITION BIAS ANALYSIS")
    if q4_first_matches and f16_first_matches:
        print(f"  When Q4 presented as A ({len(q4_first_matches)} matches):")
        print(
            f"    Q4 wins: {q4_wins_when_first}, F16 wins: {sum(1 for r in q4_first_matches if r['winner_model'] == MODEL_F16)}"
        )
        print(f"  When F16 presented as A ({len(f16_first_matches)} matches):")
        print(
            f"    Q4 wins: {q4_wins_when_second}, F16 wins: {sum(1 for r in f16_first_matches if r['winner_model'] == MODEL_F16)}"
        )

    print("\n🔍 JUDGE CONFIDENCE")
    print(f"  High:   {high_conf:3d} ({high_conf/total*100:5.1f}%)")
    print(f"  Medium: {med_conf:3d} ({med_conf/total*100:5.1f}%)")
    print(f"  Low:    {low_conf:3d} ({low_conf/total*100:5.1f}%)")

    # Interpretation
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    if is_significant:
        if q4_wins > f16_wins:
            print("  🏆 Q4_K_M (quantized) is SIGNIFICANTLY BETTER than F16")
            print("     This is unexpected - quantization usually loses quality")
        else:
            print("  🏆 F16 (non-quantized) is SIGNIFICANTLY BETTER than Q4_K_M")
            print("     Quantization causes measurable quality loss")
    else:
        if ties > (q4_wins + f16_wins):
            print("  🤝 MODELS ARE EQUIVALENT")
            print("     Majority of responses are ties - quantization has minimal impact")
        else:
            print("  📊 NO SIGNIFICANT DIFFERENCE DETECTED")
            print("     The evidence is insufficient to conclude quantization affects quality")

        # Effect size interpretation
        effect_size = abs(q4_win_rate - 0.5) * 2 if decisive > 0 else 0
        if effect_size < 0.1:
            print(f"     Effect size is negligible ({effect_size:.2f})")
        elif effect_size < 0.3:
            print(f"     Effect size is small ({effect_size:.2f})")
        else:
            print(f"     Effect size is moderate ({effect_size:.2f}) - may need more samples")

    print("=" * 70)


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

    print("\n" + "=" * 70)
    print("  QUANTIZATION COMPARISON ANALYZER")
    print("  Q4_K_M (quantized) vs F16 (non-quantized)")
    print("=" * 70)
    print(f"\nTotal batches to process: {len(batches)}")
    print("For each batch, paste the JSON response from Gemini")
    print("Type 'skip' to skip a batch, 'done' to finish early\n")

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

        raw_responses[batch_num] = text

        parsed = parse_json_response(text)
        if not parsed or "evaluations" not in parsed:
            print(f"  ⚠️  Could not parse JSON for batch {batch_num}, skipping")
            continue

        evaluations = parsed["evaluations"]
        print(f"  ✓ Parsed {len(evaluations)} evaluations")

        for eval_item in evaluations:
            example_num = eval_item["example"]
            verdict = eval_item["verdict"].upper()

            mapping_item = next((m for m in batch_items if m["example"] == example_num), None)
            if not mapping_item:
                print(f"  ⚠️  No mapping found for example {example_num}")
                continue

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
                    "order": mapping_item["order"],
                    "confidence": eval_item.get("confidence", "unknown"),
                    "reasoning": eval_item.get("reasoning", ""),
                }
            )

    print_final_results(all_results)

    # Save results
    output_file = BATCH_DIR / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Processed results saved to {output_file}")

    raw_file = BATCH_DIR / "raw_judge_responses.json"
    with open(raw_file, "w") as f:
        json.dump(
            {f"batch_{k:02d}": v for k, v in raw_responses.items()},
            f,
            indent=2,
        )
    print(f"💾 Raw judge responses saved to {raw_file}")

    audit_file = BATCH_DIR / "evaluation_audit.json"
    audit_data = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "experiment": "quantization_comparison",
        "total_batches_processed": len(raw_responses),
        "total_examples_evaluated": len(all_results),
        "model_q4": MODEL_Q4,
        "model_f16": MODEL_F16,
        "raw_responses": {f"batch_{k:02d}": v for k, v in raw_responses.items()},
        "processed_results": all_results,
        "mapping_file": "batch_mapping.json",
    }
    with open(audit_file, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"💾 Full audit trail saved to {audit_file}")


def continue_adding_batches() -> None:
    """Continue adding batches starting from the next available batch number."""
    mapping = load_mapping()

    results_file = BATCH_DIR / "evaluation_results.json"
    raw_file = BATCH_DIR / "raw_judge_responses.json"

    existing_results: list[dict[str, Any]] = []
    raw_responses: dict[str, str] = {}

    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    if raw_file.exists():
        with open(raw_file) as f:
            raw_responses = json.load(f)

    evaluated_batches: set[int] = {int(r["batch"]) for r in existing_results}

    batches: dict[int, list[dict[str, Any]]] = {}
    for item in mapping:
        batch_num = int(item["batch"])
        if batch_num not in batches:
            batches[batch_num] = []
        batches[batch_num].append(item)

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

    print("\nFor each batch, paste the JSON response from Gemini")
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
                    "order": mapping_item["order"],
                    "confidence": eval_item.get("confidence", "unknown"),
                    "reasoning": eval_item.get("reasoning", ""),
                }
            )

    existing_results.sort(key=lambda x: (x["batch"], x["example"]))

    with open(results_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"\n💾 Updated results saved to {results_file}")

    with open(raw_file, "w") as f:
        json.dump(raw_responses, f, indent=2)
    print(f"💾 Updated raw responses saved to {raw_file}")

    print_final_results(existing_results)


def add_specific_batches(batch_numbers: list[int]) -> None:
    """Add/replace specific batch results."""
    mapping = load_mapping()

    results_file = BATCH_DIR / "evaluation_results.json"
    raw_file = BATCH_DIR / "raw_judge_responses.json"

    existing_results: list[dict[str, Any]] = []
    raw_responses: dict[str, str] = {}

    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    if raw_file.exists():
        with open(raw_file) as f:
            raw_responses = json.load(f)

    # Remove existing entries for batches we're replacing
    existing_results = [r for r in existing_results if r["batch"] not in batch_numbers]
    for bn in batch_numbers:
        key = f"batch_{bn:02d}"
        if key in raw_responses:
            del raw_responses[key]

    batches: dict[int, list[dict[str, Any]]] = {}
    for item in mapping:
        batch_num = int(item["batch"])
        if batch_num not in batches:
            batches[batch_num] = []
        batches[batch_num].append(item)

    print(f"\nAdding/replacing batches: {batch_numbers}")
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
                    "order": mapping_item["order"],
                    "confidence": eval_item.get("confidence", "unknown"),
                    "reasoning": eval_item.get("reasoning", ""),
                }
            )

    existing_results.sort(key=lambda x: (x["batch"], x["example"]))

    with open(results_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"\n💾 Updated results saved to {results_file}")

    with open(raw_file, "w") as f:
        json.dump(raw_responses, f, indent=2)
    print(f"💾 Updated raw responses saved to {raw_file}")

    print_final_results(existing_results)


def file_mode(results_file: Path) -> None:
    """Analyze results from a pre-filled file."""
    load_mapping()  # Verify mapping exists

    with open(results_file) as f:
        all_results = json.load(f)

    print_final_results(all_results)


def main() -> int:
    """Main entry point."""
    results_file = BATCH_DIR / "evaluation_results.json"
    mapping_file = BATCH_DIR / "batch_mapping.json"

    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

        evaluated_batches = len({r["batch"] for r in existing_results})
        total_examples = len(existing_results)

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
