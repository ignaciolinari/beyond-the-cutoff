#!/usr/bin/env python3
"""Generate batches of pairwise comparisons for manual evaluation with Claude/GPT-4.

This script creates text files with batches of 5 examples each, ready to copy-paste
into a chat interface. Model positions (A/B) are randomized to avoid position bias.

Usage:
    python scripts/utility/generate_pairwise_batches.py

Output:
    evaluation/exports/batches/batch_01.txt
    evaluation/exports/batches/batch_02.txt
    ...
    evaluation/exports/batches/batch_mapping.json  (tracks which model is A/B)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from scripts.core.pairwise_tournament import load_responses

BATCH_SIZE = 5

BATCH_TEMPLATE = """You are an expert scientific evaluator comparing AI-generated responses to questions about recent scientific papers.

The responses are from a small language model (0.5B parameters) being tested on its ability to answer questions based on retrieved scientific context.

Focus on:
- Factual accuracy compared to the expected answer
- Completeness of key information
- Avoiding hallucinations or unsupported claims

Do NOT penalize for brevity if key points are covered.

---

I need you to compare pairs of responses to scientific questions. For each example, determine which response is better.
{examples}

---

For each example, provide your verdict. Use this EXACT JSON format:

```json
{{
  "evaluations": [
    {{"example": 1, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 2, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 3, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 4, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 5, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}}
  ]
}}
```"""

EXAMPLE_TEMPLATE = """
---
## Example {num}

### QUESTION
{question}

### EXPECTED ANSWER
{expected}

### RESPONSE A
{response_a}

### RESPONSE B
{response_b}
"""


def load_existing_mapping(output_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing mapping and get already-used task IDs."""
    mapping_file = output_dir / "batch_mapping.json"
    if not mapping_file.exists():
        return [], set()

    with open(mapping_file) as f:
        mapping: list[dict[str, Any]] = json.load(f)

    used_ids = {str(item["task_id"]) for item in mapping}
    return mapping, used_ids


def generate_batches(append_mode: bool = False) -> None:
    """Generate batches for manual evaluation.

    Args:
        append_mode: If True, add new batches without modifying existing ones.
    """
    # Try responses/ first (has more data), fallback to interleaved/
    responses_dir = Path("evaluation/responses")
    if not responses_dir.exists():
        responses_dir = Path("evaluation/results/interleaved")

    model_a_name = "rag_baseline_0p5b"
    model_b_name = "hybrid_science_0p5b_rag_trained"

    # Load responses (handles both jsonl files and subdirectories)
    responses_a = load_responses_flexible(responses_dir, model_a_name)
    responses_b = load_responses_flexible(responses_dir, model_b_name)

    common_ids = sorted(set(responses_a.keys()) & set(responses_b.keys()))
    print(f"Found {len(common_ids)} common examples between models")

    # Create output directory
    output_dir = Path("evaluation/exports/batches")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing mapping if appending
    existing_mapping: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    start_batch = 1

    if append_mode:
        existing_mapping, used_ids = load_existing_mapping(output_dir)
        if existing_mapping:
            start_batch = int(max(item["batch"] for item in existing_mapping)) + 1
            print(f"Found {len(existing_mapping)} existing examples in {start_batch - 1} batches")
            print(f"Will start from batch {start_batch}")

    # Filter out already-used IDs
    available_ids = [tid for tid in common_ids if tid not in used_ids]
    print(f"Available new examples: {len(available_ids)}")

    if not available_ids:
        print("\nWARNING:   No new examples available to generate batches")
        return

    # Shuffle for randomization (use different seed for new batches)
    random.seed(42 + start_batch)  # Reproducible but different from original
    shuffled_ids = available_ids.copy()
    random.shuffle(shuffled_ids)

    # Track mapping of which model is A/B for each example
    new_mapping: list[dict[str, Any]] = []

    batch_num = start_batch - 1
    for i in range(0, len(shuffled_ids), BATCH_SIZE):
        batch_ids = shuffled_ids[i : i + BATCH_SIZE]
        batch_num += 1

        examples_text = ""
        for j, task_id in enumerate(batch_ids):
            rec_a = responses_a[task_id]
            rec_b = responses_b[task_id]

            # Randomize which model is presented as A vs B
            if random.random() > 0.5:
                # Model A (rag_baseline) is presented as "Response A"
                response_a = rec_a["model_answer"]
                response_b = rec_b["model_answer"]
                order = "baseline_first"
                actual_a = model_a_name
                actual_b = model_b_name
            else:
                # Model B (hybrid_rag_trained) is presented as "Response A"
                response_a = rec_b["model_answer"]
                response_b = rec_a["model_answer"]
                order = "finetuned_first"
                actual_a = model_b_name
                actual_b = model_a_name

            examples_text += EXAMPLE_TEMPLATE.format(
                num=j + 1,
                question=rec_a["instruction"],
                expected=rec_a["expected_response"],
                response_a=response_a,
                response_b=response_b,
            )

            new_mapping.append(
                {
                    "batch": batch_num,
                    "example": j + 1,
                    "task_id": task_id,
                    "presented_as_A": actual_a,
                    "presented_as_B": actual_b,
                    "order": order,
                }
            )

        # Create batch file - clean, ready to copy-paste
        batch_content = BATCH_TEMPLATE.format(examples=examples_text)

        batch_file = output_dir / f"batch_{batch_num:02d}.txt"
        batch_file.write_text(batch_content)
        print(f"Created {batch_file}")

    # Combine and save mapping
    combined_mapping = existing_mapping + new_mapping
    mapping_file = output_dir / "batch_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(combined_mapping, f, indent=2)
    print(f"\nSaved mapping to {mapping_file}")

    new_batches = batch_num - start_batch + 1
    print(f"\n✓ Generated {new_batches} new batches with {len(new_mapping)} examples")
    print(f"   Total batches: {batch_num}")
    print(f"   Total examples: {len(combined_mapping)}")
    print(f"   Output directory: {output_dir}")


def load_responses_flexible(base_dir: Path, model_name: str) -> dict[str, dict[str, Any]]:
    """Load responses from either jsonl files or subdirectories."""
    # Try direct jsonl file first
    jsonl_file = base_dir / f"{model_name}.jsonl"
    if jsonl_file.exists():
        responses = {}
        with open(jsonl_file) as f:
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

    # Fallback to subdirectory with details.jsonl
    details_file = base_dir / model_name / "details.jsonl"
    if details_file.exists():
        responses = {}
        with open(details_file) as f:
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

    # Use original loader as final fallback
    return load_responses(base_dir, model_name)


def main() -> int:
    """Main entry point."""
    output_dir = Path("evaluation/exports/batches")
    mapping_file = output_dir / "batch_mapping.json"

    if mapping_file.exists():
        with open(mapping_file) as f:
            existing = json.load(f)

        num_batches = max(item["batch"] for item in existing) if existing else 0
        num_examples = len(existing)

        print(f"Found existing batches: {num_batches} batches, {num_examples} examples")
        print("\nOptions:")
        print("  1. Generate NEW batches (append to existing, don't modify)")
        print("  2. Regenerate ALL batches from scratch")
        print("  3. View batch statistics")
        print("  4. Exit")

        choice = input("\nChoice [1/2/3/4]: ").strip()

        if choice == "1":
            generate_batches(append_mode=True)
        elif choice == "2":
            confirm = input(
                "WARNING:   This will OVERWRITE existing batches. Type 'yes' to confirm: "
            )
            if confirm.lower() == "yes":
                generate_batches(append_mode=False)
            else:
                print("Cancelled.")
                return 0
        elif choice == "3":
            # Show batch statistics
            batches: dict[int, int] = {}
            for item in existing:
                b = item["batch"]
                batches[b] = batches.get(b, 0) + 1

            print("\nStats Batch Statistics:")
            for b in sorted(batches.keys()):
                print(f"  Batch {b:02d}: {batches[b]} examples")
            print(f"\n  Total: {num_examples} examples in {num_batches} batches")
            return 0
        else:
            print("Exiting.")
            return 0
    else:
        print("No existing batches found. Generating from scratch...")
        generate_batches(append_mode=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
