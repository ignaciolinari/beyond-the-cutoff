#!/usr/bin/env python3
"""Generate batches of pairwise comparisons for quantization experiment with Gemini.

Compares Q4_K_M (quantized) vs F16 (non-quantized) versions of the same model.
Creates text files with batches of 10 examples each, ready to copy-paste into Gemini.

Usage:
    python scripts/utility/generate_quantization_batches.py

Output:
    evaluation/exports/quantization_batches/batch_01.txt
    evaluation/exports/quantization_batches/batch_02.txt
    ...
    evaluation/exports/quantization_batches/batch_mapping.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

BATCH_SIZE = 10

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
    {{"example": 5, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 6, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 7, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 8, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 9, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}},
    {{"example": 10, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}}
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


def load_responses_flexible(base_dir: Path, model_name: str) -> dict[str, dict[str, Any]]:
    """Load responses from jsonl files."""
    jsonl_file = base_dir / f"{model_name}.jsonl"
    if jsonl_file.exists():
        responses = {}
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record.get("task_id", record.get("id", ""))
                    if task_id:
                        if "response" in record and "model_answer" not in record:
                            record["model_answer"] = record["response"]
                        responses[task_id] = record
        return responses

    raise FileNotFoundError(f"Could not find responses for {model_name} at {jsonl_file}")


def generate_batches() -> None:
    """Generate batches for Gemini evaluation."""
    responses_dir = Path("evaluation/responses")

    # Q4_K_M (quantized) vs F16 (non-quantized)
    model_q4 = "hybrid_science_0p5b_rag_trained"  # Quantized
    model_f16 = "hybrid_science_0p5b_rag_trained_f16"  # Non-quantized

    print("Loading responses...")
    responses_q4 = load_responses_flexible(responses_dir, model_q4)
    responses_f16 = load_responses_flexible(responses_dir, model_f16)

    common_ids = sorted(set(responses_q4.keys()) & set(responses_f16.keys()))
    print(f"Found {len(common_ids)} common examples between models")

    # Create output directory
    output_dir = Path("evaluation/exports/quantization_batches")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle for randomization
    random.seed(42)
    shuffled_ids = common_ids.copy()
    random.shuffle(shuffled_ids)

    # Track mapping of which model is A/B for each example
    mapping: list[dict[str, Any]] = []

    batch_num = 0
    for i in range(0, len(shuffled_ids), BATCH_SIZE):
        batch_ids = shuffled_ids[i : i + BATCH_SIZE]
        batch_num += 1

        examples_text = ""
        for j, task_id in enumerate(batch_ids):
            rec_q4 = responses_q4[task_id]
            rec_f16 = responses_f16[task_id]

            # Randomize which model is presented as A vs B
            if random.random() > 0.5:
                # Q4 is presented as "Response A"
                response_a = rec_q4["model_answer"]
                response_b = rec_f16["model_answer"]
                order = "q4_first"
                actual_a = model_q4
                actual_b = model_f16
            else:
                # F16 is presented as "Response A"
                response_a = rec_f16["model_answer"]
                response_b = rec_q4["model_answer"]
                order = "f16_first"
                actual_a = model_f16
                actual_b = model_q4

            examples_text += EXAMPLE_TEMPLATE.format(
                num=j + 1,
                question=rec_q4["instruction"],
                expected=rec_q4["expected_response"],
                response_a=response_a,
                response_b=response_b,
            )

            mapping.append(
                {
                    "batch": batch_num,
                    "example": j + 1,
                    "task_id": task_id,
                    "presented_as_A": actual_a,
                    "presented_as_B": actual_b,
                    "order": order,
                }
            )

        # Adjust template for partial batches
        batch_content = BATCH_TEMPLATE.format(examples=examples_text)

        # If partial batch, adjust the expected JSON format in the template
        if len(batch_ids) < BATCH_SIZE:
            # Replace the full 10-example JSON with correct number
            json_entries = ",\n    ".join(
                [
                    f'{{"example": {k+1}, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}}'
                    for k in range(len(batch_ids))
                ]
            )
            batch_content = batch_content.replace(
                '{"example": 1, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 2, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 3, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 4, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 5, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 6, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 7, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 8, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 9, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"},\n    {"example": 10, "verdict": "A or B or TIE", "confidence": "high/medium/low", "reasoning": "one sentence"}',
                json_entries,
            )

        batch_file = output_dir / f"batch_{batch_num:02d}.txt"
        batch_file.write_text(batch_content)
        print(f"Created {batch_file} ({len(batch_ids)} examples)")

    # Save mapping
    mapping_file = output_dir / "batch_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✅ Generated {batch_num} batches with {len(mapping)} examples")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Output directory: {output_dir}")
    print(f"   Mapping file: {mapping_file}")
    print("\n📋 Copy each batch file content into Gemini for evaluation")


def main() -> int:
    """Main entry point."""
    output_dir = Path("evaluation/exports/quantization_batches")

    if output_dir.exists() and any(output_dir.glob("batch_*.txt")):
        print("⚠️  Existing batches found!")
        confirm = input("Regenerate all batches? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return 0

    generate_batches()
    return 0


if __name__ == "__main__":
    sys.exit(main())
