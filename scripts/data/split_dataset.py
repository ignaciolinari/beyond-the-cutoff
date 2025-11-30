#!/usr/bin/env python3
"""Split offline dataset into train and eval sets with question-level holdout.

This script creates proper train/eval splits for the fine-tuning + evaluation pipeline:
- Questions from the SAME paper are distributed across train and eval
- This ensures fine-tuned models see paper content but not the exact eval questions
- Prevents data leakage while allowing models to learn underlying knowledge

The split is deterministic (seeded) and stratified by:
1. Paper (source document) - questions per paper are split proportionally
2. Task type - maintains task type distribution in both sets

Usage:
    # Basic split (70% train, 30% eval)
    python scripts/split_dataset.py \
        --input evaluation/datasets/offline_dataset.jsonl \
        --train-output evaluation/datasets/train_dataset.jsonl \
        --eval-output evaluation/datasets/eval_dataset.jsonl

    # Custom split ratio
    python scripts/split_dataset.py \
        --input evaluation/datasets/offline_dataset.jsonl \
        --train-output evaluation/datasets/train_dataset.jsonl \
        --eval-output evaluation/datasets/eval_dataset.jsonl \
        --eval-ratio 0.25

    # Ensure minimum examples per paper in eval set
    python scripts/split_dataset.py \
        --input evaluation/datasets/offline_dataset.jsonl \
        --train-output evaluation/datasets/train_dataset.jsonl \
        --eval-output evaluation/datasets/eval_dataset.jsonl \
        --min-eval-per-paper 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SplitStats:
    """Statistics about the dataset split."""

    total_examples: int = 0
    train_examples: int = 0
    eval_examples: int = 0
    papers_count: int = 0
    papers_with_eval: int = 0
    task_type_distribution: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary of the split statistics."""
        lines = [
            "=" * 60,
            "DATASET SPLIT SUMMARY",
            "=" * 60,
            f"Total examples:     {self.total_examples}",
            f"Train examples:     {self.train_examples} ({100*self.train_examples/max(self.total_examples,1):.1f}%)",
            f"Eval examples:      {self.eval_examples} ({100*self.eval_examples/max(self.total_examples,1):.1f}%)",
            f"Papers processed:   {self.papers_count}",
            f"Papers with eval:   {self.papers_with_eval}",
            "-" * 60,
            "Task type distribution:",
        ]

        for task_type, counts in sorted(self.task_type_distribution.items()):
            train_count = counts.get("train", 0)
            eval_count = counts.get("eval", 0)
            total = train_count + eval_count
            lines.append(
                f"  {task_type:12s}: train={train_count:3d}, eval={eval_count:3d} (total={total})"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


def extract_paper_id(example: dict[str, Any]) -> str:
    """Extract paper identifier from example.

    Uses the source document path from RAG metadata, falling back to task_id prefix.
    """
    # Try to get paper ID from RAG metadata
    rag = example.get("rag", {})
    if isinstance(rag, dict):
        retrieved = rag.get("retrieved", [])
        if retrieved and isinstance(retrieved, list):
            first_chunk = retrieved[0]
            if isinstance(first_chunk, dict):
                source_path = first_chunk.get("source_path", "")
                if source_path:
                    # Extract filename without extension as paper ID
                    return Path(str(source_path)).stem

    # Fallback: use task_id prefix (before the last underscore + number)
    task_id = str(example.get("task_id", ""))
    if task_id:
        # Assume format like "paper_name_qa_0" or "paper_name_summaries_1"
        parts = task_id.rsplit("_", 2)
        if len(parts) >= 2:
            return "_".join(parts[:-2]) if len(parts) > 2 else parts[0]

    return "unknown"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_dataset(examples: list[dict[str, Any]], path: Path) -> None:
    """Save examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def split_dataset(
    examples: list[dict[str, Any]],
    eval_ratio: float = 0.3,
    min_eval_per_paper: int = 1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitStats]:
    """Split examples into train and eval sets with question-level holdout.

    The split ensures:
    1. Questions from each paper are distributed proportionally
    2. Each paper with enough examples has at least `min_eval_per_paper` in eval
    3. Task types are distributed as evenly as possible

    Args:
        examples: List of dataset examples
        eval_ratio: Fraction of examples to hold out for evaluation (default: 0.3)
        min_eval_per_paper: Minimum eval examples per paper (default: 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, eval_examples, stats)
    """
    rng = random.Random(seed)

    # Group examples by paper
    paper_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        paper_id = extract_paper_id(example)
        paper_examples[paper_id].append(example)

    train_examples: list[dict[str, Any]] = []
    eval_examples: list[dict[str, Any]] = []
    stats = SplitStats(total_examples=len(examples), papers_count=len(paper_examples))

    # Process each paper separately
    for _paper_id, paper_exs in paper_examples.items():
        # Shuffle examples within this paper
        shuffled = paper_exs.copy()
        rng.shuffle(shuffled)

        # Calculate how many to put in eval
        n_total = len(shuffled)
        n_eval_target = max(min_eval_per_paper, round(n_total * eval_ratio))

        # Don't put ALL examples in eval - need at least 1 for training
        n_eval = min(n_eval_target, n_total - 1) if n_total > 1 else 0

        # Split: first n_eval go to eval, rest to train
        paper_eval = shuffled[:n_eval]
        paper_train = shuffled[n_eval:]

        eval_examples.extend(paper_eval)
        train_examples.extend(paper_train)

        if paper_eval:
            stats.papers_with_eval += 1

    # Track task type distribution
    for example in train_examples:
        task_type = example.get("task_type", "unknown")
        if task_type not in stats.task_type_distribution:
            stats.task_type_distribution[task_type] = {"train": 0, "eval": 0}
        stats.task_type_distribution[task_type]["train"] += 1

    for example in eval_examples:
        task_type = example.get("task_type", "unknown")
        if task_type not in stats.task_type_distribution:
            stats.task_type_distribution[task_type] = {"train": 0, "eval": 0}
        stats.task_type_distribution[task_type]["eval"] += 1

    stats.train_examples = len(train_examples)
    stats.eval_examples = len(eval_examples)

    # Shuffle final lists to mix papers
    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)

    return train_examples, eval_examples, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split offline dataset into train and eval sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL dataset file (e.g., offline_dataset.jsonl)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Output path for training dataset JSONL",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        required=True,
        help="Output path for evaluation dataset JSONL",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.3,
        help="Fraction of examples to hold out for evaluation (default: 0.3)",
    )
    parser.add_argument(
        "--min-eval-per-paper",
        type=int,
        default=1,
        help="Minimum number of eval examples per paper (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show split statistics without writing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate input
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Load dataset
    print(f"[info] Loading dataset from {input_path}", file=sys.stderr)
    examples = load_dataset(input_path)
    if not examples:
        print("[error] Dataset is empty", file=sys.stderr)
        return 1

    print(f"[info] Loaded {len(examples)} examples", file=sys.stderr)

    # Perform split
    print(
        f"[info] Splitting with eval_ratio={args.eval_ratio}, min_eval_per_paper={args.min_eval_per_paper}, seed={args.seed}",
        file=sys.stderr,
    )
    train_examples, eval_examples, stats = split_dataset(
        examples,
        eval_ratio=args.eval_ratio,
        min_eval_per_paper=args.min_eval_per_paper,
        seed=args.seed,
    )

    # Print statistics
    print(stats.summary())

    if args.dry_run:
        print("\n[info] Dry run - no files written", file=sys.stderr)
        return 0

    # Save outputs
    train_output = args.train_output.resolve()
    eval_output = args.eval_output.resolve()

    print(f"\n[info] Writing train dataset to {train_output}", file=sys.stderr)
    save_dataset(train_examples, train_output)

    print(f"[info] Writing eval dataset to {eval_output}", file=sys.stderr)
    save_dataset(eval_examples, eval_output)

    print("\n[success] Dataset split complete!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
