#!/usr/bin/env python3
"""Generate offline task definitions from processed paper text."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.datasets.tasks import TaskGenerator
from beyond_the_cutoff.models import build_generation_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline task prompts from papers")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base config file")
    parser.add_argument(
        "--generator-config",
        default=None,
        help="Optional config file whose inference section will be used for task generation",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override the processed text directory (defaults to config paths.processed_data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (defaults to config evaluation.offline_tasks_path)",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on documents")
    parser.add_argument(
        "--tasks-per-doc", type=int, default=6, help="Maximum tasks to request per document"
    )
    parser.add_argument(
        "--document-char-limit",
        type=int,
        default=4000,
        help="Maximum characters from each document to include in the prompt",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="Subset of task types to keep (default: qa summary citation)",
    )
    parser.add_argument(
        "--language-hint",
        default=None,
        help="Optional language hint passed to the generator (e.g. 'English' or 'Spanish')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling document shuffle order",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle document order before sampling (default: deterministic sorted order)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("generate_offline_tasks")

    base_cfg = load_config(args.config)
    generator_cfg = load_config(args.generator_config) if args.generator_config else base_cfg

    processed_dir = (
        Path(args.processed_dir).resolve() if args.processed_dir else base_cfg.paths.processed_data
    )
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else base_cfg.evaluation.offline_tasks_path.resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(processed_dir.rglob("*.txt"))
    if not files:
        logger.warning("No processed text files found under %s", processed_dir)
        output_path.write_text("", encoding="utf-8")
        return

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)

    if args.max_docs is not None:
        files = files[: args.max_docs]

    generator_client = build_generation_client(generator_cfg.inference)
    allowed_types = tuple(args.task_types) if args.task_types else None
    task_generator = TaskGenerator(
        generator_client,
        max_tasks_per_doc=args.tasks_per_doc,
        allowed_task_types=allowed_types,
        document_char_limit=args.document_char_limit,
        language_hint=args.language_hint,
    )

    total_tasks = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            records = task_generator.generate(text, doc_path=path)
            if not records:
                logger.info("No tasks generated for %s", path.name)
                continue
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                total_tasks += 1

    logger.info("Generated %d tasks across %d documents", total_tasks, len(files))
    logger.info("Task bank written to %s", output_path)


if __name__ == "__main__":
    main()
