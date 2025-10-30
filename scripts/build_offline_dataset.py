#!/usr/bin/env python3
"""Materialize offline RAG prompts and high-quality answers from task definitions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.datasets import OfflineDatasetBuilder, TaskRecord
from beyond_the_cutoff.models import build_generation_client
from beyond_the_cutoff.retrieval.query import RAGPipeline


def _load_options(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse options JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("Options JSON must deserialize to an object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline dataset with precomputed prompts, contexts, and answers"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base config file")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Path to task JSONL file (defaults to evaluation.offline_tasks_path)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (defaults to evaluation.offline_dataset_path)",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing index.faiss and mapping.tsv (defaults to config paths.external_data/index)",
    )
    parser.add_argument(
        "--reference-config",
        default=None,
        help="Optional config file whose inference section will be used for reference answers",
    )
    parser.add_argument(
        "--rag-config",
        default=None,
        help="Optional config file whose inference section will be used for baseline RAG answers",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip generating baseline RAG answers (only produce reference answers)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--extra-instructions",
        default=None,
        help="Additional instructions appended to every prompt",
    )
    parser.add_argument(
        "--reference-options",
        default=None,
        help="JSON dict with extra generation options for the reference model",
    )
    parser.add_argument(
        "--rag-options",
        default=None,
        help="JSON dict with extra generation options for the baseline RAG model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("build_offline_dataset")

    base_cfg = load_config(args.config)
    ref_cfg = load_config(args.reference_config) if args.reference_config else base_cfg
    rag_cfg = load_config(args.rag_config) if args.rag_config else base_cfg

    tasks_path = (
        Path(args.tasks).resolve()
        if args.tasks
        else base_cfg.evaluation.offline_tasks_path.resolve()
    )
    if not tasks_path.exists():
        raise FileNotFoundError(f"Task file not found: {tasks_path}")

    index_dir = (
        Path(args.index_dir).resolve() if args.index_dir else base_cfg.paths.external_data / "index"
    )
    index_dir = index_dir.resolve()
    index_path = index_dir / "index.faiss"
    mapping_path = index_dir / "mapping.tsv"
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            f"FAISS index or mapping TSV not found under {index_dir}. Run scripts/ingest_and_index.py first."
        )

    output_path = (
        Path(args.output).resolve()
        if args.output
        else base_cfg.evaluation.offline_dataset_path.resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_client = build_generation_client(ref_cfg.inference)
    rag_client = None
    if not args.skip_rag:
        rag_client = build_generation_client(rag_cfg.inference)

    pipeline = RAGPipeline(base_cfg, index_path=index_path, mapping_path=mapping_path)
    builder = OfflineDatasetBuilder(
        pipeline,
        reference_client=reference_client,
        rag_client=rag_client,
        reference_options=_load_options(args.reference_options),
        rag_options=_load_options(args.rag_options),
    )

    processed = 0
    written = 0

    with (
        tasks_path.open("r", encoding="utf-8") as input_handle,
        output_path.open("w", encoding="utf-8") as output_handle,
    ):
        for line in input_handle:
            if args.limit is not None and processed >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            processed += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON task on line %d: %s", processed, exc)
                continue
            try:
                task = TaskRecord.from_dict(payload)
            except ValueError as exc:
                logger.warning("Skipping invalid task definition: %s", exc)
                continue

            example = builder.build_example(task, extra_instructions=args.extra_instructions)
            if example is None:
                logger.info("Skipping task %s due to retrieval or generation failure", task.task_id)
                continue

            output_handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            written += 1

    logger.info("Processed %d tasks, wrote %d examples to %s", processed, written, output_path)


if __name__ == "__main__":
    main()
