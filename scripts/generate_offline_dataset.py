#!/usr/bin/env python3
"""Generate offline prompts and gold answers for fine-tuning/evaluation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from beyond_the_cutoff.config import ProjectConfig, load_config
from beyond_the_cutoff.retrieval.offline_dataset import OfflineDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline RAG datasets")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing index.faiss and mapping.tsv (defaults to config paths.external_data/index)",
    )
    parser.add_argument("--index-path", default=None, help="Override path to index.faiss")
    parser.add_argument("--mapping-path", default=None, help="Override path to mapping.tsv")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path for offline dataset (defaults to config.dataset_generation.output_dataset_path)",
    )
    parser.add_argument(
        "--raw-tasks",
        default=None,
        help="Output JSONL path for raw generator tasks (defaults to config.dataset_generation.raw_tasks_path)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of source documents to process",
    )
    return parser.parse_args()


def resolve_paths(cfg: ProjectConfig, args: argparse.Namespace) -> tuple[Path, Path]:
    base_index_dir = (
        Path(args.index_dir).resolve()
        if args.index_dir
        else (cfg.paths.external_data / "index").resolve()
    )

    index_path = (
        Path(args.index_path).resolve() if args.index_path else base_index_dir / "index.faiss"
    )
    mapping_path = (
        Path(args.mapping_path).resolve() if args.mapping_path else base_index_dir / "mapping.tsv"
    )
    return index_path, mapping_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    index_path, mapping_path = resolve_paths(cfg, args)

    if not index_path.exists():
        sys.exit(f"Index file not found: {index_path}")
    if not mapping_path.exists():
        sys.exit(f"Mapping file not found: {mapping_path}")

    dataset_cfg = cfg.dataset_generation
    if args.max_docs is not None:
        dataset_cfg = dataset_cfg.model_copy(update={"max_documents": args.max_docs})
        cfg = cfg.model_copy(update={"dataset_generation": dataset_cfg})

    generator = OfflineDatasetGenerator(
        cfg,
        index_path=index_path,
        mapping_path=mapping_path,
    )

    output_path = Path(args.output).resolve() if args.output else None
    raw_tasks_path = Path(args.raw_tasks).resolve() if args.raw_tasks else None
    counters = generator.generate(
        output_dataset_path=output_path,
        raw_tasks_path=raw_tasks_path,
    )

    summary = (
        f"Processed {counters.get('documents', 0)} documents | "
        f"QA: {counters.get('qa', 0)} | Summaries: {counters.get('summaries', 0)} | "
        f"Citation checks: {counters.get('citations', 0)}"
    )
    print(summary)


if __name__ == "__main__":
    main()
