#!/usr/bin/env python3
"""Generate offline prompts and gold answers for fine-tuning/evaluation experiments."""

from __future__ import annotations

import argparse
import logging
import sys
import time
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing outputs and skip documents already present in raw tasks",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=None,
        help="Number of additional retries when generator output cannot be parsed",
    )
    parser.add_argument(
        "--document",
        dest="documents",
        action="append",
        default=None,
        help=(
            "Restrict generation to specific source document paths. Can be repeated. "
            "Paths must match the mapping TSV entries."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-document progress and other informational messages",
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
    start_time = time.time()
    args = parse_args()

    # Always show INFO level for better visibility during long runs
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 70)
    print("OFFLINE DATASET GENERATION")
    print("=" * 70)

    config_load_start = time.time()
    cfg = load_config(args.config)
    config_load_time = time.time() - config_load_start
    print(f"Config: {args.config} (loaded in {config_load_time:.2f}s)")

    index_path, mapping_path = resolve_paths(cfg, args)
    print(f"Index: {index_path}")
    print(f"Mapping: {mapping_path}")

    if not index_path.exists():
        sys.exit(f"Index file not found: {index_path}")
    if not mapping_path.exists():
        sys.exit(f"Mapping file not found: {mapping_path}")

    dataset_cfg = cfg.dataset_generation
    if args.max_docs is not None:
        dataset_cfg = dataset_cfg.model_copy(update={"max_documents": args.max_docs})
        cfg = cfg.model_copy(update={"dataset_generation": dataset_cfg})
        print(f"Max documents: {args.max_docs}")

    print(f"Generator model: {cfg.dataset_generation.generator.model}")
    print(f"Timeout: {cfg.dataset_generation.generator.timeout}s")
    print(
        f"Tasks per doc: QA={dataset_cfg.questions_per_document}, "
        f"Summary={dataset_cfg.summary_prompts_per_document}, "
        f"Citation={dataset_cfg.citation_prompts_per_document}, "
        f"Contextual={dataset_cfg.contextual_prompts_per_document}"
    )

    generator_init_start = time.time()
    print("\nInitializing generator (loading models)...", flush=True)
    generator = OfflineDatasetGenerator(
        cfg,
        index_path=index_path,
        mapping_path=mapping_path,
    )
    generator_init_time = time.time() - generator_init_start
    print(f"Generator initialized in {generator_init_time:.1f}s")

    output_path = Path(args.output).resolve() if args.output else None
    raw_tasks_path = Path(args.raw_tasks).resolve() if args.raw_tasks else None

    # Count documents
    with open(mapping_path) as f:
        doc_count = len({row.split("\t")[1] for row in f.readlines()[1:]})

    print(f"\nTotal documents in corpus: {doc_count}")
    if args.resume:
        print("Mode: RESUME (will skip already processed documents)")
    else:
        print("Mode: FRESH START")

    print("\n" + "-" * 70)
    print("GENERATION IN PROGRESS")
    print("-" * 70)
    print("(Each document takes 1-5 minutes depending on length)\n", flush=True)

    generation_start = time.time()

    counters = generator.generate(
        output_dataset_path=output_path,
        raw_tasks_path=raw_tasks_path,
        resume=args.resume,
        parse_retries=args.parse_retries,
        documents=[str(Path(doc).resolve()) for doc in args.documents] if args.documents else None,
    )

    generation_time = time.time() - generation_start
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Time: {generation_time/60:.1f} minutes (total: {total_time/60:.1f} min)")
    print()
    print(f"Documents processed: {counters.get('documents', 0)}")
    print(f"Documents skipped:   {counters.get('documents_filtered', 0)}")
    print()
    print("Generated examples:")
    print(f"  QA pairs:      {counters.get('qa', 0)}")
    print(f"  Summaries:     {counters.get('summaries', 0)}")
    print(f"  Citations:     {counters.get('citations', 0)}")
    print(f"  Contextual:    {counters.get('contextual', 0)}")
    print(f"  TOTAL:         {counters.get('examples', 0)}")

    if counters.get("documents", 0) > 0:
        avg_time_per_doc = generation_time / counters.get("documents", 1)
        print(f"\nAverage time per document: {avg_time_per_doc:.1f}s")

    # Show output file locations
    dataset_out = output_path or cfg.dataset_generation.output_dataset_path
    tasks_out = raw_tasks_path or cfg.dataset_generation.raw_tasks_path
    print("\nOutput files:")
    print(f"  Dataset: {dataset_out}")
    print(f"  Raw tasks: {tasks_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
