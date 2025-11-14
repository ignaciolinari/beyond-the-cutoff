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

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting offline dataset generation")
    logger.info("=" * 80)

    config_load_start = time.time()
    cfg = load_config(args.config)
    config_load_time = time.time() - config_load_start
    logger.info(f"Loaded configuration from {args.config} ({config_load_time:.2f}s)")

    index_path, mapping_path = resolve_paths(cfg, args)
    logger.info(f"Using index: {index_path}")
    logger.info(f"Using mapping: {mapping_path}")

    if not index_path.exists():
        sys.exit(f"Index file not found: {index_path}")
    if not mapping_path.exists():
        sys.exit(f"Mapping file not found: {mapping_path}")

    dataset_cfg = cfg.dataset_generation
    if args.max_docs is not None:
        dataset_cfg = dataset_cfg.model_copy(update={"max_documents": args.max_docs})
        cfg = cfg.model_copy(update={"dataset_generation": dataset_cfg})
        logger.info(f"Limited to {args.max_docs} documents")

    generator_init_start = time.time()
    generator = OfflineDatasetGenerator(
        cfg,
        index_path=index_path,
        mapping_path=mapping_path,
    )
    generator_init_time = time.time() - generator_init_start
    logger.info(f"Initialized OfflineDatasetGenerator ({generator_init_time:.2f}s)")
    logger.info(f"Generator model: {cfg.dataset_generation.generator.model}")
    logger.info(
        f"Target tasks per document: QA={dataset_cfg.questions_per_document}, "
        f"Summaries={dataset_cfg.summary_prompts_per_document}, "
        f"Citations={dataset_cfg.citation_prompts_per_document}, "
        f"Contextual={dataset_cfg.contextual_prompts_per_document}"
    )

    output_path = Path(args.output).resolve() if args.output else None
    raw_tasks_path = Path(args.raw_tasks).resolve() if args.raw_tasks else None
    if output_path:
        logger.info(f"Output dataset: {output_path}")
    if raw_tasks_path:
        logger.info(f"Raw tasks: {raw_tasks_path}")
    if args.resume:
        logger.info("Resume mode: appending to existing outputs")

    logger.info("-" * 80)
    logger.info("Starting generation...")
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

    logger.info("-" * 80)
    logger.info("Generation completed!")
    logger.info(f"Generation time: {generation_time:.2f}s ({generation_time/60:.1f} minutes)")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    summary = (
        f"Processed {counters.get('documents', 0)} documents | "
        f"QA: {counters.get('qa', 0)} | Summaries: {counters.get('summaries', 0)} | "
        f"Contextual: {counters.get('contextual', 0)} | "
        f"Citation checks: {counters.get('citations', 0)}"
    )
    if counters.get("documents_filtered"):
        summary += f" | Skipped {counters['documents_filtered']} (filters)"
    if "documents_requested" in counters:
        summary += (
            f" | Requested {counters['documents_requested']}"
            f" (found {counters['documents_found']}, missing {counters['documents_missing']})"
        )

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    print(summary)
    if counters.get("documents", 0) > 0:
        avg_time_per_doc = generation_time / counters.get("documents", 1)
        logger.info(f"Average time per document: {avg_time_per_doc:.2f}s")


if __name__ == "__main__":
    main()
