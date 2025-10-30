#!/usr/bin/env python3
"""Ingest PDFs and build a retrieval index.

Usage:
    python scripts/ingest_and_index.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.data.pdf_loader import PDFIngestor
from beyond_the_cutoff.retrieval.index import DocumentIndexer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDFs and build FAISS index")
    p.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    p.add_argument("--source", default=None, help="Override raw data dir of PDFs")
    p.add_argument("--out", default=None, help="Override index output directory")
    p.add_argument(
        "--no-page-sidecars",
        action="store_true",
        help="Skip writing per-page JSONL sidecars during ingestion",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw_dir = Path(args.source) if args.source else cfg.paths.raw_data
    processed_dir = cfg.paths.processed_data
    index_out = Path(args.out) if args.out else (cfg.paths.external_data / "index")

    # 1) Convert PDFs to text
    ingestor = PDFIngestor(
        source_dir=raw_dir,
        target_dir=processed_dir,
        write_sidecars=not args.no_page_sidecars,
    )
    outputs = ingestor.convert_all()
    if not outputs:
        print(f"No PDFs found under {raw_dir}. Skipping index build.")
        return
    print(f"Converted {len(outputs)} PDFs to text under {processed_dir}")

    # 2) Build FAISS index
    indexer = DocumentIndexer(embedding_model=cfg.retrieval.embedding_model)
    index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_out,
        chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap,
        chunking_strategy=cfg.retrieval.chunking_strategy,
    )
    print(f"Index written to {index_path}\nMapping written to {mapping_path}")


if __name__ == "__main__":
    main()
