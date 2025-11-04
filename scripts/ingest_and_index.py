#!/usr/bin/env python3
"""Ingest PDFs and build a retrieval index.

Usage:
    python scripts/ingest_and_index.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.data.catalog import build_metadata_catalog
from beyond_the_cutoff.data.manifest import build_processed_manifest
from beyond_the_cutoff.data.pdf_loader import PDFIngestor
from beyond_the_cutoff.retrieval.index import DocumentIndexer
from beyond_the_cutoff.utils.data_quality import validate_index_artifacts


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

    # 2) Refresh processed manifest and metadata catalog
    manifest_path = build_processed_manifest(processed_dir)
    print(f"Processed manifest written to {manifest_path}")

    catalog_prefix = processed_dir / "metadata_catalog"
    artifacts = build_metadata_catalog(
        config=cfg,
        manifest_path=manifest_path,
        output_prefix=catalog_prefix,
    )
    print(
        "Metadata catalog written to "
        f"{artifacts.csv_path}, {artifacts.parquet_path}, {artifacts.corpus_path}"
    )

    # 3) Build FAISS index
    indexer = DocumentIndexer(embedding_model=cfg.retrieval.embedding_model)
    index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_out,
        chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap,
        chunking_strategy=cfg.retrieval.chunking_strategy,
    )
    print(f"Index written to {index_path}\nMapping written to {mapping_path}")

    report = validate_index_artifacts(index_path, mapping_path)
    if report.issues or report.duplicate_chunks or report.span_issues:
        print("\nIndex validation detected potential issues:")
        for issue in report.issues:
            print(f"  - [{issue.kind}] {issue.detail}")
        for dup in report.duplicate_chunks[:5]:
            indices = ",".join(str(idx) for idx in dup.chunk_indices)
            print(f"  - duplicate chunks in {dup.source_path} at indices [{indices}]")
        for span in report.span_issues[:5]:
            print(
                f"  - span issue {span.kind} in {span.source_path} chunk {span.chunk_index} "
                f"(start={span.token_start}, end={span.token_end})"
            )
        raise SystemExit("Index validation failed. Inspect the data-quality warnings above.")
    print("Index validation passed.")


if __name__ == "__main__":
    main()
