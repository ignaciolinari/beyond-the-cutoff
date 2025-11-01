#!/usr/bin/env python3
"""Generate structured metadata and corpus exports for processed documents.

Usage:
    python scripts/build_metadata_catalog.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.data.catalog import build_metadata_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed metadata catalog")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to project configuration file",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit path to processed manifest JSON",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for generated catalog files (defaults to processed dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    processed_root = cfg.paths.processed_data

    manifest_path = (
        Path(args.manifest).resolve() if args.manifest else processed_root / "manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found at {manifest_path}")

    output_prefix = (
        Path(args.output_prefix).resolve()
        if args.output_prefix
        else processed_root / "metadata_catalog"
    )
    artifacts = build_metadata_catalog(
        config=cfg,
        manifest_path=manifest_path,
        output_prefix=output_prefix,
    )

    print(f"Wrote {artifacts.csv_path}")
    if artifacts.parquet_path is not None:
        print(f"Wrote {artifacts.parquet_path}")
    else:
        print("Skipped parquet export (install 'pyarrow' or 'fastparquet' to enable)")
    print(f"Wrote {artifacts.corpus_path}")


if __name__ == "__main__":
    main()
