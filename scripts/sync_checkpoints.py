#!/usr/bin/env python3
"""Synchronise LoRA checkpoint directories between machines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from beyond_the_cutoff.utils.checkpoint_sync import sync_directories

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync LoRA checkpoints between two directories")
    parser.add_argument("--source", required=True, help="Source directory (local path)")
    parser.add_argument("--destination", required=True, help="Destination directory (local path)")
    parser.add_argument(
        "--mode",
        choices={"push", "pull"},
        default="push",
        help="Push copies source→destination, pull copies destination→source",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Optional glob patterns to include (defaults to checkpoint-friendly extensions)",
    )
    parser.add_argument("--exclude", nargs="*", help="Optional glob patterns to exclude")
    parser.add_argument(
        "--dry-run", action="store_true", help="List planned copies without writing"
    )
    parser.add_argument(
        "--checksum", action="store_true", help="Use SHA256 checksums for change detection"
    )
    parser.add_argument("--verbose", action="store_true", help="Log copied and skipped files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.mode == "push":
        src = Path(args.source)
        dst = Path(args.destination)
    else:
        src = Path(args.destination)
        dst = Path(args.source)

    result = sync_directories(
        src,
        dst,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        checksum=args.checksum,
        dry_run=args.dry_run,
    )

    for src_file, dst_file in result.copied:
        if args.verbose:
            logger.info("copied %s -> %s", src_file, dst_file)
    for src_file, _ in result.skipped:
        if args.verbose:
            logger.info("skipped %s (up-to-date)", src_file)
    for src_file, err in result.errors:
        logger.error("error copying %s: %s", src_file, err)

    summary = result.as_dict()
    logger.info(
        "Sync summary: copied=%s skipped=%s errors=%s",
        summary["copied"],
        summary["skipped"],
        summary["errors"],
    )

    if result.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
