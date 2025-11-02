#!/usr/bin/env python3
"""Lightweight health check for processed corpus exports.

Run after ingestion to ensure each processed document has a matching
`.pages.jsonl` sidecar and sensible token/page counts.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentStats:
    path: Path
    token_count: int
    page_count: int | None
    has_sidecar: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed corpus artifacts")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/arxiv_2025/papers",
        help="Directory containing processed .txt files (and optional .pages.jsonl sidecars)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2500,
        help="Warn when a document has fewer tokens than this threshold",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40000,
        help="Warn when a document has more tokens than this threshold",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit with status 1 if any warnings are encountered",
    )
    return parser.parse_args()


def load_documents(processed_dir: Path) -> list[DocumentStats]:
    documents: list[DocumentStats] = []
    for txt_path in sorted(processed_dir.glob("*.txt")):
        try:
            content = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001 - propagate read issues to report
            raise RuntimeError(f"Failed to read {txt_path}: {exc}") from exc
        tokens = len(content.split())
        sidecar_path = txt_path.with_suffix(".pages.jsonl")
        has_sidecar = sidecar_path.exists()
        page_count = None
        if has_sidecar:
            with sidecar_path.open("r", encoding="utf-8") as fh:
                page_count = sum(1 for _ in fh)
        documents.append(
            DocumentStats(
                path=txt_path,
                token_count=tokens,
                page_count=page_count,
                has_sidecar=has_sidecar,
            )
        )
    return documents


def summarise(documents: list[DocumentStats]) -> None:
    if not documents:
        print("No processed documents found.")
        return

    token_counts = [doc.token_count for doc in documents]
    page_counts = [doc.page_count for doc in documents if doc.page_count is not None]

    print(f"Documents analysed: {len(documents)}")
    print(
        f"Token stats -> min: {min(token_counts)} | median: {statistics.median(token_counts):.0f} | max: {max(token_counts)}"
    )
    if page_counts:
        print(
            f"Page stats -> min: {min(page_counts)} | median: {statistics.median(page_counts):.0f} | max: {max(page_counts)}"
        )
    missing_sidecars = [doc.path.name for doc in documents if not doc.has_sidecar]
    if missing_sidecars:
        print("Missing sidecars:")
        for name in missing_sidecars:
            print(f"  - {name}")
    else:
        print("All documents have sidecars.")


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise SystemExit(f"Processed directory not found: {processed_dir}")

    documents = load_documents(processed_dir)
    summarise(documents)

    warnings: list[str] = []
    if any(not doc.has_sidecar for doc in documents):
        warnings.append("missing_sidecars")
    for doc in documents:
        if doc.token_count < args.min_tokens:
            warnings.append(f"low_tokens:{doc.path.name}:{doc.token_count}")
        if doc.token_count > args.max_tokens:
            warnings.append(f"high_tokens:{doc.path.name}:{doc.token_count}")

    if warnings:
        print("Warnings:")
        for warn in warnings:
            print(f"  - {warn}")
        if args.fail_on_warn:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
