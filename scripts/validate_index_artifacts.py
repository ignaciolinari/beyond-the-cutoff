#!/usr/bin/env python3
"""Validate FAISS index artifacts for duplicate chunks and span integrity."""

from __future__ import annotations

import argparse
from pathlib import Path

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.utils.data_quality import validate_index_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate retrieval index artifacts")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to project configuration file",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Path to index.faiss (overrides config-derived path)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Path to mapping.tsv (overrides config-derived path)",
    )
    parser.add_argument(
        "--allow-missing-spans",
        action="store_true",
        help="Treat missing token_start/token_end values as warnings instead of errors",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    cfg = load_config(args.config)
    default_index_dir = cfg.paths.external_data / "index"
    index_path = Path(args.index) if args.index else default_index_dir / "index.faiss"
    mapping_path = Path(args.mapping) if args.mapping else default_index_dir / "mapping.tsv"
    return index_path.resolve(), mapping_path.resolve()


def main() -> None:
    args = parse_args()
    index_path, mapping_path = resolve_paths(args)

    if not index_path.exists():
        raise SystemExit(f"Index file not found: {index_path}")
    if not mapping_path.exists():
        raise SystemExit(f"Mapping file not found: {mapping_path}")

    report = validate_index_artifacts(index_path, mapping_path)

    print("=== Index Validation Report ===")
    print(f"Mapping rows: {report.mapping_count}")
    if report.vector_count is not None:
        print(f"Index vectors: {report.vector_count}")
    else:
        print("Index vectors: (unknown)")
    if report.dimension is not None:
        print(f"Vector dimension: {report.dimension}")
    if report.metadata:
        model = report.metadata.get("embedding_model")
        if model:
            print(f"Metadata embedding model: {model}")

    failure = False

    if report.issues:
        failure = True
        print("\nIssues:")
        for issue in report.issues:
            print(f"  - [{issue.kind}] {issue.detail}")
    else:
        print("\nIssues: none")

    if report.duplicate_chunks:
        failure = True
        print("\nDuplicate chunks detected:")
        for item in report.duplicate_chunks[:10]:
            indices = ",".join(str(idx) for idx in item.chunk_indices)
            print(f"  - {item.source_path} | chunks [{indices}] | sample='{item.sample_text}'")
        if len(report.duplicate_chunks) > 10:
            remaining = len(report.duplicate_chunks) - 10
            print(f"  ... and {remaining} more duplicates")
    else:
        print("\nDuplicate chunks: none")

    span_issues = list(report.span_issues)
    if args.allow_missing_spans:
        filtered = [issue for issue in span_issues if issue.kind != "missing_span"]
        if len(filtered) != len(span_issues):
            print("\nMissing spans treated as warnings (filtered)")
        span_issues = filtered

    if span_issues:
        failure = True
        print("\nSpan issues detected:")
        for span_issue in span_issues[:10]:
            print(
                f"  - {span_issue.source_path} | chunk {span_issue.chunk_index} | "
                f"{span_issue.kind} | start={span_issue.token_start} end={span_issue.token_end}"
            )
        if len(span_issues) > 10:
            remaining = len(span_issues) - 10
            print(f"  ... and {remaining} more span issues")
    else:
        print("\nSpan issues: none")

    if failure:
        raise SystemExit("Index validation failed")
    print("\nIndex validation passed without issues.")


if __name__ == "__main__":
    main()
