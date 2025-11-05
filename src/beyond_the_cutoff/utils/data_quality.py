"""Helpers for validating retrieval artifacts and generator inputs."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from . import faiss_stub as _faiss_stub


class _FaissModule(Protocol):
    def read_index(self, path: str | Path) -> Any: ...


if os.environ.get("BTC_USE_FAISS_STUB") == "1":  # pragma: no cover - test/CI path
    _faiss_module = _faiss_stub
else:  # pragma: no cover - optional dependency
    try:
        _faiss_module = import_module("faiss")
    except ModuleNotFoundError:
        _faiss_module = _faiss_stub
    except Exception:
        _faiss_module = _faiss_stub

faiss: _FaissModule = cast(_FaissModule, _faiss_module)


@dataclass(slots=True)
class ChunkRecord:
    """Minimal representation of a chunk row from the FAISS mapping TSV."""

    source_path: str
    chunk_index: int
    token_start: int | None
    token_end: int | None
    text: str


@dataclass(slots=True)
class DuplicateChunkIssue:
    """Information about duplicated chunk text within a single document."""

    source_path: str
    chunk_indices: list[int]
    normalized_text: str
    sample_text: str


@dataclass(slots=True)
class SpanIssue:
    """Details for invalid or missing token span metadata."""

    source_path: str
    chunk_index: int
    kind: str
    token_start: int | None
    token_end: int | None


@dataclass(slots=True)
class IndexValidationIssue:
    """High-level issue detected while validating index artifacts."""

    kind: str
    detail: str


@dataclass(slots=True)
class IndexValidationReport:
    """Aggregated validation results for retrieval artifacts."""

    mapping_count: int
    vector_count: int | None
    dimension: int | None
    metadata: dict[str, Any] | None
    issues: list[IndexValidationIssue]
    duplicate_chunks: list[DuplicateChunkIssue]
    span_issues: list[SpanIssue]


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalise_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def _preview_text(text: str, limit: int = 120) -> str:
    snippet = re.sub(r"\s+", " ", text).strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."


def load_mapping_rows(mapping_path: Path) -> list[ChunkRecord]:
    """Load mapping TSV rows into :class:`ChunkRecord` objects."""

    records: list[ChunkRecord] = []
    with mapping_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_path = str(row.get("source_path", "") or "").strip()
            if not source_path:
                continue
            chunk_index = _as_int(row.get("chunk_index"))
            if chunk_index is None:
                chunk_index = _as_int(row.get("id")) or 0
            token_start = _as_int(row.get("token_start"))
            token_end = _as_int(row.get("token_end"))
            text = str(row.get("text", "") or "")
            records.append(
                ChunkRecord(
                    source_path=source_path,
                    chunk_index=int(chunk_index),
                    token_start=token_start,
                    token_end=token_end,
                    text=text,
                )
            )
    return records


def detect_duplicate_chunks(
    records: Sequence[ChunkRecord],
    *,
    min_tokens: int = 8,
) -> list[DuplicateChunkIssue]:
    """Identify repeated chunks within the same document.

    Short snippets are ignored to avoid flagging boilerplate headers or footers.
    """

    by_source: dict[str, dict[str, list[int]]] = {}
    for record in records:
        tokens = record.text.split()
        if len(tokens) < min_tokens:
            continue
        normalised = _normalise_text(record.text)
        if not normalised:
            continue
        bucket = by_source.setdefault(record.source_path, {})
        bucket.setdefault(normalised, []).append(record.chunk_index)

    issues: list[DuplicateChunkIssue] = []
    for source_path, chunks in by_source.items():
        for normalised, indices in chunks.items():
            if len(indices) <= 1:
                continue
            issues.append(
                DuplicateChunkIssue(
                    source_path=source_path,
                    chunk_indices=sorted(indices),
                    normalized_text=normalised,
                    sample_text=_preview_text(normalised),
                )
            )
    return issues


def validate_citation_spans(
    records: Sequence[ChunkRecord],
    *,
    allow_missing: bool = False,
) -> list[SpanIssue]:
    """Check token spans for validity and monotonic ordering per document."""

    issues: list[SpanIssue] = []
    by_source: dict[str, list[ChunkRecord]] = {}
    for record in records:
        by_source.setdefault(record.source_path, []).append(record)

    for source_path, bucket in by_source.items():
        last_start: int | None = None
        for record in sorted(bucket, key=lambda r: r.chunk_index):
            start = record.token_start
            end = record.token_end
            if start is None or end is None:
                if not allow_missing:
                    issues.append(
                        SpanIssue(
                            source_path=source_path,
                            chunk_index=record.chunk_index,
                            kind="missing_span",
                            token_start=start,
                            token_end=end,
                        )
                    )
                continue
            if start < 0 or end < 0:
                issues.append(
                    SpanIssue(
                        source_path=source_path,
                        chunk_index=record.chunk_index,
                        kind="negative_span",
                        token_start=start,
                        token_end=end,
                    )
                )
            if end <= start:
                issues.append(
                    SpanIssue(
                        source_path=source_path,
                        chunk_index=record.chunk_index,
                        kind="non_positive_length",
                        token_start=start,
                        token_end=end,
                    )
                )
            if last_start is not None and start < last_start:
                issues.append(
                    SpanIssue(
                        source_path=source_path,
                        chunk_index=record.chunk_index,
                        kind="non_monotonic_start",
                        token_start=start,
                        token_end=end,
                    )
                )
            if start is not None:
                last_start = start
    return issues


def validate_index_artifacts(index_path: Path, mapping_path: Path) -> IndexValidationReport:
    """Validate FAISS index, mapping TSV, and associated metadata."""

    records = load_mapping_rows(mapping_path)
    mapping_count = len(records)
    duplicate_chunks = detect_duplicate_chunks(records)
    span_issues = validate_citation_spans(records)

    issues: list[IndexValidationIssue] = []
    vector_count: int | None = None
    dimension: int | None = None

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:  # pragma: no cover - handled in tests
        issues.append(IndexValidationIssue(kind="index_open_failed", detail=str(exc)))
        metadata = _load_index_metadata(index_path)
        return IndexValidationReport(
            mapping_count=mapping_count,
            vector_count=None,
            dimension=None,
            metadata=metadata,
            issues=issues,
            duplicate_chunks=duplicate_chunks,
            span_issues=span_issues,
        )

    vector_count = _extract_vector_count(index)
    dimension = _extract_dimension(index)

    if vector_count is not None and vector_count != mapping_count:
        issues.append(
            IndexValidationIssue(
                kind="mapping_size_mismatch",
                detail=f"mapping rows={mapping_count}, index vectors={vector_count}",
            )
        )

    metadata = _load_index_metadata(index_path)
    if metadata:
        meta_dim = _as_int(metadata.get("embedding_dimension"))
        if dimension is not None and meta_dim is not None and dimension != meta_dim:
            issues.append(
                IndexValidationIssue(
                    kind="dimension_mismatch",
                    detail=f"index dimension={dimension}, metadata dimension={meta_dim}",
                )
            )

    return IndexValidationReport(
        mapping_count=mapping_count,
        vector_count=vector_count,
        dimension=dimension,
        metadata=metadata,
        issues=issues,
        duplicate_chunks=duplicate_chunks,
        span_issues=span_issues,
    )


def _extract_vector_count(index: Any) -> int | None:
    count = getattr(index, "ntotal", None)
    if isinstance(count, int):
        return count
    if hasattr(index, "_vectors"):
        try:
            return int(index._vectors.shape[0])
        except Exception:  # pragma: no cover - best effort
            return None
    return None


def _extract_dimension(index: Any) -> int | None:
    dimension = getattr(index, "d", None)
    if isinstance(dimension, int):
        return dimension
    if hasattr(index, "_vectors"):
        try:
            return int(index._vectors.shape[1])
        except Exception:  # pragma: no cover - best effort
            return None
    return None


def _load_index_metadata(index_path: Path) -> dict[str, Any] | None:
    meta_path = index_path.parent / "index_meta.json"
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    return None


__all__ = [
    "ChunkRecord",
    "DuplicateChunkIssue",
    "SpanIssue",
    "IndexValidationIssue",
    "IndexValidationReport",
    "load_mapping_rows",
    "detect_duplicate_chunks",
    "validate_citation_spans",
    "validate_index_artifacts",
]
