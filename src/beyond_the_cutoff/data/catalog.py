"""Utilities for building metadata catalogs aligned with processed corpora."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ProjectConfig


@dataclass
class CatalogArtifacts:
    """Paths to generated catalog artifacts."""

    csv_path: Path
    parquet_path: Path
    corpus_path: Path


def _iter_metadata_entries(raw_root: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    for split_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        jsonl_path = split_dir / "metadata.jsonl"
        if not jsonl_path.exists():
            continue
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = line.strip()
                if not record:
                    continue
                payload = json.loads(record)
                yield split_dir.name, payload


def _canonical_id(payload: dict[str, Any]) -> str:
    value = payload.get("canonical_id")
    if isinstance(value, str) and value:
        return value
    arxiv_id = payload.get("arxiv_id")
    if isinstance(arxiv_id, str) and arxiv_id:
        return arxiv_id.split("v", 1)[0]
    raise KeyError("metadata record missing canonical identifier")


def _load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = manifest.get("documents")
    if not isinstance(docs, list):
        raise TypeError("Manifest JSON missing 'documents' list")

    by_canonical: dict[str, dict[str, Any]] = {}
    collisions: dict[str, list[str]] = defaultdict(list)

    for entry in docs:
        if not isinstance(entry, dict):
            continue
        document_id = entry.get("document_id")
        if not isinstance(document_id, str):
            continue
        canonical_id = Path(document_id).name
        collisions[canonical_id].append(document_id)
        by_canonical.setdefault(canonical_id, entry)

    dupes = {cid: paths for cid, paths in collisions.items() if len(paths) > 1}
    if dupes:
        joined = ", ".join(f"{cid}: {paths}" for cid, paths in dupes.items())
        raise ValueError(f"Duplicate canonical IDs found in manifest: {joined}")
    return by_canonical


def build_metadata_catalog(
    *,
    config: ProjectConfig,
    manifest_path: Path,
    output_prefix: Path,
) -> CatalogArtifacts:
    """Align raw metadata with processed artifacts and persist catalog exports."""

    processed_root = config.paths.processed_data
    raw_root = config.paths.raw_data

    manifest_index = _load_manifest(manifest_path)

    rows: list[dict[str, Any]] = []
    corpus_records: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped_duplicates: list[str] = []
    seen_canonical: set[str] = set()

    for split, payload in _iter_metadata_entries(raw_root):
        canonical_id = _canonical_id(payload)
        manifest_entry = manifest_index.get(canonical_id)
        if manifest_entry is None:
            missing.append(f"{split}:{canonical_id}")
            continue

        if canonical_id in seen_canonical:
            skipped_duplicates.append(f"{split}:{canonical_id}")
            continue

        seen_canonical.add(canonical_id)

        text_rel = manifest_entry.get("text_path")
        pages_rel = manifest_entry.get("pages_path")
        text_path = processed_root / text_rel if isinstance(text_rel, str) else None
        pages_path = processed_root / pages_rel if isinstance(pages_rel, str) else None

        text_bytes = manifest_entry.get("bytes")
        page_count = manifest_entry.get("page_count")

        authors = payload.get("authors") or []
        categories = payload.get("categories") or []
        summary = payload.get("summary")

        row = {
            "split": split,
            "document_id": manifest_entry.get("document_id"),
            "text_path": text_rel,
            "pages_path": pages_rel,
            "text_bytes": text_bytes,
            "page_count": page_count,
            "canonical_id": canonical_id,
            "arxiv_id": payload.get("arxiv_id"),
            "version": payload.get("version"),
            "title": payload.get("title"),
            "summary": summary,
            "authors": authors,
            "authors_joined": "; ".join(authors) if authors else "",
            "categories": categories,
            "categories_joined": "; ".join(categories) if categories else "",
            "primary_category": payload.get("primary_category"),
            "published": payload.get("published"),
            "updated": payload.get("updated"),
            "pdf_url": payload.get("pdf_url"),
            "abstract_url": payload.get("link"),
        }
        rows.append(row)

        if text_path is None or not text_path.exists():
            missing.append(f"TEXT_MISSING:{split}:{canonical_id}")
            continue

        text = text_path.read_text(encoding="utf-8")
        pages: list[dict[str, Any]] | None = None
        if pages_path and pages_path.exists():
            pages = []
            with pages_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    pages.append(json.loads(raw))

        corpus_entry = {
            "split": split,
            "canonical_id": canonical_id,
            "arxiv_id": payload.get("arxiv_id"),
            "version": payload.get("version"),
            "title": payload.get("title"),
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "primary_category": payload.get("primary_category"),
            "published": payload.get("published"),
            "updated": payload.get("updated"),
            "text": text,
            "page_count": page_count,
            "pages": pages,
            "pdf_url": payload.get("pdf_url"),
            "abstract_url": payload.get("link"),
        }
        corpus_records.append(corpus_entry)

    if missing:
        missing_report = ", ".join(sorted(missing))
        print(f"Warning: Skipped entries due to missing manifest/text: {missing_report}")
    if skipped_duplicates:
        dup_report = ", ".join(sorted(skipped_duplicates))
        print(f"Info: Skipped duplicate metadata entries: {dup_report}")

    if not rows:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(
                "missing manifest or text artifacts for: " + ", ".join(sorted(missing))
            )
        if skipped_duplicates:
            detail_parts.append(
                "duplicate metadata entries skipped for: " + ", ".join(sorted(skipped_duplicates))
            )
        detail = "; ".join(detail_parts)
        raise ValueError(
            "Metadata catalog build found no aligned entries. "
            "Ensure raw metadata JSONL files exist under"
            f" {raw_root} and that {manifest_path} was generated from processed outputs."
            + (f" Details: {detail}." if detail else "")
        )

    df = pd.DataFrame(rows)

    csv_path = output_prefix.with_suffix(".csv")
    parquet_path = output_prefix.with_suffix(".parquet")
    corpus_path = output_prefix.parent / "corpus.jsonl"

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    generated_at = datetime.now(timezone.utc).isoformat()
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in corpus_records:
            payload = {"generated_at": generated_at, **record}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return CatalogArtifacts(csv_path=csv_path, parquet_path=parquet_path, corpus_path=corpus_path)


__all__ = ["build_metadata_catalog", "CatalogArtifacts"]
