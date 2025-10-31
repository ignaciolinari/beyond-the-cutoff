"""Helpers for summarising processed corpus artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_processed_manifest(processed_root: Path) -> Path:
    """Scan processed text outputs and persist a manifest JSON file.

    Returns the path to the generated manifest.
    """

    processed_root = processed_root.resolve()
    manifest_path = processed_root / "manifest.json"

    documents: list[dict[str, object]] = []
    for txt_path in sorted(processed_root.rglob("*.txt")):
        rel = txt_path.relative_to(processed_root)
        pages_path = txt_path.with_suffix(".pages.jsonl")
        page_count = None
        if pages_path.exists():
            try:
                with pages_path.open("r", encoding="utf-8") as handle:
                    page_count = sum(1 for _ in handle)
            except UnicodeDecodeError:
                page_count = None
        documents.append(
            {
                "document_id": rel.with_suffix("").as_posix(),
                "text_path": rel.as_posix(),
                "pages_path": pages_path.relative_to(processed_root).as_posix()
                if pages_path.exists()
                else None,
                "bytes": txt_path.stat().st_size,
                "page_count": page_count,
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(documents),
        "documents": documents,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


__all__ = ["build_processed_manifest"]
