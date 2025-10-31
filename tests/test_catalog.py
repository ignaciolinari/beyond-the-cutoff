from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
import pytest

from beyond_the_cutoff.config import ProjectConfig
from beyond_the_cutoff.data.catalog import build_metadata_catalog
from beyond_the_cutoff.data.manifest import build_processed_manifest


def _write_metadata_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_build_processed_manifest(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    split = "arxiv_2025"
    canonical_id = "1234.5678"

    text_path = processed_root / split / "papers" / f"{canonical_id}.txt"
    pages_path = text_path.with_suffix(".pages.jsonl")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("foo", encoding="utf-8")
    with pages_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"page": 1, "text": "foo"}) + "\n")

    manifest_path = build_processed_manifest(processed_root)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["total_documents"] == 1
    entry = data["documents"][0]
    assert entry["document_id"] == f"{split}/papers/{canonical_id}"
    assert entry["page_count"] == 1


def test_build_metadata_catalog(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    split = "arxiv_2025"
    canonical_id = "1234.5678"
    processed_root.mkdir()

    # prepare processed artifact and manifest
    text_rel = Path(split) / "papers" / canonical_id
    text_path = processed_root / f"{text_rel}.txt"
    pages_path = processed_root / f"{text_rel}.pages.jsonl"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("document body", encoding="utf-8")
    with pages_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"page": 1, "text": "document body"}) + "\n")

    manifest_path = build_processed_manifest(processed_root)

    # raw metadata entry aligned with manifest
    record = {
        "canonical_id": canonical_id,
        "arxiv_id": f"{canonical_id}v1",
        "version": "1",
        "title": "Sample Title",
        "summary": "Abstract",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.AI"],
        "primary_category": "cs.AI",
        "published": "2025-01-01T00:00:00+00:00",
        "updated": "2025-01-01T00:00:00+00:00",
        "pdf_url": f"https://arxiv.org/pdf/{canonical_id}v1",
        "link": f"https://arxiv.org/abs/{canonical_id}v1",
    }
    _write_metadata_jsonl(raw_root / split / "metadata.jsonl", [record])

    duplicate_record = {
        **record,
        "arxiv_id": f"{canonical_id}v2",
        "version": "2",
    }
    _write_metadata_jsonl(raw_root / f"{split}_test" / "metadata.jsonl", [duplicate_record])

    paths = ProjectConfig().paths.model_copy(
        update={
            "raw_data": raw_root,
            "processed_data": processed_root,
            "external_data": tmp_path / "external",
        }
    )
    config = ProjectConfig().model_copy(update={"paths": paths})

    artifacts = build_metadata_catalog(
        config=config,
        manifest_path=manifest_path,
        output_prefix=processed_root / "metadata_catalog",
    )

    assert artifacts.csv_path.exists()
    assert artifacts.parquet_path.exists()
    assert artifacts.corpus_path.exists()

    df = pd.read_csv(artifacts.csv_path, dtype={"canonical_id": str})
    assert len(df) == 1
    assert df["canonical_id"].tolist() == [canonical_id]
    assert df["split"].tolist() == [split]
    assert df["authors_joined"].iloc[0] == "Alice; Bob"

    corpus_lines = [
        json.loads(line) for line in artifacts.corpus_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(corpus_lines) == 1
    assert corpus_lines[0]["title"] == "Sample Title"


def test_build_metadata_catalog_requires_metadata(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    text_path = processed_root / "arxiv_2025" / "papers" / "1234.5678.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("body", encoding="utf-8")

    manifest_path = build_processed_manifest(processed_root)

    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    paths = ProjectConfig().paths.model_copy(
        update={
            "raw_data": raw_root,
            "processed_data": processed_root,
            "external_data": tmp_path / "external",
        }
    )
    config = ProjectConfig().model_copy(update={"paths": paths})

    with pytest.raises(ValueError, match="Metadata catalog build found no aligned entries"):
        build_metadata_catalog(
            config=config,
            manifest_path=manifest_path,
            output_prefix=processed_root / "metadata_catalog",
        )
