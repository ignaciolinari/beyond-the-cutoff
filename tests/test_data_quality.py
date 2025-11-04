from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from beyond_the_cutoff.retrieval.index import DocumentIndexer
from beyond_the_cutoff.utils.data_quality import (
    ChunkRecord,
    detect_duplicate_chunks,
    validate_citation_spans,
    validate_index_artifacts,
)


class DummySentenceTransformer:
    def __init__(self, _model_name: str):
        self.dimension = 4

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = True,
        **_: Any,
    ) -> npt.NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for idx in range(len(texts)):
            vectors[idx, idx % self.dimension] = 1.0 + (idx % 5)
        return vectors


def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        "beyond_the_cutoff.retrieval.index.SentenceTransformer",
        DummySentenceTransformer,
    )
    yield


patch_sentence_transformer = cast(
    Callable[[pytest.MonkeyPatch], Iterator[None]],
    pytest.fixture(_patch_sentence_transformer),
)


def test_detect_duplicate_chunks_flags_similar_text() -> None:
    records = [
        ChunkRecord(
            source_path="docA",
            chunk_index=0,
            token_start=0,
            token_end=40,
            text="Alpha beta gamma delta epsilon zeta eta theta",
        ),
        ChunkRecord(
            source_path="docA",
            chunk_index=1,
            token_start=40,
            token_end=80,
            text=" alpha  beta gamma   delta epsilon zeta   eta theta ",
        ),
        ChunkRecord(
            source_path="docB",
            chunk_index=0,
            token_start=0,
            token_end=20,
            text="Unique chunk content across documents",
        ),
    ]

    issues = detect_duplicate_chunks(records, min_tokens=4)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.source_path == "docA"
    assert issue.chunk_indices == [0, 1]


def test_detect_duplicate_chunks_ignores_short_snippets() -> None:
    records = [
        ChunkRecord(
            source_path="docA",
            chunk_index=0,
            token_start=0,
            token_end=5,
            text="Figure caption",
        ),
        ChunkRecord(
            source_path="docA",
            chunk_index=1,
            token_start=5,
            token_end=10,
            text="Figure caption",
        ),
    ]

    issues = detect_duplicate_chunks(records, min_tokens=5)

    assert issues == []


def test_validate_citation_spans_reports_missing_and_invalid() -> None:
    records = [
        ChunkRecord(
            source_path="docA",
            chunk_index=0,
            token_start=0,
            token_end=50,
            text="Chunk with valid span and adequate content for testing purposes.",
        ),
        ChunkRecord(
            source_path="docA",
            chunk_index=1,
            token_start=None,
            token_end=90,
            text="Chunk missing start token span value but long enough to be realistic.",
        ),
        ChunkRecord(
            source_path="docA",
            chunk_index=2,
            token_start=10,
            token_end=5,
            text="Chunk with reversed span ordering that should trigger validation.",
        ),
        ChunkRecord(
            source_path="docA",
            chunk_index=3,
            token_start=4,
            token_end=30,
            text="This chunk starts before the previous valid span and must raise a flag.",
        ),
        ChunkRecord(
            source_path="docB",
            chunk_index=0,
            token_start=0,
            token_end=None,
            text="Different document lacking an end token span.",
        ),
    ]

    issues = validate_citation_spans(records)

    kinds = {issue.kind for issue in issues}
    assert {"missing_span", "non_positive_length", "non_monotonic_start"}.issubset(kinds)


def test_validate_index_artifacts_happy_path(
    tmp_path: Path, patch_sentence_transformer: None
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    tokens = [f"token{i}" for i in range(160)]
    (processed_dir / "paper.txt").write_text(" ".join(tokens), encoding="utf-8")

    index_dir = tmp_path / "index"
    indexer = DocumentIndexer(embedding_model="dummy")
    index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_dir,
        chunk_size=32,
        chunk_overlap=8,
        chunking_strategy="words",
    )

    report = validate_index_artifacts(index_path, mapping_path)

    assert report.mapping_count == report.vector_count
    assert report.dimension == DummySentenceTransformer("dummy").dimension
    assert report.issues == []
    assert report.duplicate_chunks == []
    assert report.span_issues == []


def test_validate_index_artifacts_detects_mapping_mismatch(
    tmp_path: Path, patch_sentence_transformer: None
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    tokens = [f"word{i}" for i in range(120)]
    (processed_dir / "paper.txt").write_text(" ".join(tokens), encoding="utf-8")

    index_dir = tmp_path / "index"
    indexer = DocumentIndexer(embedding_model="dummy")
    index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_dir,
        chunk_size=24,
        chunk_overlap=6,
        chunking_strategy="words",
    )

    broken_mapping = index_dir / "mapping_broken.tsv"
    with (
        mapping_path.open("r", encoding="utf-8") as source,
        broken_mapping.open("w", encoding="utf-8") as target,
    ):
        lines = source.readlines()
        target.writelines(lines[:-1])

    report = validate_index_artifacts(index_path, broken_mapping)

    assert any(issue.kind == "mapping_size_mismatch" for issue in report.issues)
    assert report.mapping_count + 1 == report.vector_count
