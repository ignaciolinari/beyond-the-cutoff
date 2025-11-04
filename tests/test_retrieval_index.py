import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from beyond_the_cutoff.retrieval.index import DocumentIndexer


def test_group_pages_by_section_merges_consecutive_pages() -> None:
    records = [
        {"page": 1, "text": "Intro heading\nBody", "section_title": "1 Introduction"},
        {"page": 2, "text": "More intro"},
        {"page": 3, "text": "Methods heading\nDetails", "section_title": "2 Methods"},
        {"page": 4, "text": "More methods"},
    ]

    groups = DocumentIndexer._group_pages_by_section(records)

    assert len(groups) == 2
    first, second = groups
    assert first["section_title"] == "1 Introduction"
    assert first["start_page"] == 1
    assert first["end_page"] == 2
    assert "More intro" in first["text"]

    assert second["section_title"] == "2 Methods"
    assert second["start_page"] == 3
    assert second["end_page"] == 4


def test_build_index_section_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    paper_path = input_dir / "paper.txt"
    paper_path.write_text("placeholder", encoding="utf-8")
    sidecar_path = paper_path.with_suffix(".pages.jsonl")
    pages = [
        {"page": 1, "text": "Intro heading\nIntro body", "section_title": "1 Introduction"},
        {"page": 2, "text": "Continuation of intro"},
        {"page": 3, "text": "2 Methods\nMethod body", "section_title": "2 Methods"},
    ]
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for rec in pages:
            handle.write(json.dumps(rec) + "\n")

    class DummyModel:
        def __init__(self, dimension: int = 4) -> None:
            self.dimension = dimension

        def encode(
            self,
            texts: Sequence[str],
            convert_to_numpy: bool = True,
            show_progress_bar: bool = True,
        ) -> NDArray[np.float32]:
            vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for idx in range(len(texts)):
                vectors[idx, idx % self.dimension] = 1.0
            return vectors

    class DummyIndexer(DocumentIndexer):
        def _load_model(self) -> Any:
            return DummyModel()

    indexer = DummyIndexer(embedding_model="dummy")

    _, mapping_path = indexer.build_index(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=200,
        chunk_overlap=0,
        chunking_strategy="words",
    )

    with mapping_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2

    intro_row = rows[0]
    methods_row = rows[1]

    assert intro_row["section_title"] == "1 Introduction"
    assert intro_row["page"] == "1"

    assert methods_row["section_title"] == "2 Methods"
    assert methods_row["page"] == "3"

    intro_tokens = int(intro_row["token_end"]) - int(intro_row["token_start"])
    methods_tokens = int(methods_row["token_end"]) - int(methods_row["token_start"])
    assert intro_tokens > 0
    assert methods_tokens > 0
