from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from beyond_the_cutoff.config import ProjectConfig
from beyond_the_cutoff.retrieval.index import DocumentIndexer
from beyond_the_cutoff.retrieval.offline_dataset import OfflineDatasetGenerator


class DummySentenceTransformer:
    def __init__(self, _model_name: str):
        self.dimension = 4

    def encode(self, texts: list[str], convert_to_numpy: bool = True, **_: Any) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype="float32")
            vec[0] = float(len(text))
            vec[1] = float(sum(ord(c) for c in text[:12]) % 97)
            vec[2] = float(len(text.split()))
            vec[3] = float(text.count("experiment"))
            embeddings.append(vec)
        return np.vstack(embeddings)


class DummyCrossEncoder:
    def __init__(self, _model_name: str):
        raise RuntimeError("cross-encoder unavailable")


class DummyGeneratorClient:
    model = "dummy-generator"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self, prompt: str, *, stream: bool = False, options: Any | None = None
    ) -> dict[str, Any]:
        self.calls.append(prompt)
        payload = {
            "qa": [
                {
                    "question": "What methodology does the paper use?",
                    "answer": "The study applies a retrieval-augmented approach [1].",
                }
            ],
            "summaries": [
                {
                    "instruction": "Summarize the main contribution of the paper.",
                    "response": "The paper introduces a lightweight pipeline that pairs retrieval with local generation.",
                }
            ],
            "citations": [
                {
                    "instruction": "Which section discusses evaluation results?",
                    "answer": "The evaluation results are detailed in the third paragraph [1].",
                }
            ],
        }
        return {"response": json.dumps(payload)}


def test_offline_dataset_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "beyond_the_cutoff.retrieval.index.SentenceTransformer",
        DummySentenceTransformer,
    )
    monkeypatch.setattr(
        "beyond_the_cutoff.retrieval.query.SentenceTransformer",
        DummySentenceTransformer,
    )
    monkeypatch.setattr("beyond_the_cutoff.retrieval.query.CrossEncoder", DummyCrossEncoder)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    text = "This experiment explores retrieval augmented generation. Evaluation results highlight accuracy improvements."
    (processed_dir / "paper.txt").write_text(text, encoding="utf-8")

    index_dir = tmp_path / "index"
    indexer = DocumentIndexer(embedding_model="dummy")
    index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_dir,
        chunk_size=40,
        chunk_overlap=10,
        chunking_strategy="words",
    )

    base_cfg = ProjectConfig()
    retrieval_cfg = base_cfg.retrieval.model_copy(
        update={"embedding_model": "dummy", "top_k": 2, "max_context_chars": 400}
    )
    dataset_cfg = base_cfg.dataset_generation.model_copy(
        update={
            "questions_per_document": 1,
            "summary_prompts_per_document": 1,
            "citation_prompts_per_document": 1,
            "max_documents": 1,
        }
    )
    cfg = base_cfg.model_copy(
        update={"retrieval": retrieval_cfg, "dataset_generation": dataset_cfg}
    )

    generator_client = DummyGeneratorClient()
    offline = OfflineDatasetGenerator(
        cfg,
        index_path=index_path,
        mapping_path=mapping_path,
        generator_client=generator_client,
    )

    output_dataset = tmp_path / "offline.jsonl"
    raw_tasks = tmp_path / "raw.jsonl"
    counters = offline.generate(
        output_dataset_path=output_dataset,
        raw_tasks_path=raw_tasks,
    )

    assert counters["documents"] == 1
    assert counters["qa"] == 1
    assert counters["summaries"] == 1
    assert counters["citations"] == 1

    dataset_lines = [
        json.loads(line) for line in output_dataset.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(dataset_lines) == 3

    qa_entry = next(item for item in dataset_lines if item["task_type"] == "qa")
    assert qa_entry["instruction"] == "What methodology does the paper use?"
    assert qa_entry["expected_response"].startswith("The study applies")
    assert qa_entry["rag"]["contexts"], "Expected contexts in QA entry"

    summary_entry = next(item for item in dataset_lines if item["task_type"] == "summaries")
    assert "Summarize the main contribution" in summary_entry["instruction"]
    assert not summary_entry["metadata"]["require_citations"]
    assert "retrieved_section_titles" in summary_entry["metadata"]

    citation_entry = next(item for item in dataset_lines if item["task_type"] == "citations")
    assert citation_entry["metadata"]["require_citations"]
    citation_meta = citation_entry["rag"]["citations"][0]
    assert citation_meta.get("section_title") is None
    assert "rendered_context" in citation_meta

    raw_payloads = [
        json.loads(line) for line in raw_tasks.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(raw_payloads) == 1
    assert raw_payloads[0]["parsed"]["qa"], "Expected parsed QA tasks"

    # Ensure the generator was invoked
    assert generator_client.calls, "Generator client should have been called"


def test_chunk_index_monotonic_with_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "beyond_the_cutoff.retrieval.index.SentenceTransformer",
        DummySentenceTransformer,
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    text_path = processed_dir / "paper.txt"
    text_path.write_text("placeholder", encoding="utf-8")
    pages_path = text_path.with_suffix(".pages.jsonl")
    with pages_path.open("w", encoding="utf-8") as handle:
        for page_num in range(1, 3):
            payload = {
                "page": page_num,
                "text": " ".join(["token"] * 60),
            }
            handle.write(json.dumps(payload) + "\n")

    index_dir = tmp_path / "index"
    indexer = DocumentIndexer(embedding_model="dummy")
    _index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_dir,
        chunk_size=20,
        chunk_overlap=5,
        chunking_strategy="words",
    )

    with mapping_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        chunk_indices = [int(row["chunk_index"]) for row in reader]

    assert chunk_indices == sorted(chunk_indices)
    assert len(chunk_indices) == len(set(chunk_indices))
