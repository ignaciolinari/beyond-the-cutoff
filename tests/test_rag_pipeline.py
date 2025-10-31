from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from beyond_the_cutoff.config import ProjectConfig
from beyond_the_cutoff.models.ollama import OllamaClient
from beyond_the_cutoff.retrieval.index import DocumentIndexer
from beyond_the_cutoff.retrieval.query import RAGPipeline


# pytest's decorator lacks precise typing under strict mypy, ignore the mismatch.
@pytest.fixture(autouse=True)  # type: ignore[misc]
def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "beyond_the_cutoff.retrieval.index.SentenceTransformer", DummySentenceTransformer
    )
    monkeypatch.setattr(
        "beyond_the_cutoff.retrieval.query.SentenceTransformer", DummySentenceTransformer
    )
    monkeypatch.setattr("beyond_the_cutoff.retrieval.query.CrossEncoder", DummyCrossEncoder)


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
            vec[3] = float(text.count("research"))
            embeddings.append(vec)
        return np.vstack(embeddings)


class DummyCrossEncoder:
    def __init__(self, _model_name: str):
        raise RuntimeError("cross-encoder unavailable")


class DummyOllamaClient(OllamaClient):
    def __init__(self, response: str):
        super().__init__(model="dummy")
        self._response = response

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = prompt  # unused
        _ = stream
        _ = options
        return {"response": self._response}


def test_rag_pipeline_basic(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    sample_text = " ".join(
        [
            "This research explores retrieval augmented generation for papers.",
            "The methodology section describes indexing and chunking strategies.",
            "Results indicate improvements in grounded answers across evaluations.",
        ]
    )
    text_path = processed_dir / "paper.txt"
    text_path.write_text(sample_text, encoding="utf-8")

    index_dir = tmp_path / "index"
    indexer = DocumentIndexer(embedding_model="dummy")
    index_path, mapping_path = indexer.build_index(
        input_dir=processed_dir,
        output_dir=index_dir,
        chunk_size=20,
        chunk_overlap=5,
        chunking_strategy="words",
    )

    base_cfg = ProjectConfig()
    retrieval_cfg = base_cfg.retrieval.model_copy(
        update={"embedding_model": "dummy", "top_k": 2, "reranker_model": "dummy"}
    )
    cfg = base_cfg.model_copy(update={"retrieval": retrieval_cfg})

    pipeline = RAGPipeline(cfg, index_path=index_path, mapping_path=mapping_path)
    dummy_answer = "The methodology is described in the document [1]."
    client = DummyOllamaClient(dummy_answer)
    result = pipeline.ask("What is the methodology?", client=client)

    assert result["answer"] == dummy_answer
    assert result["citations"], "Expected citations in the response"
    assert "excerpt" in result["citations"][0]
    assert "rendered_context" in result["citations"][0]

    verification = result["citation_verification"]
    assert verification["referenced"] == [1]
    assert verification["extra"] == []
    assert 1 in verification["coverage"]
