"""RAG pipeline that retrieves relevant chunks and queries an LLM via Ollama.

This is a minimal implementation intended for local experimentation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from ..config import ProjectConfig
from ..models.ollama import OllamaClient


@dataclass
class RAGPipeline:
    """Load a persisted FAISS index and answer questions with retrieved context."""

    config: ProjectConfig
    index_path: Path
    mapping_path: Path

    def __post_init__(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        self._embedder = SentenceTransformer(self.config.retrieval.embedding_model)
        # Load mapping rows: (text, source_path)
        self._texts: list[str] = []
        self._sources: list[str] = []
        with self.mapping_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text")
                source_path = row.get("source_path")
                if text is None or source_path is None:
                    raise KeyError("Mapping file is missing required columns")
                self._texts.append(text)
                self._sources.append(source_path)

    def _retrieve(self, query: str, top_k: int) -> list[tuple[str, str, float]]:
        """Return (text, source_path, score) for top_k chunks."""
        query_embedding = self._embedder.encode([query], convert_to_numpy=True)
        query_array = np.asarray(query_embedding, dtype="float32")
        query_nd = cast(npt.NDArray[np.float32], query_array)
        faiss.normalize_L2(query_nd)
        scores, idx = self._index.search(query_nd, top_k)
        results: list[tuple[str, str, float]] = []
        for j, score in zip(idx[0], scores[0], strict=True):
            if j == -1:
                continue
            results.append((self._texts[j], self._sources[j], float(score)))
        return results

    def _build_prompt(self, query: str, contexts: list[str], max_chars: int) -> str:
        # Concatenate contexts until max_chars is reached
        assembled: list[str] = []
        current = 0
        for ctx in contexts:
            if current + len(ctx) + 2 > max_chars:
                break
            assembled.append(ctx)
            current += len(ctx) + 2
        context_block = "\n\n".join(assembled)
        instructions = (
            "You are a research paper assistant. Answer the question using the provided context. "
            "Cite the sources inline as [#] based on the order of the snippets. If the answer "
            "is not in the context, say you don't know."
        )
        return f"{instructions}\n\nContext:\n{context_block}\n\n" f"Question: {query}\nAnswer:"

    def ask(self, question: str, *, client: OllamaClient | None = None) -> dict[str, Any]:
        """Answer a user question using retrieval-augmented generation.

        Returns a dict with keys: answer, contexts, sources, scores.
        """
        top_k = self.config.retrieval.top_k
        max_chars = self.config.retrieval.max_context_chars
        retrieved = self._retrieve(question, top_k)
        contexts = [t for (t, _src, _s) in retrieved]
        sources = [s for (_t, s, _s) in retrieved]
        prompt = self._build_prompt(question, contexts, max_chars)

        client = client or OllamaClient(
            model=self.config.inference.model,
            host=self.config.inference.host,
            port=self.config.inference.port,
            timeout=self.config.inference.timeout,
        )
        result = client.generate(prompt)
        return {
            "answer": result.get("response", ""),
            "contexts": contexts,
            "sources": sources,
            "scores": [s for (_t, _src, s) in retrieved],
            "model": client.model,
        }
