"""RAG pipeline that retrieves relevant chunks and queries an LLM via Ollama.

This is a minimal implementation intended for local experimentation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder, SentenceTransformer

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
        # Optional cross-encoder reranker
        self._reranker: CrossEncoder | None = None
        reranker_name = (self.config.retrieval.reranker_model or "").strip()
        if reranker_name:
            try:  # pragma: no cover - model download
                self._reranker = CrossEncoder(reranker_name)
            except Exception:
                self._reranker = None
        # Load mapping rows: (text, source_path)
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._pages: list[int | None] = []
        with self.mapping_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text")
                source_path = row.get("source_path")
                if text is None or source_path is None:
                    raise KeyError("Mapping file is missing required columns")
                page_val = row.get("page")
                page_num: int | None = None
                if isinstance(page_val, str):
                    normalized = page_val.strip()
                    if normalized and normalized.lower() != "none":
                        try:
                            page_num = int(normalized)
                        except ValueError:
                            page_num = None
                self._texts.append(text)
                self._sources.append(source_path)
                self._pages.append(page_num)

        # Validate index metadata if present
        meta_path = self.index_path.parent / "index_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                idx_model = str(meta.get("embedding_model", ""))
                if idx_model and idx_model != self.config.retrieval.embedding_model:
                    raise ValueError(
                        "Index embedding model does not match config. Found "
                        f"{idx_model!r}, expected {self.config.retrieval.embedding_model!r}. "
                        "Rebuild the index with scripts/ingest_and_index.py."
                    )
            except Exception:
                # Non-fatal; proceed with best effort
                pass

    def _retrieve(self, query: str, top_k: int) -> list[tuple[int, str, str, int | None, float]]:
        """Return (id, text, source_path, page, score) for top_k chunks."""
        query_embedding = self._embedder.encode([query], convert_to_numpy=True)
        query_array = np.asarray(query_embedding, dtype="float32")
        query_nd = cast(npt.NDArray[np.float32], query_array)
        faiss.normalize_L2(query_nd)
        scores, idx = self._index.search(query_nd, top_k)
        results: list[tuple[int, str, str, int | None, float]] = []
        for j, score in zip(idx[0], scores[0], strict=True):
            if j == -1:
                continue
            results.append((int(j), self._texts[j], self._sources[j], self._pages[j], float(score)))

        # Optional reranking with cross-encoder
        if self._reranker and results:
            pairs = [(query, t) for (_id, t, _s, _p, _sc) in results]
            try:  # pragma: no cover - model inference
                ce_scores = self._reranker.predict(pairs)
            except Exception:
                ce_scores = None
            if ce_scores is not None:
                with_scores = [(*r, float(ce)) for r, ce in zip(results, ce_scores, strict=False)]
                with_scores.sort(key=lambda x: x[-1], reverse=True)
                # Strip ce score column
                results = [r[:-1] for r in with_scores]
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
        contexts = [t for (_id, t, _src, _p, _s) in retrieved]
        # Render sources with page numbers when available
        sources = [f"{s}#page={p}" if p else s for (_id, _t, s, p, _s) in retrieved]
        prompt = self._build_prompt(question, contexts, max_chars)

        client = client or OllamaClient(
            model=self.config.inference.model,
            host=self.config.inference.host,
            port=self.config.inference.port,
            timeout=self.config.inference.timeout,
        )
        result = client.generate(prompt)
        citations = [
            {
                "id": _id,
                "source_path": _src,
                "page": _p,
                "score": _s,
            }
            for (_id, _t, _src, _p, _s) in retrieved
        ]
        return {
            "answer": result.get("response", ""),
            "contexts": contexts,
            "sources": sources,
            "scores": [s for (_id, _t, _src, _p, s) in retrieved],
            "citations": citations,
            "model": client.model,
        }
