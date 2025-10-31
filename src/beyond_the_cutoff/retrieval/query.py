"""RAG pipeline that retrieves relevant chunks and queries an LLM via Ollama.

This is a minimal implementation intended for local experimentation.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from ..config import ProjectConfig
from ..models import LLMClient, build_generation_client

try:  # pragma: no cover - optional dependency
    faiss = import_module("faiss")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    from ..utils import faiss_stub as faiss
import numpy as np
import numpy.typing as npt

CrossEncoderType = Any
SentenceTransformerType = Any

try:  # pragma: no cover - optional dependency
    _sentence_transformers = import_module("sentence_transformers")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    CrossEncoder = None
    SentenceTransformer = None
else:
    CrossEncoder = getattr(_sentence_transformers, "CrossEncoder", None)
    SentenceTransformer = getattr(_sentence_transformers, "SentenceTransformer", None)

logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline:
    """Load a persisted FAISS index and answer questions with retrieved context."""

    config: ProjectConfig
    index_path: Path
    mapping_path: Path

    def __post_init__(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required to load the retrieval encoder.")
        embedder_cls = cast(Any, SentenceTransformer)
        self._embedder = cast(
            SentenceTransformerType, embedder_cls(self.config.retrieval.embedding_model)
        )
        # Optional cross-encoder reranker
        self._reranker: CrossEncoderType | None = None
        reranker_name = (self.config.retrieval.reranker_model or "").strip()
        if reranker_name:
            if CrossEncoder is None:
                logger.warning(
                    "sentence-transformers is not installed; reranker %s is disabled.",
                    reranker_name,
                )
            else:
                try:  # pragma: no cover - model download
                    reranker_cls = cast(Any, CrossEncoder)
                    self._reranker = cast(CrossEncoderType, reranker_cls(reranker_name))
                except Exception as exc:
                    self._reranker = None
                    logger.warning(
                        "Failed to load reranker model %s; continuing without reranking. Error: %s",
                        reranker_name,
                        exc,
                    )
        # Load mapping rows: (text, source_path)
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._pages: list[int | None] = []
        self._sections: list[str | None] = []
        self._token_spans: list[tuple[int | None, int | None]] = []
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
                span_start: int | None = None
                span_end: int | None = None
                raw_start = row.get("token_start")
                raw_end = row.get("token_end")
                if isinstance(raw_start, str) and raw_start.strip():
                    try:
                        span_start = int(raw_start)
                    except ValueError:
                        span_start = None
                if isinstance(raw_end, str) and raw_end.strip():
                    try:
                        span_end = int(raw_end)
                    except ValueError:
                        span_end = None
                self._texts.append(text)
                self._sources.append(source_path)
                self._pages.append(page_num)
                section = row.get("section_title")
                self._sections.append(section if section else None)
                self._token_spans.append((span_start, span_end))

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

        self._client: LLMClient | None = None

    def _retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[int, str, str, int | None, str | None, tuple[int | None, int | None], float]]:
        """Return (id, text, source_path, page, section_title, token_span, score)."""
        query_embedding = self._embedder.encode([query], convert_to_numpy=True)
        query_array = np.asarray(query_embedding, dtype="float32")
        query_nd = cast(npt.NDArray[np.float32], query_array)
        faiss.normalize_L2(query_nd)
        scores, idx = self._index.search(query_nd, top_k)
        results: list[
            tuple[int, str, str, int | None, str | None, tuple[int | None, int | None], float]
        ] = []
        for j, score in zip(idx[0], scores[0], strict=True):
            if j == -1:
                continue
            results.append(
                (
                    int(j),
                    self._texts[j],
                    self._sources[j],
                    self._pages[j],
                    self._sections[j] if j < len(self._sections) else None,
                    self._token_spans[j] if j < len(self._token_spans) else (None, None),
                    float(score),
                )
            )

        # Optional reranking with cross-encoder
        if self._reranker and results:
            pairs = [(query, t) for (_id, t, _src, _p, _sec, _span, _sc) in results]
            try:  # pragma: no cover - model inference
                ce_scores = self._reranker.predict(pairs)
            except Exception as exc:
                ce_scores = None
                logger.warning(
                    "Cross-encoder reranker failed during prediction: %s. Falling back to dense scores.",
                    exc,
                )
            if ce_scores is not None:
                with_scores = [(*r, float(ce)) for r, ce in zip(results, ce_scores, strict=False)]
                with_scores.sort(key=lambda x: x[-1], reverse=True)
                # Strip ce score column
                results = [r[:-1] for r in with_scores]
        return results

    def _build_prompt(
        self,
        query: str,
        contexts: list[str],
        max_chars: int,
        *,
        require_citations: bool = True,
        extra_instructions: str | None = None,
    ) -> str:
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
        )
        if require_citations:
            instructions += "Cite the sources inline as [#] based on the order of the snippets. "
        else:
            instructions += "Do not fabricate citations. "
        instructions += "If the answer is not in the context, say you don't know."
        if extra_instructions:
            instructions += " " + extra_instructions.strip()
        return f"{instructions}\n\nContext:\n{context_block}\n\n" f"Question: {query}\nAnswer:"

    @staticmethod
    def _excerpt(text: str, max_chars: int = 200) -> str:
        snippet = text.strip().replace("\n", " ")
        if len(snippet) <= max_chars:
            return snippet
        return snippet[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _verify_citations(answer_text: str, contexts: list[str]) -> dict[str, Any]:
        import re

        marks = [int(m) for m in re.findall(r"\[(\d+)\]", answer_text)]
        unique_marks = sorted(set(marks))
        total = len(contexts)
        missing = [i for i in range(1, total + 1) if i not in unique_marks]
        extra = [i for i in unique_marks if i < 1 or i > total]

        answer_words = set(answer_text.lower().split())
        coverage: dict[int, float] = {}
        for idx in unique_marks:
            if idx < 1 or idx > total:
                continue
            context_words = [w for w in contexts[idx - 1].lower().split() if len(w) > 3]
            if not context_words:
                coverage[idx] = 0.0
                continue
            overlap = sum(1 for w in context_words if w in answer_words)
            coverage[idx] = overlap / max(len(context_words), 1)
        return {
            "referenced": unique_marks,
            "missing": missing,
            "extra": extra,
            "coverage": coverage,
        }

    def verify_citations(self, answer_text: str, contexts: list[str]) -> dict[str, Any]:
        """Public wrapper around :meth:`_verify_citations` for reuse in tooling."""

        return self._verify_citations(answer_text, contexts)

    def prepare_prompt(
        self,
        question: str,
        *,
        require_citations: bool = True,
        extra_instructions: str | None = None,
    ) -> dict[str, Any]:
        """Return retrieval artifacts and a rendered prompt without calling the LLM."""

        top_k = self.config.retrieval.top_k
        max_chars = self.config.retrieval.max_context_chars
        retrieved = self._retrieve(question, top_k)
        contexts = [
            self._render_context(
                text=_t,
                section_title=_sec,
                page=_p,
            )
            for (_id, _t, _src, _p, _sec, _span, _s) in retrieved
        ]
        sources = [f"{s}#page={p}" if p else s for (_id, _t, s, p, _sec, _span, _s) in retrieved]
        prompt = self._build_prompt(
            question,
            contexts,
            max_chars,
            require_citations=require_citations,
            extra_instructions=extra_instructions,
        )

        retrieved_records = [
            {
                "id": _id,
                "text": _t,
                "source_path": _src,
                "page": _p,
                "section_title": _sec,
                "token_start": _span[0],
                "token_end": _span[1],
                "score": _s,
                "excerpt": self._excerpt(_t),
                "rendered_context": contexts[idx] if idx < len(contexts) else _t,
            }
            for idx, (_id, _t, _src, _p, _sec, _span, _s) in enumerate(retrieved)
        ]

        return {
            "prompt": prompt,
            "contexts": contexts,
            "sources": sources,
            "retrieved": retrieved_records,
            "scores": [rec["score"] for rec in retrieved_records],
            "require_citations": require_citations,
            "extra_instructions": extra_instructions,
        }

    def ask(self, question: str, *, client: LLMClient | None = None) -> dict[str, Any]:
        """Answer a user question using retrieval-augmented generation.

        Returns a dict with keys: answer, contexts, sources, scores.
        """

        prepared = self.prepare_prompt(question)
        client = client or self._get_client()
        result = client.generate(prepared["prompt"])
        answer_text = result.get("response", "")

        citations = [
            {
                "id": rec["id"],
                "source_path": rec["source_path"],
                "page": rec["page"],
                "token_start": rec["token_start"],
                "token_end": rec["token_end"],
                "section_title": rec.get("section_title"),
                "score": rec["score"],
                "excerpt": rec["excerpt"],
                "rendered_context": rec.get("rendered_context"),
            }
            for rec in prepared["retrieved"]
        ]

        return {
            "answer": answer_text,
            "contexts": prepared["contexts"],
            "sources": prepared["sources"],
            "scores": prepared["scores"],
            "citations": citations,
            "model": client.model,
            "citation_verification": self._verify_citations(answer_text, prepared["contexts"]),
        }

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = build_generation_client(self.config.inference)
        return self._client

    @staticmethod
    def _render_context(
        text: str,
        *,
        section_title: str | None,
        page: int | None,
    ) -> str:
        """Add lightweight metadata headers to retrieved context chunks."""

        metadata: list[str] = []
        if section_title:
            metadata.append(f"Section: {section_title.strip()}")
        if page is not None:
            metadata.append(f"Page {page}")
        if not metadata:
            return text
        header = " | ".join(metadata)
        return f"{header}\n{text}"
