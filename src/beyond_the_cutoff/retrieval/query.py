"""RAG pipeline that retrieves relevant chunks and queries an LLM via Ollama.

This is a minimal implementation intended for local experimentation.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class RetrievedChunk:
    """Representation of a retrieved chunk with scoring metadata."""

    chunk_id: int
    text: str
    source_path: str
    page: int | None
    section_title: str | None
    token_span: tuple[int | None, int | None]
    similarity_score: float
    reranker_score: float | None = None

    @property
    def ordering_score(self) -> float:
        return self.reranker_score if self.reranker_score is not None else self.similarity_score


@dataclass(slots=True)
class RAGPipeline:
    """Load a persisted FAISS index and answer questions with retrieved context."""

    config: ProjectConfig
    index_path: Path
    mapping_path: Path
    _index: Any = field(init=False, repr=False)
    _embedder: SentenceTransformerType = field(init=False, repr=False)
    _reranker: CrossEncoderType | None = field(init=False, repr=False, default=None)
    _texts: list[str] = field(init=False, repr=False, default_factory=list)
    _sources: list[str] = field(init=False, repr=False, default_factory=list)
    _pages: list[int | None] = field(init=False, repr=False, default_factory=list)
    _sections: list[str | None] = field(init=False, repr=False, default_factory=list)
    _token_spans: list[tuple[int | None, int | None]] = field(
        init=False, repr=False, default_factory=list
    )
    _client: LLMClient | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required to load the retrieval encoder.")
        embedder_cls = cast(Any, SentenceTransformer)
        self._embedder = cast(
            SentenceTransformerType, embedder_cls(self.config.retrieval.embedding_model)
        )
        # Optional cross-encoder reranker
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
        self._texts.clear()
        self._sources.clear()
        self._pages.clear()
        self._sections.clear()
        self._token_spans.clear()
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

    def _retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return retrieved chunk metadata ordered by relevance."""

        query_embedding = self._embedder.encode([query], convert_to_numpy=True)
        query_array = np.asarray(query_embedding, dtype="float32")
        query_nd = cast(npt.NDArray[np.float32], query_array)
        faiss.normalize_L2(query_nd)
        scores, idx = self._index.search(query_nd, top_k)
        results: list[RetrievedChunk] = []
        for j, score in zip(idx[0], scores[0], strict=True):
            if j == -1:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=int(j),
                    text=self._texts[j],
                    source_path=self._sources[j],
                    page=self._pages[j],
                    section_title=self._sections[j] if j < len(self._sections) else None,
                    token_span=self._token_spans[j] if j < len(self._token_spans) else (None, None),
                    similarity_score=float(score),
                )
            )

        # Optional reranking with cross-encoder
        if self._reranker and results:
            pairs = [(query, chunk.text) for chunk in results]
            try:  # pragma: no cover - model inference
                ce_scores = self._reranker.predict(pairs)
            except Exception as exc:
                ce_scores = None
                logger.warning(
                    "Cross-encoder reranker failed during prediction: %s. Falling back to dense scores.",
                    exc,
                )
            if ce_scores is not None:
                for chunk, ce in zip(results, ce_scores, strict=False):
                    if chunk is None:
                        continue
                    chunk.reranker_score = float(ce)
                results.sort(
                    key=lambda c: c.reranker_score
                    if c.reranker_score is not None
                    else c.similarity_score,
                    reverse=True,
                )
            else:
                results.sort(key=lambda c: c.similarity_score, reverse=True)
        else:
            results.sort(key=lambda c: c.similarity_score, reverse=True)
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
        valid_marks = [idx for idx in unique_marks if 1 <= idx <= total]
        missing = [i for i in range(1, total + 1) if i not in valid_marks]
        extra = [i for i in unique_marks if i not in valid_marks]

        answer_terms = [w for w in answer_text.lower().split() if len(w) > 3]
        answer_vocab = set(answer_terms)
        coverage: dict[int, float] = {}
        for idx in valid_marks:
            context_vocab = {w for w in contexts[idx - 1].lower().split() if len(w) > 3}
            if not context_vocab or not answer_vocab:
                coverage[idx] = 0.0
                continue
            overlap = len(answer_vocab & context_vocab)
            coverage[idx] = overlap / max(len(answer_vocab), 1)
        return {
            "referenced": valid_marks,
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
        top_k_override: int | None = None,
    ) -> dict[str, Any]:
        """Return retrieval artifacts and a rendered prompt without calling the LLM."""

        top_k = top_k_override if top_k_override is not None else self.config.retrieval.top_k
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        max_chars = self.config.retrieval.max_context_chars
        retrieved = self._retrieve(question, top_k)
        contexts_raw = [
            self._render_context(
                text=chunk.text,
                section_title=chunk.section_title,
                page=chunk.page,
            )
            for chunk in retrieved
        ]
        numbered_contexts = [
            self._prefix_context_with_index(idx, ctx)
            for idx, ctx in enumerate(contexts_raw, start=1)
        ]
        sources = [
            f"{chunk.source_path}#page={chunk.page}" if chunk.page else chunk.source_path
            for chunk in retrieved
        ]
        prompt = self._build_prompt(
            question,
            numbered_contexts,
            max_chars,
            require_citations=require_citations,
            extra_instructions=extra_instructions,
        )

        retrieved_records = [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "source_path": chunk.source_path,
                "page": chunk.page,
                "section_title": chunk.section_title,
                "token_start": chunk.token_span[0],
                "token_end": chunk.token_span[1],
                "score": chunk.ordering_score,
                "similarity_score": chunk.similarity_score,
                "reranker_score": chunk.reranker_score,
                "excerpt": self._excerpt(chunk.text),
                "rendered_context": numbered_contexts[idx]
                if idx < len(numbered_contexts)
                else chunk.text,
                "ordinal": idx + 1,
            }
            for idx, chunk in enumerate(retrieved)
        ]

        return {
            "prompt": prompt,
            "contexts": numbered_contexts,
            "raw_contexts": contexts_raw,
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
                "similarity_score": rec.get("similarity_score"),
                "reranker_score": rec.get("reranker_score"),
                "ordinal": rec.get("ordinal"),
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
    def _prefix_context_with_index(index: int, context: str) -> str:
        label = f"[{index}]"
        context = context.strip()
        if not context:
            return label
        return f"{label} {context}"

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
