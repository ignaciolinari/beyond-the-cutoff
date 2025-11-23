"""Shared types for offline dataset generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MappingRow:
    """Representation of a single chunk entry from the FAISS mapping TSV."""

    chunk_id: int
    source_path: str
    page: int | None
    section_title: str | None
    chunk_index: int
    token_start: int | None
    token_end: int | None
    text: str

    def trimmed_text(self, limit: int) -> str:
        snippet = self.text.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[: max(0, limit - 3)].rstrip() + "..."


@dataclass
class DocumentStats:
    text_path: Path
    page_count: int | None = None
    token_count: int | None = None


@dataclass
class DocumentMetadata:
    """Lightweight container for structured document metadata."""

    canonical_id: str | None = None
    document_id: str | None = None
    text_path: str | None = None
    text_path_absolute: str | None = None
    pages_path: str | None = None
    title: str | None = None
    summary: str | None = None
    authors: list[str] = field(default_factory=list)
    author_details: list[dict[str, Any]] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    arxiv_id: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    published: str | None = None
    updated: str | None = None
    page_count: int | None = None
    token_count: int | None = None
    links: dict[str, str] = field(default_factory=dict)
    source_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "canonical_id": self.canonical_id,
            "document_id": self.document_id,
            "text_path": self.text_path,
            "text_path_absolute": self.text_path_absolute,
            "pages_path": self.pages_path,
            "title": self.title,
            "summary": self.summary,
            "authors": list(self.authors),
            "author_details": list(self.author_details),
            "institutions": list(self.institutions),
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "categories": list(self.categories),
            "primary_category": self.primary_category,
            "published": self.published,
            "updated": self.updated,
            "page_count": self.page_count,
            "token_count": self.token_count,
            "links": dict(self.links),
            "source_split": self.source_split,
        }
        return {key: value for key, value in payload.items() if value not in (None, [], {})}


@dataclass
class OfflineExample:
    """Output record describing a prepared offline training/eval example."""

    task_id: str
    task_type: str
    instruction: str
    expected_response: str
    rag_prompt: str
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    citations: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "expected_response": self.expected_response,
            "rag": {
                "prompt": self.rag_prompt,
                "contexts": self.contexts,
                "sources": self.sources,
                "scores": self.scores,
                "citations": self.citations,
                "retrieved": self.retrieved,
            },
            "metadata": self.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)


__all__ = ["MappingRow", "DocumentStats", "DocumentMetadata", "OfflineExample"]
