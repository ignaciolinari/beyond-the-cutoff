"""Offline dataset generation utilities for RAG and fine-tuning workflows."""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..models import LLMClient, build_generation_client
from .query import RAGPipeline

logger = logging.getLogger(__name__)

_INVALID_ESCAPE_PATTERN = re.compile(r"\\(?![\"\\/bfnrtu])")


GENERATOR_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "qa": ("question", "answer"),
    "summaries": ("instruction", "response"),
    "citations": ("instruction", "answer"),
    "contextualizations": ("instruction", "response"),
}


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


class OfflineDatasetGenerator:
    """Generate offline prompts and gold responses for downstream fine-tuning/eval."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        index_path: Path,
        mapping_path: Path,
        generator_client: LLMClient | None = None,
        pipeline: RAGPipeline | None = None,
    ) -> None:
        self.config = config
        self.index_path = index_path
        self.mapping_path = mapping_path
        self._pipeline = pipeline or RAGPipeline(
            config, index_path=index_path, mapping_path=mapping_path
        )
        self._generator = generator_client or build_generation_client(
            config.dataset_generation.generator
        )
        self._rng = random.Random(config.dataset_generation.seed)
        self._processed_root = config.paths.processed_data.resolve()
        self._document_stats_index = self._load_document_stats(self._processed_root)
        self._document_metadata_index = self._load_document_metadata(config)

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    @property
    def generator(self) -> LLMClient:
        return self._generator

    def _load_document_stats(self, processed_root: Path) -> dict[str, DocumentStats]:
        stats: dict[str, DocumentStats] = {}
        manifest_path = processed_root / "manifest.json"
        if not manifest_path.exists():
            return stats
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - manifest read failures rare
            logger.warning(
                "Failed to load processed manifest from %s: %s",
                manifest_path,
                exc,
            )
            return stats

        documents = manifest_payload.get("documents")
        if not isinstance(documents, list):
            return stats

        for entry in documents:
            if not isinstance(entry, dict):
                continue
            text_rel = entry.get("text_path")
            if not isinstance(text_rel, str) or not text_rel:
                continue
            absolute_path = (processed_root / text_rel).resolve()
            stats_entry = DocumentStats(
                text_path=absolute_path,
                page_count=self._coerce_int(entry.get("page_count")),
                token_count=self._coerce_int(entry.get("token_count")),
            )
            self._register_document_stats(stats, absolute_path, text_rel, stats_entry)

        return stats

    def _register_document_stats(
        self,
        index: dict[str, DocumentStats],
        absolute_path: Path,
        relative_path: str,
        stats_entry: DocumentStats,
    ) -> None:
        keys: set[str] = set()
        keys.add(str(absolute_path))
        try:
            resolved = absolute_path.resolve()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - resolution failure unusual
            logger.debug("Failed to resolve path %s: %s", absolute_path, exc)
            resolved = absolute_path
        keys.add(str(resolved))

        rel_obj = Path(relative_path)
        keys.add(relative_path)
        keys.add(rel_obj.as_posix())
        try:
            keys.add(rel_obj.with_suffix("").as_posix())
        except ValueError:
            pass

        for key in keys:
            index[key] = stats_entry

    def _load_document_metadata(self, config: ProjectConfig) -> dict[str, DocumentMetadata]:
        processed_root = config.paths.processed_data.resolve()
        raw_root = config.paths.raw_data.resolve()
        manifest_path = processed_root / "manifest.json"
        if not manifest_path.exists():
            return {}

        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - manifest read failures rare
            logger.warning(
                "Failed to load processed manifest for metadata from %s: %s", manifest_path, exc
            )
            return {}

        documents = manifest_payload.get("documents")
        if not isinstance(documents, list):
            return {}

        raw_metadata_index = self._collect_raw_metadata(raw_root)

        metadata_index: dict[str, DocumentMetadata] = {}
        for entry in documents:
            if not isinstance(entry, dict):
                continue
            text_rel = entry.get("text_path")
            if not isinstance(text_rel, str) or not text_rel:
                continue
            absolute_path = (processed_root / text_rel).resolve()
            canonical_id = self._canonical_id_from_manifest(entry)
            raw_metadata, source_split = raw_metadata_index.get(canonical_id, ({}, None))
            profile = self._build_document_profile(
                manifest_entry=entry,
                canonical_id=canonical_id,
                absolute_text_path=absolute_path,
                raw_metadata=raw_metadata,
                source_split=source_split,
            )
            self._register_document_metadata(metadata_index, absolute_path, text_rel, profile)

        return metadata_index

    def _collect_raw_metadata(
        self, raw_root: Path
    ) -> dict[str | None, tuple[dict[str, Any], str | None]]:
        index: dict[str | None, tuple[dict[str, Any], str | None]] = {}
        if not raw_root.exists():
            return index

        for split_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            metadata_path = split_dir / "metadata.jsonl"
            if not metadata_path.exists():
                continue
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = line.strip()
                    if not record:
                        continue
                    try:
                        payload = json.loads(record)
                    except json.JSONDecodeError:
                        continue
                    canonical_id = self._canonical_id_from_raw(payload)
                    if canonical_id is None or canonical_id in index:
                        continue
                    index[canonical_id] = (payload, split_dir.name)
        return index

    def _register_document_metadata(
        self,
        index: dict[str, DocumentMetadata],
        absolute_path: Path,
        relative_path: str,
        profile: DocumentMetadata,
    ) -> None:
        keys: set[str] = set()
        keys.add(str(absolute_path))
        try:
            resolved = absolute_path.resolve()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - resolution failure unusual
            logger.debug("Failed to resolve path %s: %s", absolute_path, exc)
            resolved = absolute_path
        keys.add(str(resolved))

        rel_obj = Path(relative_path)
        keys.add(relative_path)
        keys.add(rel_obj.as_posix())
        try:
            keys.add(rel_obj.with_suffix("").as_posix())
        except ValueError:
            pass

        for key in keys:
            index[key] = profile

    def _get_document_metadata(self, source_path: str) -> dict[str, Any] | None:
        cached = self._document_metadata_index.get(source_path)
        if cached is not None:
            return cached.to_dict()

        path = Path(source_path)
        candidates: set[str] = {source_path}
        try:
            resolved = path.resolve()
            candidates.add(str(resolved))
            try:
                rel = resolved.relative_to(self._processed_root)
                candidates.add(rel.as_posix())
                try:
                    candidates.add(rel.with_suffix("").as_posix())
                except ValueError:
                    pass
            except ValueError:
                pass
        except (OSError, RuntimeError) as exc:
            logger.debug("Failed to resolve path during metadata lookup: %s", exc)
            resolved = None

        for key in list(candidates):
            matched = self._document_metadata_index.get(key)
            if matched is not None:
                for alias in candidates:
                    self._document_metadata_index.setdefault(alias, matched)
                return matched.to_dict()

        return None

    def _build_document_profile(
        self,
        *,
        manifest_entry: Mapping[str, Any],
        canonical_id: str | None,
        absolute_text_path: Path,
        raw_metadata: Mapping[str, Any],
        source_split: str | None,
    ) -> DocumentMetadata:
        text_rel = manifest_entry.get("text_path")
        pages_rel = manifest_entry.get("pages_path")
        profile = DocumentMetadata(
            canonical_id=canonical_id,
            document_id=self._safe_str(manifest_entry.get("document_id")),
            text_path=self._safe_str(text_rel),
            text_path_absolute=str(absolute_text_path),
            pages_path=self._safe_str(pages_rel),
            title=self._safe_str(raw_metadata.get("title")),
            summary=self._safe_str(raw_metadata.get("summary") or manifest_entry.get("summary")),
            authors=self._string_list(raw_metadata.get("authors")),
            author_details=self._author_details(raw_metadata),
            institutions=self._extract_institutions(raw_metadata),
            arxiv_id=self._safe_str(raw_metadata.get("arxiv_id")),
            doi=self._safe_str(raw_metadata.get("doi")),
            journal_ref=self._safe_str(raw_metadata.get("journal_ref")),
            categories=self._string_list(raw_metadata.get("categories")),
            primary_category=self._safe_str(raw_metadata.get("primary_category")),
            published=self._safe_str(raw_metadata.get("published")),
            updated=self._safe_str(raw_metadata.get("updated")),
            page_count=self._coerce_int(manifest_entry.get("page_count")),
            token_count=self._coerce_int(manifest_entry.get("token_count")),
            links=self._build_links(raw_metadata),
            source_split=source_split,
        )

        if not profile.summary and profile.title and profile.title != profile.summary:
            abstract = self._safe_str(manifest_entry.get("abstract"))
            if abstract:
                profile.summary = abstract

        stats_entry = self._document_stats_index.get(str(absolute_text_path))
        if stats_entry is not None:
            if profile.page_count is None:
                profile.page_count = stats_entry.page_count
            if profile.token_count is None:
                profile.token_count = stats_entry.token_count

        return profile

    @staticmethod
    def _build_links(raw_metadata: Mapping[str, Any]) -> dict[str, str]:
        links: dict[str, str] = {}
        if not isinstance(raw_metadata, Mapping):
            return links
        pdf_url = raw_metadata.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url.strip():
            links["pdf"] = pdf_url.strip()
        abstract_url = raw_metadata.get("link")
        if isinstance(abstract_url, str) and abstract_url.strip():
            links["abstract"] = abstract_url.strip()
        return links

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list | tuple | set):
            results: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    results.append(text)
            return results
        return []

    @staticmethod
    def _author_details(raw_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        if not isinstance(raw_metadata, Mapping):
            return details
        parsed = raw_metadata.get("authors_parsed")
        if isinstance(parsed, list):
            for entry in parsed:
                if not isinstance(entry, list | tuple) or not entry:
                    continue
                last = str(entry[0]).strip() if len(entry) > 0 and entry[0] is not None else ""
                first = str(entry[1]).strip() if len(entry) > 1 and entry[1] is not None else ""
                affiliation = (
                    str(entry[2]).strip() if len(entry) > 2 and entry[2] is not None else ""
                )
                name_parts = [part for part in (first, last) if part]
                if not name_parts and last:
                    name_parts = [last]
                if not name_parts:
                    continue
                person = {"name": " ".join(name_parts)}
                if affiliation:
                    person["affiliation"] = affiliation
                details.append(person)
        return details

    @staticmethod
    def _extract_institutions(raw_metadata: Mapping[str, Any]) -> list[str]:
        if not isinstance(raw_metadata, Mapping):
            return []
        institutions: set[str] = set()
        parsed = raw_metadata.get("authors_parsed")
        if isinstance(parsed, list):
            for entry in parsed:
                if not isinstance(entry, list | tuple) or len(entry) < 3:
                    continue
                affiliation = str(entry[2]).strip() if entry[2] is not None else ""
                if affiliation:
                    institutions.add(affiliation)
        raw_aff = raw_metadata.get("affiliations")
        if isinstance(raw_aff, Mapping):
            for value in raw_aff.values():
                text = str(value).strip()
                if text:
                    institutions.add(text)
        elif isinstance(raw_aff, list | tuple):
            for value in raw_aff:
                text = str(value).strip()
                if text:
                    institutions.add(text)
        elif isinstance(raw_aff, str):
            text = raw_aff.strip()
            if text:
                institutions.add(text)
        return sorted(institutions)

    @staticmethod
    def _canonical_id_from_raw(payload: Mapping[str, Any]) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        value = payload.get("canonical_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        arxiv_id = payload.get("arxiv_id")
        if isinstance(arxiv_id, str) and arxiv_id.strip():
            return arxiv_id.strip().split("v", 1)[0]
        return None

    @staticmethod
    def _canonical_id_from_manifest(entry: Mapping[str, Any]) -> str | None:
        if not isinstance(entry, Mapping):
            return None
        document_id = entry.get("document_id")
        if isinstance(document_id, str) and document_id:
            return Path(document_id).name
        text_path = entry.get("text_path")
        if isinstance(text_path, str) and text_path:
            return Path(text_path).stem
        return None

    def _get_document_stats(self, source_path: str) -> DocumentStats:
        cached = self._document_stats_index.get(source_path)
        if cached is not None:
            return cached

        path = Path(source_path)
        candidates: set[str] = {source_path}
        resolved: Path | None = None
        try:
            resolved = path.resolve()
            candidates.add(str(resolved))
            try:
                rel = resolved.relative_to(self._processed_root)
                rel_posix = rel.as_posix()
                candidates.add(rel_posix)
                try:
                    candidates.add(rel.with_suffix("").as_posix())
                except ValueError:
                    pass
            except ValueError:
                pass
        except (OSError, RuntimeError) as exc:
            logger.debug("Failed to resolve path during stats lookup: %s", exc)
            resolved = None

        for key in list(candidates):
            matched = self._document_stats_index.get(key)
            if matched is not None:
                for alias in candidates:
                    self._document_stats_index.setdefault(alias, matched)
                return matched

        stats_entry = DocumentStats(text_path=resolved or path)
        for alias in candidates:
            self._document_stats_index[alias] = stats_entry
        return stats_entry

    def _compute_page_count(
        self, stats_entry: DocumentStats, rows: Sequence[MappingRow]
    ) -> int | None:
        if stats_entry.page_count is not None:
            return stats_entry.page_count

        sidecar = self._page_sidecar_path(stats_entry.text_path)
        if sidecar.exists():
            try:
                with sidecar.open("r", encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
            except Exception as exc:  # pragma: no cover - rare I/O issue
                logger.debug("Failed to read page sidecar %s: %s", sidecar, exc)
            else:
                stats_entry.page_count = count
                return stats_entry.page_count

        pages = [row.page for row in rows if row.page is not None]
        if pages:
            stats_entry.page_count = max(pages)
        return stats_entry.page_count

    def _compute_token_count(
        self, stats_entry: DocumentStats, rows: Sequence[MappingRow]
    ) -> int | None:
        if stats_entry.token_count is not None:
            return stats_entry.token_count

        text_path = stats_entry.text_path
        if text_path.exists():
            total = 0
            try:
                with text_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line:
                            total += len(line.split())
            except Exception as exc:  # pragma: no cover - rare I/O issue
                logger.debug("Failed to read %s for token counting: %s", text_path, exc)
            else:
                stats_entry.token_count = total
                return stats_entry.token_count

        estimate = self._estimate_tokens_from_rows(rows)
        stats_entry.token_count = estimate
        return stats_entry.token_count

    @staticmethod
    def _estimate_tokens_from_rows(rows: Sequence[MappingRow]) -> int | None:
        token_ceiling = 0
        for row in rows:
            if row.token_end is not None:
                token_ceiling = max(token_ceiling, row.token_end)
        if token_ceiling > 0:
            return token_ceiling

        total = 0
        for row in rows:
            if row.text:
                total += len(row.text.split())
        return total or None

    def _should_skip_document(
        self, source_path: str, rows: Sequence[MappingRow]
    ) -> dict[str, Any] | None:
        stats_entry = self._get_document_stats(source_path)
        cfg = self.config.dataset_generation

        page_limit = cfg.max_document_pages
        if page_limit is not None:
            page_count = self._compute_page_count(stats_entry, rows)
            if page_count is not None and page_count > page_limit:
                return {"kind": "page_limit", "page_count": page_count, "limit": page_limit}

        token_limit = cfg.max_document_tokens
        if token_limit is not None:
            token_count = self._compute_token_count(stats_entry, rows)
            if token_count is not None and token_count > token_limit:
                return {
                    "kind": "token_limit",
                    "token_count": token_count,
                    "limit": token_limit,
                }

        return None

    @staticmethod
    def _page_sidecar_path(text_path: Path) -> Path:
        try:
            return text_path.with_suffix(".pages.jsonl")
        except ValueError:  # pragma: no cover - unexpected suffixless path
            return text_path.parent / f"{text_path.name}.pages.jsonl"

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(float(stripped))
            except ValueError:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def generate(
        self,
        *,
        output_dataset_path: Path | None = None,
        raw_tasks_path: Path | None = None,
        resume: bool = False,
        parse_retries: int | None = None,
        documents: Sequence[str] | None = None,
    ) -> dict[str, int]:
        """Generate offline dataset artifacts and return counters."""

        dataset_cfg = self.config.dataset_generation
        dataset_path = output_dataset_path or dataset_cfg.output_dataset_path
        tasks_path = raw_tasks_path or dataset_cfg.raw_tasks_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        document_whitelist: set[str] | None = None
        if documents:
            document_whitelist = set()
            for entry in documents:
                text = str(entry)
                if text:
                    document_whitelist.add(text)
                try:
                    document_whitelist.add(str(Path(text).resolve()))
                except Exception:
                    continue
        matched_documents: set[str] = set()
        processed_documents: set[str] = set()
        dataset_mode = "w"
        tasks_mode = "w"

        if resume:
            processed_documents = self._load_processed_documents(tasks_path)
            dataset_mode = "a"
            tasks_mode = "a"

        effective_parse_retries = (
            dataset_cfg.parse_retries if parse_retries is None else max(0, parse_retries)
        )

        grouped = self._load_mapping(self.mapping_path)
        if document_whitelist is None:
            target_items = list(grouped.items())
        else:
            target_items = [(path, grouped[path]) for path in grouped if path in document_whitelist]

        total_targets = len(target_items)
        max_docs = dataset_cfg.max_documents
        progress_total = total_targets if max_docs is None else min(total_targets, max_docs)
        counters = {
            "documents": 0,
            "qa": 0,
            "summaries": 0,
            "citations": 0,
            "contextual": 0,
            "examples": 0,
            "documents_filtered": 0,
        }
        if document_whitelist is not None:
            counters.update(
                {
                    "documents_requested": len(document_whitelist),
                    "documents_found": 0,
                    "documents_missing": 0,
                }
            )

        with (
            tasks_path.open(tasks_mode, encoding="utf-8") as raw_file,
            dataset_path.open(dataset_mode, encoding="utf-8") as dataset_file,
        ):
            if progress_total:
                logger.info(
                    "Planning to process %d document(s)%s",
                    progress_total,
                    " (limited by max_documents)"
                    if max_docs is not None and progress_total < total_targets
                    else "",
                )

            for doc_index, (source_path, rows) in enumerate(target_items):
                if max_docs is not None and counters["documents"] >= max_docs:
                    break

                progress_position = doc_index + 1
                if progress_total:
                    progress_label = f"{progress_position}/{progress_total}"
                else:
                    progress_label = str(progress_position)

                if document_whitelist is not None:
                    matched_documents.add(source_path)

                if resume and source_path in processed_documents:
                    logger.info("Document %s: %s (resume)", progress_label, source_path)
                    continue

                logger.info("Document %s: %s", progress_label, source_path)

                filter_info = self._should_skip_document(source_path, rows)
                if filter_info is not None:
                    counters["documents_filtered"] += 1
                    reason_key = f"documents_filtered_{filter_info['kind']}"
                    counters[reason_key] = counters.get(reason_key, 0) + 1
                    logger.info(
                        "Document %s: %s -> skipped (%s, limit=%s, observed=%s)",
                        progress_label,
                        source_path,
                        filter_info["kind"],
                        filter_info.get("limit"),
                        filter_info.get("token_count")
                        if filter_info["kind"] == "token_limit"
                        else filter_info.get("page_count"),
                    )
                    raw_record = {
                        "document": source_path,
                        "status": "skipped",
                        "reason": filter_info["kind"],
                    }
                    details = {k: v for k, v in filter_info.items() if k != "kind"}
                    if details:
                        raw_record["details"] = details
                    raw_file.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                    continue

                examples, raw_payload = self._generate_for_document(
                    doc_index=doc_index,
                    source_path=source_path,
                    rows=rows,
                    parse_retries=effective_parse_retries,
                )

                if not examples:
                    if raw_payload:
                        raw_payload.setdefault("status", "error")
                        raw_file.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")
                    continue

                raw_payload.setdefault("status", "success")
                raw_file.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")

                for example in examples:
                    dataset_file.write(example.to_json() + "\n")
                    counters["examples"] += 1
                    counters[example.task_type] = counters.get(example.task_type, 0) + 1

                counters["documents"] += 1

        if document_whitelist is not None:
            missing_documents = document_whitelist - matched_documents
            if missing_documents:
                logger.warning(
                    "Requested documents not present in mapping: %s",
                    sorted(missing_documents),
                )
            counters["documents_found"] = len(matched_documents)
            counters["documents_missing"] = len(missing_documents)

        return counters

    def _generate_for_document(
        self,
        *,
        doc_index: int,
        source_path: str,
        rows: Sequence[MappingRow],
        parse_retries: int,
    ) -> tuple[list[OfflineExample], dict[str, Any]]:
        dataset_cfg = self.config.dataset_generation
        selected_rows = self._select_rows(rows, dataset_cfg.max_chunks_per_document)
        if not selected_rows:
            logger.debug("Skipping %s; no chunks selected", source_path)
            return [], {}

        generator_prompt = self._build_generator_prompt(
            source_path=source_path,
            rows=selected_rows,
            cfg=dataset_cfg,
        )
        attempts = max(0, parse_retries) + 1
        parsed: dict[str, Any] | None = None
        raw_text = ""
        attempt_logs: list[dict[str, Any]] = []
        validation_notes: list[dict[str, Any]] = []
        failure_reason = "unparsable_response"

        for attempt in range(1, attempts + 1):
            try:
                response = self.generator.generate(generator_prompt)
            except Exception as exc:  # pragma: no cover - network/LLM failures
                logger.warning(
                    "Generator request failed for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    exc,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "error": "request_failed",
                        "exception": repr(exc),
                    }
                )
                raw_text = ""
                parsed = None
                failure_reason = "request_failed"
                continue

            raw_text = response.get("response", "")
            parsed = self._parse_generator_response(raw_text)
            if parsed is None:
                logger.warning(
                    "Unparsable generator payload for %s (attempt %d/%d)",
                    source_path,
                    attempt,
                    attempts,
                )
                attempt_logs.append(
                    {"attempt": attempt, "response": raw_text, "error": "unparsable"}
                )
                failure_reason = "unparsable_response"
                continue

            parsed, validation_issues = self._validate_generator_payload(parsed)
            fatal_issues = [issue for issue in validation_issues if issue["fatal"]]
            if fatal_issues:
                logger.warning(
                    "Invalid generator payload for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    fatal_issues,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "response": raw_text,
                        "error": "invalid_payload",
                        "details": fatal_issues,
                    }
                )
                parsed = None
                failure_reason = "invalid_generator_payload"
                continue

            validation_notes = validation_issues
            if not self._has_tasks(parsed):
                logger.warning(
                    "Generator returned empty payload for %s (attempt %d/%d)",
                    source_path,
                    attempt,
                    attempts,
                )
                attempt_logs.append(
                    {"attempt": attempt, "response": raw_text, "error": "empty_payload"}
                )
                parsed = None
                failure_reason = "empty_payload"
                continue

            minimum_deficits = self._missing_minimum_counts(parsed, dataset_cfg)
            if minimum_deficits:
                logger.warning(
                    "Generator payload missing required counts for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    minimum_deficits,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "response": raw_text,
                        "error": "insufficient_items",
                        "details": minimum_deficits,
                    }
                )
                parsed = None
                failure_reason = "insufficient_items"
                continue

            if attempt > 1:
                logger.info(
                    "Recovered generator payload for %s after %d attempt(s)",
                    source_path,
                    attempt,
                )
            break

        if parsed is None:
            return [], {
                "document": source_path,
                "prompt": generator_prompt,
                "model": getattr(self.generator, "model", "unknown"),
                "selected_chunks": self._serialize_chunks(
                    selected_rows, dataset_cfg.max_chars_per_chunk
                ),
                "response": raw_text,
                "error": failure_reason,
                "attempts": attempt_logs,
            }

        examples: list[OfflineExample] = []
        run_id = str(uuid.uuid4())

        qa_limit = dataset_cfg.questions_per_document
        summary_limit = dataset_cfg.summary_prompts_per_document
        citation_limit = dataset_cfg.citation_prompts_per_document
        context_limit = dataset_cfg.contextual_prompts_per_document

        for idx, item in enumerate(self._take(parsed.get("qa", []), qa_limit)):
            if not isinstance(item, Mapping):
                continue
            question = item.get("question") or item.get("instruction")
            answer = item.get("answer") or item.get("response")
            if not question or not answer:
                continue
            example = self._build_example(
                task_type="qa",
                instruction=question,
                expected_response=answer,
                require_citations=True,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        for idx, item in enumerate(self._take(parsed.get("summaries", []), summary_limit)):
            if not isinstance(item, Mapping):
                continue
            instruction = item.get("instruction") or item.get("prompt") or item.get("question")
            response_text = item.get("response") or item.get("answer")
            if not instruction or not response_text:
                continue
            example = self._build_example(
                task_type="summaries",
                instruction=instruction,
                expected_response=response_text,
                require_citations=False,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        for idx, item in enumerate(self._take(parsed.get("citations", []), citation_limit)):
            if not isinstance(item, Mapping):
                continue
            instruction = item.get("instruction") or item.get("question")
            answer = item.get("answer") or item.get("response")
            if not instruction or not answer:
                continue
            example = self._build_example(
                task_type="citations",
                instruction=instruction,
                expected_response=answer,
                require_citations=True,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        for idx, item in enumerate(self._take(parsed.get("contextualizations", []), context_limit)):
            if not isinstance(item, Mapping):
                continue
            instruction = item.get("instruction") or item.get("prompt") or item.get("question")
            response_text = item.get("response") or item.get("answer")
            if not instruction or not response_text:
                continue
            example = self._build_example(
                task_type="contextual",
                instruction=instruction,
                expected_response=response_text,
                require_citations=True,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        examples, output_issues = self._validate_output_examples(examples)
        if output_issues:
            validation_notes.extend(output_issues)

        raw_payload = {
            "document": source_path,
            "model": getattr(self.generator, "model", "unknown"),
            "prompt": generator_prompt,
            "response": raw_text,
            "parsed": parsed,
            "run_id": run_id,
            "selected_chunks": self._serialize_chunks(
                selected_rows, dataset_cfg.max_chars_per_chunk
            ),
        }
        if validation_notes:
            raw_payload["validation_warnings"] = validation_notes

        if not examples:
            raw_payload["error"] = "output_validation_failed"
            return [], raw_payload

        return examples, raw_payload

    @staticmethod
    def _has_tasks(payload: Mapping[str, Any]) -> bool:
        qa_items = payload.get("qa", [])
        summary_items = payload.get("summaries", [])
        citation_items = payload.get("citations", [])
        contextual_items = payload.get("contextualizations", [])
        return any(qa_items) or any(summary_items) or any(citation_items) or any(contextual_items)

    @staticmethod
    def _load_processed_documents(tasks_path: Path) -> set[str]:
        processed: set[str] = set()
        if not tasks_path.exists():
            return processed
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                document = payload.get("document")
                if isinstance(document, str) and document:
                    status = payload.get("status")
                    if payload.get("error"):
                        continue
                    if status and status not in {"success", "skipped"}:
                        continue
                    processed.add(document)
        return processed

    @staticmethod
    def _missing_minimum_counts(payload: Mapping[str, Any], cfg: Any) -> dict[str, dict[str, int]]:
        deficits: dict[str, dict[str, int]] = {}

        def _count_items(key: str) -> int:
            items = payload.get(key, [])
            if isinstance(items, Sequence):
                return len(items)
            return 0

        requirements = {
            "qa": getattr(cfg, "min_questions_per_document", 0),
            "summaries": getattr(cfg, "min_summary_prompts_per_document", 0),
            "citations": getattr(cfg, "min_citation_prompts_per_document", 0),
            "contextualizations": getattr(cfg, "min_contextual_prompts_per_document", 0),
        }

        for key, required in requirements.items():
            if required <= 0:
                continue
            observed = _count_items(key)
            if observed < required:
                deficits[key] = {"required": int(required), "observed": observed}

        return deficits

    def _build_example(
        self,
        *,
        task_type: str,
        instruction: str,
        expected_response: str,
        require_citations: bool,
        doc_index: int,
        task_index: int,
        run_id: str,
        source_path: str,
        selected_rows: Sequence[MappingRow],
        extra_metadata: Mapping[str, Any] | None,
    ) -> OfflineExample | None:
        instruction_clean = self._coerce_text(instruction).strip()
        expected_clean = self._coerce_text(expected_response).strip()
        if not instruction_clean or not expected_clean:
            return None

        extra_instructions = None
        top_k_override: int | None = None
        if require_citations:
            extra_instructions = (
                "Use the numbered excerpts as explicit evidence. When more than one excerpt is relevant, cite each distinct snippet at least once using its [#] marker. "
                "Write complete sentences grounded in the excerpt content, and avoid recycling the same citation for unrelated statements or fabricating references."
            )
            configured_top_k = getattr(self.config.retrieval, "top_k", 4)
            top_k_override = max(2, int(configured_top_k)) if configured_top_k else 2

        prepared = self.pipeline.prepare_prompt(
            instruction_clean,
            require_citations=require_citations,
            extra_instructions=extra_instructions,
            top_k_override=top_k_override,
        )

        enforcement_meta: dict[str, Any] | None = None
        if require_citations:
            enforced = self._ensure_citation_compliance(
                question=instruction_clean,
                answer=expected_clean,
                contexts=prepared["contexts"],
                source_path=source_path,
            )
            if enforced is None:
                return None
            expected_clean, enforcement_meta = enforced

        metadata = {
            "source_path": source_path,
            "generator_model": getattr(self.generator, "model", "unknown"),
            "generator_run_id": run_id,
            "document_index": doc_index,
            "task_index": task_index,
            "require_citations": require_citations,
            "retrieved_chunk_ids": [rec.get("id") for rec in prepared["retrieved"]],
            "retrieved_section_titles": [rec.get("section_title") for rec in prepared["retrieved"]],
            "selected_chunk_ids": [row.chunk_id for row in selected_rows],
        }
        if extra_metadata:
            metadata["generator_metadata"] = dict(extra_metadata)
        if enforcement_meta:
            metadata["citation_enforcement"] = enforcement_meta

        if prepared.get("retrieved"):
            metadata["context_map"] = self._build_context_map(prepared["retrieved"])
        raw_contexts = prepared.get("raw_contexts")
        if isinstance(raw_contexts, list) and raw_contexts:
            metadata["raw_contexts"] = list(raw_contexts)
        extra_block = prepared.get("extra_instructions")
        if isinstance(extra_block, str) and extra_block.strip():
            metadata["retrieval_extra_instructions"] = extra_block.strip()

        profile = self._get_document_metadata(source_path)
        if profile:
            metadata["paper_profile"] = profile

        return OfflineExample(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            instruction=instruction_clean,
            expected_response=expected_clean,
            rag_prompt=prepared["prompt"],
            contexts=prepared["contexts"],
            sources=prepared["sources"],
            scores=prepared["scores"],
            citations=[
                {
                    "id": rec.get("id"),
                    "source_path": rec.get("source_path"),
                    "page": rec.get("page"),
                    "section_title": rec.get("section_title"),
                    "token_start": rec.get("token_start"),
                    "token_end": rec.get("token_end"),
                    "score": rec.get("score"),
                    "excerpt": rec.get("excerpt"),
                    "rendered_context": rec.get("rendered_context"),
                }
                for rec in prepared["retrieved"]
            ],
            retrieved=prepared["retrieved"],
            metadata=metadata,
        )

    @staticmethod
    def _truncate(text: str, limit: int = 120) -> str:
        snippet = text.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[: max(0, limit - 3)].rstrip() + "..."

    def _validate_output_examples(
        self, examples: Sequence[OfflineExample]
    ) -> tuple[list[OfflineExample], list[dict[str, Any]]]:
        filtered: list[OfflineExample] = []
        issues: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for example in examples:
            instruction_text = example.instruction.strip()
            response_text = example.expected_response.strip()
            if not instruction_text or not response_text:
                issues.append(
                    {
                        "field": example.task_type,
                        "kind": "empty_text",
                        "fatal": True,
                        "instruction": self._truncate(example.instruction),
                    }
                )
                continue

            key = (example.task_type, instruction_text.lower())
            if key in seen:
                issues.append(
                    {
                        "field": example.task_type,
                        "kind": "duplicate_instruction",
                        "fatal": False,
                        "instruction": self._truncate(example.instruction),
                    }
                )
                continue

            if example.metadata.get("require_citations"):
                citation_issue = self._check_citation_coverage(example)
                if citation_issue is not None:
                    issues.append(citation_issue)
                    if citation_issue.get("fatal", True):
                        continue

            filtered.append(example)
            seen.add(key)

        return filtered, issues

    def _check_citation_coverage(self, example: OfflineExample) -> dict[str, Any] | None:
        enforcement_meta = example.metadata.get("citation_enforcement")
        verification: Mapping[str, Any] | None = None
        if isinstance(enforcement_meta, Mapping):
            verification = enforcement_meta.get("verification")
            if not isinstance(verification, Mapping):
                verification = None

        if verification is None:
            verification = self.pipeline.verify_citations(
                example.expected_response, list(example.contexts)
            )
            updated_meta = dict(enforcement_meta) if isinstance(enforcement_meta, Mapping) else {}
            updated_meta.setdefault("status", updated_meta.get("status", "post_validation"))
            updated_meta["verification"] = verification
            example.metadata["citation_enforcement"] = updated_meta

        referenced = verification.get("referenced") or []
        extra = verification.get("extra") or []
        coverage = verification.get("coverage") or {}
        min_coverage = max(0.0, min(1.0, self.config.dataset_generation.min_citation_coverage))

        if not referenced:
            return {
                "field": example.task_type,
                "kind": "missing_citations",
                "fatal": True,
                "instruction": self._truncate(example.instruction),
            }

        insufficient = [idx for idx in referenced if coverage.get(idx, 0.0) < min_coverage]
        if insufficient:
            return {
                "field": example.task_type,
                "kind": "low_citation_coverage",
                "fatal": True,
                "instruction": self._truncate(example.instruction),
                "details": {"referenced": referenced, "insufficient": insufficient},
            }

        if extra:
            return {
                "field": example.task_type,
                "kind": "invalid_citation_reference",
                "fatal": True,
                "instruction": self._truncate(example.instruction),
                "details": {"extra": extra},
            }

        return None

    def _select_rows(
        self,
        rows: Sequence[MappingRow],
        max_chunks: int,
    ) -> list[MappingRow]:
        if not rows:
            return []
        ordered = sorted(rows, key=lambda r: r.chunk_index)
        if len(ordered) <= max_chunks:
            return ordered
        indices = list(range(len(ordered)))
        self._rng.shuffle(indices)
        chosen = sorted(indices[:max_chunks])
        return [ordered[i] for i in chosen]

    @staticmethod
    def _take(items: Iterable[Any], limit: int) -> list[Any]:
        result: list[Any] = []
        if limit <= 0:
            return result
        for item in items:
            result.append(item)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _build_context_map(retrieved_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        context_map: list[dict[str, Any]] = []
        for idx, rec in enumerate(retrieved_records, start=1):
            entry = {
                "index": rec.get("ordinal", idx),
                "chunk_id": rec.get("id"),
                "source_path": rec.get("source_path"),
                "page": rec.get("page"),
                "section_title": rec.get("section_title"),
                "token_start": rec.get("token_start"),
                "token_end": rec.get("token_end"),
                "rendered_context": rec.get("rendered_context"),
                "score": rec.get("score"),
                "similarity_score": rec.get("similarity_score"),
                "reranker_score": rec.get("reranker_score"),
            }
            context_map.append(entry)
        return context_map

    def _serialize_chunks(self, rows: Sequence[MappingRow], max_chars: int) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": row.chunk_id,
                "chunk_index": row.chunk_index,
                "page": row.page,
                "section_title": row.section_title,
                "token_start": row.token_start,
                "token_end": row.token_end,
                "text": row.trimmed_text(max_chars),
            }
            for row in rows
        ]

    def _build_generator_prompt(
        self,
        *,
        source_path: str,
        rows: Sequence[MappingRow],
        cfg: Any,
    ) -> str:
        def _range_phrase(noun: str, minimum: int, maximum: int) -> str:
            if maximum <= 0:
                return f"0 {noun}"
            if minimum <= 0:
                return f"up to {maximum} {noun}"
            if minimum == maximum:
                return f"{maximum} {noun}"
            return f"{minimum} to {maximum} {noun}"

        preamble = (
            "You are assisting in building a dataset for a retrieval-augmented research assistant. "
            "Given the numbered excerpts from a scientific paper, create diverse tasks."
        )
        qa_phrase = _range_phrase(
            "question-answer pairs", cfg.min_questions_per_document, cfg.questions_per_document
        )
        summary_phrase = _range_phrase(
            "summary instructions",
            cfg.min_summary_prompts_per_document,
            cfg.summary_prompts_per_document,
        )
        context_phrase = _range_phrase(
            "contextualization prompts that capture broader themes or connections",
            cfg.min_contextual_prompts_per_document,
            cfg.contextual_prompts_per_document,
        )
        citation_phrase = _range_phrase(
            "citation-check tasks",
            cfg.min_citation_prompts_per_document,
            cfg.citation_prompts_per_document,
        )
        instructions = (
            f"Produce {qa_phrase}, {summary_phrase}, {context_phrase}, and {citation_phrase}. "
            "Vary difficulty and focus so the set covers methods, results, limitations, and implications. "
            "All answers must be grounded in the text. Citation and contextualization responses must include inline [#] markers that reference the numbered excerpts, using multiple distinct markers when more than one excerpt is relevant. "
            "Return ONLY valid JSON with keys 'qa', 'summaries', 'contextualizations', and 'citations'."
        )
        schema = (
            "- Each item in 'qa' must include 'question' and 'answer'. Answers should cite supporting excerpts with inline [#] markers.\n"
            "- Each item in 'summaries' must include 'instruction' and 'response', covering contributions, methods, and notable limitations.\n"
            "- Each item in 'contextualizations' must include 'instruction' and 'response'. Use these to relate the paper to broader themes, contrasting approaches, or key author/institution highlights. Responses must include inline citations for every supported claim.\n"
            "- Each item in 'citations' must include non-empty 'instruction' and 'answer'. Citation answers must cite relevant excerpts using inline markers such as [1], [2] and should draw on multiple distinct snippets when evidence exists.\n"
            "- Avoid null values. If a field is unknown, omit the item instead of returning null.\n"
            "- Use concise academic tone. Avoid speculative statements."
        )

        few_shot_examples = (
            "Example output (use it as a style guide, but do not reuse the wording):\n"
            "{\n"
            '  "qa": [\n'
            "    {\n"
            '      "question": "What retrieval backbone do the authors deploy?",\n'
            '      "answer": "They fine-tune a bi-encoder and then rerank with a cross-encoder calibrated on the validation split [1][2]."\n'
            "    }\n"
            "  ],\n"
            '  "summaries": [\n'
            "    {\n"
            '      "instruction": "Write a concise summary covering objectives, methods, and findings.",\n'
            '      "response": "The paper proposes a lightweight retrieval pipeline, couples it with local generation, and reports significant accuracy gains on scientific benchmarks [1][3]."\n'
            "    }\n"
            "  ],\n"
            '  "contextualizations": [\n'
            "    {\n"
            '      "instruction": "Relate the authors\' approach to prior institution-scale retrieval systems.",\n'
            '      "response": "The authors contrast their decentralized deployment with earlier centralized retrieval services, highlighting alignment with institutional compliance requirements [2][3]."\n'
            "    }\n"
            "  ],\n"
            '  "citations": [\n'
            "    {\n"
            '      "instruction": "Identify the paragraph describing evaluation metrics.",\n'
            '      "answer": "The evaluation metrics and reporting strategy appear in the section titled ‘Results and Analysis’ [3]."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "In your own output, ensure every [#] marker refers to an existing excerpt number from this document, and only add markers when the claim is explicitly supported. Never include citation markers inside questions or instructions."
        )
        context_lines = []
        for idx, row in enumerate(rows, start=1):
            meta_bits: list[str] = []
            if row.section_title:
                meta_bits.append(f"Section: {row.section_title}")
            if row.page is not None:
                meta_bits.append(f"Page {row.page}")
            entry_header = f"[{idx}]"
            if meta_bits:
                entry_header = f"{entry_header} {' | '.join(meta_bits)}"
            context_lines.append(f"{entry_header}\n{row.trimmed_text(cfg.max_chars_per_chunk)}")
        context_block = "\n".join(context_lines)

        return (
            f"{preamble}\n{instructions}\n{schema}\n\n"
            f"{few_shot_examples}\n\n"
            f"Document: {source_path}\n"
            f"Context Excerpts:\n{context_block}\n\n"
            "Return the JSON now."
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
        if stripped.startswith("```json"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```json"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _ensure_citation_compliance(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        source_path: str,
    ) -> tuple[str, dict[str, Any]] | None:
        enforcement: dict[str, Any] = {"status": "pass", "attempts": 0}
        cleaned_answer = answer.strip()
        if not cleaned_answer:
            return None

        if not contexts:
            enforcement["verification"] = {"referenced": []}
            return cleaned_answer, enforcement

        min_coverage = max(0.0, min(1.0, self.config.dataset_generation.min_citation_coverage))

        def _is_compliant(payload: Mapping[str, Any]) -> bool:
            referenced = payload.get("referenced") or []
            if not referenced or payload.get("extra"):
                return False
            if min_coverage <= 0.0:
                return True
            coverage = payload.get("coverage") or {}
            return all(coverage.get(idx, 0.0) >= min_coverage for idx in referenced)

        verification = self.pipeline.verify_citations(cleaned_answer, list(contexts))
        enforcement["verification"] = verification
        if _is_compliant(verification):
            return cleaned_answer, enforcement

        attempts = max(0, self.config.dataset_generation.citation_rewrite_attempts)
        candidate = cleaned_answer
        for attempt in range(1, attempts + 1):
            rewrite_prompt = self._build_citation_rewrite_prompt(
                question=question,
                contexts=contexts,
                answer=candidate,
            )
            try:
                response = self.generator.generate(rewrite_prompt)
            except Exception as exc:  # pragma: no cover - network/LLM failures
                logger.warning(
                    "Citation rewrite failed for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    exc,
                )
                enforcement.setdefault("errors", []).append(
                    {
                        "attempt": attempt,
                        "error": "rewrite_failed",
                        "exception": repr(exc),
                    }
                )
                continue
            candidate = (response.get("response", "") or "").strip()
            enforcement["attempts"] = attempt
            if not candidate:
                continue
            verification = self.pipeline.verify_citations(candidate, list(contexts))
            if _is_compliant(verification):
                enforcement.update(
                    {
                        "status": "rewrite_success",
                        "verification": verification,
                    }
                )
                return candidate, enforcement

        enforcement.update({"status": "failed"})
        logger.warning("Dropping task for %s due to missing citations", source_path)
        return None

    @staticmethod
    def _build_citation_rewrite_prompt(
        *,
        question: str,
        contexts: Sequence[str],
        answer: str,
    ) -> str:
        context_block = "\n".join(contexts)
        return (
            "You are revising an answer for a retrieval-augmented research assistant. "
            "Ensure the answer includes inline citations using square brackets with numbers, such as [1], "
            "that refer to the numbered contexts. Cite only relevant contexts, do not fabricate, and draw on multiple different snippets when they contain supporting evidence. "
            "Reuse key terminology from the contexts so the overlap with cited evidence is explicit. "
            "Rewrite the answer if needed to include distinct citations and keep it concise.\n\n"
            f"Question: {question}\n\n"
            "Numbered contexts:\n"
            f"{context_block}\n\n"
            "Draft answer:\n"
            f"{answer}\n\n"
            "Rewritten answer with citations:"
        )

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:  # pragma: no cover - unexpected encoding
                return value.decode(errors="ignore")
        if isinstance(value, Sequence):
            return " ".join(str(part) for part in value if part)
        return str(value)

    def _parse_generator_response(self, text: str) -> dict[str, Any] | None:
        candidate = self._strip_fences(text)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if "Invalid \\escape" in exc.msg:
                cleaned = _INVALID_ESCAPE_PATTERN.sub("", candidate)
                if cleaned != candidate:
                    try:
                        data = json.loads(cleaned)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Failed to decode generator JSON after escape fix: %s",
                            candidate,
                        )
                        return None
                    else:
                        logger.debug("Recovered generator JSON after sanitizing invalid escapes")
                else:
                    logger.debug("Failed to decode generator JSON: %s", candidate)
                    return None
            else:
                logger.debug("Failed to decode generator JSON: %s", candidate)
                return None
        if not isinstance(data, dict):
            return None
        # Normalise expected keys
        data.setdefault("qa", [])
        data.setdefault("summaries", [])
        data.setdefault("citations", [])
        data.setdefault("contextualizations", [])
        return data

    def _validate_generator_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        cleaned = dict(payload)
        issues: list[dict[str, Any]] = []

        for key, required_fields in GENERATOR_REQUIRED_FIELDS.items():
            items = payload.get(key, [])
            if not isinstance(items, list):
                issues.append(
                    {
                        "field": key,
                        "kind": "not_list",
                        "fatal": key != "qa",
                    }
                )
                cleaned[key] = []
                continue

            valid_items: list[dict[str, Any]] = []
            for idx, item in enumerate(items):
                if not isinstance(item, Mapping):
                    issues.append(
                        {
                            "field": f"{key}[{idx}]",
                            "kind": "not_mapping",
                            "fatal": key != "qa",
                        }
                    )
                    continue

                normalized = dict(item)
                missing: list[str] = []
                for required_field in required_fields:
                    value = item.get(required_field)
                    if value is None:
                        missing.append(required_field)
                        continue
                    text_value = self._coerce_text(value).strip()
                    if not text_value:
                        missing.append(required_field)
                        continue
                    normalized[required_field] = text_value

                if missing:
                    issues.append(
                        {
                            "field": f"{key}[{idx}]",
                            "kind": "missing_required",
                            "missing": missing,
                            "fatal": key != "qa",
                        }
                    )
                    continue

                valid_items.append(normalized)

            cleaned[key] = valid_items

        return cleaned, issues

    def _load_mapping(self, mapping_path: Path) -> dict[str, list[MappingRow]]:
        import csv

        grouped: dict[str, list[MappingRow]] = {}
        with mapping_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                try:
                    chunk_id = int(row.get("id", "0"))
                except (TypeError, ValueError):
                    continue
                source_path = str(row.get("source_path", ""))
                if not source_path:
                    continue
                page = self._maybe_int(row.get("page"))
                chunk_index = self._maybe_int(row.get("chunk_index"), default=chunk_id) or chunk_id
                token_start = self._maybe_int(row.get("token_start"))
                token_end = self._maybe_int(row.get("token_end"))
                text = str(row.get("text", ""))
                section_title = row.get("section_title")
                mapping_row = MappingRow(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    page=page,
                    section_title=section_title if section_title else None,
                    chunk_index=chunk_index,
                    token_start=token_start,
                    token_end=token_end,
                    text=text,
                )
                grouped.setdefault(source_path, []).append(mapping_row)

        # Deterministic ordering for reproducibility
        for bucket in grouped.values():
            bucket.sort(key=lambda r: r.chunk_index)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    @staticmethod
    def _maybe_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default


__all__ = ["OfflineDatasetGenerator", "OfflineExample", "MappingRow"]
