"""Document metadata and statistics management."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .types import DocumentMetadata, DocumentStats, MappingRow

logger = logging.getLogger(__name__)


class DocumentMetadataManager:
    """Manage document metadata and statistics from manifests."""

    def __init__(self, config: Any):
        """
        Args:
            config: ProjectConfig with paths to processed and raw data
        """
        self.config = config
        self._processed_root = config.paths.processed_data.resolve()
        self._raw_root = config.paths.raw_data.resolve()
        self._document_stats_index = self._load_document_stats()
        self._document_metadata_index = self._load_document_metadata()

    def get_document_metadata(self, source_path: str) -> dict[str, Any] | None:
        """Get metadata dict for a document path."""
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

        for key in list(candidates):
            matched = self._document_metadata_index.get(key)
            if matched is not None:
                for alias in candidates:
                    self._document_metadata_index.setdefault(alias, matched)
                return matched.to_dict()

        return None

    def get_document_stats(self, source_path: str) -> DocumentStats:
        """Get stats for a document, computing if not cached."""
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
                candidates.add(rel.as_posix())
                try:
                    candidates.add(rel.with_suffix("").as_posix())
                except ValueError:
                    pass
            except ValueError:
                pass
        except (OSError, RuntimeError) as exc:
            logger.debug("Failed to resolve path during stats lookup: %s", exc)

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

    def compute_page_count(
        self, stats_entry: DocumentStats, rows: Sequence[MappingRow]
    ) -> int | None:
        """Compute page count from sidecar or mapping rows."""
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

    def compute_token_count(
        self, stats_entry: DocumentStats, rows: Sequence[MappingRow]
    ) -> int | None:
        """Compute token count from text file or estimate from rows."""
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

    def should_skip_document(
        self, source_path: str, rows: Sequence[MappingRow]
    ) -> dict[str, Any] | None:
        """Check if document should be skipped based on size limits."""
        stats_entry = self.get_document_stats(source_path)
        cfg = self.config.dataset_generation

        page_limit = cfg.max_document_pages
        if page_limit is not None:
            page_count = self.compute_page_count(stats_entry, rows)
            if page_count is not None and page_count > page_limit:
                return {"kind": "page_limit", "page_count": page_count, "limit": page_limit}

        token_limit = cfg.max_document_tokens
        if token_limit is not None:
            token_count = self.compute_token_count(stats_entry, rows)
            if token_count is not None and token_count > token_limit:
                return {
                    "kind": "token_limit",
                    "token_count": token_count,
                    "limit": token_limit,
                }

        return None

    def _load_document_stats(self) -> dict[str, DocumentStats]:
        """Load document statistics from processed manifest."""
        stats: dict[str, DocumentStats] = {}
        manifest_path = self._processed_root / "manifest.json"
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
            absolute_path = (self._processed_root / text_rel).resolve()
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
        """Register stats under multiple path variants."""
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

    def _load_document_metadata(self) -> dict[str, DocumentMetadata]:
        """Load document metadata from processed and raw manifests."""
        manifest_path = self._processed_root / "manifest.json"
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

        raw_metadata_index = self._collect_raw_metadata()

        metadata_index: dict[str, DocumentMetadata] = {}
        for entry in documents:
            if not isinstance(entry, dict):
                continue
            text_rel = entry.get("text_path")
            if not isinstance(text_rel, str) or not text_rel:
                continue
            absolute_path = (self._processed_root / text_rel).resolve()
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
        self,
    ) -> dict[str | None, tuple[dict[str, Any], str | None]]:
        """Collect raw metadata from raw data splits."""
        index: dict[str | None, tuple[dict[str, Any], str | None]] = {}
        if not self._raw_root.exists():
            return index

        for split_dir in sorted(path for path in self._raw_root.iterdir() if path.is_dir()):
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
        """Register metadata under multiple path variants."""
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

    def _build_document_profile(
        self,
        *,
        manifest_entry: Mapping[str, Any],
        canonical_id: str | None,
        absolute_text_path: Path,
        raw_metadata: Mapping[str, Any],
        source_split: str | None,
    ) -> DocumentMetadata:
        """Build complete document metadata profile."""
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
        """Extract PDF and abstract links."""
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
        """Safely convert value to string or None."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        """Convert value to list of strings."""
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
        """Extract structured author details."""
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
        """Extract unique institutions from author affiliations."""
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
        """Extract canonical ID from raw metadata."""
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
        """Extract canonical ID from manifest entry."""
        if not isinstance(entry, Mapping):
            return None
        document_id = entry.get("document_id")
        if isinstance(document_id, str) and document_id:
            return Path(document_id).name
        text_path = entry.get("text_path")
        if isinstance(text_path, str) and text_path:
            return Path(text_path).stem
        return None

    @staticmethod
    def _page_sidecar_path(text_path: Path) -> Path:
        """Get path to page sidecar file."""
        try:
            return text_path.with_suffix(".pages.jsonl")
        except ValueError:  # pragma: no cover - unexpected suffixless path
            return text_path.parent / f"{text_path.name}.pages.jsonl"

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        """Safely coerce value to int or None."""
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

    @staticmethod
    def _estimate_tokens_from_rows(rows: Sequence[MappingRow]) -> int | None:
        """Estimate token count from mapping rows."""
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


__all__ = ["DocumentMetadataManager"]
