"""Document indexing using sentence-transformers and FAISS.

This module builds a dense vector index over text chunks and persists:
- a FAISS index file (binary)
- a TSV mapping file with fields: id, source_path, chunk_index, text
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..utils import faiss_stub

if os.environ.get("BTC_USE_FAISS_STUB") == "1":  # pragma: no cover - test/CI path
    faiss = faiss_stub
else:  # pragma: no cover - optional dependency
    try:
        faiss = import_module("faiss")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        faiss = faiss_stub
    except Exception:
        faiss = faiss_stub
import numpy as np
import numpy.typing as npt

try:  # pragma: no cover - torch is optional until encoding time
    import torch
except ImportError:  # pragma: no cover - runtime guard
    torch = None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
else:
    SentenceTransformerType = Any

try:  # pragma: no cover - optional dependency
    SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from ..utils.chunking import chunk_text, chunk_text_sentences


@dataclass
class DocumentIndexer:
    """Build and persist a FAISS index over text chunks."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 2
    device: str | None = None

    def _load_model(self) -> SentenceTransformerType:
        if SentenceTransformer is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("sentence-transformers is required to build the retrieval index.")
        return cast(SentenceTransformerType, SentenceTransformer(self.embedding_model))

    def build_index(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        pattern: str = "*.txt",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunking_strategy: str | None = None,
    ) -> tuple[Path, Path]:
        """Build a FAISS index from text files.

        Returns a tuple of (index_path, mapping_path).
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = output_dir / "mapping.tsv"
        index_path = output_dir / "index.faiss"
        meta_path = output_dir / "index_meta.json"

        texts: list[str] = []
        meta_rows: list[dict[str, object]] = []

        strategy = (chunking_strategy or "words").lower()

        for path in sorted(input_dir.rglob(pattern)):
            chunk_counter = 0
            absolute_cursor = 0
            # Prefer page-aware sidecar if present
            pages_sidecar = path.with_suffix(".pages.jsonl")
            if pages_sidecar.exists():
                with pages_sidecar.open("r", encoding="utf-8") as f:
                    page_records: list[dict[str, Any]] = []
                    for line in f:
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(record, dict):
                            continue
                        page_records.append(record)

                for group in self._group_pages_by_section(page_records):
                    section_title = group["section_title"]
                    section_text = group["text"]
                    start_page = group["start_page"]
                    if not section_text.strip():
                        continue

                    if strategy == "sentences":
                        chunks = chunk_text_sentences(
                            section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                    else:
                        chunks = chunk_text(
                            section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )

                    overlap_with_previous = 0
                    for ch in chunks:
                        tokens = ch.split()
                        token_len = len(tokens)
                        if token_len == 0:
                            continue

                        token_start = absolute_cursor
                        token_end = token_start + token_len
                        texts.append(ch)
                        meta_rows.append(
                            {
                                "source_path": str(path),
                                "page": start_page,
                                "chunk_index": chunk_counter,
                                "token_start": token_start,
                                "token_end": token_end,
                                "section_title": section_title,
                            }
                        )
                        chunk_counter += 1

                        unique_tokens = (
                            token_len
                            if overlap_with_previous == 0
                            else token_len - overlap_with_previous
                        )
                        if unique_tokens <= 0:
                            unique_tokens = 1
                        absolute_cursor += unique_tokens
                        overlap_with_previous = min(chunk_overlap, token_len)
                    overlap_with_previous = 0
                continue

            # Fallback: whole-document text
            content = path.read_text(encoding="utf-8", errors="ignore")
            if strategy == "sentences":
                chunks = chunk_text_sentences(
                    content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            else:
                chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            overlap_with_previous = 0
            for ch in chunks:
                tokens = ch.split()
                token_len = len(tokens)
                if token_len == 0:
                    continue

                token_start = absolute_cursor
                token_end = token_start + token_len
                texts.append(ch)
                meta_rows.append(
                    {
                        "source_path": str(path),
                        "page": None,
                        "chunk_index": chunk_counter,
                        "token_start": token_start,
                        "token_end": token_end,
                        "section_title": None,
                    }
                )
                chunk_counter += 1

                unique_tokens = (
                    token_len if overlap_with_previous == 0 else token_len - overlap_with_previous
                )
                if unique_tokens <= 0:
                    unique_tokens = 1
                absolute_cursor += unique_tokens
                overlap_with_previous = min(chunk_overlap, token_len)

        if not texts:
            raise ValueError(f"No text files matching {pattern!r} found under {input_dir}")

        model = self._load_model()
        if torch is not None:
            try:
                max_threads = int(os.environ.get("BTC_TORCH_THREADS", "1"))
            except ValueError:
                max_threads = 1
            torch.set_num_threads(max(1, max_threads))

        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }
        if self.device:
            encode_kwargs["device"] = self.device

        embeddings = model.encode(texts, **encode_kwargs)
        embeddings_array = np.asarray(embeddings, dtype="float32")
        embeddings_nd = cast(npt.NDArray[np.float32], embeddings_array)

        index = faiss.IndexFlatIP(embeddings_nd.shape[1])
        faiss.normalize_L2(embeddings_nd)
        index.add(embeddings_nd)
        faiss.write_index(index, str(index_path))

        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "id",
                    "source_path",
                    "page",
                    "chunk_index",
                    "token_start",
                    "token_end",
                    "section_title",
                    "text",
                ]
            )  # header
            for idx, row in enumerate(meta_rows):
                writer.writerow(
                    [
                        idx,
                        row.get("source_path", ""),
                        row.get("page", ""),
                        row.get("chunk_index", 0),
                        row.get("token_start", ""),
                        row.get("token_end", ""),
                        row.get("section_title", ""),
                        texts[idx],
                    ]
                )

        # Persist index metadata for validation on load
        meta = {
            "embedding_model": self.embedding_model,
            "embedding_dimension": int(embeddings_nd.shape[1]),
            "normalized": True,
            "chunking_strategy": strategy,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "mapping_fields": [
                "id",
                "source_path",
                "page",
                "chunk_index",
                "token_start",
                "token_end",
                "section_title",
                "text",
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return index_path, mapping_path

    @staticmethod
    def _group_pages_by_section(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group page JSONL records into section-aware blocks."""

        groups: list[dict[str, Any]] = []

        def _start_section(title: str | None, page: int) -> dict[str, Any]:
            return {
                "section_title": title.strip()
                if isinstance(title, str) and title.strip()
                else None,
                "start_page": page,
                "end_page": page,
                "texts": [],
            }

        def _flush(current: dict[str, Any] | None) -> None:
            if not current:
                return
            combined = "\n\n".join(current.get("texts", [])).strip()
            if combined:
                groups.append(
                    {
                        "section_title": current.get("section_title"),
                        "start_page": current.get("start_page"),
                        "end_page": current.get("end_page"),
                        "text": combined,
                    }
                )

        current_section: dict[str, Any] | None = None
        last_page_num = 0

        for record in records:
            raw_page = record.get("page")
            if isinstance(raw_page, int):
                page_num = raw_page
            elif isinstance(raw_page, str) and raw_page.strip():
                try:
                    page_num = int(raw_page.strip())
                except ValueError:
                    page_num = last_page_num + 1 if last_page_num else 1
            else:
                page_num = last_page_num + 1 if last_page_num else 1
            last_page_num = page_num

            raw_title = record.get("section_title")
            title = raw_title.strip() if isinstance(raw_title, str) else None
            page_text = str(record.get("text", "") or "")

            if current_section is None:
                current_section = _start_section(title, page_num)
            else:
                current_title = current_section.get("section_title")
                if title and title != current_title:
                    _flush(current_section)
                    current_section = _start_section(title, page_num)
                elif current_title is None and title:
                    _flush(current_section)
                    current_section = _start_section(title, page_num)
                else:
                    current_section["end_page"] = page_num

            if current_section is None:
                current_section = _start_section(title, page_num)

            if title and not current_section.get("section_title"):
                current_section["section_title"] = title

            if page_text:
                current_section.setdefault("texts", []).append(page_text)
            current_section["end_page"] = page_num

        _flush(current_section)
        return groups
