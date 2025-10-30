"""Document indexing using sentence-transformers and FAISS.

This module builds a dense vector index over text chunks and persists:
- a FAISS index file (binary)
- a TSV mapping file with fields: id, source_path, chunk_index, text
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from ..utils.chunking import chunk_text, chunk_text_sentences


@dataclass
class DocumentIndexer:
    """Build and persist a FAISS index over text chunks."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def _load_model(self) -> SentenceTransformer:
        return SentenceTransformer(self.embedding_model)

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
            # Prefer page-aware sidecar if present
            pages_sidecar = path.with_suffix(".pages.jsonl")
            if pages_sidecar.exists():
                page_idx = 0
                with pages_sidecar.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        page_idx = int(rec.get("page", page_idx + 1))
                        page_text = str(rec.get("text", ""))
                        if not page_text.strip():
                            continue
                        if strategy == "sentences":
                            chunks = chunk_text_sentences(
                                page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                            )
                        else:
                            chunks = chunk_text(
                                page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                            )
                        start_token = 0
                        for ci, ch in enumerate(chunks):
                            token_len = len(ch.split())
                            texts.append(ch)
                            meta_rows.append(
                                {
                                    "source_path": str(path),
                                    "page": page_idx,
                                    "chunk_index": ci,
                                    "token_start": start_token,
                                    "token_end": start_token + token_len,
                                }
                            )
                            # next chunk starts advance by (chunk_size - overlap)
                            start_token += max(1, chunk_size - chunk_overlap)
                continue

            # Fallback: whole-document text
            content = path.read_text(encoding="utf-8", errors="ignore")
            if strategy == "sentences":
                chunks = chunk_text_sentences(
                    content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            else:
                chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            start_token = 0
            for ci, ch in enumerate(chunks):
                token_len = len(ch.split())
                texts.append(ch)
                meta_rows.append(
                    {
                        "source_path": str(path),
                        "page": None,
                        "chunk_index": ci,
                        "token_start": start_token,
                        "token_end": start_token + token_len,
                    }
                )
                start_token += max(1, chunk_size - chunk_overlap)

        if not texts:
            raise ValueError(f"No text files matching {pattern!r} found under {input_dir}")

        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
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
                "text",
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return index_path, mapping_path
