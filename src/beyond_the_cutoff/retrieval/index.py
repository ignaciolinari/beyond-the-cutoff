"""Document indexing using sentence-transformers and FAISS.

This module builds a dense vector index over text chunks and persists:
- a FAISS index file (binary)
- a TSV mapping file with fields: id, source_path, chunk_index, text
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from ..utils.chunking import chunk_text


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
    ) -> tuple[Path, Path]:
        """Build a FAISS index from text files.

        Returns a tuple of (index_path, mapping_path).
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = output_dir / "mapping.tsv"
        index_path = output_dir / "index.faiss"

        texts: list[str] = []
        meta: list[tuple[str, int]] = []  # (source_path, chunk_index)

        for path in sorted(input_dir.rglob(pattern)):
            content = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, ch in enumerate(chunks):
                texts.append(ch)
                meta.append((str(path), i))

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
            writer.writerow(["id", "source_path", "chunk_index", "text"])  # header
            for idx, (src, ci) in enumerate(meta):
                writer.writerow([idx, src, ci, texts[idx]])

        return index_path, mapping_path
