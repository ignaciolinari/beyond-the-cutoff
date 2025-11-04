"""Minimal FAISS-compatible shim backed by NumPy.

This is intended purely for local testing environments where the real FAISS
Python bindings are unavailable. It only implements the methods used inside the
project (``IndexFlatIP``, ``normalize_L2``, ``write_index``, and ``read_index``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt


def normalize_L2(array: npt.NDArray[np.float32]) -> None:  # noqa: N802 - mirror faiss API
    """Row-wise L2 normalisation (in place)."""

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    array /= norms


@dataclass
class IndexFlatIP:
    """Simplified in-memory index using inner product search."""

    dimension: int
    _vectors: npt.NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, vectors: npt.NDArray[np.float32]) -> None:
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension {self.dimension}, got {vectors.shape[1]}"
            )
        if self._vectors.size == 0:
            self._vectors = vectors.astype(np.float32, copy=True)
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(
        self, queries: npt.NDArray[np.float32], k: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Expected queries with dimension {self.dimension}, got {queries.shape[1]}"
            )
        if self._vectors.size == 0:
            scores = np.full((queries.shape[0], k), -1.0, dtype=np.float32)
            indices = np.full((queries.shape[0], k), -1, dtype=np.int64)
            return scores, indices

        scores = queries @ self._vectors.T
        topk_idx = np.argsort(scores, axis=1)[:, ::-1]
        topk_idx = topk_idx[:, :k]
        topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
        return topk_scores.astype(np.float32, copy=False), topk_idx.astype(np.int64)


def write_index(index: IndexFlatIP, path: str | Path) -> None:
    target = Path(path)
    with target.open("wb") as handle:
        np.savez_compressed(handle, vectors=index._vectors)


def read_index(path: str | Path) -> IndexFlatIP:
    source = Path(path)
    with source.open("rb") as handle, np.load(handle) as data:
        vectors = data["vectors"].astype(np.float32)
    restored = IndexFlatIP(vectors.shape[1])
    restored._vectors = vectors
    return restored


__all__ = [
    "IndexFlatIP",
    "normalize_L2",
    "read_index",
    "write_index",
]
