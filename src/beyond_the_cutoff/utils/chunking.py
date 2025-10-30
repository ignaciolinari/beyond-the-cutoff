"""Utilities for chunking text into overlapping windows for retrieval.

The chunking strategy is simple and fast: split on whitespace into tokens, then
reconstruct chunks with a sliding window of fixed size and overlap. This is
robust to long paragraphs and avoids splitting mid-word.
"""

from __future__ import annotations

from collections.abc import Iterable


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Target number of tokens (whitespace-delimited words) per chunk.
        chunk_overlap: Number of tokens to overlap between adjacent chunks.

    Returns:
        A list of chunk strings. If the input is short, returns a single chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = text.split()
    if not tokens:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(tokens):
            break
    return chunks


def iter_chunk_texts(texts: Iterable[str], chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    """Yield chunks for a sequence of texts with the same parameters."""

    for text in texts:
        yield from chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
