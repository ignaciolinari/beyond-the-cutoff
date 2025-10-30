"""Utilities for chunking text into overlapping windows for retrieval.

The chunking strategy is simple and fast: split on whitespace into tokens, then
reconstruct chunks with a sliding window of fixed size and overlap. This is
robust to long paragraphs and avoids splitting mid-word.
"""

from __future__ import annotations

import re
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


_ABBREVIATIONS = {
    "al.",
    "e.g.",
    "i.e.",
    "cf.",
    "fig.",
    "eq.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "sr.",
    "jr.",
    "vs.",
    "prof.",
}


def split_into_sentences(text: str) -> list[str]:
    """Heuristic sentence splitter with optional NLTK fallback."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    # Prefer NLTK Punkt if available (no hard dependency)
    try:  # pragma: no cover - optional dependency
        import nltk

        if hasattr(nltk, "sent_tokenize"):
            return [s.strip() for s in nltk.sent_tokenize(cleaned) if s.strip()]
    except Exception:
        pass

    # Fallback heuristic splitter with abbreviation awareness
    pieces: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]", cleaned):
        end = match.end()
        token = cleaned[start:end].strip()
        if not token:
            continue
        lower_token = token.lower()
        if any(lower_token.endswith(abbr) for abbr in _ABBREVIATIONS):
            continue
        pieces.append(token)
        start = end
    if start < len(cleaned):
        tail = cleaned[start:].strip()
        if tail:
            pieces.append(tail)

    if not pieces:
        return [cleaned]

    sentences: list[str] = []
    for part in pieces:
        s = part.strip()
        if not s:
            continue
        if sentences and len(s.split()) < 3:
            sentences[-1] = (sentences[-1] + " " + s).strip()
        else:
            sentences.append(s)
    return sentences


def chunk_text_sentences(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Sentence-aware chunking with token-based overlap.

    Chunks pack whole sentences up to approximately `chunk_size` tokens.
    Overlap is implemented by repeating the last `chunk_overlap` tokens of
    the previous chunk at the beginning of the next chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    prev_tail_tokens: list[str] = []
    current_sentences: list[str] = []

    def count_tokens(items: list[str]) -> int:
        return sum(len(s.split()) for s in items)

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = len(sent.split())
        base_tokens = len(prev_tail_tokens)
        current_tokens = base_tokens + count_tokens(current_sentences)
        if current_tokens + sent_tokens <= chunk_size or not current_sentences:
            current_sentences.append(sent)
            i += 1
            continue

        # finalize current chunk
        chunk_tokens = prev_tail_tokens + (" ".join(current_sentences)).split()
        chunk_text_str = " ".join(chunk_tokens)
        if chunk_text_str:
            chunks.append(chunk_text_str)
            prev_tail_tokens = chunk_text_str.split()[-chunk_overlap:] if chunk_overlap else []
        else:
            prev_tail_tokens = []
        current_sentences = []

    # last chunk
    if current_sentences or prev_tail_tokens:
        chunk_tokens = prev_tail_tokens + (" ".join(current_sentences)).split()
        chunk_text_str = " ".join(chunk_tokens)
        if chunk_text_str:
            chunks.append(chunk_text_str)

    return chunks
