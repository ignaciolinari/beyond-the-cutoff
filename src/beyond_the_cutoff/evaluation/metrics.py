"""Utility functions for automatic evaluation metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from statistics import mean
from typing import Any

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CONTEXT_NUMBER_PATTERN = re.compile(r"^\s*\[(\d+)\]")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def normalize_contexts(contexts: Iterable[Any]) -> list[str]:
    """Return contexts as trimmed strings with explicit numbering."""
    processed = [str(ctx).strip() for ctx in contexts if ctx is not None]
    if not processed:
        return []
    if all(_CONTEXT_NUMBER_PATTERN.match(ctx) for ctx in processed):
        return processed
    return [f"[{idx + 1}] {ctx}" for idx, ctx in enumerate(processed)]


def evaluate_citations(answer: str, contexts: Sequence[str]) -> dict[str, Any]:
    """Compute simple citation coverage diagnostics for ``answer``.

    Coverage is computed as the fraction of meaningful words (>3 chars) from
    the cited context that appear in the answer. This measures how well the
    answer actually uses the cited source material.
    """
    marks = [int(match) for match in _CITATION_PATTERN.findall(answer)]
    unique_marks = sorted(set(marks))
    total = len(contexts)
    missing = [i for i in range(1, total + 1) if i not in unique_marks]
    extra = [i for i in unique_marks if i < 1 or i > total]

    # Filter answer words to meaningful tokens (>3 chars)
    answer_filtered = [w for w in answer.lower().split() if len(w) > 3]
    answer_words = set(answer_filtered)

    coverage: dict[int, float] = {}
    for idx in unique_marks:
        if idx < 1 or idx > total:
            continue
        # Strip citation markers like "[1]" from context before word extraction
        context_text = _CONTEXT_NUMBER_PATTERN.sub("", contexts[idx - 1])
        context_words = [w for w in context_text.lower().split() if len(w) > 3]
        if not context_words:
            coverage[idx] = 0.0
            continue
        overlap = sum(1 for w in context_words if w in answer_words)
        coverage[idx] = overlap / len(context_words)

    mean_coverage = mean(coverage.values()) if coverage else 0.0
    return {
        "referenced": unique_marks,
        "missing": missing,
        "extra": extra,
        "coverage": coverage,
        "mean_coverage": mean_coverage,
        "total_contexts": total,
        "unique_citations": len(unique_marks),
    }


def citation_precision(metrics: Mapping[str, Any]) -> float:
    """Return precision-style score derived from :func:`evaluate_citations`."""
    referenced = metrics.get("referenced", [])
    extra = metrics.get("extra", [])
    total = len(referenced)
    if total == 0:
        return 1.0
    valid = total - len(extra)
    return max(min(valid / total, 1.0), 0.0)


def citation_recall(metrics: Mapping[str, Any], *, total_contexts: int) -> float:
    """Return recall-style score derived from :func:`evaluate_citations`."""
    if total_contexts == 0:
        return 1.0
    missing = metrics.get("missing", [])
    covered = total_contexts - len(missing)
    return max(min(covered / total_contexts, 1.0), 0.0)


def grounded_fraction(answer: str, contexts: Sequence[str], *, min_token_length: int = 3) -> float:
    """Estimate groundedness via lexical overlap with ``contexts``."""
    tokens = [tok.lower() for tok in _TOKEN_PATTERN.findall(answer)]
    meaningful = [tok for tok in tokens if len(tok) >= min_token_length]
    if not meaningful:
        return 0.0
    context_tokens = {
        tok.lower()
        for ctx in contexts
        for tok in _TOKEN_PATTERN.findall(ctx)
        if len(tok) >= min_token_length
    }
    if not context_tokens:
        return 0.0
    grounded = sum(1 for tok in meaningful if tok in context_tokens)
    return grounded / len(meaningful)


@lru_cache(maxsize=1)
def _load_bleu_metric() -> Any:  # pragma: no cover - integration heavy
    try:
        from evaluate import load

        return load("bleu")
    except ImportError as e:
        raise ImportError(
            "The 'evaluate' library is required for BLEU computation. "
            "Install it with: pip install evaluate"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load BLEU metric: {e}") from e


@lru_cache(maxsize=1)
def _load_bertscore_metric() -> Any:  # pragma: no cover - integration heavy
    try:
        from evaluate import load

        return load("bertscore")
    except ImportError as e:
        raise ImportError(
            "The 'evaluate' library is required for BERTScore computation. "
            "Install it with: pip install evaluate bert-score"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load BERTScore metric: {e}") from e


def compute_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute BLEU score for given predictions and references."""
    if not predictions or not references:
        return 0.0
    metric = _load_bleu_metric()
    formatted_refs = [[ref] for ref in references]
    result = metric.compute(predictions=list(predictions), references=formatted_refs)
    return float(result.get("bleu", 0.0))


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    lang: str = "en",
) -> dict[str, float]:
    """Compute mean BERTScore precision/recall/F1 for the provided pairs."""
    if not predictions or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    metric = _load_bertscore_metric()
    result = metric.compute(predictions=list(predictions), references=list(references), lang=lang)
    precision = result.get("precision", [])
    recall = result.get("recall", [])
    f1 = result.get("f1", [])
    return {
        "precision": _safe_mean(precision),
        "recall": _safe_mean(recall),
        "f1": _safe_mean(f1),
    }


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


# End of module
