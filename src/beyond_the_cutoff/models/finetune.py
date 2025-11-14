"""Utilities to support local LoRA/PEFT fine-tuning workflows."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SupervisedSample:
    """Minimal prompt/response pair extracted from an offline dataset row."""

    prompt: str
    response: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class TokenisationStats:
    """Summary describing how raw samples were transformed into tokenised inputs."""

    total_samples: int
    usable_samples: int
    skipped_empty_response: int
    skipped_prompt_too_long: int
    truncated_samples: int
    avg_prompt_tokens: float
    avg_response_tokens: float


def load_supervised_samples(
    dataset_path: Path,
    *,
    allowed_task_types: Sequence[str] | None = None,
    use_rag_contexts: bool = True,
) -> list[SupervisedSample]:
    """Parse a JSONL offline dataset into prompt/response pairs.

    Args:
        dataset_path: Path to the JSONL offline dataset file
        allowed_task_types: Optional sequence of task types to include (e.g., ["qa", "summary"])
        use_rag_contexts: If True, use RAG prompts with contexts when available.
                          If False, extract instruction-only prompts (for instruction-only training).
    """

    path = dataset_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    allowed: set[str] | None = None
    if allowed_task_types:
        allowed = {item.strip().lower() for item in allowed_task_types if item.strip()}

    samples: list[SupervisedSample] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_no}: {exc}") from exc

            task_type_raw = str(record.get("task_type", "")).strip().lower()
            if allowed is not None and task_type_raw not in allowed:
                continue

            prompt = _extract_prompt(record, use_rag_contexts=use_rag_contexts)
            response = _extract_response(record)
            if not prompt or not response:
                continue

            metadata: dict[str, Any] = {
                "task_id": record.get("task_id"),
                "task_type": record.get("task_type"),
                "source_path": (record.get("metadata") or {}).get("source_path"),
            }
            samples.append(SupervisedSample(prompt=prompt, response=response, metadata=metadata))

    if not samples:
        logger.warning("No usable samples were extracted from %s", path)
    return samples


def summarise_task_types(samples: Sequence[SupervisedSample]) -> dict[str, int]:
    """Return a histogram of task types for logging/reporting."""

    counts: dict[str, int] = {}
    for sample in samples:
        task_type = str(sample.metadata.get("task_type") or "unknown").lower()
        counts[task_type] = counts.get(task_type, 0) + 1
    return counts


def tokenise_samples(
    samples: Sequence[SupervisedSample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_length: int,
    seed: int = 42,
) -> tuple[Dataset, TokenisationStats]:
    """Convert prompt/response pairs into tokenised tensors for causal LM training."""

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be > 0")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    features: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []

    skipped_empty = 0
    skipped_too_long = 0
    truncated = 0

    for sample in shuffled:
        prompt_ids = _tokenize_to_ids(tokenizer, sample.prompt)
        response_text = sample.response.strip()
        if not response_text:
            skipped_empty += 1
            continue

        answer_ids = _tokenize_to_ids(tokenizer, response_text)
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        if not answer_ids:
            skipped_empty += 1
            continue

        if len(prompt_ids) >= max_seq_length:
            skipped_too_long += 1
            continue

        available_answer_slots = max_seq_length - len(prompt_ids)
        if len(answer_ids) > available_answer_slots:
            answer_ids = answer_ids[:available_answer_slots]
            truncated += 1
            if not answer_ids:
                skipped_too_long += 1
                continue

        input_ids = list(prompt_ids + answer_ids)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + list(answer_ids)

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(answer_ids))

    if not features:
        empty_dataset = Dataset.from_dict(
            {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
        )
        stats = TokenisationStats(
            total_samples=len(samples),
            usable_samples=0,
            skipped_empty_response=skipped_empty,
            skipped_prompt_too_long=skipped_too_long,
            truncated_samples=truncated,
            avg_prompt_tokens=0.0,
            avg_response_tokens=0.0,
        )
        return empty_dataset, stats

    dataset = Dataset.from_list(features)

    stats = TokenisationStats(
        total_samples=len(samples),
        usable_samples=len(features),
        skipped_empty_response=skipped_empty,
        skipped_prompt_too_long=skipped_too_long,
        truncated_samples=truncated,
        avg_prompt_tokens=float(_safe_mean(prompt_lengths)),
        avg_response_tokens=float(_safe_mean(response_lengths)),
    )
    return dataset, stats


def _extract_prompt(record: dict[str, Any], *, use_rag_contexts: bool = True) -> str:
    """Extract prompt from an offline dataset record.

    Args:
        record: Dictionary containing task data from offline_dataset.jsonl
        use_rag_contexts: If True, prefer RAG prompts with contexts when available.
                         If False, extract instruction-only prompts (for instruction-only training).

    Returns:
        Extracted prompt string, or empty string if no valid prompt found.
    """
    instruction = str(record.get("instruction") or "").strip()

    # For instruction-only mode, return just the instruction
    if not use_rag_contexts:
        if instruction:
            return instruction
        return ""

    # For RAG mode, prefer the pre-built RAG prompt if available
    rag = record.get("rag") or {}
    prompt = rag.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    # Fallback: construct prompt from instruction + contexts
    contexts: Iterable[str] = rag.get("contexts") or []
    context_text = "\n\n".join(str(item).strip() for item in contexts if str(item).strip())
    if context_text:
        return f"Instruction: {instruction}\n\nContext:\n{context_text}\n\nAnswer:"
    if instruction:
        return f"Instruction: {instruction}\nAnswer:"
    return ""


def _extract_response(record: dict[str, Any]) -> str:
    response = record.get("expected_response") or record.get("response")
    return str(response or "").strip()


def _safe_mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / max(len(values), 1)


def _tokenize_to_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids)


__all__ = [
    "SupervisedSample",
    "TokenisationStats",
    "load_supervised_samples",
    "summarise_task_types",
    "tokenise_samples",
]
