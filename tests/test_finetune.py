from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

try:
    from transformers import PreTrainedTokenizerBase

    from beyond_the_cutoff.models.finetune import (
        SupervisedSample,
        load_supervised_samples,
        summarise_task_types,
        tokenise_samples,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name in {"datasets", "transformers"}:
        pytest.skip(f"{exc.name} package not installed", allow_module_level=True)
    raise


class ToyTokenizer:
    """Minimal tokenizer for tests assigning incremental IDs to whitespace tokens."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}
        self._next_id = 0
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.eos_token_id = self._get_or_add(self.eos_token)
        self.pad_token_id = self._get_or_add(self.pad_token)
        self.padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = [piece for piece in text.strip().split() if piece]
        token_ids = [self._get_or_add(piece) for piece in pieces]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def _get_or_add(self, token: str) -> int:
        if token not in self._vocab:
            idx = self._next_id
            self._vocab[token] = idx
            self._reverse[idx] = token
            self._next_id += 1
        return self._vocab[token]


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "offline_dataset.jsonl"
    records = [
        {
            "task_id": "doc-1-qa-0",
            "task_type": "qa",
            "instruction": "What is the main finding?",
            "expected_response": "The method improves accuracy by 5%.",
            "rag": {
                "prompt": "Context: ...\n\nQuestion: What is the main finding?\nAnswer:",
                "contexts": ["Snippet A", "Snippet B"],
            },
            "metadata": {"source_path": "docs/doc-1.txt"},
        },
        {
            "task_id": "doc-1-summary-0",
            "task_type": "summaries",
            "instruction": "Summarise the dataset section.",
            "expected_response": "The dataset includes 1200 labelled examples.",
            "rag": {
                "prompt": "Context: ...\n\nInstruction: Summarise the dataset section.\nAnswer:",
                "contexts": ["Dataset section"],
            },
            "metadata": {"source_path": "docs/doc-1.txt"},
        },
    ]
    with dataset.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return dataset


def test_load_supervised_samples_filters_and_extracts(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    samples = load_supervised_samples(dataset_path, allowed_task_types=["qa"])

    assert len(samples) == 1
    sample = samples[0]
    assert sample.prompt.startswith("Context:")
    assert sample.response.startswith("The method")
    assert sample.metadata["task_type"] == "qa"

    histogram = summarise_task_types(samples)
    assert histogram == {"qa": 1}


def test_tokenise_samples_produces_labels_mask(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    samples = load_supervised_samples(dataset_path)

    tokenizer = ToyTokenizer()
    dataset, stats = tokenise_samples(
        samples,
        cast(PreTrainedTokenizerBase, tokenizer),
        max_seq_length=64,
    )

    assert dataset.num_rows == stats.usable_samples == 2
    instance = dataset[0]
    assert len(instance["input_ids"]) == len(instance["attention_mask"]) == len(instance["labels"])
    prompt_mask_count = instance["labels"].count(-100)
    assert 0 < prompt_mask_count < len(instance["labels"])


def test_tokenise_samples_skips_when_prompt_too_long() -> None:
    samples = [
        SupervisedSample(prompt="word " * 100, response="short answer", metadata={}),
    ]
    tokenizer = ToyTokenizer()

    dataset, stats = tokenise_samples(
        samples,
        cast(PreTrainedTokenizerBase, tokenizer),
        max_seq_length=10,
    )

    assert dataset.num_rows == 0
    assert stats.usable_samples == 0
    assert stats.skipped_prompt_too_long == 1
