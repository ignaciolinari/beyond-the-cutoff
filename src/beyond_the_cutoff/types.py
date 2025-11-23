"""Shared type definitions and enumerations for Beyond the Cutoff."""

from __future__ import annotations

from enum import Enum


class ModelType(str, Enum):
    """Model type classification for evaluation and prompt formatting."""

    BASE = "base"
    INSTRUCTION_ONLY = "instruction_only"
    RAG_TRAINED = "rag_trained"

    def __str__(self) -> str:
        return self.value


class PromptMode(str, Enum):
    """Prompt mode for evaluation."""

    RAG = "rag"
    INSTRUCTION = "instruction"

    def __str__(self) -> str:
        return self.value


__all__ = ["ModelType", "PromptMode"]
