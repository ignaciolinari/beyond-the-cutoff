"""Model backends for Beyond the Cutoff."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMClient
from .ollama import OllamaClient
from .transformers_local import TransformersClient

if TYPE_CHECKING:  # pragma: no cover
    from ..config import InferenceConfig

__all__ = [
    "LLMClient",
    "OllamaClient",
    "TransformersClient",
    "build_generation_client",
]


def build_generation_client(config: InferenceConfig) -> LLMClient:
    """Instantiate the generation backend declared in the config."""

    provider = config.provider.strip().lower()
    if provider == "ollama":
        return OllamaClient(
            model=config.model,
            host=config.host,
            port=config.port,
            timeout=config.timeout,
        )
    if provider in {"transformers", "hf", "huggingface"}:
        return TransformersClient(
            model=config.model,
            device=config.device,
            torch_dtype=config.torch_dtype,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            stop_sequences=config.stop_sequences,
        )
    raise ValueError(f"Unsupported inference provider: {config.provider!r}")
