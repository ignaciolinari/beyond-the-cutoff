"""Shared interfaces for local generation clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class LLMClient(Protocol):
    """Protocol describing the minimal interface for generation backends."""

    model: str

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return a dictionary containing at least a ``response`` field."""
