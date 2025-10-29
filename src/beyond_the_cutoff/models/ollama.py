"""Minimal Ollama HTTP client tailored for local experimentation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

Message = Mapping[str, str]


class OllamaError(RuntimeError):
    """Raised when the Ollama service returns an unexpected response."""


@dataclass
class OllamaClient:
    """Convenience wrapper around the Ollama HTTP API."""

    model: str = "phi3:mini"
    host: str = "http://localhost"
    port: int | None = 11434
    timeout: float = 60.0
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        """Return the base URL where the Ollama daemon listens."""
        root = self.host.rstrip("/")
        if self.port is None:
            return root
        return f"{root}:{self.port}"

    def generate(
        self, prompt: str, *, stream: bool = False, options: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call `/api/generate` with the provided prompt."""
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": stream}
        if options:
            payload["options"] = dict(options)
        return self._request("POST", "/api/generate", payload)

    def chat(
        self,
        messages: Iterable[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call `/api/chat` with a list of role/content messages."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": stream,
        }
        if options:
            payload["options"] = dict(options)
        return self._request("POST", "/api/chat", payload)

    def list_models(self) -> dict[str, Any]:
        """Return the tags known to the local Ollama daemon."""
        return self._request("GET", "/api/tags")

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                response = client.request(
                    method, url, json=dict(payload) if payload is not None else None
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failures
            raise OllamaError(f"Ollama request to {url} failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned non-JSON response") from exc

        if isinstance(data, dict):
            return data
        raise OllamaError(f"Unexpected payload type from Ollama: {type(data)!r}")
