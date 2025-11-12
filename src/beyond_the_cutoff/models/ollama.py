"""Minimal Ollama HTTP client tailored for local experimentation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

httpx: Any
_HTTPX_IMPORT_ERROR: ModuleNotFoundError | None

try:  # pragma: no cover - optional dependency
    httpx = import_module("httpx")
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    httpx = None
    _HTTPX_IMPORT_ERROR = exc
else:
    _HTTPX_IMPORT_ERROR = None

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
    temperature: float | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    num_predict: int | None = None
    stop_sequences: tuple[str, ...] = ()

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
        extra_options, stop = self._prepare_options(options)
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": stream}
        if extra_options:
            payload["options"] = extra_options
        if stop:
            payload["stop"] = stop
        return self._request("POST", "/api/generate", payload)

    def chat(
        self,
        messages: Iterable[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call `/api/chat` with a list of role/content messages."""
        extra_options, stop = self._prepare_options(options)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": stream,
        }
        if extra_options:
            payload["options"] = extra_options
        if stop:
            payload["stop"] = stop
        return self._request("POST", "/api/chat", payload)

    def list_models(self) -> dict[str, Any]:
        """Return the tags known to the local Ollama daemon."""
        return self._request("GET", "/api/tags")

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if httpx is None:  # pragma: no cover - optional dependency guard
            raise OllamaError(
                "httpx is not installed. Install the 'httpx' extra to use the Ollama backend."
            ) from _HTTPX_IMPORT_ERROR
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

    def _prepare_options(
        self, options: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any] | None, list[str] | None]:
        defaults: dict[str, Any] = {}
        if self.temperature is not None:
            defaults["temperature"] = self.temperature
        if self.top_p is not None:
            defaults["top_p"] = self.top_p
        if self.repeat_penalty is not None:
            defaults["repeat_penalty"] = self.repeat_penalty
        if self.num_predict is not None and self.num_predict > 0:
            defaults["num_predict"] = self.num_predict

        merged = dict(defaults)
        stop_values: list[str] = list(self.stop_sequences)

        if options:
            incoming = dict(options)
            if "stop" in incoming:
                stop_values.extend(self._normalise_stop(incoming.pop("stop")))
            merged.update(incoming)

        cleaned_options = merged if merged else None
        cleaned_stop = stop_values if stop_values else None
        return cleaned_options, cleaned_stop

    @staticmethod
    def _normalise_stop(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(item) for item in value if str(item)]
        return [str(value)]
