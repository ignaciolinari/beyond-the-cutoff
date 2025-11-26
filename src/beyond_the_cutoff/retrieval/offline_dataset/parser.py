"""JSON parsing utilities for generator LLM responses."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_INVALID_ESCAPE_PATTERN = re.compile(r"\\(?![\"\\/bfnrtu])")


class ResponseParser:
    """Parse and clean generator LLM responses."""

    @staticmethod
    def strip_fences(text: str) -> str:
        """Remove markdown code fences from JSON responses."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
        if stripped.startswith("```json"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```json"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    @staticmethod
    def coerce_text(value: Any) -> str:
        """Convert various types to string."""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:  # pragma: no cover - unexpected encoding
                return value.decode(errors="ignore")
        if isinstance(value, Sequence):
            return " ".join(str(part) for part in value if part)
        return str(value)

    def parse_generator_response(self, text: str) -> dict[str, Any] | None:
        """Parse JSON response from generator LLM with error recovery."""
        candidate = self.strip_fences(text)

        # Handle primed responses that start with "qa": (missing opening brace)
        stripped_candidate = candidate.strip()
        if stripped_candidate.startswith('"qa"') or stripped_candidate.startswith("'qa'"):
            candidate = "{" + candidate
            logger.debug("Prepended opening brace for primed JSON response")

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if "Invalid \\escape" in exc.msg:
                cleaned = _INVALID_ESCAPE_PATTERN.sub("", candidate)
                if cleaned != candidate:
                    try:
                        data = json.loads(cleaned)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Failed to decode generator JSON after escape fix: %s",
                            candidate,
                        )
                        return None
                    else:
                        logger.debug("Recovered generator JSON after sanitizing invalid escapes")
                else:
                    logger.debug("Failed to decode generator JSON: %s", candidate)
                    return None
            else:
                logger.debug("Failed to decode generator JSON: %s", candidate)
                return None
        if not isinstance(data, dict):
            return None
        # Normalise expected keys
        data.setdefault("qa", [])
        data.setdefault("summaries", [])
        data.setdefault("citations", [])
        data.setdefault("contextualizations", [])
        return data


__all__ = ["ResponseParser"]
