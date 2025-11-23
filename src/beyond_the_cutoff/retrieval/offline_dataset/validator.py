"""Validation logic for generator payloads and output examples."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from .parser import ResponseParser
from .types import OfflineExample

logger = logging.getLogger(__name__)

GENERATOR_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "qa": ("question", "answer"),
    "summaries": ("instruction", "response"),
    "citations": ("instruction", "answer"),
    "contextualizations": ("instruction", "response"),
}


class PayloadValidator:
    """Validate generator payloads and output examples."""

    def __init__(self, parser: ResponseParser):
        self.parser = parser

    def validate_generator_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Validate and clean generator JSON payload."""
        cleaned = dict(payload)
        issues: list[dict[str, Any]] = []

        for key, required_fields in GENERATOR_REQUIRED_FIELDS.items():
            items = payload.get(key, [])
            if not isinstance(items, list):
                issues.append(
                    {
                        "field": key,
                        "kind": "not_list",
                        "fatal": key != "qa",
                    }
                )
                cleaned[key] = []
                continue

            valid_items: list[dict[str, Any]] = []
            for idx, item in enumerate(items):
                if not isinstance(item, Mapping):
                    issues.append(
                        {
                            "field": f"{key}[{idx}]",
                            "kind": "not_mapping",
                            "fatal": key != "qa",
                        }
                    )
                    continue

                normalized = dict(item)
                missing: list[str] = []
                for required_field in required_fields:
                    value = item.get(required_field)
                    if value is None:
                        missing.append(required_field)
                        continue
                    text_value = self.parser.coerce_text(value).strip()
                    if not text_value:
                        missing.append(required_field)
                        continue
                    normalized[required_field] = text_value

                if missing:
                    issues.append(
                        {
                            "field": f"{key}[{idx}]",
                            "kind": "missing_required",
                            "missing": missing,
                            "fatal": key != "qa",
                        }
                    )
                    continue

                valid_items.append(normalized)

            cleaned[key] = valid_items

        return cleaned, issues

    def validate_output_examples(
        self,
        examples: Sequence[OfflineExample],
        pipeline: Any,  # RAGPipeline - avoid circular import
        min_citation_coverage: float,
    ) -> tuple[list[OfflineExample], list[dict[str, Any]]]:
        """Validate final output examples with citation checks."""
        filtered: list[OfflineExample] = []
        issues: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for example in examples:
            instruction_text = example.instruction.strip()
            response_text = example.expected_response.strip()
            if not instruction_text or not response_text:
                issues.append(
                    {
                        "field": example.task_type,
                        "kind": "empty_text",
                        "fatal": True,
                        "instruction": self._truncate(example.instruction),
                    }
                )
                continue

            key = (example.task_type, instruction_text.lower())
            if key in seen:
                issues.append(
                    {
                        "field": example.task_type,
                        "kind": "duplicate_instruction",
                        "fatal": False,
                        "instruction": self._truncate(example.instruction),
                    }
                )
                continue

            if example.metadata.get("require_citations"):
                citation_issue = self._check_citation_coverage(
                    example, pipeline, min_citation_coverage
                )
                if citation_issue is not None:
                    issues.append(citation_issue)
                    if citation_issue.get("fatal", True):
                        continue

            filtered.append(example)
            seen.add(key)

        return filtered, issues

    def _check_citation_coverage(
        self, example: OfflineExample, pipeline: Any, min_coverage: float
    ) -> dict[str, Any] | None:
        """Check citation coverage for an example requiring citations."""
        enforcement_meta = example.metadata.get("citation_enforcement")
        verification: Mapping[str, Any] | None = None
        if isinstance(enforcement_meta, Mapping):
            verification = enforcement_meta.get("verification")
            if not isinstance(verification, Mapping):
                verification = None

        if verification is None:
            verification = pipeline.verify_citations(
                example.expected_response, list(example.contexts)
            )
            updated_meta = dict(enforcement_meta) if isinstance(enforcement_meta, Mapping) else {}
            updated_meta.setdefault("status", updated_meta.get("status", "post_validation"))
            updated_meta["verification"] = verification
            example.metadata["citation_enforcement"] = updated_meta

        # verification is guaranteed to be a Mapping at this point
        assert verification is not None
        referenced = verification.get("referenced") or []
        extra = verification.get("extra") or []
        coverage = verification.get("coverage") or {}

        if not referenced:
            return {
                "field": example.task_type,
                "kind": "missing_citations",
                "fatal": True,
                "instruction": self._truncate(example.instruction),
            }

        insufficient = [idx for idx in referenced if coverage.get(idx, 0.0) < min_coverage]
        if insufficient:
            return {
                "field": example.task_type,
                "kind": "low_citation_coverage",
                "fatal": True,
                "instruction": self._truncate(example.instruction),
                "details": {"referenced": referenced, "insufficient": insufficient},
            }

        if extra:
            return {
                "field": example.task_type,
                "kind": "invalid_citation_reference",
                "fatal": True,
                "instruction": self._truncate(example.instruction),
                "details": {"extra": extra},
            }

        return None

    @staticmethod
    def _truncate(text: str, limit: int = 120) -> str:
        snippet = text.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def has_tasks(payload: Mapping[str, Any]) -> bool:
        """Check if payload contains any tasks."""
        qa_items = payload.get("qa", [])
        summary_items = payload.get("summaries", [])
        citation_items = payload.get("citations", [])
        contextual_items = payload.get("contextualizations", [])
        return any(qa_items) or any(summary_items) or any(citation_items) or any(contextual_items)

    @staticmethod
    def missing_minimum_counts(payload: Mapping[str, Any], cfg: Any) -> dict[str, dict[str, int]]:
        """Check if payload meets minimum task counts."""
        deficits: dict[str, dict[str, int]] = {}

        def _count_items(key: str) -> int:
            items = payload.get(key, [])
            if isinstance(items, Sequence):
                return len(items)
            return 0

        requirements = {
            "qa": getattr(cfg, "min_questions_per_document", 0),
            "summaries": getattr(cfg, "min_summary_prompts_per_document", 0),
            "citations": getattr(cfg, "min_citation_prompts_per_document", 0),
            "contextualizations": getattr(cfg, "min_contextual_prompts_per_document", 0),
        }

        for key, required in requirements.items():
            if required <= 0:
                continue
            observed = _count_items(key)
            if observed < required:
                deficits[key] = {"required": int(required), "observed": observed}

        return deficits


__all__ = ["PayloadValidator", "GENERATOR_REQUIRED_FIELDS"]
