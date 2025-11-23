"""Citation enforcement and rewriting logic."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


class CitationEnforcer:
    """Enforce citation compliance with automatic rewriting."""

    def __init__(self, generator_client: Any, config: Any, pipeline: Any):
        """
        Args:
            generator_client: LLM client for rewriting
            config: ProjectConfig with dataset_generation settings
            pipeline: RAGPipeline for verification
        """
        self.generator = generator_client
        self.config = config
        self.pipeline = pipeline

    def ensure_citation_compliance(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        source_path: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Verify and potentially rewrite answer to ensure citation compliance.

        Returns:
            Tuple of (cleaned_answer, enforcement_metadata) or None if compliance failed
        """
        enforcement: dict[str, Any] = {"status": "pass", "attempts": 0}
        cleaned_answer = answer.strip()
        if not cleaned_answer:
            return None

        if not contexts:
            enforcement["verification"] = {"referenced": []}
            return cleaned_answer, enforcement

        min_coverage = max(0.0, min(1.0, self.config.dataset_generation.min_citation_coverage))

        def _is_compliant(payload: Mapping[str, Any]) -> bool:
            referenced = payload.get("referenced") or []
            if not referenced or payload.get("extra"):
                return False
            if min_coverage <= 0.0:
                return True
            coverage = payload.get("coverage") or {}
            return all(coverage.get(idx, 0.0) >= min_coverage for idx in referenced)

        verification = self.pipeline.verify_citations(cleaned_answer, list(contexts))
        enforcement["verification"] = verification
        if _is_compliant(verification):
            return cleaned_answer, enforcement

        attempts = max(0, self.config.dataset_generation.citation_rewrite_attempts)
        candidate = cleaned_answer
        for attempt in range(1, attempts + 1):
            rewrite_prompt = self._build_citation_rewrite_prompt(
                question=question,
                contexts=contexts,
                answer=candidate,
            )
            try:
                response = self.generator.generate(rewrite_prompt)
            except Exception as exc:  # pragma: no cover - network/LLM failures
                logger.warning(
                    "Citation rewrite failed for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    exc,
                )
                enforcement.setdefault("errors", []).append(
                    {
                        "attempt": attempt,
                        "error": "rewrite_failed",
                        "exception": repr(exc),
                    }
                )
                continue
            candidate = (response.get("response", "") or "").strip()
            enforcement["attempts"] = attempt
            if not candidate:
                continue
            verification = self.pipeline.verify_citations(candidate, list(contexts))
            if _is_compliant(verification):
                enforcement.update(
                    {
                        "status": "rewrite_success",
                        "verification": verification,
                    }
                )
                return candidate, enforcement

        enforcement.update({"status": "failed"})
        logger.warning("Dropping task for %s due to missing citations", source_path)
        return None

    @staticmethod
    def _build_citation_rewrite_prompt(
        *,
        question: str,
        contexts: Sequence[str],
        answer: str,
    ) -> str:
        """Build prompt for LLM to rewrite answer with proper citations."""
        context_block = "\n".join(contexts)
        return (
            "You are revising an answer for a retrieval-augmented research assistant. "
            "Ensure the answer includes inline citations using square brackets with numbers, such as [1], "
            "that refer to the numbered contexts. Cite only relevant contexts, do not fabricate, and draw on multiple different snippets when they contain supporting evidence. "
            "Reuse key terminology from the contexts so the overlap with cited evidence is explicit. "
            "Rewrite the answer if needed to include distinct citations and keep it concise.\n\n"
            f"Question: {question}\n\n"
            "Numbered contexts:\n"
            f"{context_block}\n\n"
            "Draft answer:\n"
            f"{answer}\n\n"
            "Rewritten answer with citations:"
        )


__all__ = ["CitationEnforcer"]
