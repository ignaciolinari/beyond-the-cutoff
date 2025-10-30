"""Assemble offline datasets that pair prompts, contexts, and answers."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..models import LLMClient
from ..retrieval.query import RAGPipeline
from .tasks import TaskRecord

logger = logging.getLogger(__name__)


@dataclass
class OfflineExample:
    """Container for a fully-materialized offline example."""

    task: TaskRecord
    prompt: str
    contexts: list[str]
    sources: list[str]
    retrieved: list[dict[str, Any]]
    reference_answer: str
    reference_model: str
    reference_citation_verification: dict[str, Any] | None
    rag_answer: str | None = None
    rag_model: str | None = None
    rag_citation_verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        task_payload = self.task.to_dict()
        return {
            **task_payload,
            "prompt": self.prompt,
            "contexts": self.contexts,
            "sources": self.sources,
            "retrieved": self.retrieved,
            "reference_answer": self.reference_answer,
            "reference_model": self.reference_model,
            "reference_citation_verification": self.reference_citation_verification,
            "rag_answer": self.rag_answer,
            "rag_model": self.rag_model,
            "rag_citation_verification": self.rag_citation_verification,
            "fine_tune_input": self.prompt,
            "fine_tune_output": self.reference_answer,
        }


class OfflineDatasetBuilder:
    """Materialize offline prompt/answer pairs from task definitions."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        reference_client: LLMClient,
        rag_client: LLMClient | None = None,
        reference_options: Mapping[str, Any] | None = None,
        rag_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._reference_client = reference_client
        self._rag_client = rag_client
        self._reference_options = dict(reference_options or {})
        self._rag_options = dict(rag_options or {})

    def build_example(
        self,
        task: TaskRecord,
        *,
        extra_instructions: str | None = None,
    ) -> OfflineExample | None:
        """Return an :class:`OfflineExample` for *task* or ``None`` when retrieval fails."""

        merged_instructions_parts = [
            part.strip()
            for part in (task.answer_guidance, extra_instructions)
            if isinstance(part, str) and part.strip()
        ]
        merged_instructions = (
            " ".join(merged_instructions_parts) if merged_instructions_parts else None
        )

        prepared = self._pipeline.prepare_prompt(
            task.instruction,
            require_citations=task.requires_citations,
            extra_instructions=merged_instructions,
        )
        if not prepared["contexts"]:
            logger.warning("No contexts retrieved for task %s", task.task_id)
            return None

        reference_answer = self._call_client(
            self._reference_client, prepared["prompt"], self._reference_options
        )
        if reference_answer is None:
            logger.warning("Reference model failed to answer task %s", task.task_id)
            return None

        rag_answer: str | None = None
        rag_verification: dict[str, Any] | None = None
        rag_model_name: str | None = None
        if self._rag_client is not None:
            rag_answer = self._call_client(self._rag_client, prepared["prompt"], self._rag_options)
            if rag_answer is not None:
                rag_verification = self._pipeline.verify_citations(rag_answer, prepared["contexts"])
                rag_model_name = getattr(self._rag_client, "model", "unknown")

        reference_verification = self._pipeline.verify_citations(
            reference_answer, prepared["contexts"]
        )

        retrieved = [copy.deepcopy(entry) for entry in prepared["retrieved"]]

        return OfflineExample(
            task=task,
            prompt=prepared["prompt"],
            contexts=list(prepared["contexts"]),
            sources=list(prepared["sources"]),
            retrieved=retrieved,
            reference_answer=reference_answer,
            reference_model=getattr(self._reference_client, "model", "unknown"),
            reference_citation_verification=reference_verification,
            rag_answer=rag_answer,
            rag_model=rag_model_name,
            rag_citation_verification=rag_verification,
        )

    @staticmethod
    def _call_client(
        client: LLMClient,
        prompt: str,
        options: Mapping[str, Any] | None = None,
    ) -> str | None:
        try:
            result = client.generate(prompt, options=options)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Generation failed for model %s: %s", getattr(client, "model", "?"), exc)
            return None
        response = result.get("response") if isinstance(result, Mapping) else None
        if response is None:
            logger.warning(
                "Model %s returned payload without 'response' key", getattr(client, "model", "?")
            )
            return None
        text = str(response).strip()
        return text or None
