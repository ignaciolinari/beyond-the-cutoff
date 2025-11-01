"""Task generation utilities for offline dataset preparation."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import LLMClient

logger = logging.getLogger(__name__)

_TASK_TYPE_ALIASES = {
    "qa": "qa",
    "questionanswer": "qa",
    "questionanswering": "qa",
    "summary": "summaries",
    "summaries": "summaries",
    "summarization": "summaries",
    "citation": "citations",
    "citations": "citations",
    "citationcheck": "citations",
}


@dataclass
class TaskRecord:
    """Normalized representation of a task to be answered by the assistant."""

    task_id: str
    task_type: str
    instruction: str
    doc_path: str | None = None
    requires_citations: bool = True
    answer_guidance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "doc_path": self.doc_path,
            "requires_citations": self.requires_citations,
            "answer_guidance": self.answer_guidance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskRecord:
        task_id = str(data.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("task_id is required to deserialize TaskRecord")
        task_type = str(data.get("task_type") or "").strip().lower()
        instruction = str(data.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"TaskRecord {task_id!r} is missing an instruction")
        requires_citations = bool(data.get("requires_citations", True))
        answer_guidance_raw = data.get("answer_guidance")
        answer_guidance = (
            str(answer_guidance_raw).strip() if isinstance(answer_guidance_raw, str) else None
        )
        metadata = data.get("metadata")
        if isinstance(metadata, Mapping):
            meta_dict = {str(k): v for k, v in metadata.items()}
        else:
            meta_dict = {}
        return cls(
            task_id=task_id,
            task_type=task_type,
            instruction=instruction,
            doc_path=str(data.get("doc_path")) if data.get("doc_path") else None,
            requires_citations=requires_citations,
            answer_guidance=answer_guidance,
            metadata=meta_dict,
        )


class TaskGenerator:
    """Generate diverse evaluation or training tasks from document excerpts."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tasks_per_doc: int = 6,
        allowed_task_types: Iterable[str] | None = None,
        document_char_limit: int = 4000,
        language_hint: str | None = None,
    ) -> None:
        self._client = client
        self.max_tasks_per_doc = max(1, max_tasks_per_doc)
        allowed = allowed_task_types or ("qa", "summaries", "citations")
        self.allowed_task_types = {self._canonical_task_type(t.strip().lower()) for t in allowed}
        self.document_char_limit = max(512, document_char_limit)
        self.language_hint = language_hint

    def generate(self, doc_text: str, *, doc_path: Path | str | None = None) -> list[TaskRecord]:
        """Return a list of :class:`TaskRecord` objects for *doc_text*."""

        if not doc_text.strip():
            return []

        excerpt = self._select_excerpt(doc_text)
        prompt = self._render_prompt(excerpt, doc_path)
        response = self._client.generate(prompt)
        raw_output = str(response.get("response", "")).strip()

        tasks_payload = self._parse_response(raw_output)
        if not isinstance(tasks_payload, Sequence):
            logger.warning("Task generator returned a non-sequence payload: %s", raw_output)
            return []

        slug = self._slugify(doc_path)
        collected: list[TaskRecord] = []
        counter = 1
        for item in tasks_payload:
            if not isinstance(item, Mapping):
                continue
            if counter > self.max_tasks_per_doc:
                break
            task = self._record_from_item(item, slug, counter, doc_path)
            if task.task_type not in self.allowed_task_types:
                continue
            collected.append(task)
            counter += 1

        return collected

    def _select_excerpt(self, doc_text: str) -> str:
        if len(doc_text) <= self.document_char_limit:
            return doc_text
        head = doc_text[: self.document_char_limit // 2]
        tail = doc_text[-self.document_char_limit // 2 :]
        return head + "\n\n...\n\n" + tail

    def _render_prompt(self, excerpt: str, doc_path: Path | str | None) -> str:
        doc_label = Path(doc_path).name if doc_path else "document"
        allowed_types = ", ".join(sorted(self.allowed_task_types))
        language_hint = (
            f"The document appears to be written in {self.language_hint}."
            if self.language_hint
            else ""
        )
        schema_example = {
            "task_type": "qa | summary | citation",
            "instruction": "Precise instruction grounded in the excerpt",
            "requires_citations": True,
            "answer_guidance": "Optional hint describing the preferred answer style.",
            "notes": "Optional metadata such as key sections or reasoning focus.",
        }
        return textwrap.dedent(
            f"""
            You design benchmarking tasks for a retrieval-augmented assistant that helps with
            scientific papers. Given the excerpt of "{doc_label}", write up to {self.max_tasks_per_doc}
            diverse tasks covering: {allowed_types}. {language_hint}

            Requirements:
            - Instructions must be grounded in details from the excerpt.
            - Use a JSON array as the *only* output.
            - Each item must include the keys shown below, with the specified types.
            - Use instructions in the same language as the excerpt.
            - If the excerpt lacks enough content, return an empty JSON array []
              without explanation.

            JSON schema (example values):
            {json.dumps(schema_example, indent=2)}

            Document excerpt:
            <<<
            {excerpt}
            >>>
            """
        ).strip()

    @staticmethod
    def _parse_response(payload: str) -> Any:
        if not payload:
            return []
        payload = payload.strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("[")
            end = payload.rfind("]")
            if start != -1 and end != -1 and end > start:
                snippet = payload[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse snippet as JSON: %s", snippet)
            logger.warning("Could not parse task generator output as JSON: %s", payload)
            return []

    def _record_from_item(
        self,
        item: Mapping[str, Any],
        slug: str,
        counter: int,
        doc_path: Path | str | None,
    ) -> TaskRecord:
        raw_type = str(item.get("task_type") or item.get("type") or "qa").strip().lower()
        normalized_type = re.sub(r"[^a-z]+", "", raw_type) or "qa"
        canonical_type = self._canonical_task_type(normalized_type)

        instruction_val = item.get("instruction") or item.get("prompt") or ""
        instruction = str(instruction_val).strip()
        if not instruction:
            instruction = "Summarize the key contributions of the paper."

        requires_citations = self._coerce_bool(item.get("requires_citations"), default=True)
        guidance_raw = item.get("answer_guidance") or item.get("answer_style")
        answer_guidance = (
            str(guidance_raw).strip()
            if isinstance(guidance_raw, str) and guidance_raw.strip()
            else None
        )

        metadata: dict[str, Any] = {}
        for key in ("notes", "difficulty", "focus", "expected_citations", "raw_source"):
            if key in item and item[key] is not None:
                metadata[key] = item[key]

        metadata.setdefault("raw_task", item)

        task_id = f"{slug}-{canonical_type}-{counter:02d}"

        return TaskRecord(
            task_id=task_id,
            task_type=canonical_type,
            instruction=instruction,
            doc_path=str(doc_path) if doc_path else None,
            requires_citations=requires_citations,
            answer_guidance=answer_guidance,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1"}:
                return True
            if normalized in {"false", "no", "n", "0"}:
                return False
        return default

    @staticmethod
    def _slugify(doc_path: Path | str | None) -> str:
        if not doc_path:
            return "doc"
        name = Path(doc_path).stem
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
        return slug or "doc"

    @staticmethod
    def _canonical_task_type(value: str) -> str:
        value = value.strip().lower()
        return _TASK_TYPE_ALIASES.get(value, value or "qa")
