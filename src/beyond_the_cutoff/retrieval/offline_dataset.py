"""Offline dataset generation utilities for RAG and fine-tuning workflows."""

from __future__ import annotations

import json
import logging
import random
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from ..models import LLMClient, build_generation_client
from .query import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class MappingRow:
    """Representation of a single chunk entry from the FAISS mapping TSV."""

    chunk_id: int
    source_path: str
    page: int | None
    section_title: str | None
    chunk_index: int
    token_start: int | None
    token_end: int | None
    text: str

    def trimmed_text(self, limit: int) -> str:
        snippet = self.text.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[: max(0, limit - 3)].rstrip() + "..."


@dataclass
class OfflineExample:
    """Output record describing a prepared offline training/eval example."""

    task_id: str
    task_type: str
    instruction: str
    expected_response: str
    rag_prompt: str
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    citations: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "expected_response": self.expected_response,
            "rag": {
                "prompt": self.rag_prompt,
                "contexts": self.contexts,
                "sources": self.sources,
                "scores": self.scores,
                "citations": self.citations,
                "retrieved": self.retrieved,
            },
            "metadata": self.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)


class OfflineDatasetGenerator:
    """Generate offline prompts and gold responses for downstream fine-tuning/eval."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        index_path: Path,
        mapping_path: Path,
        generator_client: LLMClient | None = None,
        pipeline: RAGPipeline | None = None,
    ) -> None:
        self.config = config
        self.index_path = index_path
        self.mapping_path = mapping_path
        self._pipeline = pipeline or RAGPipeline(
            config, index_path=index_path, mapping_path=mapping_path
        )
        self._generator = generator_client or build_generation_client(
            config.dataset_generation.generator
        )
        self._rng = random.Random(config.dataset_generation.seed)

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    @property
    def generator(self) -> LLMClient:
        return self._generator

    def generate(
        self,
        *,
        output_dataset_path: Path | None = None,
        raw_tasks_path: Path | None = None,
        resume: bool = False,
        parse_retries: int | None = None,
    ) -> dict[str, int]:
        """Generate offline dataset artifacts and return counters."""

        dataset_cfg = self.config.dataset_generation
        dataset_path = output_dataset_path or dataset_cfg.output_dataset_path
        tasks_path = raw_tasks_path or dataset_cfg.raw_tasks_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        processed_documents: set[str] = set()
        dataset_mode = "w"
        tasks_mode = "w"

        if resume:
            processed_documents = self._load_processed_documents(tasks_path)
            dataset_mode = "a"
            tasks_mode = "a"

        effective_parse_retries = (
            dataset_cfg.parse_retries if parse_retries is None else max(0, parse_retries)
        )

        grouped = self._load_mapping(self.mapping_path)
        max_docs = dataset_cfg.max_documents
        counters = {"documents": 0, "qa": 0, "summaries": 0, "citations": 0, "examples": 0}

        with (
            tasks_path.open(tasks_mode, encoding="utf-8") as raw_file,
            dataset_path.open(dataset_mode, encoding="utf-8") as dataset_file,
        ):
            for doc_index, (source_path, rows) in enumerate(grouped.items()):
                if max_docs is not None and counters["documents"] >= max_docs:
                    break

                if resume and source_path in processed_documents:
                    logger.debug(
                        "Skipping %s; already present in raw tasks (resume mode)", source_path
                    )
                    continue

                examples, raw_payload = self._generate_for_document(
                    doc_index=doc_index,
                    source_path=source_path,
                    rows=rows,
                    parse_retries=effective_parse_retries,
                )

                if not examples:
                    if raw_payload:
                        raw_payload.setdefault("status", "error")
                        raw_file.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")
                    continue

                raw_payload.setdefault("status", "success")
                raw_file.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")

                for example in examples:
                    dataset_file.write(example.to_json() + "\n")
                    counters["examples"] += 1
                    counters[example.task_type] = counters.get(example.task_type, 0) + 1

                counters["documents"] += 1

        return counters

    def _generate_for_document(
        self,
        *,
        doc_index: int,
        source_path: str,
        rows: Sequence[MappingRow],
        parse_retries: int,
    ) -> tuple[list[OfflineExample], dict[str, Any]]:
        dataset_cfg = self.config.dataset_generation
        selected_rows = self._select_rows(rows, dataset_cfg.max_chunks_per_document)
        if not selected_rows:
            logger.debug("Skipping %s; no chunks selected", source_path)
            return [], {}

        generator_prompt = self._build_generator_prompt(
            source_path=source_path,
            rows=selected_rows,
            cfg=dataset_cfg,
        )
        attempts = max(0, parse_retries) + 1
        parsed: dict[str, Any] | None = None
        raw_text = ""
        attempt_logs: list[dict[str, Any]] = []

        for attempt in range(1, attempts + 1):
            try:
                response = self.generator.generate(generator_prompt)
            except Exception as exc:  # pragma: no cover - network/LLM failures
                logger.warning(
                    "Generator request failed for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    exc,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "error": "request_failed",
                        "exception": repr(exc),
                    }
                )
                raw_text = ""
                parsed = None
                continue

            raw_text = response.get("response", "")
            parsed = self._parse_generator_response(raw_text)
            if parsed is None:
                logger.warning(
                    "Unparsable generator payload for %s (attempt %d/%d)",
                    source_path,
                    attempt,
                    attempts,
                )
                attempt_logs.append(
                    {"attempt": attempt, "response": raw_text, "error": "unparsable"}
                )
                continue

            if not self._has_tasks(parsed):
                logger.warning(
                    "Generator returned empty payload for %s (attempt %d/%d)",
                    source_path,
                    attempt,
                    attempts,
                )
                attempt_logs.append(
                    {"attempt": attempt, "response": raw_text, "error": "empty_payload"}
                )
                parsed = None
                continue

            if attempt > 1:
                logger.info(
                    "Recovered generator payload for %s after %d attempt(s)",
                    source_path,
                    attempt,
                )
            break

        if parsed is None:
            return [], {
                "document": source_path,
                "prompt": generator_prompt,
                "model": getattr(self.generator, "model", "unknown"),
                "selected_chunks": self._serialize_chunks(
                    selected_rows, dataset_cfg.max_chars_per_chunk
                ),
                "response": raw_text,
                "error": "unparsable_response",
                "attempts": attempt_logs,
            }

        examples: list[OfflineExample] = []
        run_id = str(uuid.uuid4())

        qa_limit = dataset_cfg.questions_per_document
        summary_limit = dataset_cfg.summary_prompts_per_document
        citation_limit = dataset_cfg.citation_prompts_per_document

        for idx, item in enumerate(self._take(parsed.get("qa", []), qa_limit)):
            if not isinstance(item, Mapping):
                continue
            question = item.get("question") or item.get("instruction")
            answer = item.get("answer") or item.get("response")
            if not question or not answer:
                continue
            example = self._build_example(
                task_type="qa",
                instruction=question,
                expected_response=answer,
                require_citations=True,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        for idx, item in enumerate(self._take(parsed.get("summaries", []), summary_limit)):
            if not isinstance(item, Mapping):
                continue
            instruction = item.get("instruction") or item.get("prompt") or item.get("question")
            response_text = item.get("response") or item.get("answer")
            if not instruction or not response_text:
                continue
            example = self._build_example(
                task_type="summaries",
                instruction=instruction,
                expected_response=response_text,
                require_citations=False,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        for idx, item in enumerate(self._take(parsed.get("citations", []), citation_limit)):
            if not isinstance(item, Mapping):
                continue
            instruction = item.get("instruction") or item.get("question")
            answer = item.get("answer") or item.get("response")
            if not instruction or not answer:
                continue
            example = self._build_example(
                task_type="citations",
                instruction=instruction,
                expected_response=answer,
                require_citations=True,
                doc_index=doc_index,
                task_index=idx,
                run_id=run_id,
                source_path=source_path,
                selected_rows=selected_rows,
                extra_metadata=item,
            )
            if example:
                examples.append(example)

        raw_payload = {
            "document": source_path,
            "model": getattr(self.generator, "model", "unknown"),
            "prompt": generator_prompt,
            "response": raw_text,
            "parsed": parsed,
            "run_id": run_id,
            "selected_chunks": self._serialize_chunks(
                selected_rows, dataset_cfg.max_chars_per_chunk
            ),
        }

        return examples, raw_payload

    @staticmethod
    def _has_tasks(payload: Mapping[str, Any]) -> bool:
        qa_items = payload.get("qa", [])
        summary_items = payload.get("summaries", [])
        citation_items = payload.get("citations", [])
        return any(qa_items) or any(summary_items) or any(citation_items)

    @staticmethod
    def _load_processed_documents(tasks_path: Path) -> set[str]:
        processed: set[str] = set()
        if not tasks_path.exists():
            return processed
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                document = payload.get("document")
                if isinstance(document, str) and document:
                    status = payload.get("status")
                    if status and status != "success":
                        continue
                    if payload.get("error"):
                        continue
                    processed.add(document)
        return processed

    def _build_example(
        self,
        *,
        task_type: str,
        instruction: str,
        expected_response: str,
        require_citations: bool,
        doc_index: int,
        task_index: int,
        run_id: str,
        source_path: str,
        selected_rows: Sequence[MappingRow],
        extra_metadata: Mapping[str, Any] | None,
    ) -> OfflineExample | None:
        instruction_clean = self._coerce_text(instruction).strip()
        expected_clean = self._coerce_text(expected_response).strip()
        if not instruction_clean or not expected_clean:
            return None

        extra_instructions = None
        top_k_override: int | None = None
        if require_citations:
            extra_instructions = (
                "Use the numbered excerpts as explicit evidence. When more than one excerpt is relevant, cite each distinct snippet at least once using its [#] marker. "
                "Write complete sentences grounded in the excerpt content, and avoid recycling the same citation for unrelated statements or fabricating references."
            )
            top_k_override = 1

        prepared = self.pipeline.prepare_prompt(
            instruction_clean,
            require_citations=require_citations,
            extra_instructions=extra_instructions,
            top_k_override=top_k_override,
        )

        enforcement_meta: dict[str, Any] | None = None
        if require_citations:
            enforced = self._ensure_citation_compliance(
                question=instruction_clean,
                answer=expected_clean,
                contexts=prepared["contexts"],
                source_path=source_path,
            )
            if enforced is None:
                return None
            expected_clean, enforcement_meta = enforced

        metadata = {
            "source_path": source_path,
            "generator_model": getattr(self.generator, "model", "unknown"),
            "generator_run_id": run_id,
            "document_index": doc_index,
            "task_index": task_index,
            "require_citations": require_citations,
            "retrieved_chunk_ids": [rec.get("id") for rec in prepared["retrieved"]],
            "retrieved_section_titles": [rec.get("section_title") for rec in prepared["retrieved"]],
            "selected_chunk_ids": [row.chunk_id for row in selected_rows],
        }
        if extra_metadata:
            metadata["generator_metadata"] = dict(extra_metadata)
        if enforcement_meta:
            metadata["citation_enforcement"] = enforcement_meta

        return OfflineExample(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            instruction=instruction_clean,
            expected_response=expected_clean,
            rag_prompt=prepared["prompt"],
            contexts=prepared["contexts"],
            sources=prepared["sources"],
            scores=prepared["scores"],
            citations=[
                {
                    "id": rec.get("id"),
                    "source_path": rec.get("source_path"),
                    "page": rec.get("page"),
                    "section_title": rec.get("section_title"),
                    "token_start": rec.get("token_start"),
                    "token_end": rec.get("token_end"),
                    "score": rec.get("score"),
                    "excerpt": rec.get("excerpt"),
                    "rendered_context": rec.get("rendered_context"),
                }
                for rec in prepared["retrieved"]
            ],
            retrieved=prepared["retrieved"],
            metadata=metadata,
        )

    def _select_rows(
        self,
        rows: Sequence[MappingRow],
        max_chunks: int,
    ) -> list[MappingRow]:
        if not rows:
            return []
        ordered = sorted(rows, key=lambda r: r.chunk_index)
        if len(ordered) <= max_chunks:
            return ordered
        indices = list(range(len(ordered)))
        self._rng.shuffle(indices)
        chosen = sorted(indices[:max_chunks])
        return [ordered[i] for i in chosen]

    @staticmethod
    def _take(items: Iterable[Any], limit: int) -> list[Any]:
        result: list[Any] = []
        if limit <= 0:
            return result
        for item in items:
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def _serialize_chunks(self, rows: Sequence[MappingRow], max_chars: int) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": row.chunk_id,
                "chunk_index": row.chunk_index,
                "page": row.page,
                "section_title": row.section_title,
                "token_start": row.token_start,
                "token_end": row.token_end,
                "text": row.trimmed_text(max_chars),
            }
            for row in rows
        ]

    def _build_generator_prompt(
        self,
        *,
        source_path: str,
        rows: Sequence[MappingRow],
        cfg: Any,
    ) -> str:
        preamble = (
            "You are assisting in building a dataset for a retrieval-augmented research assistant. "
            "Given the numbered excerpts from a scientific paper, create diverse tasks."
        )
        instructions = (
            f"Produce up to {cfg.questions_per_document} question-answer pairs, "
            f"{cfg.summary_prompts_per_document} summary instructions, and "
            f"{cfg.citation_prompts_per_document} citation-check tasks. "
            "Answers must be grounded in the text. For citation tasks, ensure responses contain inline [#] markers that reference the numbered excerpts. Return ONLY valid JSON with keys 'qa', 'summaries', 'citations'."
        )
        schema = (
            "- Each item in 'qa' must include 'question' and 'answer'.\n"
            "- Each item in 'summaries' must include 'instruction' and 'response'.\n"
            "- Each item in 'citations' must include non-empty 'instruction' and 'answer'. Citation answers must cite relevant excerpts using inline markers such as [1], [2] and should draw on multiple distinct snippets when evidence exists.\n"
            "- Avoid null values. If a field is unknown, omit the item instead of returning null.\n"
            "- Use concise academic tone. Avoid speculative statements."
        )
        context_lines = []
        for idx, row in enumerate(rows, start=1):
            meta_bits: list[str] = []
            if row.section_title:
                meta_bits.append(f"Section: {row.section_title}")
            if row.page is not None:
                meta_bits.append(f"Page {row.page}")
            entry_header = f"[{idx}]"
            if meta_bits:
                entry_header = f"{entry_header} {' | '.join(meta_bits)}"
            context_lines.append(f"{entry_header}\n{row.trimmed_text(cfg.max_chars_per_chunk)}")
        context_block = "\n".join(context_lines)

        return (
            f"{preamble}\n{instructions}\n{schema}\n\n"
            f"Document: {source_path}\n"
            f"Context Excerpts:\n{context_block}\n\n"
            "Return the JSON now."
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
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

    def _ensure_citation_compliance(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        source_path: str,
    ) -> tuple[str, dict[str, Any]] | None:
        enforcement: dict[str, Any] = {"status": "pass", "attempts": 0}
        cleaned_answer = answer.strip()
        if not cleaned_answer:
            return None

        if not contexts:
            enforcement["verification"] = {"referenced": []}
            return cleaned_answer, enforcement

        verification = self.pipeline.verify_citations(cleaned_answer, list(contexts))
        enforcement["verification"] = verification
        if verification.get("referenced") and not verification.get("extra"):
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
            if verification.get("referenced") and not verification.get("extra"):
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
        context_block = "\n".join(contexts)
        return (
            "You are revising an answer for a retrieval-augmented research assistant. "
            "Ensure the answer includes inline citations using square brackets with numbers, such as [1], "
            "that refer to the numbered contexts. Cite only relevant contexts, do not fabricate, and draw on multiple different snippets when they contain supporting evidence. "
            "Rewrite the answer if needed to include distinct citations and keep it concise.\n\n"
            f"Question: {question}\n\n"
            "Numbered contexts:\n"
            f"{context_block}\n\n"
            "Draft answer:\n"
            f"{answer}\n\n"
            "Rewritten answer with citations:"
        )

    @staticmethod
    def _coerce_text(value: Any) -> str:
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

    def _parse_generator_response(self, text: str) -> dict[str, Any] | None:
        candidate = self._strip_fences(text)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            logger.debug("Failed to decode generator JSON: %s", candidate)
            return None
        if not isinstance(data, dict):
            return None
        # Normalise expected keys
        data.setdefault("qa", [])
        data.setdefault("summaries", [])
        data.setdefault("citations", [])
        return data

    def _load_mapping(self, mapping_path: Path) -> dict[str, list[MappingRow]]:
        import csv

        grouped: dict[str, list[MappingRow]] = {}
        with mapping_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                try:
                    chunk_id = int(row.get("id", "0"))
                except (TypeError, ValueError):
                    continue
                source_path = str(row.get("source_path", ""))
                if not source_path:
                    continue
                page = self._maybe_int(row.get("page"))
                chunk_index = self._maybe_int(row.get("chunk_index"), default=chunk_id) or chunk_id
                token_start = self._maybe_int(row.get("token_start"))
                token_end = self._maybe_int(row.get("token_end"))
                text = str(row.get("text", ""))
                section_title = row.get("section_title")
                mapping_row = MappingRow(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    page=page,
                    section_title=section_title if section_title else None,
                    chunk_index=chunk_index,
                    token_start=token_start,
                    token_end=token_end,
                    text=text,
                )
                grouped.setdefault(source_path, []).append(mapping_row)

        # Deterministic ordering for reproducibility
        for bucket in grouped.values():
            bucket.sort(key=lambda r: r.chunk_index)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    @staticmethod
    def _maybe_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default


__all__ = ["OfflineDatasetGenerator", "OfflineExample", "MappingRow"]
