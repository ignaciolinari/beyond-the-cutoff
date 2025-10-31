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
    ) -> dict[str, int]:
        """Generate offline dataset artifacts and return counters."""

        dataset_cfg = self.config.dataset_generation
        dataset_path = output_dataset_path or dataset_cfg.output_dataset_path
        tasks_path = raw_tasks_path or dataset_cfg.raw_tasks_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        grouped = self._load_mapping(self.mapping_path)
        max_docs = dataset_cfg.max_documents
        counters = {"documents": 0, "qa": 0, "summaries": 0, "citations": 0, "examples": 0}

        with (
            tasks_path.open("w", encoding="utf-8") as raw_file,
            dataset_path.open("w", encoding="utf-8") as dataset_file,
        ):
            for doc_index, (source_path, rows) in enumerate(grouped.items()):
                if max_docs is not None and counters["documents"] >= max_docs:
                    break

                examples, raw_payload = self._generate_for_document(
                    doc_index=doc_index,
                    source_path=source_path,
                    rows=rows,
                )

                if not examples:
                    continue

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
        response = self.generator.generate(generator_prompt)
        raw_text = response.get("response", "")
        parsed = self._parse_generator_response(raw_text)
        if parsed is None:
            logger.warning("Generator returned unparsable payload for %s", source_path)
            return [], {
                "document": source_path,
                "prompt": generator_prompt,
                "model": getattr(self.generator, "model", "unknown"),
                "selected_chunks": self._serialize_chunks(
                    selected_rows, dataset_cfg.max_chars_per_chunk
                ),
                "response": raw_text,
                "error": "unparsable_response",
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
        instruction_clean = instruction.strip()
        expected_clean = expected_response.strip()
        if not instruction_clean or not expected_clean:
            return None

        prepared = self.pipeline.prepare_prompt(
            instruction_clean,
            require_citations=require_citations,
        )

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
            "Answers must be grounded in the text. Return ONLY valid JSON with keys 'qa', 'summaries', 'citations'."
        )
        schema = (
            "- Each item in 'qa' must include 'question' and 'answer'.\n"
            "- Each item in 'summaries' must include 'instruction' and 'response'.\n"
            "- Each item in 'citations' must include 'instruction' and 'answer'.\n"
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
