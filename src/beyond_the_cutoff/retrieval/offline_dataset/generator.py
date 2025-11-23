"""Main generator coordinating offline dataset creation."""

from __future__ import annotations

import csv
import json
import logging
import random
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...config import ProjectConfig
from ...models import LLMClient, build_generation_client
from ..query import RAGPipeline
from .citation_enforcer import CitationEnforcer
from .document_metadata import DocumentMetadataManager
from .parser import ResponseParser
from .types import MappingRow, OfflineExample
from .validator import PayloadValidator

logger = logging.getLogger(__name__)


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

        # Initialize components
        self.parser = ResponseParser()
        self.validator = PayloadValidator(self.parser)
        self.metadata_manager = DocumentMetadataManager(config)
        self.citation_enforcer = CitationEnforcer(self._generator, config, self._pipeline)

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
        documents: Sequence[str] | None = None,
    ) -> dict[str, int]:
        """Generate offline dataset artifacts and return counters."""

        dataset_cfg = self.config.dataset_generation
        dataset_path = output_dataset_path or dataset_cfg.output_dataset_path
        tasks_path = raw_tasks_path or dataset_cfg.raw_tasks_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        document_whitelist: set[str] | None = None
        if documents:
            document_whitelist = set()
            for entry in documents:
                text = str(entry)
                if text:
                    document_whitelist.add(text)
                try:
                    document_whitelist.add(str(Path(text).resolve()))
                except Exception:
                    continue
        matched_documents: set[str] = set()
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
        if document_whitelist is None:
            target_items = list(grouped.items())
        else:
            target_items = [(path, grouped[path]) for path in grouped if path in document_whitelist]

        total_targets = len(target_items)
        max_docs = dataset_cfg.max_documents
        progress_total = total_targets if max_docs is None else min(total_targets, max_docs)
        counters = {
            "documents": 0,
            "qa": 0,
            "summaries": 0,
            "citations": 0,
            "contextual": 0,
            "examples": 0,
            "documents_filtered": 0,
        }
        if document_whitelist is not None:
            counters.update(
                {
                    "documents_requested": len(document_whitelist),
                    "documents_found": 0,
                    "documents_missing": 0,
                }
            )

        with (
            tasks_path.open(tasks_mode, encoding="utf-8") as raw_file,
            dataset_path.open(dataset_mode, encoding="utf-8") as dataset_file,
        ):
            if progress_total:
                logger.info(
                    "Planning to process %d document(s)%s",
                    progress_total,
                    " (limited by max_documents)"
                    if max_docs is not None and progress_total < total_targets
                    else "",
                )

            for doc_index, (source_path, rows) in enumerate(target_items):
                if max_docs is not None and counters["documents"] >= max_docs:
                    break

                progress_position = doc_index + 1
                if progress_total:
                    progress_label = f"{progress_position}/{progress_total}"
                else:
                    progress_label = str(progress_position)

                if document_whitelist is not None:
                    matched_documents.add(source_path)

                if resume and source_path in processed_documents:
                    logger.info("Document %s: %s (resume)", progress_label, source_path)
                    continue

                logger.info("Document %s: %s", progress_label, source_path)

                filter_info = self.metadata_manager.should_skip_document(source_path, rows)
                if filter_info is not None:
                    counters["documents_filtered"] += 1
                    reason_key = f"documents_filtered_{filter_info['kind']}"
                    counters[reason_key] = counters.get(reason_key, 0) + 1
                    logger.info(
                        "Document %s: %s -> skipped (%s, limit=%s, observed=%s)",
                        progress_label,
                        source_path,
                        filter_info["kind"],
                        filter_info.get("limit"),
                        filter_info.get("token_count")
                        if filter_info["kind"] == "token_limit"
                        else filter_info.get("page_count"),
                    )
                    raw_record = {
                        "document": source_path,
                        "status": "skipped",
                        "reason": filter_info["kind"],
                    }
                    details = {k: v for k, v in filter_info.items() if k != "kind"}
                    if details:
                        raw_record["details"] = details
                    raw_file.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
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

        if document_whitelist is not None:
            missing_documents = document_whitelist - matched_documents
            if missing_documents:
                logger.warning(
                    "Requested documents not present in mapping: %s",
                    sorted(missing_documents),
                )
            counters["documents_found"] = len(matched_documents)
            counters["documents_missing"] = len(missing_documents)

        return counters

    def _generate_for_document(
        self,
        *,
        doc_index: int,
        source_path: str,
        rows: Sequence[MappingRow],
        parse_retries: int,
    ) -> tuple[list[OfflineExample], dict[str, Any]]:
        """Generate tasks for a single document."""
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
        validation_notes: list[dict[str, Any]] = []
        failure_reason = "unparsable_response"

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
                failure_reason = "request_failed"
                continue

            raw_text = response.get("response", "")
            parsed = self.parser.parse_generator_response(raw_text)
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
                failure_reason = "unparsable_response"
                continue

            parsed, validation_issues = self.validator.validate_generator_payload(parsed)
            fatal_issues = [issue for issue in validation_issues if issue["fatal"]]
            if fatal_issues:
                logger.warning(
                    "Invalid generator payload for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    fatal_issues,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "response": raw_text,
                        "error": "invalid_payload",
                        "details": fatal_issues,
                    }
                )
                parsed = None
                failure_reason = "invalid_generator_payload"
                continue

            validation_notes = validation_issues
            if not self.validator.has_tasks(parsed):
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
                failure_reason = "empty_payload"
                continue

            minimum_deficits = self.validator.missing_minimum_counts(parsed, dataset_cfg)
            if minimum_deficits:
                logger.warning(
                    "Generator payload missing required counts for %s (attempt %d/%d): %s",
                    source_path,
                    attempt,
                    attempts,
                    minimum_deficits,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "response": raw_text,
                        "error": "insufficient_items",
                        "details": minimum_deficits,
                    }
                )
                parsed = None
                failure_reason = "insufficient_items"
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
                "error": failure_reason,
                "attempts": attempt_logs,
            }

        examples: list[OfflineExample] = []
        run_id = str(uuid.uuid4())

        qa_limit = dataset_cfg.questions_per_document
        summary_limit = dataset_cfg.summary_prompts_per_document
        citation_limit = dataset_cfg.citation_prompts_per_document
        context_limit = dataset_cfg.contextual_prompts_per_document

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

        for idx, item in enumerate(self._take(parsed.get("contextualizations", []), context_limit)):
            if not isinstance(item, Mapping):
                continue
            instruction = item.get("instruction") or item.get("prompt") or item.get("question")
            response_text = item.get("response") or item.get("answer")
            if not instruction or not response_text:
                continue
            example = self._build_example(
                task_type="contextual",
                instruction=instruction,
                expected_response=response_text,
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

        examples, output_issues = self.validator.validate_output_examples(
            examples, self._pipeline, dataset_cfg.min_citation_coverage
        )
        if output_issues:
            validation_notes.extend(output_issues)

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
        if validation_notes:
            raw_payload["validation_warnings"] = validation_notes

        if not examples:
            raw_payload["error"] = "output_validation_failed"
            return [], raw_payload

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
        """Build a single offline example with RAG context."""
        instruction_clean = self.parser.coerce_text(instruction).strip()
        expected_clean = self.parser.coerce_text(expected_response).strip()
        if not instruction_clean or not expected_clean:
            return None

        extra_instructions = None
        top_k_override: int | None = None
        if require_citations:
            extra_instructions = (
                "Use the numbered excerpts as explicit evidence. When more than one excerpt is relevant, cite each distinct snippet at least once using its [#] marker. "
                "Write complete sentences grounded in the excerpt content, and avoid recycling the same citation for unrelated statements or fabricating references."
            )
            configured_top_k = getattr(self.config.retrieval, "top_k", 4)
            top_k_override = max(2, int(configured_top_k)) if configured_top_k else 2

        prepared = self.pipeline.prepare_prompt(
            instruction_clean,
            require_citations=require_citations,
            extra_instructions=extra_instructions,
            top_k_override=top_k_override,
        )

        enforcement_meta: dict[str, Any] | None = None
        if require_citations:
            enforced = self.citation_enforcer.ensure_citation_compliance(
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

        if prepared.get("retrieved"):
            metadata["context_map"] = self._build_context_map(prepared["retrieved"])
        raw_contexts = prepared.get("raw_contexts")
        if isinstance(raw_contexts, list) and raw_contexts:
            metadata["raw_contexts"] = list(raw_contexts)
        extra_block = prepared.get("extra_instructions")
        if isinstance(extra_block, str) and extra_block.strip():
            metadata["retrieval_extra_instructions"] = extra_block.strip()

        profile = self.metadata_manager.get_document_metadata(source_path)
        if profile:
            metadata["paper_profile"] = profile

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

    def _build_generator_prompt(
        self,
        *,
        source_path: str,
        rows: Sequence[MappingRow],
        cfg: Any,
    ) -> str:
        """Build prompt for generator LLM."""

        def _range_phrase(noun: str, minimum: int, maximum: int) -> str:
            if maximum <= 0:
                return f"0 {noun}"
            if minimum <= 0:
                return f"up to {maximum} {noun}"
            if minimum == maximum:
                return f"{maximum} {noun}"
            return f"{minimum} to {maximum} {noun}"

        preamble = (
            "You are assisting in building a dataset for a retrieval-augmented research assistant. "
            "Given the numbered excerpts from a scientific paper, create diverse tasks."
        )
        qa_phrase = _range_phrase(
            "question-answer pairs", cfg.min_questions_per_document, cfg.questions_per_document
        )
        summary_phrase = _range_phrase(
            "summary instructions",
            cfg.min_summary_prompts_per_document,
            cfg.summary_prompts_per_document,
        )
        context_phrase = _range_phrase(
            "contextualization prompts that capture broader themes or connections",
            cfg.min_contextual_prompts_per_document,
            cfg.contextual_prompts_per_document,
        )
        citation_phrase = _range_phrase(
            "citation-check tasks",
            cfg.min_citation_prompts_per_document,
            cfg.citation_prompts_per_document,
        )
        instructions = (
            f"Produce {qa_phrase}, {summary_phrase}, {context_phrase}, and {citation_phrase}. "
            "Vary difficulty and focus so the set covers methods, results, limitations, and implications. "
            "All answers must be grounded in the text. Citation and contextualization responses must include inline [#] markers that reference the numbered excerpts, using multiple distinct markers when more than one excerpt is relevant. "
            "Return ONLY valid JSON with keys 'qa', 'summaries', 'contextualizations', and 'citations'."
        )
        schema = (
            "- Each item in 'qa' must include 'question' and 'answer'. Answers should cite supporting excerpts with inline [#] markers.\n"
            "- Each item in 'summaries' must include 'instruction' and 'response', covering contributions, methods, and notable limitations.\n"
            "- Each item in 'contextualizations' must include 'instruction' and 'response'. Use these to relate the paper to broader themes, contrasting approaches, or key author/institution highlights. Responses must include inline citations for every supported claim.\n"
            "- Each item in 'citations' must include non-empty 'instruction' and 'answer'. Citation answers must cite relevant excerpts using inline markers such as [1], [2] and should draw on multiple distinct snippets when evidence exists.\n"
            "- Avoid null values. If a field is unknown, omit the item instead of returning null.\n"
            "- Use concise academic tone. Avoid speculative statements."
        )

        few_shot_examples = (
            "Example output (use it as a style guide, but do not reuse the wording):\n"
            "{\n"
            '  "qa": [\n'
            "    {\n"
            '      "question": "What retrieval backbone do the authors deploy?",\n'
            '      "answer": "They fine-tune a bi-encoder and then rerank with a cross-encoder calibrated on the validation split [1][2]."\n'
            "    }\n"
            "  ],\n"
            '  "summaries": [\n'
            "    {\n"
            '      "instruction": "Write a concise summary covering objectives, methods, and findings.",\n'
            '      "response": "The paper proposes a lightweight retrieval pipeline, couples it with local generation, and reports significant accuracy gains on scientific benchmarks [1][3]."\n'
            "    }\n"
            "  ],\n"
            '  "contextualizations": [\n'
            "    {\n"
            '      "instruction": "Relate the authors\' approach to prior institution-scale retrieval systems.",\n'
            '      "response": "The authors contrast their decentralized deployment with earlier centralized retrieval services, highlighting alignment with institutional compliance requirements [2][3]."\n'
            "    }\n"
            "  ],\n"
            '  "citations": [\n'
            "    {\n"
            '      "instruction": "Identify the paragraph describing evaluation metrics.",\n'
            '      "answer": "The evaluation metrics and reporting strategy appear in the section titled \'Results and Analysis\' [3]."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "In your own output, ensure every [#] marker refers to an existing excerpt number from this document, and only add markers when the claim is explicitly supported. Never include citation markers inside questions or instructions."
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
            f"{few_shot_examples}\n\n"
            f"Document: {source_path}\n"
            f"Context Excerpts:\n{context_block}\n\n"
            "Return the JSON now."
        )

    def _select_rows(
        self,
        rows: Sequence[MappingRow],
        max_chunks: int,
    ) -> list[MappingRow]:
        """Select chunk rows for generation, shuffling if needed."""
        if not rows:
            return []
        ordered = sorted(rows, key=lambda r: r.chunk_index)
        if len(ordered) <= max_chunks:
            return ordered
        indices = list(range(len(ordered)))
        self._rng.shuffle(indices)
        chosen = sorted(indices[:max_chunks])
        return [ordered[i] for i in chosen]

    def _serialize_chunks(self, rows: Sequence[MappingRow], max_chars: int) -> list[dict[str, Any]]:
        """Serialize chunk rows to dict format."""
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

    def _load_mapping(self, mapping_path: Path) -> dict[str, list[MappingRow]]:
        """Load and group mapping rows by source path."""
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
    def _load_processed_documents(tasks_path: Path) -> set[str]:
        """Load set of already processed document paths."""
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
                    if payload.get("error"):
                        continue
                    if status and status not in {"success", "skipped"}:
                        continue
                    processed.add(document)
        return processed

    @staticmethod
    def _build_context_map(retrieved_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Build context map from retrieved records."""
        context_map: list[dict[str, Any]] = []
        for idx, rec in enumerate(retrieved_records, start=1):
            entry = {
                "index": rec.get("ordinal", idx),
                "chunk_id": rec.get("id"),
                "source_path": rec.get("source_path"),
                "page": rec.get("page"),
                "section_title": rec.get("section_title"),
                "token_start": rec.get("token_start"),
                "token_end": rec.get("token_end"),
                "rendered_context": rec.get("rendered_context"),
                "score": rec.get("score"),
                "similarity_score": rec.get("similarity_score"),
                "reranker_score": rec.get("reranker_score"),
            }
            context_map.append(entry)
        return context_map

    @staticmethod
    def _take(items: Iterable[Any], limit: int) -> list[Any]:
        """Take up to limit items from iterable."""
        result: list[Any] = []
        if limit <= 0:
            return result
        for item in items:
            result.append(item)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _maybe_int(value: Any, *, default: int | None = None) -> int | None:
        """Safely convert value to int."""
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default


__all__ = ["OfflineDatasetGenerator"]
