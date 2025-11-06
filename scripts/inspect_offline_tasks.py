#!/usr/bin/env python3
"""Interactively inspect offline dataset artifacts grouped by source document."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RAW = Path("evaluation/datasets/cog_psych_offline_tasks.jsonl")
DEFAULT_DATASET = Path("evaluation/datasets/cog_psych_offline_dataset.jsonl")


@dataclass
class RawRecord:
    document: str
    status: str
    error: str | None
    attempts: list[dict[str, Any]]
    payload: dict[str, Any]


@dataclass
class TaskRecord:
    task_id: str
    task_type: str
    instruction: str
    response: str
    citations: list[dict[str, Any]]
    contexts: list[str]
    metadata: dict[str, Any]


@dataclass
class DocumentBundle:
    document: str
    raw_records: list[RawRecord]
    tasks: list[TaskRecord]

    @property
    def status_summary(self) -> Counter[str]:
        return Counter(record.status for record in self.raw_records)

    @property
    def error_summary(self) -> Counter[str]:
        return Counter(record.error for record in self.raw_records if record.error)


def load_raw_records(path: Path) -> dict[str, list[RawRecord]]:
    grouped: dict[str, list[RawRecord]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            document = payload.get("document")
            if not isinstance(document, str):
                continue
            grouped[document].append(
                RawRecord(
                    document=document,
                    status=str(payload.get("status", "unknown")),
                    error=payload.get("error"),
                    attempts=list(payload.get("attempts", [])),
                    payload=payload,
                )
            )
    return grouped


def load_tasks(path: Path) -> dict[str, list[TaskRecord]]:
    grouped: dict[str, list[TaskRecord]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metadata = payload.get("metadata") or {}
            document = metadata.get("source_path")
            if not isinstance(document, str):
                continue
            rag = payload.get("rag") or {}
            grouped[document].append(
                TaskRecord(
                    task_id=str(payload.get("task_id")),
                    task_type=str(payload.get("task_type")),
                    instruction=str(payload.get("instruction", "")),
                    response=str(payload.get("expected_response", "")),
                    citations=list(payload.get("rag", {}).get("citations", [])),
                    contexts=list(rag.get("contexts", [])),
                    metadata=metadata,
                )
            )
    return grouped


def build_bundles(
    raw_map: dict[str, list[RawRecord]], task_map: dict[str, list[TaskRecord]]
) -> list[DocumentBundle]:
    documents = sorted({*raw_map.keys(), *task_map.keys()})
    return [
        DocumentBundle(
            document=doc,
            raw_records=raw_map.get(doc, []),
            tasks=task_map.get(doc, []),
        )
        for doc in documents
    ]


def format_status(bundle: DocumentBundle) -> str:
    status_counts = bundle.status_summary
    parts = [f"{status}:{count}" for status, count in sorted(status_counts.items())]
    if not parts:
        parts.append("status:0")
    task_types = Counter(task.task_type for task in bundle.tasks)
    task_summary = ", ".join(
        f"{task_type}:{count}" for task_type, count in sorted(task_types.items())
    )
    return f"{' '.join(parts)} | tasks {len(bundle.tasks)} ({task_summary or 'none'})"


def truncate(text: str, width: int = 80) -> str:
    plain = text.strip().replace("\n", " ")
    if len(plain) <= width:
        return plain
    return f"{plain[: width - 3].rstrip()}..."


def render_task(task: TaskRecord, index: int) -> str:
    citation_flag = "[cite]" if "[" in task.response and "]" in task.response else "    "
    return f"[{index}] {task.task_type:<9} {citation_flag} {truncate(task.instruction, 70)}"


def render_attempts(attempts: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for attempt in attempts:
        line = f"attempt {attempt.get('attempt')} -> {attempt.get('error', 'unknown')}"
        if attempt.get("exception"):
            line += f" ({truncate(str(attempt['exception']), 60)})"
        lines.append(line)
    return "\n".join(lines) if lines else "  (none)"


def show_document(bundle: DocumentBundle) -> None:
    print(f"\n=== Document: {bundle.document}")
    print(format_status(bundle))
    if bundle.error_summary:
        error_parts = ", ".join(f"{err}:{count}" for err, count in bundle.error_summary.items())
        print(f"errors       -> {error_parts}")
    for record in bundle.raw_records:
        print(f"raw status   -> {record.status} | error={record.error or '-'}")
        attempt_text = render_attempts(record.attempts)
        if attempt_text.strip():
            print(textwrap.indent(attempt_text, prefix="    "))
    if not bundle.tasks:
        print("No generated tasks for this document.")
        return
    print("-- Tasks --")
    for index, task in enumerate(bundle.tasks):
        print(render_task(task, index))


def show_task_detail(task: TaskRecord) -> None:
    print(f"\nTask {task.task_id} [{task.task_type}]")
    print("Instruction:")
    print(textwrap.indent(textwrap.fill(task.instruction, width=100), prefix="  "))
    print("Response:")
    print(textwrap.indent(textwrap.fill(task.response, width=100), prefix="  "))
    if task.citations:
        print("Citations:")
        for citation in task.citations:
            desc = truncate(citation.get("rendered_context", citation.get("excerpt", "")), 120)
            print(f"  - [id {citation.get('id')}] {desc}")
    else:
        print("Citations: none")
    if task.contexts:
        print("Contexts:")
        for idx, context in enumerate(task.contexts, start=1):
            print(f"  [{idx}] {truncate(context, 120)}")
    else:
        print("Contexts: none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect offline dataset artifacts")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Path to raw tasks JSONL")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to curated offline dataset JSONL",
    )
    return parser.parse_args()


def repl(bundles: list[DocumentBundle]) -> None:
    if not bundles:
        print("No documents found.")
        return
    current: DocumentBundle | None = None
    help_text = (
        "Commands: list (l), view <index>, task <index>, search <text>, errors, help (h), quit (q)"
    )
    print(help_text)
    while True:
        try:
            raw_input = input("inspect> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw_input:
            continue
        cmd, *rest = raw_input.split(maxsplit=1)
        arg = rest[0] if rest else ""
        if cmd in {"quit", "q", "exit"}:
            break
        if cmd in {"help", "h"}:
            print(help_text)
            continue
        if cmd in {"list", "l"}:
            for index, bundle in enumerate(bundles):
                print(f"[{index}] {truncate(bundle.document, 80)} -> {format_status(bundle)}")
            continue
        if cmd in {"view", "v"}:
            if not arg.isdigit():
                print("Provide a numeric index (see list).")
                continue
            index = int(arg)
            if not 0 <= index < len(bundles):
                print("Index out of range.")
                continue
            current = bundles[index]
            show_document(current)
            continue
        if cmd in {"task", "t"}:
            if current is None:
                print("Select a document with 'view <index>' first.")
                continue
            if not arg.isdigit():
                print("Provide a numeric task index.")
                continue
            index = int(arg)
            if not 0 <= index < len(current.tasks):
                print("Task index out of range.")
                continue
            show_task_detail(current.tasks[index])
            continue
        if cmd in {"search", "find"}:
            if not arg:
                print("Provide text to search for.")
                continue
            needle = arg.lower()
            matches = [
                (idx, bundle)
                for idx, bundle in enumerate(bundles)
                if needle in bundle.document.lower()
                or any(needle in task.instruction.lower() for task in bundle.tasks)
            ]
            if not matches:
                print("No matches.")
                continue
            for index, bundle in matches:
                print(f"[{index}] {truncate(bundle.document, 80)} -> tasks {len(bundle.tasks)}")
            continue
        if cmd == "errors":
            for index, bundle in enumerate(bundles):
                if bundle.error_summary:
                    err_str = ", ".join(
                        f"{err}:{count}" for err, count in bundle.error_summary.items()
                    )
                    print(f"[{index}] {truncate(bundle.document, 80)} -> {err_str}")
            continue
        print("Unknown command. Type 'help' for options.")


def main() -> None:
    args = parse_args()
    raw_map = load_raw_records(args.raw)
    task_map = load_tasks(args.dataset)
    bundles = build_bundles(raw_map, task_map)
    repl(bundles)


if __name__ == "__main__":
    main()
