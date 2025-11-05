#!/usr/bin/env python3
"""Build curated evaluation splits (QA, summaries) from the offline dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from beyond_the_cutoff.config import ProjectConfig, load_config

DEFAULT_MANIFEST_PATH = Path("evaluation/datasets/manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialise evaluation dataset splits")
    parser.add_argument("--config", default="configs/default.yaml", help="Project config path")
    parser.add_argument(
        "--offline-dataset",
        default=None,
        help="Offline dataset JSONL (defaults to config.evaluation.offline_dataset_path)",
    )
    parser.add_argument(
        "--qa-output",
        default=None,
        help="Output path for QA split (defaults to config.evaluation.qa_dataset_path)",
    )
    parser.add_argument(
        "--summaries-output",
        default=None,
        help="Output path for summaries split (defaults to config.evaluation.summary_dataset_path)",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Manifest path (defaults to evaluation/datasets/manifest.json)",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Version identifier recorded in the manifest (default: v1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of offline examples to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files and manifest",
    )
    return parser.parse_args()


def iter_offline_examples(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _stringify_list(items: Any) -> list[str]:
    if not items:
        return []
    if isinstance(items, list | tuple | set):
        return [str(value) for value in items if value is not None]
    return [str(items)]


def _merge_metadata(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate
    return {}


def _base_payload(example: dict[str, Any]) -> dict[str, Any]:
    rag_block = example.get("rag", {})
    contexts = rag_block.get("contexts") or example.get("contexts") or []
    sources = rag_block.get("sources") or example.get("sources") or []
    citations = rag_block.get("citations") or example.get("citations") or []
    retrieved = rag_block.get("retrieved") or example.get("retrieved") or []

    return {
        "task_id": str(example.get("task_id", "")),
        "task_type": str(example.get("task_type", "")),
        "rag_prompt": str(rag_block.get("prompt") or example.get("rag_prompt") or ""),
        "contexts": _stringify_list(contexts),
        "sources": _stringify_list(sources),
        "retrieved": retrieved if isinstance(retrieved, list) else [],
        "citations": citations if isinstance(citations, list) else [],
        "metadata": _merge_metadata(example.get("metadata")),
    }


def build_qa_record(example: dict[str, Any]) -> dict[str, Any] | None:
    base = _base_payload(example)
    question = example.get("instruction") or example.get("question")
    reference = example.get("expected_response") or example.get("answer")
    if not question or not reference:
        return None
    qa_payload = {
        "question": str(question).strip(),
        "reference_answer": str(reference).strip(),
        "answer_citations": example.get("citations")
        if isinstance(example.get("citations"), list)
        else [],
    }
    base.update(qa_payload)
    return base


def build_summary_record(example: dict[str, Any]) -> dict[str, Any] | None:
    base = _base_payload(example)
    instruction = example.get("instruction") or example.get("prompt") or example.get("question")
    reference = example.get("expected_response") or example.get("response") or example.get("answer")
    if not instruction or not reference:
        return None
    summary_payload = {
        "instruction": str(instruction).strip(),
        "reference_summary": str(reference).strip(),
    }
    base.update(summary_payload)
    return base


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    project_cfg: ProjectConfig = load_config(config_path)

    offline_path = (
        Path(args.offline_dataset)
        if args.offline_dataset
        else project_cfg.evaluation.offline_dataset_path
    )
    offline_path = offline_path.resolve()
    if not offline_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {offline_path}")

    qa_path = Path(args.qa_output) if args.qa_output else project_cfg.evaluation.qa_dataset_path
    qa_path = qa_path.resolve()
    summaries_path = (
        Path(args.summaries_output)
        if args.summaries_output
        else project_cfg.evaluation.summary_dataset_path
    ).resolve()
    manifest_path = (
        Path(args.manifest_output) if args.manifest_output else DEFAULT_MANIFEST_PATH
    ).resolve()

    _ensure_can_write(qa_path, force=args.force)
    _ensure_can_write(summaries_path, force=args.force)
    _ensure_can_write(manifest_path, force=args.force)

    qa_count = 0
    summary_count = 0
    unsupported_counts: dict[str, int] = {}

    with (
        qa_path.open("w", encoding="utf-8") as qa_handle,
        summaries_path.open("w", encoding="utf-8") as summary_handle,
    ):
        for example in iter_offline_examples(offline_path, limit=args.limit):
            task_type = str(example.get("task_type", "")).strip()
            if task_type == "qa":
                record = build_qa_record(example)
                if record:
                    qa_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    qa_count += 1
            elif task_type == "summaries":
                record = build_summary_record(example)
                if record:
                    summary_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary_count += 1
            else:
                key = task_type or "unknown"
                unsupported_counts[key] = unsupported_counts.get(key, 0) + 1

    manifest_payload = {
        "version": args.version,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_offline_dataset": str(offline_path),
        "filters": {
            key: value for key, value in {"limit": args.limit}.items() if value is not None
        },
        "splits": {
            "qa": {"path": str(qa_path), "count": qa_count},
            "summaries": {"path": str(summaries_path), "count": summary_count},
        },
        "excluded_task_types": unsupported_counts,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "qa_count": qa_count,
                "summaries_count": summary_count,
                "manifest": str(manifest_path),
                "unsupported_task_types": unsupported_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
