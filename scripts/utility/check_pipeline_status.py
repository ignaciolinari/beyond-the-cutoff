#!/usr/bin/env python3
"""Report pipeline artifact coverage for each data run."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
EXTERNAL_ROOT = ROOT / "data" / "external"
EVAL_DATASETS = ROOT / "evaluation" / "datasets"

ARTIFACTS = {
    "raw_manifest": "manifest.json",
    "processed_manifest": "manifest.json",
    "index_meta": "index/index_meta.json",
    "index_mapping": "index/mapping.tsv",
}

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def append_note(entry: dict[str, Any], message: str | None) -> None:
    if not message:
        return
    existing = entry.get("note")
    if existing:
        entry["note"] = f"{existing}; {message}"
    else:
        entry["note"] = message


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return -1


def extract_manifest_counts(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    info: dict[str, Any] = {}

    field_map = {
        "total_papers": "papers_total",
        "requested_total": "papers_requested",
        "total_documents": "documents_total",
        "total_chunks": "chunks_total",
    }
    for src, dst in field_map.items():
        value = payload.get(src)
        if isinstance(value, int):
            info[dst] = value

    documents = payload.get("documents")
    if isinstance(documents, list):
        info["documents_listed"] = len(documents)

    run_dir = path.parent
    papers_dir = run_dir / "papers"
    if papers_dir.exists():
        pdf_count = sum(1 for _ in papers_dir.glob("*.pdf"))
        txt_count = sum(1 for _ in papers_dir.glob("*.txt"))
        jsonl_count = sum(1 for _ in papers_dir.glob("*.jsonl"))
        info["pdf_files"] = pdf_count
        info["text_files"] = txt_count
        if jsonl_count:
            info["jsonl_files"] = jsonl_count

    corpus_file = run_dir / "corpus.jsonl"
    if corpus_file.exists():
        info["corpus_lines"] = count_lines(corpus_file)

    return info


def _candidate_eval_bases(run: str) -> list[str]:
    sanitized = run.replace("/", "_")
    pieces = [part for part in sanitized.split("_") if part]

    base_variants: list[str] = []

    def _append(value: str | None) -> None:
        if not value:
            return
        if value not in base_variants:
            base_variants.append(value)

    _append(sanitized)
    _append(run.split("_run")[0])
    _append(sanitized.replace("_run", "_"))

    # Add progressively shorter prefixes (e.g., cog_psych_2025 -> cog_psych).
    for length in range(len(pieces), 0, -1):
        _append("_".join(pieces[:length]))

    return base_variants


def candidate_eval_names(run: str, suffix: str) -> list[str]:
    slug_variants = _candidate_eval_bases(run)
    unique: list[str] = []
    for variant in slug_variants:
        candidate = f"{variant}{suffix}"
        if candidate not in unique:
            unique.append(candidate)
    return unique


def iter_runs() -> list[str]:
    """Collect candidate run identifiers based on raw data folders."""
    if not RAW_ROOT.exists():
        return []
    runs: list[str] = []
    for child in sorted(RAW_ROOT.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            runs.append(child.name)
    return runs


def fmt_time(timestamp: str | None) -> str:
    if not timestamp:
        return "?"
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return timestamp


def load_json_timestamp(path: Path, field: str) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get(field)
    if isinstance(value, str):
        return value
    return None


def find_eval_file(run: str, suffix: str) -> Path | None:
    # Try a few slug variants since naming differs per run.
    slug = run
    for candidate in _candidate_eval_bases(slug):
        candidate_path = EVAL_DATASETS / f"{candidate}{suffix}"
        if candidate_path.exists():
            return candidate_path
    # Fallback: look for any file that starts with slug and ends with suffix.
    prefixes = _candidate_eval_bases(slug)
    for candidate_path in sorted(EVAL_DATASETS.glob(f"*{suffix}")):
        if any(candidate_path.name.startswith(prefix) for prefix in prefixes):
            return candidate_path
    return None


def describe_file(path: Path | None, *, label: str | None = None) -> dict[str, Any]:
    info: dict[str, Any]
    if not path:
        info = {"exists": False}
        append_note(info, label or "path unresolved")
        return info
    if not path.exists():
        info = {"exists": False}
        append_note(info, label)
        return info

    info = {"exists": True, "size": path.stat().st_size}
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".tsv"}:
        info["lines"] = count_lines(path)
    if suffix == ".json" and "manifest" in path.name:
        info.update(extract_manifest_counts(path))
    return info


def report_run(run: str) -> dict[str, dict[str, str | int | bool]]:
    data: dict[str, dict[str, str | int | bool]] = {"run": {"id": run}}

    raw_manifest = RAW_ROOT / run / ARTIFACTS["raw_manifest"]
    processed_manifest = PROCESSED_ROOT / run / ARTIFACTS["processed_manifest"]
    index_meta = EXTERNAL_ROOT / run / ARTIFACTS["index_meta"]
    index_mapping = EXTERNAL_ROOT / run / ARTIFACTS["index_mapping"]

    data["raw_manifest"] = describe_file(raw_manifest)
    data["processed_manifest"] = describe_file(processed_manifest)
    data["index_meta"] = describe_file(index_meta)
    data["index_mapping"] = describe_file(index_mapping)

    data["raw_manifest"]["timestamp"] = fmt_time(load_json_timestamp(raw_manifest, "fetched_at"))
    data["processed_manifest"]["timestamp"] = fmt_time(
        load_json_timestamp(processed_manifest, "generated_at")
    )
    if index_meta.exists():
        data["index_meta"]["timestamp"] = fmt_time(load_json_timestamp(index_meta, "built_at"))

    task_candidates = candidate_eval_names(run, "_offline_tasks.jsonl")
    dataset_candidates = candidate_eval_names(run, "_offline_dataset.jsonl")

    tasks_file = find_eval_file(run, "_offline_tasks.jsonl")
    dataset_file = find_eval_file(run, "_offline_dataset.jsonl")

    data["offline_tasks"] = describe_file(
        tasks_file,
        label=f"expected in evaluation/datasets: {', '.join(task_candidates)}",
    )
    data["offline_dataset"] = describe_file(
        dataset_file,
        label=f"expected in evaluation/datasets: {', '.join(dataset_candidates)}",
    )

    # Aggregate run-level metrics for quick progress snapshots.
    run_stats: dict[str, Any] = {}
    raw_info = data["raw_manifest"]
    processed_info = data["processed_manifest"]

    def first_available(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return None

    run_stats_candidates = {
        "papers_expected": first_available(
            raw_info, ("papers_total", "papers_requested", "documents_total")
        ),
        "papers_available": first_available(raw_info, ("pdf_files", "text_files")),
        "documents_expected": first_available(
            processed_info, ("documents_total", "documents_listed")
        ),
        "documents_ready": first_available(processed_info, ("text_files", "pdf_files")),
        "documents_in_corpus": processed_info.get("corpus_lines"),
    }

    for key, value in run_stats_candidates.items():
        if value is not None:
            run_stats[key] = value

    if (
        run_stats.get("documents_ready") is not None
        and run_stats.get("documents_expected") is not None
    ):
        run_stats["documents_progress"] = (
            f"{run_stats['documents_ready']}/{run_stats['documents_expected']}"
        )
    elif (
        run_stats.get("documents_ready") is not None
        and run_stats.get("papers_available") is not None
    ):
        run_stats["documents_progress"] = (
            f"{run_stats['documents_ready']}/{run_stats['papers_available']}"
        )

    data["run"].update(run_stats)

    # Provide progress hints when artifacts are missing but we can infer status.
    documents_progress = data["run"].get("documents_progress")
    if data["offline_tasks"].get("exists") and not data["offline_dataset"].get("exists"):
        append_note(
            data["offline_dataset"],
            "dataset generation pending (tasks exist)"
            + (f"; document progress {documents_progress}" if documents_progress else ""),
        )
    if data["offline_dataset"].get("exists") and not data["offline_tasks"].get("exists"):
        append_note(
            data["offline_tasks"],
            "tasks index missing (resume with --raw-tasks)",
        )
    if not data["offline_tasks"].get("exists") and not data["offline_dataset"].get("exists"):
        append_note(
            data["offline_tasks"],
            f"documents ready {documents_progress}" if documents_progress else None,
        )
        append_note(
            data["offline_dataset"],
            f"documents ready {documents_progress}" if documents_progress else None,
        )

    return data


def main() -> int:
    runs = iter_runs()
    if not runs:
        print("No runs detected under data/raw", file=sys.stderr)
        return 1

    for run in runs:
        payload = report_run(run)
        print(f"\n=== {run} ===")
        run_summary = {k: v for k, v in payload.get("run", {}).items() if k != "id"}
        if run_summary:
            print(f"run summary     -> {run_summary}")
        for key, value in payload.items():
            if key == "run":
                continue
            print(f"{key:16s} -> {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
