#!/usr/bin/env python3
"""Summarise offline dataset artifacts (QA, summaries, citations).

Example usage:
    python scripts/analyze_offline_dataset.py \
        --dataset data/subset20/evaluation/offline_dataset.jsonl \
        --raw-tasks data/subset20/evaluation/offline_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse offline dataset JSONL files")
    parser.add_argument("--dataset", required=True, help="Path to offline_dataset.jsonl")
    parser.add_argument(
        "--raw-tasks",
        required=False,
        help="Optional path to offline_tasks.jsonl for generator diagnostics",
    )
    parser.add_argument(
        "--show-citation-details",
        action="store_true",
        help="Print detailed rows for citation tasks that failed verification",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def describe_numeric(values: list[int]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    report = [f"min={min(vals)}", f"p50={int(statistics.median(vals))}"]
    if len(vals) >= 3:
        try:
            p90 = statistics.quantiles(vals, n=10)[8]
            report.append(f"p90={int(p90)}")
        except (statistics.StatisticsError, IndexError):
            pass
    report.append(f"max={max(vals)}")
    return " | ".join(report)


def format_distribution(counter: Counter[str]) -> str:
    total = sum(counter.values())
    parts = [f"{key}: {value} ({value / total:.1%})" for key, value in counter.most_common()]
    return ", ".join(parts) if parts else "(empty)"


def analyse_dataset(dataset_path: Path, show_citation_details: bool) -> None:
    per_type_lengths: dict[str, dict[str, list[int]]] = {}
    tasks_per_doc: dict[str, Counter[str]] = {}
    duplicates: list[tuple[str, str, str]] = []
    seen_instructions: set[tuple[str, str]] = set()
    citation_stats: dict[str, Any] = {
        "responses_with_citations": 0,
        "total_citation_markers": 0,
        "verification_success": 0,
        "verification_total": 0,
        "coverage_scores": [],
        "missing_reference_cases": 0,
        "extra_reference_cases": 0,
        "failures": [],
    }

    citation_marker_pattern = re.compile(r"\[(\d+)\]")

    for record in load_jsonl(dataset_path):
        task_type = record.get("task_type", "unknown")
        instruction = (record.get("instruction") or "").strip()
        response = (record.get("expected_response") or "").strip()
        metadata = record.get("metadata", {})
        source_path = metadata.get("source_path")
        doc_label = Path(source_path).name if source_path else "unknown"
        doc_counter = tasks_per_doc.setdefault(doc_label, Counter())
        doc_counter[task_type] += 1

        metrics = per_type_lengths.setdefault(
            task_type,
            {
                "instruction_words": [],
                "response_words": [],
                "response_chars": [],
            },
        )

        metrics["instruction_words"].append(word_count(instruction))
        metrics["response_words"].append(word_count(response))
        metrics["response_chars"].append(len(response))

        key = (task_type, instruction.lower())
        if key in seen_instructions:
            duplicates.append((task_type, instruction, doc_label))
        else:
            seen_instructions.add(key)

        if task_type == "citations":
            markers = citation_marker_pattern.findall(response)
            if markers:
                citation_stats["responses_with_citations"] += 1
                citation_stats["total_citation_markers"] += len(markers)
            verification = metadata.get("citation_enforcement", {}).get("verification", {})
            if verification:
                citation_stats["verification_total"] += 1
                referenced = verification.get("referenced", []) or []
                missing = verification.get("missing", []) or []
                extra = verification.get("extra", []) or []
                coverage = verification.get("coverage", {}) or {}

                has_valid = bool(referenced)
                has_extra = bool(extra)
                if has_valid and not has_extra:
                    citation_stats["verification_success"] += 1
                if missing:
                    citation_stats["missing_reference_cases"] += 1
                if has_extra:
                    citation_stats["extra_reference_cases"] += 1
                if missing or has_extra:
                    citation_stats["failures"].append(
                        {
                            "instruction": instruction,
                            "doc": doc_label,
                            "missing": missing,
                            "extra": extra,
                            "coverage": coverage,
                        }
                    )
                citation_stats["coverage_scores"].extend(float(v) for v in coverage.values())

    print("=== Offline dataset summary ===")
    total_tasks = sum(sum(counter.values()) for counter in tasks_per_doc.values())
    type_counts: Counter[str] = Counter()
    for counter in tasks_per_doc.values():
        type_counts.update(counter)
    print(f"Total tasks: {total_tasks}")
    print("Breakdown by type:", format_distribution(type_counts))

    print("\nPer-document task counts (top 10 by total):")
    doc_totals = sorted(tasks_per_doc.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
    for doc, counter in doc_totals[:10]:
        total = sum(counter.values())
        print(f"  {doc}: total {total} -> {format_distribution(counter)}")

    print("\nLength statistics per task type:")
    for task_type, metrics in per_type_lengths.items():
        print(f"[{task_type}]")
        for metric_name, values in metrics.items():
            print(f"  {metric_name}: {describe_numeric(values)}")

    if duplicates:
        print("\nPotential duplicate instructions (up to 5 shown):")
        for task_type, instruction, doc_label in duplicates[:5]:
            print(f"  {task_type} | {doc_label} | {instruction[:100]}")
    else:
        print("\nNo duplicate instructions detected.")

    if type_counts.get("citations"):
        print("\nCitation diagnostics:")
        print(
            f"  Responses containing citation markers: {citation_stats['responses_with_citations']}/"
            f"{type_counts['citations']}"
        )
        if citation_stats["responses_with_citations"]:
            avg_markers = (
                citation_stats["total_citation_markers"]
                / citation_stats["responses_with_citations"]
            )
            print(f"  Avg markers per citing response: {avg_markers:.2f}")
        if citation_stats["verification_total"]:
            success_rate = (
                citation_stats["verification_success"] / citation_stats["verification_total"]
            )
            print(
                f"  Citation verification success: {citation_stats['verification_success']}/"
                f"{citation_stats['verification_total']} ({success_rate:.1%})"
            )
            if citation_stats["coverage_scores"]:
                mean_cov = statistics.mean(citation_stats["coverage_scores"])
                print(f"  Mean snippet coverage (metadata.coverage): {mean_cov:.2f}")
            if citation_stats["missing_reference_cases"]:
                print(
                    f"  Records with missing references: {citation_stats['missing_reference_cases']}"
                )
            if citation_stats["extra_reference_cases"]:
                print(
                    f"  Records with extra/out-of-range markers: {citation_stats['extra_reference_cases']}"
                )
            if citation_stats["failures"] and show_citation_details:
                print("  Sample failures (up to 5):")
                for failure in citation_stats["failures"][:5]:
                    coverage_summary = ", ".join(
                        f"{key}:{float(value):.2f}"
                        for key, value in sorted(failure["coverage"].items())
                    )
                    print(
                        "    - {doc} | {inst} | missing={missing} | extra={extra} | coverage=[{coverage}]".format(
                            doc=failure["doc"],
                            inst=failure["instruction"][:60],
                            missing=",".join(map(str, failure["missing"])) or "-",
                            extra=",".join(map(str, failure.get("extra", []))) or "-",
                            coverage=coverage_summary or "n/a",
                        )
                    )


def analyse_raw_tasks(raw_tasks_path: Path) -> None:
    statuses: Counter[str] = Counter()
    parsed_item_counts: dict[str, int] = {}

    for record in load_jsonl(raw_tasks_path):
        status = record.get("status", "unknown")
        statuses[status] += 1
        parsed = record.get("parsed") or {}
        for key, items in parsed.items():
            parsed_item_counts[key] = parsed_item_counts.get(key, 0) + len(items)

    print("\n=== Generator diagnostics ===")
    print("Run status counts:", format_distribution(statuses))
    if parsed_item_counts:
        for key, total in parsed_item_counts.items():
            print(f"  Parsed {key}: total items {total}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")
    analyse_dataset(dataset_path, args.show_citation_details)

    if args.raw_tasks:
        raw_tasks_path = Path(args.raw_tasks)
        if not raw_tasks_path.exists():
            raise SystemExit(f"Raw tasks file not found: {raw_tasks_path}")
        analyse_raw_tasks(raw_tasks_path)


if __name__ == "__main__":
    main()
