"""CLI for fetching a 2025 arXiv corpus with metadata and PDFs."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, cast

import typer  # type: ignore[import-not-found]

from beyond_the_cutoff.data import ArxivClient, ArxivPaper, build_category_query

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Simpler type annotation to avoid ParamSpec/TypeVar issues with some type checkers
command = cast(Callable[[Callable[..., object]], Callable[..., object]], app.command())

DEFAULT_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "stat.ML"]
DEFAULT_START = datetime(2025, 7, 1, tzinfo=timezone.utc)
DEFAULT_OUTPUT_DIR = Path("data/raw/arxiv_2025")


def _parse_date(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _paper_to_serialisable(paper: ArxivPaper) -> dict[str, object]:
    data = asdict(paper)
    data["canonical_id"] = paper.canonical_id
    data["published"] = paper.published.isoformat()
    data["updated"] = paper.updated.isoformat()
    return data


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, records: Iterable[dict[str, object]]) -> None:
    snapshot = list(records)
    if not snapshot:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(snapshot[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(snapshot)


@command
def main(  # noqa: PLR0913 - CLI signature prioritises explicit options
    contact_email: Annotated[
        str,
        typer.Option(
            ...,
            "--contact-email",
            "-e",
            help="Contact email for arXiv User-Agent.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to store metadata, manifest, and PDFs.",
        ),
    ] = DEFAULT_OUTPUT_DIR,
    category: Annotated[
        list[str],
        typer.Option(
            "--category",
            "-c",
            help="arXiv categories to include (repeatable).",
        ),
    ] = DEFAULT_CATEGORIES,
    total: Annotated[
        int,
        typer.Option(
            "--total",
            "-n",
            help="Total number of unique papers to fetch.",
        ),
    ] = 100,
    start_date: Annotated[
        str,
        typer.Option(
            "--start-date",
            help="Lower bound (inclusive) for submission date (YYYY-MM-DD).",
        ),
    ] = DEFAULT_START.strftime("%Y-%m-%d"),
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end-date",
            help="Upper bound (inclusive) for submission date (YYYY-MM-DD). Defaults to today UTC.",
        ),
    ] = None,
    oversample: Annotated[
        float,
        typer.Option(
            "--oversample",
            help="Multiplier per category to mitigate duplicates when pooling results.",
        ),
    ] = 1.6,
    max_results: Annotated[
        int,
        typer.Option(
            "--max-results",
            help="Maximum results to request per category query (capped at arXiv limit 2000).",
        ),
    ] = 200,
    download_pdfs: Annotated[
        bool,
        typer.Option(
            "--download-pdfs/--skip-pdfs",
            help="Toggle PDF downloads.",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-download PDFs even if they already exist.",
        ),
    ] = False,
    download_retries: Annotated[
        int,
        typer.Option(
            "--download-retries",
            help="Number of attempts per PDF before recording a failure.",
        ),
    ] = 3,
    retry_backoff: Annotated[
        float,
        typer.Option(
            "--retry-backoff",
            help="Base backoff (seconds) used for exponential retry delays.",
        ),
    ] = 2.0,
) -> None:
    """Fetch a post-2025 arXiv corpus and write metadata + PDFs."""

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date) if end_date else datetime.now(timezone.utc)

    if end_dt <= start_dt:
        raise typer.BadParameter("--end-date must be after --start-date")
    if total <= 0:
        raise typer.BadParameter("--total must be positive")
    categories = list(dict.fromkeys(category))  # preserve order, drop duplicates
    if not categories:
        raise typer.BadParameter("At least one --category is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    csv_path = output_dir / "metadata.csv"
    manifest_path = output_dir / "manifest.json"
    pdf_dir = output_dir / "papers"
    failures_path = output_dir / "download_failures.jsonl"

    typer.echo(
        f"Fetching target of {total} papers across {len(categories)} categories: {', '.join(categories)}"
    )

    client = ArxivClient(contact_email=contact_email)
    try:
        papers = _collect_papers(
            client,
            categories=categories,
            total=total,
            start_dt=start_dt,
            end_dt=end_dt,
            oversample=oversample,
            per_query_limit=max_results,
        )
    finally:
        client.close()

    records = [_paper_to_serialisable(p) for p in papers]
    _write_jsonl(metadata_path, records)
    _write_csv(csv_path, records)

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_papers": len(papers),
        "requested_total": total,
        "categories": categories,
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "oversample": oversample,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    typer.echo(f"Wrote metadata for {len(papers)} papers to {metadata_path}")

    if not download_pdfs:
        typer.echo("Skipping PDF downloads per flag --skip-pdfs")
        return

    failures: list[dict[str, object]] = []
    client = ArxivClient(contact_email=contact_email)
    try:
        for idx, paper in enumerate(papers, start=1):
            target_path = pdf_dir / f"{paper.canonical_id}.pdf"
            success = False
            for attempt in range(1, max(1, download_retries) + 1):
                try:
                    client.download_pdf(paper, target_path, force=force)
                    typer.echo(f"[{idx}/{len(papers)}] Downloaded {paper.arxiv_id}")
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001 - network and HTTP failures are logged
                    if attempt == max(1, download_retries):
                        typer.echo(f"[{idx}/{len(papers)}] Failed {paper.arxiv_id}: {exc}")
                        failures.append(
                            {
                                "arxiv_id": paper.arxiv_id,
                                "canonical_id": paper.canonical_id,
                                "pdf_url": paper.pdf_url,
                                "error": str(exc),
                            }
                        )
                    else:
                        delay = retry_backoff * (2 ** (attempt - 1))
                        typer.echo(
                            f"[{idx}/{len(papers)}] Retry {attempt}/{download_retries} for {paper.arxiv_id} in {delay:.1f}s"
                        )
                        time.sleep(delay)
            if not success:
                continue
    finally:
        client.close()

    if failures:
        _write_jsonl(failures_path, failures)
        typer.echo(f"Logged {len(failures)} download failures to {failures_path}")


def _collect_papers(
    client: ArxivClient,
    *,
    categories: list[str],
    total: int,
    start_dt: datetime,
    end_dt: datetime,
    oversample: float,
    per_query_limit: int,
) -> list[ArxivPaper]:
    desired_per_category = max(1, total // len(categories))
    if desired_per_category * len(categories) < total:
        desired_per_category += 1

    canonical_map: dict[str, ArxivPaper] = {}
    for category in categories:
        limit = min(per_query_limit, int(desired_per_category * oversample))
        limit = max(desired_per_category, limit)
        typer.echo(f"Querying {category} with max_results={limit}")
        query = build_category_query([category], submitted_after=start_dt, submitted_before=end_dt)
        result = client.search(search_query=query, max_results=limit)
        papers_for_category = result.papers
        if not papers_for_category:
            typer.echo(
                f"No results for category {category} using submittedDate filter; retrying without date filter"
            )
            query = build_category_query([category], submitted_after=None, submitted_before=None)
            result = client.search(search_query=query, max_results=limit)
            papers_for_category = result.papers
        if not papers_for_category:
            typer.echo(f"No results for category {category}")
        for paper in papers_for_category:
            if paper.published < start_dt or paper.published > end_dt:
                continue
            canonical_map.setdefault(paper.canonical_id, paper)

    papers = list(canonical_map.values())
    papers.sort(key=lambda p: p.published, reverse=True)
    if len(papers) > total:
        papers = papers[:total]
    return papers


if __name__ == "__main__":
    app()
