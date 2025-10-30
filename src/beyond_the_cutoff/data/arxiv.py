"""Utilities for querying and downloading papers from the arXiv API."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


def _require_contact_email(contact_email: str) -> str:
    if "@" not in contact_email:
        msg = "arXiv requires a valid contact email in the User-Agent."
        raise ValueError(msg)
    return contact_email


def _parse_datetime(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return dt.replace(tzinfo=timezone.utc)


def _strip_arxiv_version(arxiv_id: str) -> str:
    return arxiv_id.split("v", maxsplit=1)[0] if "v" in arxiv_id else arxiv_id


@dataclass(slots=True)
class ArxivPaper:
    """Container for arXiv metadata."""

    arxiv_id: str
    version: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str | None
    published: datetime
    updated: datetime
    pdf_url: str
    link: str

    @property
    def canonical_id(self) -> str:
        return _strip_arxiv_version(self.arxiv_id)


@dataclass(slots=True)
class ArxivQueryResult:
    """Parsed response for an arXiv API query."""

    papers: list[ArxivPaper]
    total_results: int


class ArxivClient:
    """Thin wrapper around the arXiv export API with rate limiting."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        contact_email: str,
        timeout: float = 30.0,
        rate_limit_seconds: float = 3.0,
        user_agent_app: str = "BeyondTheCutoff/0.1",
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import-time guard
            msg = "httpx is required to use ArxivClient. Install it via `pip install httpx`."
            raise ImportError(msg) from exc

        self.contact_email = _require_contact_email(contact_email)
        self.rate_limit_seconds = rate_limit_seconds
        agent = f"{user_agent_app} (mailto:{self.contact_email})"
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": agent})
        self._last_request_ts = 0.0

    def close(self) -> None:
        self._client.close()

    def _respect_rate_limit(self) -> None:
        if self.rate_limit_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        remaining = self.rate_limit_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def search(
        self,
        *,
        search_query: str,
        start: int = 0,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> ArxivQueryResult:
        self._respect_rate_limit()
        params = {
            "search_query": search_query,
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        response = self._client.get(self.BASE_URL, params=params)
        self._last_request_ts = time.monotonic()
        response.raise_for_status()
        return _parse_response(response.text)

    def download_pdf(self, paper: ArxivPaper, target_path: Path, *, force: bool = False) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not force:
            return target_path
        self._respect_rate_limit()
        response = self._client.get(paper.pdf_url)
        self._last_request_ts = time.monotonic()
        response.raise_for_status()
        target_path.write_bytes(response.content)
        return target_path


def build_category_query(
    categories: Iterable[str], *, submitted_after: datetime, submitted_before: datetime
) -> str:
    cats = "+OR+".join(f"cat:{cat}" for cat in categories)
    window = (
        f"submittedDate:[{submitted_after.strftime('%Y%m%d%H%M%S')}+TO+"
        f"{submitted_before.strftime('%Y%m%d%H%M%S')}]"
    )
    return f"({cats})+AND+{window}"


def format_query_params_for_logging(params: dict[str, str]) -> str:
    """Deterministically encode params for logging/debugging."""

    return urlencode(sorted(params.items()))


def _parse_response(payload: str) -> ArxivQueryResult:
    root = ET.fromstring(payload)
    total_elem = root.find(f"{OPENSEARCH_NS}totalResults")
    total_text = total_elem.text if total_elem is not None else None
    total_results = int(total_text) if total_text is not None else 0
    papers: list[ArxivPaper] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper)
    return ArxivQueryResult(papers=papers, total_results=total_results)


def _parse_entry(entry: ET.Element) -> ArxivPaper | None:
    entry_id = entry.findtext(f"{ATOM_NS}id")
    if not entry_id:
        return None
    title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
    summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip()
    updated_text = entry.findtext(f"{ATOM_NS}updated")
    published_text = entry.findtext(f"{ATOM_NS}published")
    if not updated_text or not published_text:
        return None
    updated = _parse_datetime(updated_text)
    published = _parse_datetime(published_text)
    authors = [
        (author.findtext(f"{ATOM_NS}name") or "").strip()
        for author in entry.findall(f"{ATOM_NS}author")
    ]
    categories = [cat.get("term", "") for cat in entry.findall(f"{ATOM_NS}category")]
    primary_elem = entry.find(f"{ARXIV_NS}primary_category")
    primary_category = primary_elem.get("term") if primary_elem is not None else None
    pdf_url = _extract_pdf_url(entry)
    link_elem = entry.find(f"{ATOM_NS}link[@rel='alternate']")
    link = link_elem.get("href") if link_elem is not None else None
    if not link:
        link = entry_id

    arxiv_id = entry_id.rsplit("/", maxsplit=1)[-1]
    version = arxiv_id.split("v", maxsplit=1)[-1] if "v" in arxiv_id else "1"

    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        summary=summary,
        authors=[a for a in authors if a],
        categories=[c for c in categories if c],
        primary_category=primary_category,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
        link=link,
    )


def _extract_pdf_url(entry: ET.Element) -> str:
    pdf_links: list[str] = []
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.get("type") == "application/pdf":
            href = link.get("href")
            if href:
                pdf_links.append(href)
    if pdf_links:
        return pdf_links[0]
    entry_id = entry.findtext(f"{ATOM_NS}id")
    if entry_id:
        return entry_id.replace("/abs/", "/pdf/") + ".pdf"
    msg = "PDF link not found in entry and fallback failed"
    raise ValueError(msg)


__all__ = [
    "ArxivClient",
    "ArxivPaper",
    "ArxivQueryResult",
    "build_category_query",
    "format_query_params_for_logging",
]
