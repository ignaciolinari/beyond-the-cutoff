"""Data ingestion and dataset utilities."""

from .arxiv import ArxivClient, ArxivPaper, ArxivQueryResult, build_category_query

__all__ = [
    "ArxivClient",
    "ArxivPaper",
    "ArxivQueryResult",
    "build_category_query",
]
