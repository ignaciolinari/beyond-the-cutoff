"""Data ingestion and dataset utilities."""

from .arxiv import ArxivClient, ArxivPaper, ArxivQueryResult, build_category_query
from .catalog import CatalogArtifacts, build_metadata_catalog
from .manifest import build_processed_manifest

__all__ = [
    "ArxivClient",
    "ArxivPaper",
    "ArxivQueryResult",
    "CatalogArtifacts",
    "build_category_query",
    "build_metadata_catalog",
    "build_processed_manifest",
]
