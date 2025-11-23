"""Offline dataset generation for RAG and fine-tuning workflows."""

from .generator import OfflineDatasetGenerator
from .types import DocumentMetadata, DocumentStats, MappingRow, OfflineExample

__all__ = [
    "OfflineDatasetGenerator",
    "OfflineExample",
    "MappingRow",
    "DocumentStats",
    "DocumentMetadata",
]
