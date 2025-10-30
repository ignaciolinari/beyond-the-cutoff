"""Beyond the Cutoff research assistant toolkit."""

from importlib import import_module
from typing import Any

from .config import ProjectConfig, load_config
from .models import OllamaClient


def _optional_attr(module_path: str, attr_name: str) -> Any:
    """Safely import *attr_name* from *module_path*, returning None on failure."""

    try:  # pragma: no cover - optional dependency loading
        module = import_module(module_path)
    except Exception:
        return None
    return getattr(module, attr_name, None)


RAGPipeline = _optional_attr("beyond_the_cutoff.retrieval.query", "RAGPipeline")
DocumentIndexer = _optional_attr("beyond_the_cutoff.retrieval.index", "DocumentIndexer")

__all__ = [
    "ProjectConfig",
    "load_config",
    "OllamaClient",
    "RAGPipeline",
    "DocumentIndexer",
    "__version__",
]

__version__ = "0.1.0"
