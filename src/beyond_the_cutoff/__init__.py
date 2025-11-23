"""Beyond the Cutoff research assistant toolkit."""

from typing import TYPE_CHECKING, Any

from .config import ProjectConfig, load_config
from .models import LLMClient, OllamaClient, TransformersClient, build_generation_client
from .types import ModelType, PromptMode

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from .retrieval.index import DocumentIndexer as _DocumentIndexer
    from .retrieval.query import RAGPipeline as _RAGPipeline

    RAGPipeline = _RAGPipeline
    DocumentIndexer = _DocumentIndexer
else:  # pragma: no cover - sentinel values replaced via __getattr__
    RAGPipeline = None  # type: ignore[assignment]
    DocumentIndexer = None  # type: ignore[assignment]

__all__ = [
    "ProjectConfig",
    "load_config",
    "LLMClient",
    "OllamaClient",
    "TransformersClient",
    "build_generation_client",
    "RAGPipeline",
    "DocumentIndexer",
    "ModelType",
    "PromptMode",
    "__version__",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazily import optional heavy dependencies."""

    if name == "RAGPipeline":  # pragma: no cover - import side effects exercised elsewhere
        from .retrieval.query import RAGPipeline as _RAGPipeline

        return _RAGPipeline
    if name == "DocumentIndexer":  # pragma: no cover - import side effects exercised elsewhere
        from .retrieval.index import DocumentIndexer as _DocumentIndexer

        return _DocumentIndexer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - reflection helper
    return sorted(set(globals()) | {"RAGPipeline", "DocumentIndexer"})
