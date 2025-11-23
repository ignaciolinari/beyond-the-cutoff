"""Configuration models and loaders for Beyond the Cutoff."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    OmegaConf = import_module("omegaconf").OmegaConf
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OmegaConf = None
from pydantic import BaseModel, Field, field_validator, model_validator

from .types import ModelType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ProjectMetadata(BaseModel):
    """High-level project metadata."""

    name: str = Field(default="Beyond the Cutoff")
    seed: int = Field(default=42)


class PathsConfig(BaseModel):
    """Filesystem layout for the project.

    Paths are resolved relative to the config file's parent directory.
    For configs in `configs/`, use `../data/raw` to reach the repo root.
    """

    raw_data: Path = Field(default=Path("../data/raw"))
    processed_data: Path = Field(default=Path("../data/processed"))
    external_data: Path = Field(default=Path("../data/external"))

    @field_validator("raw_data", "processed_data", "external_data", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def with_base(self, base_dir: Path) -> PathsConfig:
        """Return a new config with absolute paths resolved against *base_dir*."""
        return self.model_copy(
            update={
                "raw_data": self._resolve_path(self.raw_data, base_dir),
                "processed_data": self._resolve_path(self.processed_data, base_dir),
                "external_data": self._resolve_path(self.external_data, base_dir),
            }
        )

    @staticmethod
    def _resolve_path(path: Path, base_dir: Path) -> Path:
        return path if path.is_absolute() else (base_dir / path).resolve()


class RetrievalConfig(BaseModel):
    """Settings for the retrieval pipeline."""

    vector_store: str = Field(default="faiss")
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_batch_size: int = Field(default=8, ge=1)
    embedding_device: str = Field(default="auto")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64, ge=0)
    top_k: int = Field(default=4, ge=1)
    max_context_chars: int = Field(
        default=6000, ge=512
    )  # soft cap to keep prompts within model context
    # Chunking strategy: "words" (sliding window over whitespace tokens) or
    # "sentences" (packs full sentences up to chunk_size tokens)
    chunking_strategy: str = Field(default="words")
    # Optional cross-encoder reranker model name (empty disables reranking)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")


class FineTuningConfig(BaseModel):
    """Fine-tuning hyperparameters.

    Paths are resolved relative to the config file's parent directory.
    For configs in `configs/`, use `../outputs/...` to reach the repo root.
    """

    base_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    adapter_output_dir: Path = Field(default=Path("../outputs/adapters"))
    lora_rank: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_steps: int = Field(default=1000, ge=1)

    @field_validator("adapter_output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def with_base(self, base_dir: Path) -> FineTuningConfig:
        """Return a copy with absolute adapter output directory."""
        return self.model_copy(update={"adapter_output_dir": self._resolve_dir(base_dir)})

    def _resolve_dir(self, base_dir: Path) -> Path:
        directory = self.adapter_output_dir
        return directory if directory.is_absolute() else (base_dir / directory).resolve()


class EvaluationConfig(BaseModel):
    """Evaluation dataset and metric configuration.

    Paths are resolved relative to the config file's parent directory.
    For configs in `configs/`, use `../evaluation/...` to reach the repo root.
    """

    metrics: list[str] = Field(default_factory=lambda: ["factuality", "citation_accuracy"])
    offline_tasks_path: Path = Field(default=Path("../evaluation/datasets/offline_tasks.jsonl"))
    offline_dataset_path: Path = Field(default=Path("../evaluation/datasets/offline_dataset.jsonl"))

    @field_validator(
        "offline_tasks_path",
        "offline_dataset_path",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def with_base(self, base_dir: Path) -> EvaluationConfig:
        """Return a copy with absolute dataset paths."""
        return self.model_copy(
            update={
                "offline_tasks_path": self._resolve_path(self.offline_tasks_path, base_dir),
                "offline_dataset_path": self._resolve_path(self.offline_dataset_path, base_dir),
            }
        )

    @staticmethod
    def _resolve_path(path: Path, base_dir: Path) -> Path:
        return path if path.is_absolute() else (base_dir / path).resolve()


class InferenceConfig(BaseModel):
    """Settings for local inference backends (defaults to Ollama)."""

    provider: str = Field(default="ollama")
    model: str = Field(default="qwen2.5:0.5b-instruct")
    # Explicit model type: "base", "instruction_only", or "rag_trained"
    # If not specified, will be inferred from config file name or model name
    model_type: ModelType | None = Field(
        default=None,
        description="Explicit model type: 'base', 'instruction_only', or 'rag_trained'. "
        "If None, will be inferred from config filename or model name.",
    )
    host: str = Field(default="http://localhost")
    port: int | None = Field(default=11434)
    timeout: float = Field(default=120.0, gt=0.0)  # Reduced from 480s for better responsiveness
    device: str = Field(default="auto")
    torch_dtype: str | None = Field(default="auto")
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, gt=0.0)
    stop_sequences: list[str] = Field(default_factory=list)

    @field_validator("model_type", mode="before")
    @classmethod
    def _validate_model_type(cls, value: Any) -> ModelType | None:
        if value is None:
            return None
        if isinstance(value, ModelType):
            return value
        value_str = str(value).strip().lower()
        try:
            return ModelType(value_str)
        except ValueError:
            raise ValueError(
                f"model_type must be 'base', 'instruction_only', or 'rag_trained', got {value!r}"
            ) from None

    def base_url(self) -> str:
        """Return the full base URL for the inference endpoint."""
        root = self.host.rstrip("/")
        if self.port is None:
            return root
        return f"{root}:{self.port}"

    @field_validator("stop_sequences", mode="before")
    @classmethod
    def _validate_stop_sequences(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        return [str(value)]


def _default_generator_config() -> InferenceConfig:
    """Return the default generator backend (7B Ollama tag)."""

    return InferenceConfig(
        provider="ollama",
        model="qwen2.5:7b-instruct-q4_K_M",
        host="http://localhost",
        port=11434,
        timeout=120.0,
        device="auto",
        torch_dtype="auto",
        max_new_tokens=768,
        temperature=0.5,
        top_p=0.95,
        repetition_penalty=1.05,
        stop_sequences=[],
    )


class DatasetGenerationConfig(BaseModel):
    """Settings controlling offline dataset generation.

    Paths are resolved relative to the config file's parent directory.
    For configs in `configs/`, use `../evaluation/...` to reach the repo root.
    """

    generator: InferenceConfig = Field(default_factory=_default_generator_config)
    output_dataset_path: Path = Field(default=Path("../evaluation/datasets/offline_dataset.jsonl"))
    raw_tasks_path: Path = Field(default=Path("../evaluation/datasets/offline_tasks.jsonl"))
    questions_per_document: int = Field(default=4, ge=0)
    summary_prompts_per_document: int = Field(default=1, ge=0)
    citation_prompts_per_document: int = Field(default=1, ge=0)
    contextual_prompts_per_document: int = Field(default=1, ge=0)
    min_questions_per_document: int = Field(default=3, ge=0)
    min_summary_prompts_per_document: int = Field(default=1, ge=0)
    min_citation_prompts_per_document: int = Field(default=1, ge=0)
    min_contextual_prompts_per_document: int = Field(default=1, ge=0)
    max_chunks_per_document: int = Field(default=4, ge=1)
    max_chars_per_chunk: int = Field(default=1200, ge=256)
    max_documents: int | None = Field(default=None, ge=1)
    max_document_tokens: int | None = Field(default=25000)
    max_document_pages: int | None = Field(default=50)
    seed: int = Field(default=42)
    parse_retries: int = Field(default=2, ge=0)
    citation_rewrite_attempts: int = Field(default=1, ge=0)
    min_citation_coverage: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("output_dataset_path", "raw_tasks_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def with_base(self, base_dir: Path) -> DatasetGenerationConfig:
        """Return a copy with resolved output paths."""

        return self.model_copy(
            update={
                "output_dataset_path": self._resolve_path(self.output_dataset_path, base_dir),
                "raw_tasks_path": self._resolve_path(self.raw_tasks_path, base_dir),
            }
        )

    @staticmethod
    def _resolve_path(path: Path, base_dir: Path) -> Path:
        return path if path.is_absolute() else (base_dir / path).resolve()

    @field_validator("max_document_tokens", "max_document_pages", mode="before")
    @classmethod
    def _coerce_positive_optional(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                candidate = int(float(stripped))
            except ValueError as exc:  # pragma: no cover - validation guard
                raise ValueError("must be an integer or null") from exc
        elif isinstance(value, bool):  # pragma: no cover - defensive guard
            raise ValueError("must be an integer or null")
        else:
            try:
                candidate = int(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - validation guard
                raise ValueError("must be an integer or null") from exc

        if candidate <= 0:
            raise ValueError("must be greater than zero or null")
        return candidate

    @model_validator(mode="after")
    def _validate_minimums(self) -> DatasetGenerationConfig:
        if self.min_questions_per_document > self.questions_per_document:
            raise ValueError("min_questions_per_document cannot exceed questions_per_document")
        if self.min_summary_prompts_per_document > self.summary_prompts_per_document:
            raise ValueError(
                "min_summary_prompts_per_document cannot exceed summary_prompts_per_document"
            )
        if self.min_citation_prompts_per_document > self.citation_prompts_per_document:
            raise ValueError(
                "min_citation_prompts_per_document cannot exceed citation_prompts_per_document"
            )
        if self.min_contextual_prompts_per_document > self.contextual_prompts_per_document:
            raise ValueError(
                "min_contextual_prompts_per_document cannot exceed contextual_prompts_per_document"
            )
        return self


class ProjectConfig(BaseModel):
    """Top-level configuration container."""

    project: ProjectMetadata = Field(default_factory=ProjectMetadata)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dataset_generation: DatasetGenerationConfig = Field(default_factory=DatasetGenerationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    def with_base(self, base_dir: Path) -> ProjectConfig:
        """Return a copy with filesystem paths resolved against *base_dir*."""
        return self.model_copy(
            update={
                "paths": self.paths.with_base(base_dir),
                "fine_tuning": self.fine_tuning.with_base(base_dir),
                "evaluation": self.evaluation.with_base(base_dir),
                "dataset_generation": self.dataset_generation.with_base(base_dir),
            }
        )


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if OmegaConf is not None:
        omega_conf = OmegaConf.load(str(path))
        container = OmegaConf.to_container(omega_conf, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(
                f"Expected config to deserialize into a mapping, got {type(container)!r}"
            )
        return {str(key): value for key, value in container.items()}

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected config to deserialize into a mapping, got {type(data)!r}")
    return {str(key): value for key, value in data.items()}


def load_config(path: Path | str | None = None) -> ProjectConfig:
    """Load a project configuration from *path* (defaults to configs/default.yaml)."""
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    raw_config = _load_raw_config(config_path)
    config = ProjectConfig.model_validate(raw_config)
    return config.with_base(config_path.parent)


__all__ = [
    "ProjectMetadata",
    "PathsConfig",
    "RetrievalConfig",
    "FineTuningConfig",
    "EvaluationConfig",
    "DatasetGenerationConfig",
    "InferenceConfig",
    "ProjectConfig",
    "load_config",
]
