"""Configuration models and loaders for Beyond the Cutoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ProjectMetadata(BaseModel):
    """High-level project metadata."""

    name: str = Field(default="Beyond the Cutoff")
    seed: int = Field(default=42)


class PathsConfig(BaseModel):
    """Filesystem layout for the project."""

    raw_data: Path = Field(default=Path("data/raw"))
    processed_data: Path = Field(default=Path("data/processed"))
    external_data: Path = Field(default=Path("data/external"))

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
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64, ge=0)


class FineTuningConfig(BaseModel):
    """Fine-tuning hyperparameters."""

    base_model: str = Field(default="microsoft/Phi-3-mini-4k-instruct")
    adapter_output_dir: Path = Field(default=Path("outputs/adapters"))
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
    """Evaluation dataset and metric configuration."""

    metrics: list[str] = Field(default_factory=lambda: ["factuality", "citation_accuracy"])
    qa_dataset_path: Path = Field(default=Path("evaluation/datasets/qa_pairs.jsonl"))
    summary_dataset_path: Path = Field(default=Path("evaluation/datasets/summaries.jsonl"))

    @field_validator("qa_dataset_path", "summary_dataset_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    def with_base(self, base_dir: Path) -> EvaluationConfig:
        """Return a copy with absolute dataset paths."""
        return self.model_copy(
            update={
                "qa_dataset_path": self._resolve_path(self.qa_dataset_path, base_dir),
                "summary_dataset_path": self._resolve_path(self.summary_dataset_path, base_dir),
            }
        )

    @staticmethod
    def _resolve_path(path: Path, base_dir: Path) -> Path:
        return path if path.is_absolute() else (base_dir / path).resolve()


class InferenceConfig(BaseModel):
    """Settings for local inference backends (defaults to Ollama)."""

    provider: str = Field(default="ollama")
    model: str = Field(default="phi3:mini")
    host: str = Field(default="http://localhost")
    port: int | None = Field(default=11434)
    timeout: float = Field(default=60.0, gt=0.0)

    def base_url(self) -> str:
        """Return the full base URL for the inference endpoint."""
        root = self.host.rstrip("/")
        if self.port is None:
            return root
        return f"{root}:{self.port}"


class ProjectConfig(BaseModel):
    """Top-level configuration container."""

    project: ProjectMetadata = Field(default_factory=ProjectMetadata)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    def with_base(self, base_dir: Path) -> ProjectConfig:
        """Return a copy with filesystem paths resolved against *base_dir*."""
        return self.model_copy(
            update={
                "paths": self.paths.with_base(base_dir),
                "fine_tuning": self.fine_tuning.with_base(base_dir),
                "evaluation": self.evaluation.with_base(base_dir),
            }
        )


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    omega_conf = OmegaConf.load(str(path))
    container = OmegaConf.to_container(omega_conf, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected config to deserialize into a mapping, got {type(container)!r}")
    return {str(key): value for key, value in container.items()}


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
    "InferenceConfig",
    "ProjectConfig",
    "load_config",
]
