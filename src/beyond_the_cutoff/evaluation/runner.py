"""Utilities for executing evaluation runs across models and judges."""

from __future__ import annotations

import datetime
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from beyond_the_cutoff.config import InferenceConfig, ProjectConfig
from beyond_the_cutoff.evaluation.metrics import evaluate_citations, normalize_contexts
from beyond_the_cutoff.models import LLMClient, build_generation_client
from beyond_the_cutoff.types import ModelType
from beyond_the_cutoff.utils.experiment_logging import append_experiment_record
from beyond_the_cutoff.utils.validation import (
    print_validation_result,
    validate_configuration,
    validate_evaluation_sanity,
)


@dataclass
class JudgePrompt:
    """Parsed judge configuration."""

    name: str
    prompt: str
    criteria: list[dict[str, Any]]
    output_format: dict[str, Any]
    references: dict[str, Any]


@dataclass
class EvaluationResult:
    """Container summarising the outcome of an evaluation run."""

    summary: dict[str, Any]
    score_rows: list[dict[str, Any]]
    metrics_path: Path | None
    details_path: Path | None
    metadata_path: Path


@cache
def load_inference_from_yaml(path: Path) -> InferenceConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected inference YAML to be a mapping, got {type(data)!r}")
    return InferenceConfig.model_validate(data)


@cache
def load_judge_prompt(path: Path) -> JudgePrompt:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Judge config must be a mapping, got {type(payload)!r}")
    return JudgePrompt(
        name=str(payload.get("name", "judge")),
        prompt=str(payload.get("prompt", "")),
        criteria=list(payload.get("criteria", [])),
        output_format=dict(payload.get("format", {})),
        references=dict(payload.get("references", {})),
    )


def render_judge_prompt(
    template: str, question: str, contexts: Iterable[Any], answer: str, *, has_contexts: bool = True
) -> str:
    numbered_contexts = normalize_contexts(contexts) if has_contexts else []
    context_block = (
        "\n\n".join(numbered_contexts) if numbered_contexts else "(No contexts provided)"
    )
    rendered = template.replace("QUESTION", question.strip())
    rendered = rendered.replace("ASSISTANT_RESPONSE", answer.strip())
    rendered = rendered.replace("CONTEXTS", context_block)
    return rendered


def _detect_model_type(
    model_config_path: Path | None,
    model_name: str,
    model_cfg: InferenceConfig | None = None,
    *,
    warn_on_inference: bool = True,
) -> ModelType:
    """Detect model type from config path or model name.

    Args:
        model_config_path: Optional path to model config file
        model_name: Model name/Ollama tag
        model_cfg: Optional InferenceConfig with explicit model_type field
        warn_on_inference: If True, warn when model_type is inferred rather than explicit

    Returns:
        ModelType enum: INSTRUCTION_ONLY, RAG_TRAINED, or BASE

    Detection priority:
    1. Explicit model_type in InferenceConfig (most reliable)
    2. Config file name
    3. Model name/Ollama tag
    4. Default to BASE
    """
    # First, check explicit model_type in config (most reliable)
    if model_cfg and model_cfg.model_type:
        return model_cfg.model_type

    # Model type was not explicit - warn if requested
    if warn_on_inference:
        inference_source = []
        if model_config_path:
            inference_source.append(f"config filename '{model_config_path.name}'")
        inference_source.append(f"model name '{model_name}'")
        source_str = " and ".join(inference_source)
        print(
            f"[warn] model_type not explicitly set in config. Inferring from {source_str}. "
            "Consider adding explicit 'model_type' field to config for reliability.",
            file=sys.stderr,
        )
    # First, check config file name (most reliable indicator)
    if model_config_path:
        config_name = model_config_path.name.lower()
        # Check for RAG-trained indicators (must come before instruction_only check)
        if "rag_trained" in config_name or "rag-trained" in config_name:
            return ModelType.RAG_TRAINED
        # Check for instruction-only indicators
        if "instruction_only" in config_name or "instruction-only" in config_name:
            return ModelType.INSTRUCTION_ONLY
        # Check for hybrid configs (these use RAG-trained models)
        if "hybrid" in config_name and "instruction" not in config_name:
            # Hybrid configs typically use RAG-trained models unless explicitly instruction-only
            return ModelType.RAG_TRAINED

    # Second, check model name/Ollama tag
    model_lower = model_name.lower()

    # Check for explicit instruction-only indicators in model name
    if "instruction_only" in model_lower or "instruction-only" in model_lower:
        return ModelType.INSTRUCTION_ONLY

    # Check for RAG-trained indicators (but exclude instruction-only variants)
    # Common patterns: "lora_science_0p5" (without instruction_only suffix) = RAG-trained
    #                  "lora_science_0p5_instruction_only" = instruction-only
    if "lora_science" in model_lower:
        # If it contains "lora_science" but NOT "instruction_only", it's likely RAG-trained
        if "instruction" not in model_lower:
            return ModelType.RAG_TRAINED
        # If it has both, check which comes first or is more specific
        if "instruction_only" in model_lower:
            return ModelType.INSTRUCTION_ONLY

    # Check for base model indicators
    if any(
        base_indicator in model_lower
        for base_indicator in ["qwen2.5:0.5b", "qwen2.5:3b", "qwen2.5:7b"]
    ):
        # Base models from Ollama (not fine-tuned)
        if "lora" not in model_lower and "science" not in model_lower:
            return ModelType.BASE

    # Default to base model if we can't determine
    # This is conservative - better to treat as base than misclassify
    return ModelType.BASE


def _build_instruction_only_prompt(
    instruction: str,
    *,
    model_type: ModelType | None = None,
    model_config_path: Path | None = None,
    model_name: str = "",
    model_cfg: InferenceConfig | None = None,
) -> str:
    """Build a prompt for instruction-only mode (no RAG contexts).

    Args:
        instruction: The instruction/question text
        model_type: Model type (INSTRUCTION_ONLY, RAG_TRAINED, or BASE).
                    If None, will be inferred from model_config_path and model_name.
        model_config_path: Optional path to model config file for type detection
        model_name: Optional model name for type detection

    Note: System message comes from the Ollama Modelfile. User content should NOT
    duplicate the system message - it should only contain the question and answer prompt.
    The Modelfile provides the system message, so user content is simplified to avoid duplication.

    For RAG-trained models evaluated in instruction-only mode, we use a format
    that doesn't contradict the system message (which mentions citations).
    """
    instruction_text = instruction.strip()
    if not instruction_text:
        raise ValueError("Instruction cannot be empty for instruction-only mode")

    # Detect model type if not provided
    if model_type is None:
        model_type = _detect_model_type(model_config_path, model_name, model_cfg=model_cfg)
        # Log detection for debugging
        if model_config_path:
            print(
                f"[info] Detected model type '{model_type}' from config '{model_config_path.name}' "
                f"and model name '{model_name}'",
                file=sys.stderr,
            )

    # Match training format exactly for instruction-only models
    # For RAG-trained models evaluated without contexts (Condition 5), use a format that
    # acknowledges the mismatch: model was trained WITH contexts, evaluated WITHOUT.
    if model_type == ModelType.RAG_TRAINED:
        # RAG-trained model has system message about citations, so we need to explicitly
        # tell it not to cite sources since no contexts are provided. The model was trained
        # with RAG prompts, but when evaluated without contexts (Condition 5), we use a format
        # that acknowledges we're asking for knowledge-based answers without contexts.
        # This tests distribution shift: model trained WITH contexts, evaluated WITHOUT.
        # We explicitly instruct not to cite to prevent spurious citations.
        return (
            "Answer the following question based on your knowledge. "
            "Do not include citations or source references as no sources are provided. "
            "Provide a clear and concise response.\n\n"
            f"Question: {instruction_text}\n\nAnswer:"
        )
    elif model_type == ModelType.INSTRUCTION_ONLY:
        # Instruction-only trained model: Modelfile provides system message, user content is just question/answer
        return f"Question: {instruction_text}\n\nAnswer:"
    else:
        # Base model: Modelfile provides system message, user content is just question/answer
        return f"Question: {instruction_text}\n\nAnswer:"


def _build_rag_prompt_for_instruction_only_model(
    instruction: str,
    contexts: list[str],
    *,
    model_config_path: Path | None = None,
    model_name: str = "",
) -> str:
    """Build a RAG prompt for instruction-only models evaluated with RAG contexts.

    This is used for Condition 4 (FT+RAG instruction-only) where an instruction-only
    trained model is evaluated WITH RAG contexts. The Modelfile provides the system
    message, so user content should not duplicate it. User content includes contexts
    and citation instructions.

    Args:
        instruction: The instruction/question text
        contexts: List of numbered context strings (e.g., "[1] context text")
        model_config_path: Optional path to model config file for type detection
        model_name: Optional model name for type detection

    Returns:
        Formatted prompt string that matches instruction-only training format but includes contexts
    """
    instruction_text = instruction.strip()
    if not instruction_text:
        raise ValueError("Instruction cannot be empty")

    if not contexts:
        raise ValueError("Contexts cannot be empty for RAG prompt")

    # Build context block from numbered contexts
    context_block = "\n\n".join(contexts)

    # Hybrid format: Preserves training instruction structure ("Question: ... Answer:")
    # while adding RAG context requirements. This bridges the training format with RAG needs.
    # Training format: "Question: X\n\nAnswer:" (system message provided separately)
    # This hybrid format ensures the model sees familiar instruction patterns while learning to use contexts.
    prompt = (
        f"Question: {instruction_text}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer using the provided context. Cite sources inline as [#] based on the order of the snippets. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Answer:"
    )

    return prompt


def parse_judge_output(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = payload[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
                return {"raw": parsed}
            except json.JSONDecodeError:
                return {"raw": payload}
        return {"raw": payload}


def summarise_scores(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    coverage_values: list[float] = []
    instruction_only_count = 0
    rag_count = 0

    # Track error categories
    error_category_counts: dict[str, int] = defaultdict(int)
    error_stage_counts: dict[str, int] = defaultdict(int)

    for row in score_rows:
        scores = row.get("judge_scores", {})
        for key, value in scores.items():
            if isinstance(value, int | float):
                metrics[key].append(float(value))
        citation_metrics = row.get("citation_metrics", {})

        # Validate citation metrics mode
        citation_mode = citation_metrics.get("mode")
        if citation_mode == "instruction_only":
            instruction_only_count += 1
            # Ensure citation metrics are not aggregated for instruction-only mode
            if "validated" not in citation_metrics:
                print(
                    "[warn] Citation metrics for instruction-only mode missing validation flag. "
                    "This may indicate a bug in the evaluation pipeline.",
                    file=sys.stderr,
                )
        elif citation_mode == "rag":
            rag_count += 1
            # Only aggregate citation metrics for RAG mode
            coverage = citation_metrics.get("mean_coverage")
            if isinstance(coverage, int | float):
                coverage_values.append(float(coverage))
        elif citation_metrics:  # Has citation metrics but no mode specified
            print(
                "[warn] Citation metrics missing mode specification. "
                "Assuming RAG mode for aggregation.",
                file=sys.stderr,
            )
            coverage = citation_metrics.get("mean_coverage")
            if isinstance(coverage, int | float):
                coverage_values.append(float(coverage))

        # Track error information
        if row.get("errors"):
            errors = row.get("errors", {})
            error_categories = row.get("error_categories", {})

            # Count errors by stage
            for stage in errors.keys():
                error_stage_counts[stage] += 1

            # Count errors by category
            for _stage, category in error_categories.items():
                error_category_counts[category] += 1
            # If no category specified, count as unknown
            for stage in errors.keys():
                if stage not in error_categories:
                    error_category_counts["unknown"] += 1

    summary: dict[str, Any] = {
        key: mean(values) if values else 0.0 for key, values in metrics.items()
    }
    summary["citation_mean_coverage"] = mean(coverage_values) if coverage_values else 0.0
    summary["citation_metrics_mode_counts"] = {
        "instruction_only": instruction_only_count,
        "rag": rag_count,
    }

    # Add error statistics
    if error_category_counts or error_stage_counts:
        summary["error_statistics"] = {
            "by_category": dict(error_category_counts),
            "by_stage": dict(error_stage_counts),
        }

    return summary


def _categorize_error(error_message: str, error_type: type[Exception]) -> str:
    """Categorize an error into a structured error type.

    Args:
        error_message: Error message string
        error_type: Exception type

    Returns:
        Error category: 'network', 'parsing', 'model', 'validation', 'timeout', 'unknown'
    """
    error_str = str(error_message).lower()
    error_type_name = error_type.__name__.lower()

    # Network-related errors
    if any(
        indicator in error_str or indicator in error_type_name
        for indicator in [
            "connection",
            "network",
            "timeout",
            "http",
            "socket",
            "refused",
            "unreachable",
        ]
    ):
        if "timeout" in error_str or "timeout" in error_type_name:
            return "timeout"
        return "network"

    # Parsing errors
    if any(
        indicator in error_str or indicator in error_type_name
        for indicator in ["json", "parse", "decode", "syntax", "invalid format", "malformed"]
    ):
        return "parsing"

    # Model/API errors
    if any(
        indicator in error_str or indicator in error_type_name
        for indicator in [
            "model",
            "api",
            "rate limit",
            "quota",
            "unauthorized",
            "forbidden",
            "401",
            "403",
            "429",
        ]
    ):
        return "model"

    # Validation errors
    if any(
        indicator in error_str or indicator in error_type_name
        for indicator in ["validation", "invalid", "missing", "required", "valueerror"]
    ):
        return "validation"

    # Default to unknown
    return "unknown"


def _call_with_retries(
    func: Callable[[], Any],
    *,
    stage: str,
    max_retries: int,
    retry_delay: float,
    use_exponential_backoff: bool = True,
) -> tuple[Any | None, str | None, str | None]:
    """Call a function with retries and exponential backoff.

    Args:
        func: Function to call (no arguments)
        stage: Description of the operation (for logging)
        max_retries: Maximum number of retries (total attempts = max_retries + 1)
        retry_delay: Base delay in seconds (doubles with exponential backoff)
        use_exponential_backoff: If True, use exponential backoff; otherwise use linear

    Returns:
        Tuple of (result, error_message, error_category). Result is None if all attempts failed.
    """
    attempts = max_retries + 1
    last_error: str | None = None
    last_error_type: type[Exception] | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func(), None, None
        except KeyboardInterrupt:  # pragma: no cover - allow user aborts
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            last_error_type = type(exc)
            if attempt == attempts:
                break

            # Calculate wait time: exponential backoff by default, linear fallback
            if use_exponential_backoff:
                # Exponential backoff: base_delay * 2^(attempt-1)
                # Cap at 300 seconds (5 minutes) to avoid extremely long waits
                wait_seconds = min(max(retry_delay, 0.0) * (2 ** (attempt - 1)), 300.0)
            else:
                # Linear backoff: base_delay * attempt
                wait_seconds = max(retry_delay, 0.0) * attempt

            print(
                f"[warn] {stage} failed on attempt {attempt}/{attempts}: {last_error}. "
                f"Retrying in {wait_seconds:.1f}s...",
                file=sys.stderr,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    # Categorize the error before returning
    error_category = (
        _categorize_error(last_error or "", last_error_type or Exception) if last_error else None
    )
    return None, last_error, error_category


def _save_checkpoint(
    checkpoint_path: Path,
    score_rows: list[dict[str, Any]],
    evaluated_task_ids: set[str],
    *,
    model_label: str,
    dataset_path: Path,
) -> None:
    """Save evaluation checkpoint to disk.

    Args:
        checkpoint_path: Path where checkpoint should be saved
        score_rows: List of evaluation results so far
        evaluated_task_ids: Set of task IDs that have been evaluated
        model_label: Label for the model being evaluated
        dataset_path: Path to the dataset being evaluated
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute dataset hash for validation
    from beyond_the_cutoff.utils.experiment_logging import compute_file_sha256

    try:
        dataset_sha256 = compute_file_sha256(dataset_path)
        dataset_size = dataset_path.stat().st_size
    except Exception:
        dataset_sha256 = None
        dataset_size = None

    checkpoint_data = {
        "model_label": model_label,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_size_bytes": dataset_size,
        "evaluated_task_ids": sorted(evaluated_task_ids),
        "score_rows": score_rows,
        "timestamp": time.time(),
        "checkpoint_version": "1.1",  # Version for checkpoint format
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_checkpoint(
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]] | None:
    """Load evaluation checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (score_rows, evaluated_task_ids, checkpoint_metadata) if checkpoint exists, None otherwise.
        checkpoint_metadata contains summary information about the checkpoint.
    """
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        score_rows = checkpoint_data.get("score_rows", [])
        evaluated_task_ids = set(checkpoint_data.get("evaluated_task_ids", []))

        # Extract checkpoint metadata for summary
        checkpoint_metadata = {
            "model_label": checkpoint_data.get("model_label", "unknown"),
            "dataset_path": checkpoint_data.get("dataset_path", "unknown"),
            "timestamp": checkpoint_data.get("timestamp"),
            "examples_count": len(score_rows),
            "task_ids_count": len(evaluated_task_ids),
            "errors_count": sum(1 for row in score_rows if row.get("errors")),
            "empty_responses_count": sum(
                1
                for row in score_rows
                if not str(row.get("model_answer", "")).strip()
                and (not row.get("errors") or "generation" not in row.get("errors", {}))
            ),
        }

        return score_rows, evaluated_task_ids, checkpoint_metadata
    except Exception as exc:
        print(
            f"[warn] Failed to load checkpoint from {checkpoint_path}: {exc}. "
            "Starting fresh evaluation.",
            file=sys.stderr,
        )
        return None


def run_evaluation(
    *,
    project_config: ProjectConfig,
    dataset_path: Path,
    model_cfg: InferenceConfig,
    judge_prompt_path: Path,
    judge_inference_cfg: InferenceConfig,
    model_label: str,
    config_path: Path,
    model_config_path: Path | None = None,
    judge_inference_path: Path | None = None,
    limit: int | None = None,
    output_path: Path | None = None,
    details_output_path: Path | None = None,
    metadata_output_path: Path | None = None,
    max_retries: int = 2,
    retry_delay: float = 15.0,
    prompt_mode: str = "rag",
    validate: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 10,
    resume_from_checkpoint: bool = True,
) -> EvaluationResult:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {dataset_path}")

    # Validate configuration before running evaluation
    if validate:
        config_validation = validate_configuration(
            config_path,
            model_config_path=model_config_path,
            judge_config_path=judge_prompt_path,
            judge_inference_path=judge_inference_path,
            prompt_mode=prompt_mode,
        )
        if not config_validation.passed:
            print_validation_result(config_validation)
            # Only fail on errors, warnings are informational
            if any(i.severity == "error" for i in config_validation.issues):
                raise ValueError("Configuration validation failed. See errors above.")

    judge_prompt = load_judge_prompt(judge_prompt_path.resolve())

    # Warn if judge config might not match prompt mode
    judge_name_lower = judge_prompt.name.lower()
    is_instruction_judge = "instruction" in judge_name_lower and "rag" not in judge_name_lower
    is_rag_judge = "rag" in judge_name_lower and "instruction" not in judge_name_lower
    if prompt_mode == "instruction" and is_rag_judge:
        print(
            f"[warn] prompt_mode='instruction' but judge config '{judge_prompt.name}' appears to be for RAG evaluation. "
            "Consider using an instruction-only judge config.",
            file=sys.stderr,
        )
    elif prompt_mode == "rag" and is_instruction_judge:
        print(
            f"[warn] prompt_mode='rag' but judge config '{judge_prompt.name}' appears to be for instruction-only evaluation. "
            "Consider using a RAG judge config.",
            file=sys.stderr,
        )

    model_client: LLMClient = build_generation_client(model_cfg)
    judge_client: LLMClient = build_generation_client(judge_inference_cfg)

    if prompt_mode not in ("rag", "instruction"):
        raise ValueError(f"prompt_mode must be 'rag' or 'instruction', got {prompt_mode!r}")

    # Validate dataset structure before processing
    _validate_dataset_structure(dataset_path, prompt_mode, limit=limit)

    # Initialize checkpoint path if not provided
    if checkpoint_path is None and output_path:
        checkpoint_path = output_path.parent / f"{output_path.stem}.checkpoint.json"
    elif checkpoint_path is None:
        checkpoint_path = Path("evaluation/results") / model_label / "checkpoint.json"

    # Try to load checkpoint if resume is enabled
    score_rows: list[dict[str, Any]] = []
    evaluated_task_ids: set[str] = set()
    checkpoint_loaded = False
    checkpoint_metadata: dict[str, Any] | None = None

    if resume_from_checkpoint and checkpoint_path:
        checkpoint_result = _load_checkpoint(checkpoint_path)
        if checkpoint_result:
            score_rows, evaluated_task_ids, checkpoint_metadata = checkpoint_result

            # Validate checkpoint dataset path matches current dataset path
            checkpoint_dataset_path = checkpoint_metadata.get("dataset_path", "")
            checkpoint_dataset_sha256 = checkpoint_metadata.get("dataset_sha256")
            current_dataset_path_str = str(dataset_path.resolve())
            checkpoint_dataset_path_resolved = (
                str(Path(checkpoint_dataset_path).resolve())
                if checkpoint_dataset_path != "unknown"
                else None
            )

            # Also validate SHA256 if available (more reliable than path)
            dataset_mismatch = False
            if checkpoint_dataset_sha256:
                try:
                    from beyond_the_cutoff.utils.experiment_logging import compute_file_sha256

                    current_dataset_sha256 = compute_file_sha256(dataset_path)
                    if checkpoint_dataset_sha256 != current_dataset_sha256:
                        dataset_mismatch = True
                        print(
                            "[warn] Checkpoint dataset SHA256 mismatch detected:",
                            file=sys.stderr,
                        )
                        print(
                            f"  Checkpoint SHA256: {checkpoint_dataset_sha256[:16]}...",
                            file=sys.stderr,
                        )
                        print(
                            f"  Current SHA256: {current_dataset_sha256[:16]}...",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(
                        f"[warn] Could not compute dataset SHA256 for validation: {exc}",
                        file=sys.stderr,
                    )

            if (
                checkpoint_dataset_path_resolved
                and checkpoint_dataset_path_resolved != current_dataset_path_str
            ):
                dataset_mismatch = True

            if dataset_mismatch:
                print(
                    "[warn] Checkpoint dataset path mismatch detected:",
                    file=sys.stderr,
                )
                print(
                    f"  Checkpoint dataset: {checkpoint_dataset_path_resolved}",
                    file=sys.stderr,
                )
                print(
                    f"  Current dataset: {current_dataset_path_str}",
                    file=sys.stderr,
                )
                print(
                    "[warn] Dataset paths differ - checkpoint may be from a different dataset. "
                    "Starting fresh evaluation to avoid incorrect resumption.",
                    file=sys.stderr,
                )
                # Reset checkpoint data - don't resume from mismatched dataset
                score_rows = []
                evaluated_task_ids = set()
                checkpoint_metadata = None
                checkpoint_loaded = False
            else:
                checkpoint_loaded = True

            if checkpoint_loaded and checkpoint_metadata:
                # Print detailed checkpoint resumption summary
                print("-" * 80, file=sys.stderr)
                print("[info] Checkpoint resumption summary:", file=sys.stderr)
                print(
                    f"  Model: {checkpoint_metadata.get('model_label', 'unknown')}", file=sys.stderr
                )
                print(
                    f"  Dataset: {checkpoint_metadata.get('dataset_path', 'unknown')}",
                    file=sys.stderr,
                )
                print(
                    f"  Examples already evaluated: {checkpoint_metadata.get('examples_count', 0)}",
                    file=sys.stderr,
                )
                print(
                    f"  Unique task IDs: {checkpoint_metadata.get('task_ids_count', 0)}",
                    file=sys.stderr,
                )
                if checkpoint_metadata.get("errors_count", 0) > 0:
                    print(
                        f"  Examples with errors: {checkpoint_metadata.get('errors_count', 0)}",
                        file=sys.stderr,
                    )
                if checkpoint_metadata.get("empty_responses_count", 0) > 0:
                    print(
                        f"  Empty responses: {checkpoint_metadata.get('empty_responses_count', 0)}",
                        file=sys.stderr,
                    )
                if checkpoint_metadata.get("timestamp"):
                    try:
                        timestamp = checkpoint_metadata["timestamp"]
                        if isinstance(timestamp, int | float):
                            dt = datetime.datetime.fromtimestamp(timestamp)
                            print(
                                f"  Checkpoint timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}",
                                file=sys.stderr,
                            )
                    except Exception:
                        pass
                print("-" * 80, file=sys.stderr)

    # Count total examples for progress reporting
    total_examples = _count_dataset_examples(dataset_path, limit=limit)

    # Track initial count from checkpoint for accurate timing calculations
    initial_examples_count = len(score_rows) if checkpoint_loaded else 0

    # Start timing
    evaluation_start_time = time.time()

    print("=" * 80, file=sys.stderr)
    print("[info] Starting evaluation run", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"[info] Model: {model_cfg.model} ({model_label})", file=sys.stderr)
    print(f"[info] Prompt mode: {prompt_mode}", file=sys.stderr)
    print(f"[info] Judge: {judge_prompt.name} ({judge_inference_cfg.model})", file=sys.stderr)
    print(f"[info] Dataset: {dataset_path}", file=sys.stderr)
    print(f"[info] Total examples: {total_examples}", file=sys.stderr)
    if checkpoint_loaded and initial_examples_count > 0:
        print(
            f"[info] Resuming from checkpoint: {initial_examples_count} examples already evaluated",
            file=sys.stderr,
        )
        print(
            f"[info] Remaining examples: {total_examples - initial_examples_count}", file=sys.stderr
        )
    if limit:
        print(f"[info] Limit: {limit} examples", file=sys.stderr)
    print(f"[info] Max retries: {max_retries}, Retry delay: {retry_delay}s", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    # Detect model type once before the loop for logging purposes
    detected_model_type_for_logging: str | None = None
    if prompt_mode == "rag":
        # Only warn on inference during initial detection, not during loop
        detected_model_type_for_logging = _detect_model_type(
            model_config_path, model_cfg.model, model_cfg=model_cfg, warn_on_inference=True
        )
        print(f"[info] Detected model type: {detected_model_type_for_logging}", file=sys.stderr)
        if detected_model_type_for_logging == "instruction_only":
            print(
                "[info] Using hybrid RAG prompt format for instruction-only model (Condition 4). "
                "Prompt format matches training while including RAG contexts.",
                file=sys.stderr,
            )
    print("-" * 80, file=sys.stderr)
    if checkpoint_path:
        print(f"[info] Checkpoint path: {checkpoint_path}", file=sys.stderr)
        print(f"[info] Checkpoint interval: every {checkpoint_interval} examples", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    examples_processed = 0
    for idx, example in enumerate(_iter_dataset(dataset_path, limit=limit), start=1):
        task_id = str(example.get("task_id", f"line_{idx}"))

        # Skip if already evaluated (from checkpoint)
        if task_id in evaluated_task_ids:
            continue

        examples_processed += 1
        evaluated_task_ids.add(task_id)
        instruction = example.get("instruction", "")
        rag = example.get("rag", {})

        # Determine prompt based on mode
        if prompt_mode == "instruction":
            if not instruction:
                raise KeyError(
                    f"Example {task_id} missing instruction field for instruction-only mode"
                )
            # Check if dataset has contexts that will be ignored
            dataset_contexts = rag.get("contexts") or example.get("contexts") or []
            if dataset_contexts:
                print(
                    f"[info] Instruction-only mode: ignoring {len(dataset_contexts)} context(s) from dataset for example {task_id}",
                    file=sys.stderr,
                )
            # Detect model type for instruction-only prompt (warn only on first call)
            # Check if we already detected it (for RAG mode) to avoid duplicate warnings
            if prompt_mode == "instruction" and detected_model_type_for_logging is None:
                # First time detecting - warn if inferred
                _detect_model_type(
                    model_config_path, model_cfg.model, model_cfg=model_cfg, warn_on_inference=True
                )
            prompt_text = _build_instruction_only_prompt(
                instruction,
                model_config_path=model_config_path,
                model_name=model_cfg.model,
                model_cfg=model_cfg,
            )
            contexts_raw: list[Any] = []  # No contexts for instruction-only mode
        else:  # prompt_mode == "rag"
            contexts_raw = rag.get("contexts") or example.get("contexts") or []

            # Use pre-detected model type (detected before loop for consistent logging)
            # Fallback to detection if not pre-detected (shouldn't happen in normal flow)
            # Don't warn again during loop - warning already shown during initial detection
            detected_model_type = detected_model_type_for_logging or _detect_model_type(
                model_config_path, model_cfg.model, model_cfg=model_cfg, warn_on_inference=False
            )

            if detected_model_type == "instruction_only":
                # Condition 4: Instruction-only model evaluated WITH RAG contexts
                # Use hybrid format that matches training format while including contexts
                if not instruction:
                    raise KeyError(f"Example {task_id} missing instruction field for RAG mode")
                if not contexts_raw:
                    print(
                        f"[warn] RAG mode enabled for instruction-only model but example {task_id} has no contexts. "
                        "Falling back to instruction-only prompt format.",
                        file=sys.stderr,
                    )
                    prompt_text = _build_instruction_only_prompt(
                        instruction,
                        model_config_path=model_config_path,
                        model_name=model_cfg.model,
                        model_cfg=model_cfg,
                    )
                else:
                    # Normalize contexts to numbered format if needed
                    contexts_numbered = normalize_contexts(contexts_raw)
                    prompt_text = _build_rag_prompt_for_instruction_only_model(
                        instruction,
                        contexts_numbered,
                        model_config_path=model_config_path,
                        model_name=model_cfg.model,
                    )
            else:
                # Standard RAG mode: use pre-built RAG prompt from dataset
                prompt = rag.get("prompt") or example.get("rag_prompt")
                if not prompt:
                    raise KeyError(f"Example {task_id} missing prompt field for RAG mode")
                prompt_text = str(prompt)

                # Validate: RAG prompt should not include system text (Modelfile provides it)
                # This prevents distribution shift and system message duplication
                prompt_lower = prompt_text.lower()
                system_text_indicators = [
                    "you are a research paper assistant",
                    "you are a scientific research assistant",
                    "you are an assistant",
                ]
                for indicator in system_text_indicators:
                    if indicator in prompt_lower:
                        print(
                            f"[warn] RAG prompt for example {task_id} contains system text '{indicator}'. "
                            "System message is provided separately via Modelfile, so user content should not duplicate it. "
                            "This may cause distribution shift between training and evaluation.",
                            file=sys.stderr,
                        )
                        break  # Only warn once per prompt

                # Warn if RAG mode but no contexts retrieved
                if not contexts_raw:
                    print(
                        f"[warn] RAG mode enabled but example {task_id} has no contexts. "
                        "Citation metrics may not be meaningful.",
                        file=sys.stderr,
                    )

        record_errors: dict[str, str] = {}
        error_categories: dict[str, str] = {}

        # Time generation
        generation_start = time.time()
        model_response, generation_error, generation_error_category = _call_with_retries(
            partial(model_client.generate, prompt_text),
            stage=f"generation (task {task_id or idx})",
            max_retries=max(max_retries, 0),
            retry_delay=retry_delay,
        )
        generation_time = time.time() - generation_start
        assistant_answer = ""
        if model_response is not None:
            assistant_answer = str(model_response.get("response", "")).strip()
        if generation_error:
            record_errors["generation"] = generation_error
            if generation_error_category:
                error_categories["generation"] = generation_error_category

        # Track empty responses
        has_empty_response = not assistant_answer.strip()
        if has_empty_response and not generation_error:
            record_errors.setdefault("generation", "Empty response received")
            error_categories.setdefault("generation", "validation")

        contexts_numbered = normalize_contexts(contexts_raw)
        # has_contexts should be False for instruction mode, True for RAG mode only if contexts exist
        has_contexts = prompt_mode == "rag" and len(contexts_raw) > 0

        # For instruction-only mode, citation metrics are not meaningful
        # We still compute them but mark them as N/A
        citation_metrics = evaluate_citations(assistant_answer, contexts_numbered)
        if prompt_mode == "instruction":
            # Validate: instruction-only mode should not have contexts
            if contexts_raw:
                print(
                    f"[warn] Instruction-only mode but example {task_id or idx} has contexts. "
                    "This may indicate a configuration error. Citation metrics will be marked as N/A.",
                    file=sys.stderr,
                )
            # Mark citation metrics as not applicable for instruction-only mode
            citation_metrics = {
                **citation_metrics,
                "mode": "instruction_only",
                "note": "Citation metrics not applicable - no contexts provided",
                "validated": True,  # Flag to prevent accidental aggregation
            }
        else:
            # RAG mode: validate that contexts exist
            if not contexts_raw:
                print(
                    f"[warn] RAG mode but example {task_id or idx} has no contexts. "
                    "Citation metrics may not be meaningful.",
                    file=sys.stderr,
                )
            citation_metrics["mode"] = "rag"
            citation_metrics["validated"] = True

        judge_prompt_text = render_judge_prompt(
            judge_prompt.prompt,
            instruction,
            contexts_numbered,
            assistant_answer,
            has_contexts=has_contexts,
        )
        judge_response = None
        judge_payload: dict[str, Any] = {}
        judge_time = 0.0

        if generation_error:
            record_errors.setdefault("judge", "Skipped due to generation failure")
            error_categories.setdefault("judge", "validation")
        else:
            # Time judging
            judge_start = time.time()
            judge_response, judge_error, judge_error_category = _call_with_retries(
                partial(judge_client.generate, judge_prompt_text),
                stage=f"judging (task {task_id or idx})",
                max_retries=max(max_retries, 0),
                retry_delay=retry_delay,
            )
            judge_time = time.time() - judge_start
            if judge_error:
                record_errors["judge"] = judge_error
                if judge_error_category:
                    error_categories["judge"] = judge_error_category
            if judge_response is not None:
                judge_payload = parse_judge_output(str(judge_response.get("response", "")))

        judge_scores = judge_payload.get("scores") if isinstance(judge_payload, dict) else {}

        row_data: dict[str, Any] = {
            "task_id": task_id,
            "task_type": example.get("task_type"),
            "model_answer": assistant_answer,
            "judge_scores": judge_scores if isinstance(judge_scores, dict) else {},
            "judge_verdict": judge_payload.get("verdict"),
            "citation_metrics": citation_metrics,
            "model_label": model_label,
            "timing": {
                "generation_seconds": generation_time,
                "judge_seconds": judge_time,
                "total_seconds": generation_time + judge_time,
            },
        }

        # Add error information if present
        if record_errors:
            row_data["errors"] = record_errors
            if error_categories:
                row_data["error_categories"] = error_categories

        score_rows.append(row_data)

        # Save checkpoint periodically
        if (
            checkpoint_path
            and examples_processed > 0
            and examples_processed % checkpoint_interval == 0
        ):
            _save_checkpoint(
                checkpoint_path,
                score_rows,
                evaluated_task_ids,
                model_label=model_label,
                dataset_path=dataset_path,
            )
            print(
                f"[info] Checkpoint saved: {len(score_rows)} examples evaluated",
                file=sys.stderr,
            )

        # Progress reporting every 10 examples or at milestones
        # Use total examples evaluated (len(score_rows)) for consistent reporting
        total_evaluated = len(score_rows)
        if total_evaluated % 10 == 0 or total_evaluated == total_examples:
            error_count = sum(1 for row in score_rows if row.get("errors"))
            elapsed = time.time() - evaluation_start_time
            # Calculate average time per example based on NEW examples processed in this run
            # This gives accurate timing when resuming from checkpoint
            if examples_processed > 0:
                avg_time_per_example = elapsed / examples_processed
                remaining_examples = total_examples - total_evaluated
                remaining = remaining_examples * avg_time_per_example
            else:
                # No new examples processed yet (shouldn't happen, but handle gracefully)
                avg_time_per_example = 0.0
                remaining = 0.0

            # Show checkpoint info if resuming
            checkpoint_info = (
                f" (resumed from {initial_examples_count})"
                if checkpoint_loaded and initial_examples_count > 0
                else ""
            )
            print(
                f"[info] Progress: {total_evaluated}/{total_examples} examples evaluated{checkpoint_info} "
                f"({error_count} with errors) | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Avg: {avg_time_per_example:.2f}s/example | "
                f"ETA: {remaining/60:.1f}min",
                file=sys.stderr,
            )

    evaluation_time = time.time() - evaluation_start_time

    print("-" * 80, file=sys.stderr)
    print(
        f"[info] Evaluation completed in {evaluation_time:.2f}s ({evaluation_time/60:.1f} minutes)",
        file=sys.stderr,
    )
    print(f"[info] Processed {len(score_rows)} examples", file=sys.stderr)

    summary = summarise_scores(score_rows)

    # Warn if citation_mean_coverage is computed for instruction-only mode
    if prompt_mode == "instruction" and summary.get("citation_mean_coverage", 0.0) > 0.0:
        print(
            "[warn] citation_mean_coverage > 0 detected for instruction-only mode. "
            "Citation metrics should not be aggregated for instruction-only evaluations. "
            "This may indicate a bug in the evaluation pipeline.",
            file=sys.stderr,
        )

    # Compute timing statistics
    timing_stats = _compute_timing_stats(score_rows)

    print(
        f"[info] Average generation time: {timing_stats.get('mean_generation_seconds', 0):.2f}s",
        file=sys.stderr,
    )
    print(
        f"[info] Average judge time: {timing_stats.get('mean_judge_seconds', 0):.2f}s",
        file=sys.stderr,
    )
    print(
        f"[info] Average total time per example: {timing_stats.get('mean_total_seconds', 0):.2f}s",
        file=sys.stderr,
    )

    summary.update(
        {
            "model_label": model_label,
            "dataset": str(dataset_path),
            "examples_evaluated": len(score_rows),
            "judge_name": judge_prompt.name,
            "judge_model": judge_inference_cfg.model,
            "examples_with_errors": sum(1 for row in score_rows if row.get("errors")),
            "examples_with_empty_responses": sum(
                1
                for row in score_rows
                if not str(row.get("model_answer", "")).strip()
                and (not row.get("errors") or "generation" not in row.get("errors", {}))
            ),
            "error_rate": (
                sum(1 for row in score_rows if row.get("errors")) / len(score_rows)
                if score_rows
                else 0.0
            ),
            "prompt_mode": prompt_mode,
            "evaluated_task_ids": sorted(evaluated_task_ids),
            "timing": timing_stats,
        }
    )

    # Add checkpoint resumption info if applicable
    if checkpoint_loaded and initial_examples_count > 0:
        summary["checkpoint_resumed"] = True
        summary["checkpoint_examples_count"] = initial_examples_count
        summary["new_examples_processed"] = examples_processed
    else:
        summary["checkpoint_resumed"] = False

    # Warn if error rate is high
    error_rate = summary.get("error_rate", 0.0)
    if error_rate > 0.1:  # More than 10% errors
        print(
            f"[warn] High error rate detected: {error_rate:.1%} of examples had errors. "
            "Consider reviewing error logs.",
            file=sys.stderr,
        )

    metrics_path: Path | None = None
    if output_path:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {"summary": summary, "examples": score_rows}
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        metrics_path = output_path

    details_path_resolved: Path | None = None
    if details_output_path:
        details_path = details_output_path.resolve()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for row in score_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        details_path_resolved = details_path

    metadata_path = _resolve_metadata_path(
        metadata_output_path,
        metrics_path,
        model_label,
    )

    append_experiment_record(
        metadata_path,
        project_config=project_config,
        dataset_path=dataset_path,
        model_config=model_cfg,
        judge_config=judge_inference_cfg,
        metrics=summary,
        score_rows=score_rows,
        model_label=model_label,
        config_path=config_path,
        model_config_path=model_config_path,
        judge_prompt_path=judge_prompt_path.resolve(),
        judge_inference_path=judge_inference_path.resolve() if judge_inference_path else None,
        details_path=details_path_resolved,
        metrics_path=metrics_path,
    )

    # Save final checkpoint and clean up if successful
    if checkpoint_path:
        _save_checkpoint(
            checkpoint_path,
            score_rows,
            evaluated_task_ids,
            model_label=model_label,
            dataset_path=dataset_path,
        )
        # Clean up checkpoint file if evaluation completed successfully
        # (keep it if there were errors so user can resume)
        if metrics_path and metrics_path.exists() and not summary.get("error_rate", 0.0) > 0.1:
            try:
                checkpoint_path.unlink()
                print(
                    "[info] Checkpoint file cleaned up (evaluation completed successfully)",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[warn] Failed to clean up checkpoint file: {exc}", file=sys.stderr)

    # Run final evaluation sanity check after writing files
    if validate:
        final_sanity_check = validate_evaluation_sanity(
            metrics_path=metrics_path,
            details_path=details_path_resolved,
            max_error_rate=0.1,
            min_examples=1,
        )
        if final_sanity_check.issues:
            print_validation_result(final_sanity_check)

    return EvaluationResult(
        summary=summary,
        score_rows=score_rows,
        metrics_path=metrics_path,
        details_path=details_path_resolved,
        metadata_path=metadata_path,
    )


def _validate_dataset_structure(
    dataset_path: Path, prompt_mode: str, *, limit: int | None = None
) -> None:
    """Validate that dataset has required fields for the given prompt_mode.

    Also performs pre-flight checks to warn about potential configuration issues,
    such as using instruction-only mode with a dataset containing RAG contexts.
    """
    issues: list[str] = []
    warnings: list[str] = []
    sample_count = 0
    instruction_mode_with_contexts_count = 0
    rag_mode_without_contexts_count = 0

    for idx, example in enumerate(_iter_dataset(dataset_path, limit=limit), start=1):
        sample_count += 1
        task_id = example.get("task_id", f"line_{idx}")

        if prompt_mode == "instruction":
            if not example.get("instruction"):
                issues.append(
                    f"Example {task_id} missing 'instruction' field (required for instruction mode)"
                )
            # Pre-flight check: warn if dataset has RAG contexts that will be ignored
            rag = example.get("rag", {})
            contexts = rag.get("contexts") or example.get("contexts") or []
            if contexts:
                instruction_mode_with_contexts_count += 1
        else:  # prompt_mode == "rag"
            rag = example.get("rag", {})
            if not rag.get("prompt") and not example.get("rag_prompt"):
                issues.append(
                    f"Example {task_id} missing 'rag.prompt' or 'rag_prompt' field (required for RAG mode)"
                )
            # Pre-flight check: warn if RAG mode but no contexts available
            contexts = rag.get("contexts") or example.get("contexts") or []
            if not contexts:
                rag_mode_without_contexts_count += 1

    # Report warnings for potential configuration issues
    if prompt_mode == "instruction" and instruction_mode_with_contexts_count > 0:
        warnings.append(
            f"Pre-flight check: {instruction_mode_with_contexts_count}/{sample_count} examples have RAG contexts "
            "that will be ignored in instruction-only mode. This may indicate a configuration mismatch. "
            "Consider using 'rag' prompt_mode if you want to use the contexts."
        )

    if prompt_mode == "rag" and rag_mode_without_contexts_count > 0:
        warnings.append(
            f"Pre-flight check: {rag_mode_without_contexts_count}/{sample_count} examples have no RAG contexts "
            "in RAG mode. Citation metrics may not be meaningful for these examples."
        )

    # Print warnings (non-fatal)
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)

    # Raise errors for critical issues
    if issues:
        error_msg = f"Dataset validation failed for {prompt_mode} mode:\n" + "\n".join(issues[:10])
        if len(issues) > 10:
            error_msg += f"\n... and {len(issues) - 10} more issues"
        raise ValueError(error_msg)

    if sample_count == 0:
        raise ValueError(f"Dataset {dataset_path} appears to be empty")


def _compute_timing_stats(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute timing statistics from score rows."""
    generation_times: list[float] = []
    judge_times: list[float] = []
    total_times: list[float] = []

    for row in score_rows:
        timing = row.get("timing", {})
        if isinstance(timing, dict):
            gen_time = timing.get("generation_seconds")
            judge_time = timing.get("judge_seconds")
            total_time = timing.get("total_seconds")

            if isinstance(gen_time, int | float):
                generation_times.append(float(gen_time))
            if isinstance(judge_time, int | float):
                judge_times.append(float(judge_time))
            if isinstance(total_time, int | float):
                total_times.append(float(total_time))

    stats: dict[str, Any] = {}
    if generation_times:
        stats["generation"] = {
            "mean_seconds": mean(generation_times),
            "min_seconds": min(generation_times),
            "max_seconds": max(generation_times),
            "total_seconds": sum(generation_times),
        }
    if judge_times:
        stats["judge"] = {
            "mean_seconds": mean(judge_times),
            "min_seconds": min(judge_times),
            "max_seconds": max(judge_times),
            "total_seconds": sum(judge_times),
        }
    if total_times:
        stats["total"] = {
            "mean_seconds": mean(total_times),
            "min_seconds": min(total_times),
            "max_seconds": max(total_times),
            "total_seconds": sum(total_times),
        }

    return stats


def _count_dataset_examples(dataset_path: Path, *, limit: int | None = None) -> int:
    """Count total examples in dataset (for progress reporting)."""
    count = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                count += 1
    return count


def _iter_dataset(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            payload = line.strip()
            if not payload:
                continue
            yield json.loads(payload)


def _resolve_metadata_path(
    requested: Path | None,
    metrics_path: Path | None,
    model_label: str,
) -> Path:
    if requested:
        metadata_path = requested.resolve()
    elif metrics_path is not None:
        metadata_path = metrics_path.with_name("metadata.jsonl")
    else:
        metadata_path = Path("evaluation/results") / model_label / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    return metadata_path
