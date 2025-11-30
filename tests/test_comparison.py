from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.evaluation.comparison import (
    build_comparison_report,
    describe_plan,
    execute_comparison_plan,
    load_comparison_plan,
)
from beyond_the_cutoff.evaluation.runner import EvaluationResult


def test_load_comparison_plan_defaults() -> None:
    plan_path = Path("configs/evaluation/six_condition_experiment.yaml")
    plan = load_comparison_plan(plan_path)
    assert plan.defaults.metrics_filename == "metrics.json"
    assert len(plan.runs) >= 2
    assert plan.runs[0].label


def test_describe_plan_includes_expected_paths() -> None:
    project_cfg = load_config()
    plan_path = Path("configs/evaluation/six_condition_experiment.yaml")
    plan = load_comparison_plan(plan_path)
    rows = describe_plan(plan, project_cfg)
    labels = {row["label"] for row in rows}
    # Check for all 6 conditions in the comparison plan
    expected_labels = {
        "base_baseline_0p5b",
        "rag_baseline_0p5b",
        "lora_science_0p5b_ft_only",
        "hybrid_science_0p5b_instruction_only",
        "lora_science_0p5b_rag_trained_ft_only",
        "hybrid_science_0p5b_rag_trained",
    }
    assert expected_labels.issubset(
        labels
    ), f"Missing labels. Found: {labels}, Expected: {expected_labels}"
    baseline = next(row for row in rows if row["label"] == "rag_baseline_0p5b")
    assert baseline["metrics_path"].endswith("evaluation/results/rag_baseline_0p5b/metrics.json")


def test_execute_comparison_plan_invokes_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"task_id": "t1", "instruction": "demo", "rag": {"prompt": "Q"}}\n')

    judge_config_path = repo_root / "configs" / "judges" / "rag.yaml"
    judge_inference_path = repo_root / "configs" / "judges" / "archive" / "ollama_qwen7b.yaml"
    model_config_path = repo_root / "configs" / "models" / "base_ollama.yaml"

    plan_content = f"""
defaults:
  dataset: dataset.jsonl
  judge_config: {judge_config_path}
  judge_inference: {judge_inference_path}
  output_dir: artifacts
  skip_if_exists: false
runs:
  - label: demo
    model_config: {model_config_path}
"""
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(plan_content)

    plan = load_comparison_plan(plan_path)
    project_cfg = load_config()

    call_args: list[dict[str, Any]] = []

    def _fake_run_evaluation(**kwargs: Any) -> EvaluationResult:
        call_args.append(kwargs)
        output_path = kwargs.get("output_path")
        metadata_output_path = kwargs.get("metadata_output_path")
        if metadata_output_path is None:
            if output_path is not None:
                metadata_output_path = output_path.with_name("metadata.jsonl")
            else:
                metadata_output_path = tmp_path / "meta" / f"{kwargs['model_label']}.jsonl"
        return EvaluationResult(
            summary={"model_label": kwargs["model_label"], "examples_evaluated": 1},
            score_rows=[{"task_id": "t1"}],
            metrics_path=output_path,
            details_path=kwargs.get("details_output_path"),
            metadata_path=metadata_output_path,
        )

    monkeypatch.setattr(
        "beyond_the_cutoff.evaluation.comparison.run_evaluation",
        _fake_run_evaluation,
    )

    results = execute_comparison_plan(
        plan,
        project_config=project_cfg,
        config_path=repo_root / "configs" / "default.yaml",
        force=True,
    )

    assert len(call_args) == 1
    invoked = call_args[0]
    assert invoked["dataset_path"] == dataset_path.resolve()
    assert invoked["model_label"] == "demo"
    assert results[0].summary["model_label"] == "demo"
    assert results[0].metrics_path == invoked["output_path"]

    report = build_comparison_report(results)
    assert report.as_dict()["runs"][0]["label"] == "demo"


def test_comparison_plan_uses_dual_judge_system() -> None:
    """Verify that the comparison plan uses appropriate judge configs for each condition."""
    plan_path = Path("configs/evaluation/six_condition_experiment.yaml")
    plan = load_comparison_plan(plan_path)

    # Check that default judge is RAG judge
    assert plan.defaults.judge_config is not None
    assert "rag.yaml" in str(plan.defaults.judge_config)

    # Check instruction-only conditions use instruction judge
    instruction_only_labels = {
        "base_baseline_0p5b",
        "lora_science_0p5b_ft_only",
        "lora_science_0p5b_rag_trained_ft_only",
    }
    for run in plan.runs:
        if run.label in instruction_only_labels:
            assert run.judge_config is not None
            assert "instruction.yaml" in str(
                run.judge_config
            ), f"Condition {run.label} should use instruction judge"
            assert (
                run.prompt_mode == "instruction"
            ), f"Condition {run.label} should use instruction prompt mode"

    # Check RAG conditions use RAG judge
    rag_labels = {
        "rag_baseline_0p5b",
        "hybrid_science_0p5b_instruction_only",
        "hybrid_science_0p5b_rag_trained",
    }
    for run in plan.runs:
        if run.label in rag_labels:
            assert run.judge_config is not None
            assert "rag.yaml" in str(
                run.judge_config
            ), f"Condition {run.label} should use RAG judge"
            assert run.prompt_mode == "rag", f"Condition {run.label} should use RAG prompt mode"
