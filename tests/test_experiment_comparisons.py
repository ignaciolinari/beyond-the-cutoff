"""Tests for scientifically valid experiment comparisons.

This module tests:
1. Comparison configuration loading and validation
2. Within-group comparison logic (same eval mode)
3. Cross-group comparison restrictions (different eval modes)
4. Metrics extraction and comparison computation
5. Report generation
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "validation"))

from validate_experiment_comparisons import (  # type: ignore[import-not-found]
    ComparisonResult,
    ModelCondition,
    compute_comparison,
    extract_metrics,
    generate_comparison_summary,
    generate_full_report,
    generate_within_group_table,
    validate_experiment,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_instruction_metrics() -> dict[str, float]:
    """Sample metrics for instruction-mode evaluation."""
    return {
        "factuality": 7.5,
        "completeness": 6.8,
        "communication": 7.2,
        "overall": 7.2,  # Weighted average
    }


@pytest.fixture
def sample_rag_metrics() -> dict[str, float]:
    """Sample metrics for RAG-mode evaluation."""
    return {
        "factuality": 8.0,
        "completeness": 7.5,
        "communication": 7.0,
        "grounding": 8.5,
        "overall": 7.8,  # Different weighting
    }


@pytest.fixture
def sample_conditions() -> dict[str, ModelCondition]:
    """Sample conditions for testing."""
    return {
        "base_baseline": ModelCondition(
            name="base_baseline",
            result_dir=Path("evaluation/results/base_baseline_0p5b"),
            eval_mode="instruction",
            training_mode="none",
            description="Base model without FT or RAG",
            metrics={
                "factuality": 5.0,
                "completeness": 4.5,
                "communication": 5.5,
                "weighted_total": 5.0,
            },
        ),
        "rag_baseline": ModelCondition(
            name="rag_baseline",
            result_dir=Path("evaluation/results/rag_baseline_0p5b"),
            eval_mode="rag",
            training_mode="none",
            description="Base model with RAG",
            metrics={
                "factuality": 7.5,
                "completeness": 7.0,
                "communication": 6.5,
                "grounding": 8.0,
                "weighted_total": 7.3,
            },
        ),
        "ft_only_instruction": ModelCondition(
            name="ft_only_instruction",
            result_dir=Path("evaluation/results/lora_science_0p5b_ft_only"),
            eval_mode="instruction",
            training_mode="instruction",
            description="Instruction-trained model without RAG",
            metrics={
                "factuality": 6.5,
                "completeness": 6.0,
                "communication": 6.5,
                "weighted_total": 6.3,
            },
        ),
        "ft_rag_instruction": ModelCondition(
            name="ft_rag_instruction",
            result_dir=Path("evaluation/results/hybrid_science_0p5b_instruction_only"),
            eval_mode="rag",
            training_mode="instruction",
            description="Instruction-trained model with RAG",
            metrics={
                "factuality": 7.8,
                "completeness": 7.2,
                "communication": 6.8,
                "grounding": 6.5,
                "weighted_total": 7.1,
            },
        ),
        "ft_only_rag_trained": ModelCondition(
            name="ft_only_rag_trained",
            result_dir=Path("evaluation/results/lora_science_0p5b_rag_trained_ft_only"),
            eval_mode="instruction",
            training_mode="rag",
            description="RAG-trained model without RAG",
            metrics={
                "factuality": 5.5,
                "completeness": 5.0,
                "communication": 5.8,
                "weighted_total": 5.4,
            },
        ),
        "ft_rag_trained": ModelCondition(
            name="ft_rag_trained",
            result_dir=Path("evaluation/results/hybrid_science_0p5b_rag_trained"),
            eval_mode="rag",
            training_mode="rag",
            description="RAG-trained model with RAG (optimal)",
            metrics={
                "factuality": 8.5,
                "completeness": 8.0,
                "communication": 7.5,
                "grounding": 9.0,
                "weighted_total": 8.3,
            },
        ),
    }


@pytest.fixture
def temp_results_dir(
    sample_instruction_metrics: dict[str, float], sample_rag_metrics: dict[str, float]
) -> Generator[Path, None, None]:
    """Create temporary results directory with sample metrics files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create instruction-mode results
        instr_dir = tmppath / "evaluation/results/base_baseline_0p5b"
        instr_dir.mkdir(parents=True)
        with open(instr_dir / "metrics.json", "w") as f:
            json.dump(sample_instruction_metrics, f)

        # Create RAG-mode results
        rag_dir = tmppath / "evaluation/results/rag_baseline_0p5b"
        rag_dir.mkdir(parents=True)
        with open(rag_dir / "metrics.json", "w") as f:
            json.dump(sample_rag_metrics, f)

        yield tmppath


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "models": {
            "base_baseline": {
                "result_dir": "evaluation/results/base_baseline_0p5b",
                "eval_mode": "instruction",
                "training_mode": "none",
                "description": "Base model",
            },
            "rag_baseline": {
                "result_dir": "evaluation/results/rag_baseline_0p5b",
                "eval_mode": "rag",
                "training_mode": "none",
                "description": "Base model with RAG",
            },
        },
        "within_group_comparisons": {
            "instruction_group": {
                "comparisons": [
                    {
                        "name": "test_comparison",
                        "model_a": "base_baseline",
                        "model_b": "ft_only_instruction",
                        "research_question": "Test question",
                    }
                ]
            }
        },
        "cross_group_comparisons": {
            "comparisons": [
                {
                    "name": "cross_test",
                    "model_a": "base_baseline",
                    "model_b": "rag_baseline",
                    "research_question": "Cross group test",
                    "warning": "Do not compare weighted totals",
                }
            ]
        },
    }


# =============================================================================
# Tests: Metrics Extraction
# =============================================================================


class TestMetricsExtraction:
    """Tests for metrics extraction from result files."""

    def test_extract_instruction_metrics(
        self, temp_results_dir: Path, sample_instruction_metrics: dict[str, float]
    ) -> None:
        """Test extracting metrics from instruction-mode results."""
        result_dir = temp_results_dir / "evaluation/results/base_baseline_0p5b"
        metrics = extract_metrics(result_dir, "instruction")

        assert "factuality" in metrics
        assert "completeness" in metrics
        assert "communication" in metrics
        assert metrics["factuality"] == sample_instruction_metrics["factuality"]

    def test_extract_rag_metrics(
        self, temp_results_dir: Path, sample_rag_metrics: dict[str, float]
    ) -> None:
        """Test extracting metrics from RAG-mode results."""
        result_dir = temp_results_dir / "evaluation/results/rag_baseline_0p5b"
        metrics = extract_metrics(result_dir, "rag")

        assert "factuality" in metrics
        assert "grounding" in metrics
        assert metrics["grounding"] == sample_rag_metrics["grounding"]

    def test_extract_metrics_missing_file(self, temp_results_dir: Path) -> None:
        """Test that missing metrics file returns empty dict."""
        result_dir = temp_results_dir / "evaluation/results/nonexistent"
        metrics = extract_metrics(result_dir, "instruction")

        assert metrics == {}

    def test_extract_metrics_handles_nested_dimensions(self, temp_results_dir: Path) -> None:
        """Test extracting metrics from nested 'dimensions' structure."""
        result_dir = temp_results_dir / "evaluation/results/nested_metrics"
        result_dir.mkdir(parents=True)

        nested_metrics = {
            "dimensions": {
                "factuality": 7.0,
                "completeness": 6.5,
                "communication": 7.2,
            },
            "overall": 6.9,
        }

        with open(result_dir / "metrics.json", "w") as f:
            json.dump(nested_metrics, f)

        metrics = extract_metrics(result_dir, "instruction")

        assert metrics["factuality"] == 7.0
        assert metrics["weighted_total"] == 6.9


# =============================================================================
# Tests: Within-Group Comparisons
# =============================================================================


class TestWithinGroupComparisons:
    """Tests for within-group comparison logic."""

    def test_instruction_group_comparison_includes_weighted_total(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Within instruction group, weighted_total should be valid."""
        model_a = sample_conditions["base_baseline"]
        model_b = sample_conditions["ft_only_instruction"]

        comparison_config: dict[str, Any] = {
            "name": "test_within_instruction",
            "research_question": "Does FT help?",
        }

        result = compute_comparison(model_a, model_b, comparison_config, "within-group")

        assert "weighted_total" in result.valid_metrics
        assert result.comparison_type == "within-group"
        assert "weighted_total" not in result.invalid_metrics

    def test_rag_group_comparison_includes_grounding(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Within RAG group, grounding should be valid."""
        model_a = sample_conditions["rag_baseline"]
        model_b = sample_conditions["ft_rag_trained"]

        comparison_config: dict[str, Any] = {
            "name": "test_within_rag",
            "research_question": "Does FT improve RAG?",
        }

        result = compute_comparison(model_a, model_b, comparison_config, "within-group")

        assert "grounding" in result.valid_metrics
        assert "weighted_total" in result.valid_metrics

    def test_within_group_metric_differences_computed(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Test that metric differences are correctly computed."""
        model_a = sample_conditions["base_baseline"]  # factuality: 5.0
        model_b = sample_conditions["ft_only_instruction"]  # factuality: 6.5

        comparison_config: dict[str, Any] = {"name": "test"}

        result = compute_comparison(model_a, model_b, comparison_config, "within-group")

        # model_b - model_a = 6.5 - 5.0 = 1.5
        assert result.metric_differences["factuality"] == pytest.approx(1.5)

    def test_within_group_determines_winner(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Test that winner is determined based on factuality."""
        model_a = sample_conditions["base_baseline"]  # factuality: 5.0
        model_b = sample_conditions["ft_only_instruction"]  # factuality: 6.5

        comparison_config: dict[str, Any] = {"name": "test"}

        result = compute_comparison(model_a, model_b, comparison_config, "within-group")

        # 6.5 - 5.0 = 1.5 > 0.1, so model_b wins
        assert result.winner == "ft_only_instruction"


# =============================================================================
# Tests: Cross-Group Comparisons
# =============================================================================


class TestCrossGroupComparisons:
    """Tests for cross-group comparison restrictions."""

    def test_cross_group_excludes_weighted_total(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Cross-group comparisons should NOT include weighted_total."""
        model_a = sample_conditions["ft_only_instruction"]  # instruction mode
        model_b = sample_conditions["rag_baseline"]  # rag mode

        comparison_config: dict[str, Any] = {
            "name": "cross_test",
            "metrics_to_compare": ["factuality", "completeness", "communication"],
            "warning": "Do not compare weighted totals",
        }

        result = compute_comparison(model_a, model_b, comparison_config, "cross-group")

        assert "weighted_total" in result.invalid_metrics
        assert "weighted_total" not in result.valid_metrics

    def test_cross_group_excludes_grounding(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Cross-group comparisons should NOT include grounding."""
        model_a = sample_conditions["base_baseline"]  # instruction mode
        model_b = sample_conditions["ft_rag_trained"]  # rag mode

        comparison_config: dict[str, Any] = {
            "name": "cross_test",
            "metrics_to_compare": ["factuality", "completeness", "communication"],
        }

        result = compute_comparison(model_a, model_b, comparison_config, "cross-group")

        assert "grounding" in result.invalid_metrics

    def test_cross_group_only_raw_dimensions_valid(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Cross-group should only allow raw dimension scores."""
        model_a = sample_conditions["ft_only_instruction"]
        model_b = sample_conditions["ft_rag_instruction"]

        comparison_config: dict[str, Any] = {
            "name": "cross_test",
            "metrics_to_compare": ["factuality", "completeness", "communication"],
        }

        result = compute_comparison(model_a, model_b, comparison_config, "cross-group")

        assert set(result.valid_metrics) == {"factuality", "completeness", "communication"}

    def test_cross_group_preserves_warning(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Cross-group comparison should preserve warning from config."""
        model_a = sample_conditions["base_baseline"]
        model_b = sample_conditions["rag_baseline"]

        comparison_config: dict[str, Any] = {
            "name": "cross_test",
            "warning": "Do NOT compare weighted totals - different judge weights",
        }

        result = compute_comparison(model_a, model_b, comparison_config, "cross-group")

        assert result.warning == "Do NOT compare weighted totals - different judge weights"

    def test_cross_group_same_eval_mode_allows_all_metrics(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """If both models have same eval_mode, it's not really cross-group."""
        # This tests a special case in the config where both use RAG
        model_a = sample_conditions["rag_baseline"]
        model_b = sample_conditions["ft_rag_trained"]

        # Even though config says cross-group, same eval_mode should work
        comparison_config: dict[str, Any] = {
            "name": "same_eval_mode",
            "metrics_to_compare": [
                "factuality",
                "completeness",
                "communication",
                "grounding",
                "weighted_total",
            ],
            "warning": None,  # No warning needed - same eval mode
        }

        # Using within-group because same eval mode
        result = compute_comparison(model_a, model_b, comparison_config, "within-group")

        assert "weighted_total" in result.valid_metrics
        assert "grounding" in result.valid_metrics


# =============================================================================
# Tests: Scientific Validity
# =============================================================================


class TestScientificValidity:
    """Tests ensuring comparisons follow scientific validity rules."""

    def test_instruction_vs_rag_cannot_compare_totals(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Instruction mode vs RAG mode cannot compare weighted totals."""
        # Condition 3 (instruction) vs Condition 2 (RAG)
        model_a = sample_conditions["ft_only_instruction"]
        model_b = sample_conditions["rag_baseline"]

        result = compute_comparison(model_a, model_b, {"name": "ft_vs_rag"}, "cross-group")

        # Should explicitly mark weighted_total as invalid
        assert "weighted_total" in result.invalid_metrics
        assert "weighted_total" not in result.metric_differences

    def test_all_six_conditions_have_factuality(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """All conditions should have factuality metric (the common ground)."""
        for name, condition in sample_conditions.items():
            assert "factuality" in condition.metrics, f"{name} missing factuality"

    def test_only_rag_conditions_have_grounding(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Only RAG-mode conditions should have grounding metric."""
        for name, condition in sample_conditions.items():
            if condition.eval_mode == "rag":
                assert "grounding" in condition.metrics, f"{name} (RAG) should have grounding"
            else:
                assert (
                    "grounding" not in condition.metrics
                ), f"{name} (instruction) should not have grounding"

    def test_comparison_identifies_evaluation_mode_mismatch(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Cross-group comparison should identify when eval modes differ."""
        model_a = sample_conditions["ft_only_instruction"]  # instruction
        model_b = sample_conditions["ft_rag_instruction"]  # rag

        # Different eval modes -> should be treated as cross-group
        assert model_a.eval_mode != model_b.eval_mode


# =============================================================================
# Tests: Validation
# =============================================================================


class TestValidation:
    """Tests for experiment setup validation."""

    def test_validation_detects_missing_results(self, sample_config: dict[str, Any]) -> None:
        """Validation should detect missing result directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validation, conditions = validate_experiment(sample_config, Path(tmpdir))

            assert not validation.all_results_exist
            assert len(validation.missing_results) > 0

    def test_validation_loads_existing_conditions(
        self, temp_results_dir: Path, sample_config: dict[str, Any]
    ) -> None:
        """Validation should load conditions when results exist."""
        validation, conditions = validate_experiment(sample_config, temp_results_dir)

        # Should load the two conditions that have results
        assert validation.conditions_loaded == 2
        assert "base_baseline" in conditions
        assert "rag_baseline" in conditions


# =============================================================================
# Tests: Report Generation
# =============================================================================


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_within_group_table_markdown(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Test markdown table generation."""
        conditions = [
            sample_conditions["base_baseline"],
            sample_conditions["ft_only_instruction"],
        ]

        table = generate_within_group_table(
            "Test Group", conditions, ["factuality", "completeness"]
        )

        assert "### Test Group" in table
        assert "base_baseline" in table
        assert "ft_only_instruction" in table
        assert "|" in table  # Has table formatting

    def test_generate_comparison_summary_includes_warnings(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Test that comparison summary includes warnings."""
        result = ComparisonResult(
            name="test_comparison",
            model_a="base_baseline",
            model_b="rag_baseline",
            research_question="Test question",
            comparison_type="cross-group",
            valid_metrics=["factuality"],
            invalid_metrics=["weighted_total"],
            metric_differences={"factuality": 2.5},
            winner="rag_baseline",
            warning="Do NOT compare weighted totals",
            expected="RAG should win",
        )

        summary = generate_comparison_summary([result])

        assert "⚠️" in summary
        assert "Do NOT compare weighted totals" in summary

    def test_generate_full_report_structure(
        self, sample_conditions: dict[str, ModelCondition]
    ) -> None:
        """Test full report has correct structure."""
        config: dict[str, Any] = {"models": {}}
        results = [
            ComparisonResult(
                name="test",
                model_a="a",
                model_b="b",
                research_question="q",
                comparison_type="within-group",
                valid_metrics=["factuality"],
                invalid_metrics=[],
                metric_differences={"factuality": 1.0},
                winner="b",
                warning=None,
                expected=None,
            )
        ]

        report = generate_full_report(config, sample_conditions, results)

        assert "experiment" in report
        assert "conditions" in report
        assert "comparisons" in report
        assert "summary" in report
        assert report["summary"]["total_comparisons"] == 1


# =============================================================================
# Tests: Configuration Loading
# =============================================================================


class TestConfigurationLoading:
    """Tests for loading comparison configuration."""

    def test_config_yaml_is_valid(self) -> None:
        """Test that the actual config file is valid YAML."""
        config_path = (
            Path(__file__).parent.parent / "configs/evaluation/six_condition_comparisons.yaml"
        )

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "models" in config
            assert "within_group_comparisons" in config
            assert "cross_group_comparisons" in config
            assert "pairwise_evaluation" in config

    def test_config_has_six_models(self) -> None:
        """Test that config defines all six conditions."""
        config_path = (
            Path(__file__).parent.parent / "configs/evaluation/six_condition_comparisons.yaml"
        )

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert len(config["models"]) == 6

    def test_config_within_group_has_correct_conditions(self) -> None:
        """Test that within-group comparisons only include same-mode conditions."""
        config_path = (
            Path(__file__).parent.parent / "configs/evaluation/six_condition_comparisons.yaml"
        )

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Instruction group should have instruction-mode conditions
            instruction_conditions = config["within_group_comparisons"]["instruction_group"][
                "conditions"
            ]
            assert "base_baseline" in instruction_conditions
            assert "ft_only_instruction" in instruction_conditions
            assert "ft_only_rag_trained" in instruction_conditions

            # RAG group should have RAG-mode conditions
            rag_conditions = config["within_group_comparisons"]["rag_group"]["conditions"]
            assert "rag_baseline" in rag_conditions
            assert "ft_rag_instruction" in rag_conditions
            assert "ft_rag_trained" in rag_conditions

    def test_cross_group_comparisons_have_warnings(self) -> None:
        """Test that cross-group comparisons include appropriate warnings."""
        config_path = (
            Path(__file__).parent.parent / "configs/evaluation/six_condition_comparisons.yaml"
        )

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            cross_comparisons = config["cross_group_comparisons"]["comparisons"]

            # Check that most cross-group comparisons have warnings
            warnings_count = sum(1 for c in cross_comparisons if c.get("warning"))

            # At least some should have warnings
            assert warnings_count >= 3


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_comparison_with_missing_metrics(self) -> None:
        """Test comparison when one model has missing metrics."""
        model_a = ModelCondition(
            name="partial_a",
            result_dir=Path("/tmp"),
            eval_mode="instruction",
            training_mode="none",
            description="Partial metrics",
            metrics={"factuality": 6.0},  # Missing completeness, communication
        )
        model_b = ModelCondition(
            name="partial_b",
            result_dir=Path("/tmp"),
            eval_mode="instruction",
            training_mode="none",
            description="Partial metrics",
            metrics={"factuality": 7.0, "completeness": 6.5},
        )

        result = compute_comparison(model_a, model_b, {"name": "test"}, "within-group")

        # Should only compute diff for metrics both have
        assert "factuality" in result.metric_differences
        assert "completeness" not in result.metric_differences  # a doesn't have it

    def test_comparison_tie_detection(self) -> None:
        """Test that ties are properly detected."""
        model_a = ModelCondition(
            name="tie_a",
            result_dir=Path("/tmp"),
            eval_mode="instruction",
            training_mode="none",
            description="Tie test",
            metrics={"factuality": 7.0},
        )
        model_b = ModelCondition(
            name="tie_b",
            result_dir=Path("/tmp"),
            eval_mode="instruction",
            training_mode="none",
            description="Tie test",
            metrics={"factuality": 7.05},  # Within 0.1 threshold
        )

        result = compute_comparison(model_a, model_b, {"name": "test"}, "within-group")

        assert result.winner == "tie"

    def test_negative_difference_determines_model_a_winner(self) -> None:
        """Test that model_a wins when it has higher score."""
        model_a = ModelCondition(
            name="winner_a",
            result_dir=Path("/tmp"),
            eval_mode="instruction",
            training_mode="none",
            description="Higher score",
            metrics={"factuality": 8.0},
        )
        model_b = ModelCondition(
            name="loser_b",
            result_dir=Path("/tmp"),
            eval_mode="instruction",
            training_mode="none",
            description="Lower score",
            metrics={"factuality": 6.0},
        )

        result = compute_comparison(model_a, model_b, {"name": "test"}, "within-group")

        # diff = 6.0 - 8.0 = -2.0 < -0.1, so model_a wins
        assert result.winner == "winner_a"
