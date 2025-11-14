"""Tests for prompt format validation utilities."""

from __future__ import annotations

from pathlib import Path

from beyond_the_cutoff.utils.prompt_validation import (
    validate_modelfile_system_message,
    validate_prompt_format_consistency,
)


def test_validate_prompt_format_consistency_instruction_mode_matches() -> None:
    """Test that validation passes when instruction-only format matches training."""
    training_system = "You are a research paper assistant."
    modelfile_system = "You are a research paper assistant."
    evaluation_prompt = (
        "You are a research paper assistant. Answer the following question based on your knowledge.\n\n"
        "Question: What is the main contribution?\n\nAnswer:"
    )

    issues = validate_prompt_format_consistency(
        training_system_message=training_system,
        modelfile_system_message=modelfile_system,
        evaluation_prompt=evaluation_prompt,
        mode="instruction",
    )

    assert issues == [], f"Expected no issues but got: {issues}"


def test_validate_prompt_format_consistency_instruction_mode_missing_prefix() -> None:
    """Test that validation fails when instruction-only format is missing training prefix."""
    training_system = "You are a research paper assistant."
    modelfile_system = "You are a research paper assistant."
    # Missing the training prefix - this is the old buggy format
    evaluation_prompt = "Question: What is the main contribution?\n\nAnswer:"

    issues = validate_prompt_format_consistency(
        training_system_message=training_system,
        modelfile_system_message=modelfile_system,
        evaluation_prompt=evaluation_prompt,
        mode="instruction",
    )

    assert len(issues) > 0, "Expected validation issues for missing training format prefix"
    assert any("missing expected training format prefix" in issue.lower() for issue in issues)


def test_validate_prompt_format_consistency_system_message_mismatch() -> None:
    """Test that validation fails when system messages don't match."""
    training_system = "You are a research paper assistant."
    modelfile_system = "You are a scientific assistant."  # Different!
    evaluation_prompt = (
        "You are a research paper assistant. Answer the following question based on your knowledge.\n\n"
        "Question: What is the main contribution?\n\nAnswer:"
    )

    issues = validate_prompt_format_consistency(
        training_system_message=training_system,
        modelfile_system_message=modelfile_system,
        evaluation_prompt=evaluation_prompt,
        mode="instruction",
    )

    assert len(issues) > 0, "Expected validation issues for system message mismatch"
    assert any("system message mismatch" in issue.lower() for issue in issues)


def test_validate_modelfile_system_message_matches(tmp_path: Path) -> None:
    """Test that modelfile validation passes when system message matches."""
    modelfile_path = tmp_path / "Modelfile"
    modelfile_path.write_text(
        'SYSTEM "You are a research paper assistant."\n' "PARAMETER temperature 0\n",
        encoding="utf-8",
    )

    issues = validate_modelfile_system_message(
        modelfile_path=modelfile_path,
        expected_system_message="You are a research paper assistant.",
    )

    assert issues == [], f"Expected no issues but got: {issues}"


def test_validate_modelfile_system_message_mismatch(tmp_path: Path) -> None:
    """Test that modelfile validation fails when system message doesn't match."""
    modelfile_path = tmp_path / "Modelfile"
    modelfile_path.write_text(
        'SYSTEM "You are a scientific assistant."\n' "PARAMETER temperature 0\n",
        encoding="utf-8",
    )

    issues = validate_modelfile_system_message(
        modelfile_path=modelfile_path,
        expected_system_message="You are a research paper assistant.",
    )

    assert len(issues) > 0, "Expected validation issues for system message mismatch"
    assert any("system message mismatch" in issue.lower() for issue in issues)


def test_validate_modelfile_missing_system(tmp_path: Path) -> None:
    """Test that modelfile validation fails when SYSTEM directive is missing."""
    modelfile_path = tmp_path / "Modelfile"
    modelfile_path.write_text(
        "PARAMETER temperature 0\n",
        encoding="utf-8",
    )

    issues = validate_modelfile_system_message(
        modelfile_path=modelfile_path,
        expected_system_message="You are a research paper assistant.",
    )

    assert len(issues) > 0, "Expected validation issues for missing SYSTEM directive"
    assert any("missing system directive" in issue.lower() for issue in issues)
