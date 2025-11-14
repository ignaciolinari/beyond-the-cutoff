"""Utilities for validating prompt format consistency between training and evaluation."""

from __future__ import annotations

from pathlib import Path


def validate_prompt_format_consistency(
    training_system_message: str,
    modelfile_system_message: str | None,
    evaluation_prompt: str,
    *,
    mode: str = "instruction",
) -> list[str]:
    """Validate that prompt formats are consistent between training and evaluation.

    Args:
        training_system_message: System message used during training
        modelfile_system_message: System message from Ollama Modelfile (if using Ollama)
        evaluation_prompt: Prompt used during evaluation (user content only)
        mode: Either "instruction" or "rag"

    Returns:
        List of validation warnings/errors (empty if all checks pass)
    """
    issues: list[str] = []

    # Check 1: System message consistency
    if modelfile_system_message:
        if training_system_message.strip() != modelfile_system_message.strip():
            issues.append(
                f"System message mismatch:\n"
                f"  Training: {training_system_message!r}\n"
                f"  Modelfile: {modelfile_system_message!r}\n"
                f"  This can cause distribution shift between training and evaluation."
            )

    # Check 2: No system-like text in user content (for instruction mode)
    if mode == "instruction":
        system_indicators = [
            "You are a",
            "You are an",
            "You are the",
            "Act as",
            "Your role is",
        ]
        evaluation_lower = evaluation_prompt.lower()
        for indicator in system_indicators:
            if indicator.lower() in evaluation_lower:
                issues.append(
                    f"System-like text found in user content: {indicator!r}\n"
                    f"  This may cause duplication when Ollama applies Modelfile system message.\n"
                    f"  User content should only contain the question/instruction."
                )
                break  # Only report once

    # Check 3: Instruction-only prompts should be simple
    if mode == "instruction":
        if "Question:" not in evaluation_prompt and "Answer:" not in evaluation_prompt:
            issues.append(
                "Instruction-only prompt should follow format: 'Question: ...\\n\\nAnswer:'\n"
                "  This ensures consistency with training format."
            )

    return issues


def validate_modelfile_system_message(
    modelfile_path: Path,
    expected_system_message: str,
) -> list[str]:
    """Validate that a Modelfile contains the expected system message.

    Args:
        modelfile_path: Path to the Modelfile
        expected_system_message: System message that should be in the Modelfile

    Returns:
        List of validation issues (empty if valid)
    """
    issues: list[str] = []

    if not modelfile_path.exists():
        issues.append(f"Modelfile not found: {modelfile_path}")
        return issues

    content = modelfile_path.read_text(encoding="utf-8")

    # Extract SYSTEM line
    system_lines = [line for line in content.splitlines() if line.strip().startswith("SYSTEM")]

    if not system_lines:
        issues.append("Modelfile missing SYSTEM directive")
        return issues

    # Extract the system message (handle quoted strings)
    system_line = system_lines[0]
    if '"' in system_line:
        # Extract content between quotes
        start = system_line.find('"') + 1
        end = system_line.rfind('"')
        if end > start:
            actual_message = system_line[start:end]
        else:
            issues.append("Malformed SYSTEM directive in Modelfile")
            return issues
    else:
        # No quotes, take everything after SYSTEM
        parts = system_line.split("SYSTEM", 1)
        if len(parts) > 1:
            actual_message = parts[1].strip()
        else:
            issues.append("Malformed SYSTEM directive in Modelfile")
            return issues

    if actual_message.strip() != expected_system_message.strip():
        issues.append(
            f"Modelfile system message mismatch:\n"
            f"  Expected: {expected_system_message!r}\n"
            f"  Actual: {actual_message!r}\n"
            f"  Update the Modelfile to match training system message."
        )

    return issues


__all__ = [
    "validate_prompt_format_consistency",
    "validate_modelfile_system_message",
]
