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

    # Check 2: Instruction-only mode format consistency
    if mode == "instruction":
        # Expected format: System message is separate, user content should NOT duplicate it
        expected_question_marker = "Question:"
        expected_answer_marker = "Answer:"
        system_text_markers = [
            "you are a research paper assistant",
            "you are a scientific research assistant",
        ]

        evaluation_lower = evaluation_prompt.lower()
        has_question = expected_question_marker.lower() in evaluation_lower
        has_answer = expected_answer_marker.lower() in evaluation_lower
        has_system_duplication = any(marker in evaluation_lower for marker in system_text_markers)

        # Check for system text duplication (should NOT be present)
        if has_system_duplication:
            issues.append(
                f"Evaluation prompt contains system text duplication.\n"
                f"  System message is provided separately (via Modelfile), so user content should not duplicate it.\n"
                f"  User content should only contain: 'Question: ...\\n\\nAnswer:'\n"
                f"  Current prompt: {evaluation_prompt[:100]!r}..."
            )
        elif not has_question or not has_answer:
            issues.append(
                "Evaluation prompt missing required markers.\n"
                "  Expected format: 'Question: ...\\n\\nAnswer:'\n"
                "  System message is provided separately, user content should be simplified."
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


def validate_rag_prompt_no_system_text(rag_prompt: str) -> list[str]:
    """Validate that a RAG prompt does not contain system text.

    RAG prompts stored in the dataset should contain only user content, as the system
    message is provided separately via Modelfile during evaluation. Including system
    text in the prompt causes duplication and potential distribution shift.

    Args:
        rag_prompt: The RAG prompt string to validate

    Returns:
        List of validation issues (empty if valid)
    """
    issues: list[str] = []

    prompt_lower = rag_prompt.lower()
    system_text_indicators = [
        "you are a research paper assistant",
        "you are a scientific research assistant",
        "you are an assistant",
    ]

    for indicator in system_text_indicators:
        if indicator in prompt_lower:
            issues.append(
                f"RAG prompt contains system text '{indicator}'. "
                "System message is provided separately via Modelfile, so user content should not duplicate it. "
                "This may cause distribution shift between training and evaluation."
            )
            break  # Only report once per prompt

    return issues


__all__ = [
    "validate_prompt_format_consistency",
    "validate_modelfile_system_message",
    "validate_rag_prompt_no_system_text",
]
