"""Tests for training/evaluation prompt format compatibility.

This test suite ensures that prompts used during evaluation match the formats
used during training, preventing distribution shift and ensuring valid experimental comparisons.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beyond_the_cutoff.evaluation.runner import (
    _build_instruction_only_prompt,
    _build_rag_prompt_for_instruction_only_model,
    _detect_model_type,
)


class TestInstructionOnlyPromptCompatibility:
    """Test that instruction-only evaluation prompts match training format."""

    def test_instruction_only_prompt_matches_training_format(self) -> None:
        """Test that instruction-only prompt format matches training format."""
        instruction = "What is the main contribution of this paper?"

        # This should match the training format from notebooks/finetuning/lora_science_v1_instruction_only.ipynb
        prompt = _build_instruction_only_prompt(
            instruction,
            model_type="instruction_only",
        )

        # Training format: System message is separate, user content is just question/answer
        # Should NOT include system text duplication
        assert "You are a research paper assistant" not in prompt
        assert "Question:" in prompt
        assert instruction in prompt
        assert "Answer:" in prompt

        # Should not include RAG-specific instructions
        assert "provided context" not in prompt.lower()
        assert "cite the sources" not in prompt.lower()

    def test_instruction_only_prompt_for_base_model(self) -> None:
        """Test that base model gets appropriate instruction-only prompt."""
        instruction = "What is machine learning?"

        prompt = _build_instruction_only_prompt(
            instruction,
            model_type="base",
        )

        # Base model: System message is separate, user content is just question/answer
        assert "You are a research paper assistant" not in prompt
        assert "Question:" in prompt
        assert instruction in prompt
        assert "Answer:" in prompt

    def test_instruction_only_prompt_for_rag_trained_model(self) -> None:
        """Test that RAG-trained model evaluated without contexts gets appropriate format."""
        instruction = "What is the main finding?"

        prompt = _build_instruction_only_prompt(
            instruction,
            model_type="rag_trained",
        )

        # RAG-trained model evaluated without contexts should acknowledge knowledge-based answer
        assert "Answer the following question based on your knowledge" in prompt
        assert "Question:" in prompt
        assert instruction in prompt
        assert "Answer:" in prompt

        # Should not contradict system message about citations
        assert "cite" not in prompt.lower()


class TestCondition4HybridPrompt:
    """Test Condition 4: Instruction-only model evaluated WITH RAG contexts."""

    def test_hybrid_prompt_matches_training_format(self) -> None:
        """Test that hybrid prompt matches instruction-only training format."""
        instruction = "What are the key findings?"
        contexts = [
            "[1] Section: Introduction\nThis paper presents a novel approach...",
            "[2] Section: Methods\nWe used a dataset of 1000 samples...",
        ]

        prompt = _build_rag_prompt_for_instruction_only_model(
            instruction,
            contexts,
        )

        # System message is separate, user content should not duplicate it
        assert "You are a research paper assistant" not in prompt

        # Must include question first (matching training format)
        assert "Question:" in prompt
        assert instruction in prompt

        # Must include RAG contexts and citation instructions
        assert "Context:" in prompt
        assert contexts[0] in prompt
        assert contexts[1] in prompt
        assert "using the provided context" in prompt.lower()
        assert "Cite sources inline as [#]" in prompt

        # Must include answer marker at the end (matching training format)
        assert "Answer:" in prompt

    def test_hybrid_prompt_preserves_training_instruction_style(self) -> None:
        """Test that hybrid prompt preserves the training instruction style."""
        instruction = "Summarize the methodology."
        contexts = ["[1] The method uses gradient descent..."]

        prompt = _build_rag_prompt_for_instruction_only_model(
            instruction,
            contexts,
        )

        # System message is separate, user content should not duplicate it
        # Should start with Question: (matching training format)
        assert prompt.strip().startswith(
            "Question:"
        ), "Hybrid prompt should start with 'Question:' to match training format"

        # Should include context usage instructions
        assert "using the provided context" in prompt.lower()

        # Should preserve training structure: Question: ... Answer:
        # (not the standard RAG format that starts with "Answer the question")
        assert not prompt.strip().startswith("Answer the question"), (
            "Hybrid prompt should preserve training structure (Question: ... Answer:), "
            "not use standard RAG format that starts with 'Answer the question'"
        )

    def test_hybrid_prompt_requires_contexts(self) -> None:
        """Test that hybrid prompt requires contexts."""
        instruction = "What is X?"

        with pytest.raises(ValueError, match="cannot be empty"):
            _build_rag_prompt_for_instruction_only_model(
                instruction,
                contexts=[],
            )

    def test_hybrid_prompt_handles_numbered_contexts(self) -> None:
        """Test that hybrid prompt correctly handles pre-numbered contexts."""
        instruction = "What is the result?"
        contexts = [
            "[1] First context with citation marker",
            "[2] Second context with citation marker",
        ]

        prompt = _build_rag_prompt_for_instruction_only_model(
            instruction,
            contexts,
        )

        # Should preserve the numbered format
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "First context" in prompt
        assert "Second context" in prompt


class TestModelTypeDetection:
    """Test model type detection for prompt format selection."""

    def test_detect_instruction_only_from_config_path(self) -> None:
        """Test detection of instruction-only model from config path."""
        config_path = Path("configs/lora_science_v1_instruction_only_ollama.yaml")
        model_name = "lora_science_0p5_instruction_only"

        model_type = _detect_model_type(config_path, model_name)

        assert model_type == "instruction_only"

    def test_detect_rag_trained_from_config_path(self) -> None:
        """Test detection of RAG-trained model from config path."""
        config_path = Path("configs/lora_science_v1_rag_trained_ollama.yaml")
        model_name = "lora_science_0p5"

        model_type = _detect_model_type(config_path, model_name)

        assert model_type == "rag_trained"

    def test_detect_instruction_only_from_model_name(self) -> None:
        """Test detection of instruction-only model from model name when config is None."""
        model_name = "lora_science_0p5_instruction_only"

        model_type = _detect_model_type(None, model_name)

        assert model_type == "instruction_only"

    def test_detect_rag_trained_from_model_name(self) -> None:
        """Test detection of RAG-trained model from model name when config is None."""
        model_name = "lora_science_0p5"  # No instruction_only suffix

        model_type = _detect_model_type(None, model_name)

        assert model_type == "rag_trained"

    def test_detect_base_model(self) -> None:
        """Test detection of base model."""
        model_name = "qwen2.5:0.5b-instruct"

        model_type = _detect_model_type(None, model_name)

        assert model_type == "base"

    def test_detect_hybrid_config_uses_rag_trained(self) -> None:
        """Test that hybrid configs are detected as RAG-trained (unless explicitly instruction-only)."""
        config_path = Path("configs/hybrid_science_v1_ollama.yaml")
        model_name = "lora_science_0p5"

        model_type = _detect_model_type(config_path, model_name)

        # Hybrid configs typically use RAG-trained models
        assert model_type == "rag_trained"


class TestPromptFormatConsistency:
    """Integration tests for prompt format consistency across the pipeline."""

    def test_condition_4_uses_hybrid_format(self) -> None:
        """Test that Condition 4 (instruction-only + RAG) uses hybrid format.

        This is the critical test that ensures the bug we fixed doesn't regress.
        """
        instruction = "What is the main contribution?"
        contexts = [
            "[1] Section: Abstract\nThis paper introduces a new method...",
            "[2] Section: Introduction\nPrevious work has shown...",
        ]

        # Simulate Condition 4: instruction-only model evaluated with RAG
        prompt = _build_rag_prompt_for_instruction_only_model(
            instruction,
            contexts,
        )

        # Verify it doesn't duplicate system message
        assert "You are a research paper assistant" not in prompt

        # Verify it includes RAG elements
        assert "using the provided context" in prompt.lower()
        assert "Cite sources inline" in prompt
        assert "Context:" in prompt

    def test_instruction_only_evaluation_matches_training(self) -> None:
        """Test that instruction-only evaluation format exactly matches training format."""
        instruction = "What is the methodology?"

        # Training format: System message is separate, user content is just question/answer
        training_user_content = f"Question: {instruction}\n\nAnswer:"

        # Evaluation format
        eval_prompt = _build_instruction_only_prompt(
            instruction,
            model_type="instruction_only",
        )

        # Should match exactly (modulo whitespace)
        assert eval_prompt.strip() == training_user_content.strip(), (
            f"Evaluation prompt must match training format exactly.\n"
            f"Training: {training_user_content!r}\n"
            f"Evaluation: {eval_prompt!r}"
        )

    def test_rag_trained_evaluation_uses_rag_format(self) -> None:
        """Test that RAG-trained models use standard RAG prompt format."""
        # This would come from the dataset's rag.prompt field
        # We're testing that the evaluation runner correctly uses it
        instruction = "What are the findings?"

        # When evaluated without contexts (Condition 5), should use knowledge-based format
        prompt = _build_instruction_only_prompt(
            instruction,
            model_type="rag_trained",
        )

        # Should acknowledge knowledge-based answer (doesn't contradict citation system message)
        assert "Answer the following question based on your knowledge" in prompt
        assert "Question:" in prompt
        assert instruction in prompt


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_instruction_raises_error(self) -> None:
        """Test that empty instruction raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _build_instruction_only_prompt("", model_type="instruction_only")

    def test_hybrid_prompt_empty_instruction_raises_error(self) -> None:
        """Test that hybrid prompt with empty instruction raises ValueError."""
        contexts = ["[1] Some context"]

        with pytest.raises(ValueError, match="cannot be empty"):
            _build_rag_prompt_for_instruction_only_model("", contexts)

    def test_whitespace_only_instruction(self) -> None:
        """Test that whitespace-only instruction is handled."""
        instruction = "   \n\t  "

        with pytest.raises(ValueError, match="cannot be empty"):
            _build_instruction_only_prompt(instruction, model_type="instruction_only")

    def test_model_type_detection_falls_back_to_base(self) -> None:
        """Test that unknown model types default to base."""
        model_type = _detect_model_type(None, "unknown_model_xyz")

        assert model_type == "base", "Unknown models should default to 'base' for safety"
