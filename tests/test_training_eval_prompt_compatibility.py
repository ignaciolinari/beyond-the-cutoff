"""Tests for training/evaluation prompt format compatibility.

This test suite ensures that prompts used during evaluation match the formats
used during training, preventing distribution shift and ensuring valid experimental comparisons.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beyond_the_cutoff.config import ModelType
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
            model_type=ModelType.INSTRUCTION_ONLY,
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
            model_type=ModelType.BASE,
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
            model_type=ModelType.RAG_TRAINED,
        )

        # RAG-trained model evaluated without contexts should acknowledge knowledge-based answer
        assert "Answer the following question based on your knowledge" in prompt
        assert "Question:" in prompt
        assert instruction in prompt
        assert "Answer:" in prompt

        # Should explicitly instruct not to cite since no sources are provided
        assert "do not include citations" in prompt.lower() or "do not cite" in prompt.lower()
        assert "no sources are provided" in prompt.lower() or "no sources" in prompt.lower()


class TestCondition4HybridPrompt:
    """Test Condition 4: Instruction-only model evaluated WITH RAG contexts.

    DESIGN DECISION: Condition 4 uses the STANDARD RAG prompt format (same as
    Conditions 2 and 6) rather than a hybrid format that preserves training structure.

    Note: Condition 5 uses INSTRUCTION mode (no RAG contexts), not RAG mode.
    Only Conditions 2, 4, 6 use RAG evaluation mode.

    Rationale: The transfer learning test should isolate whether an instruction-only
    trained model can benefit from RAG contexts - NOT whether it can handle a unique
    prompt format. Using the same format as other RAG conditions ensures valid
    comparison across the RAG condition group (2, 4, 6).

    The model WILL experience distribution shift (it was trained on "Question: X\\nAnswer:"
    but now sees the full RAG format), which is the intended test.
    """

    def test_condition4_uses_standard_rag_format(self) -> None:
        """Test that Condition 4 uses the standard RAG format for fair comparison."""
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

        # Must use STANDARD RAG format: Instructions -> Context -> Question -> Answer
        # This matches the format used for RAG evaluation conditions (2, 4, 6)
        # Note: Condition 5 is in the instruction group (evaluated WITHOUT contexts)
        assert prompt.strip().startswith(
            "Answer the question using the provided context"
        ), "Condition 4 should use standard RAG format (same as Conditions 2 and 6)"

        # Must include question
        assert "Question:" in prompt
        assert instruction in prompt

        # Must include RAG contexts and citation instructions
        assert "Context:" in prompt
        assert contexts[0] in prompt
        assert contexts[1] in prompt
        assert "Cite the sources inline as [#]" in prompt

        # Must include answer marker at the end
        assert "Answer:" in prompt
        assert prompt.strip().endswith("Answer:")

    def test_condition4_format_matches_other_rag_conditions(self) -> None:
        """Test that Condition 4 format matches standard RAG format (Conditions 2 and 6).

        RAG evaluation conditions: 2, 4, 6 (evaluated WITH contexts)
        Instruction evaluation conditions: 1, 3, 5 (evaluated WITHOUT contexts)
        """
        instruction = "Summarize the methodology."
        contexts = ["[1] The method uses gradient descent..."]

        prompt = _build_rag_prompt_for_instruction_only_model(
            instruction,
            contexts,
        )

        # Should use standard RAG format (not hybrid format)
        # Standard format: "Answer the question using the provided context..."
        assert prompt.strip().startswith("Answer the question"), (
            "Condition 4 should use standard RAG format for fair comparison "
            "with other RAG evaluation conditions (2 and 6)"
        )

        # Should follow standard format: Instructions -> Context -> Question -> Answer
        # NOT the old hybrid format: Question -> Context -> Instructions -> Answer
        instructions_idx = prompt.find("Answer the question")
        context_idx = prompt.find("Context:")
        question_idx = prompt.find("Question:")
        answer_idx = prompt.rfind("Answer:")

        assert instructions_idx < context_idx < question_idx < answer_idx, (
            "Format should be: Instructions -> Context -> Question -> Answer "
            "(matching standard RAG format)"
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
        """Test that hybrid configs are detected as RAG-trained based on filename.

        Note: This test uses a legacy config file that has been moved to vintage/configs/.
        When model_cfg is not provided, detection falls back to filename-based inference.
        Configs with 'hybrid' in the name (but not 'instruction') are detected as rag_trained.
        """
        config_path = Path("vintage/configs/hybrid_science_v1_ollama.yaml")
        model_name = "lora_science_0p5"

        # Test filename-based detection (model_cfg=None)
        model_type = _detect_model_type(config_path, model_name, model_cfg=None)

        # Filename-based detection: "hybrid" without "instruction" -> rag_trained
        assert model_type == "rag_trained"

        # Test that explicit model_type takes precedence when config is loaded
        from beyond_the_cutoff.evaluation.runner import load_inference_from_yaml

        model_cfg = load_inference_from_yaml(config_path)
        model_type_with_config = _detect_model_type(config_path, model_name, model_cfg=model_cfg)

        # With explicit model_type: instruction_only, detection should use that
        assert model_type_with_config == "instruction_only"


class TestPromptFormatConsistency:
    """Integration tests for prompt format consistency across the pipeline."""

    def test_condition_4_uses_standard_rag_format(self) -> None:
        """Test that Condition 4 (instruction-only + RAG) uses standard RAG format.

        DESIGN DECISION: Condition 4 uses the SAME prompt format as other RAG evaluation
        conditions (2 and 6) rather than a hybrid format. This ensures valid comparison
        across the RAG condition group (2, 4, 6) by isolating the training mode variable.

        Note: Condition 5 uses INSTRUCTION mode (no RAG), not RAG mode.

        The model experiences distribution shift (trained on "Question: X\\nAnswer:" but
        now sees full RAG format), which is the intended transfer learning test.
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

        # Verify it uses STANDARD RAG format (same as Conditions 2 and 6)
        # RAG evaluation conditions: 2, 4, 6 | Instruction conditions: 1, 3, 5
        assert prompt.strip().startswith("Answer the question using the provided context")
        assert "Cite the sources inline as [#]" in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert instruction in prompt

    def test_instruction_only_evaluation_matches_training(self) -> None:
        """Test that instruction-only evaluation format exactly matches training format."""
        instruction = "What is the methodology?"

        # Training format: System message is separate, user content is just question/answer
        training_user_content = f"Question: {instruction}\n\nAnswer:"

        # Evaluation format
        eval_prompt = _build_instruction_only_prompt(
            instruction,
            model_type=ModelType.INSTRUCTION_ONLY,
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
            model_type=ModelType.RAG_TRAINED,
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
            _build_instruction_only_prompt("", model_type=ModelType.INSTRUCTION_ONLY)

    def test_hybrid_prompt_empty_instruction_raises_error(self) -> None:
        """Test that hybrid prompt with empty instruction raises ValueError."""
        contexts = ["[1] Some context"]

        with pytest.raises(ValueError, match="cannot be empty"):
            _build_rag_prompt_for_instruction_only_model("", contexts)

    def test_whitespace_only_instruction(self) -> None:
        """Test that whitespace-only instruction is handled."""
        instruction = "   \n\t  "

        with pytest.raises(ValueError, match="cannot be empty"):
            _build_instruction_only_prompt(instruction, model_type=ModelType.INSTRUCTION_ONLY)

    def test_model_type_detection_falls_back_to_base(self) -> None:
        """Test that unknown model types default to base."""
        model_type = _detect_model_type(None, "unknown_model_xyz")

        assert model_type == "base", "Unknown models should default to 'base' for safety"
