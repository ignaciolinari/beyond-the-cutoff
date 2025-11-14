# Ollama Modelfiles

This directory contains Modelfiles for registering fine-tuned models with Ollama.

## Modelfiles

### `Modelfile` (Default - RAG-trained)
- **Model**: RAG-trained fine-tuned model (trained WITH RAG contexts)
- **Status**: NOT used in 0.5B experiment
- **Location**: Moved to `vintage/ollama/` for reference
- **Ollama tag**: `lora_science_0p5` (or similar)
- **System prompt**: Mentions citations and contexts

### `Modelfile.rag_trained`
- Same as `Modelfile` - explicit version for clarity
- **Status**: ✅ **USED** in 0.5B experiment (for conditions 5-6)
- **Location**: `ollama/Modelfile.rag_trained`

### `Modelfile.instruction_only`
- **Model**: Instruction-only fine-tuned model (trained WITHOUT RAG contexts)
- **Use for**: BOTH FT-only AND RAG+FT evaluation
  - FT-only: `lora_science_0p5b_ft_only` (without contexts)
  - RAG+FT: `hybrid_science_0p5b` (with contexts)
- **Ollama tag**: `lora_science_0p5_instruction_only` (or similar)
- **System prompt**: Flexible - works with or without contexts

## Usage

### Register Models for 6-Condition Experiment

**You need TWO fine-tuned models for the complete 6-condition experiment:**

```bash
# 1. Register instruction-only model
# Update FROM path in Modelfile.instruction_only to point to your GGUF file
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
ollama push lora_science_0p5_instruction_only

# This model is used for:
# - FT-only evaluation (without RAG contexts)
# - FT+RAG (instruction-only) evaluation (with RAG contexts)

# 2. Register RAG-trained model
# Update FROM path in ollama/Modelfile.rag_trained to point to your RAG-trained GGUF file
ollama create lora_science_0p5 -f ollama/Modelfile.rag_trained
ollama push lora_science_0p5

# This model is used for:
# - RAG-trained FT-only evaluation (without RAG contexts)
# - RAG-trained FT+RAG evaluation (with RAG contexts - optimal setup)
```

## Key Differences

| Aspect | RAG-trained | Instruction-only |
|--------|-------------|------------------|
| Training | With contexts | Without contexts |
| System Prompt | Mentions citations | Flexible (works with/without contexts) |
| Evaluation | RAG+FT mode only | FT-only AND RAG+FT modes |
| Modelfile | `Modelfile` or `Modelfile.rag_trained` | `Modelfile.instruction_only` |

**Note**: For the complete 6-condition experiment, you need BOTH models:
- Instruction-only model (for conditions 3-4)
- RAG-trained model (for conditions 5-6)

## Quick Reference

| Evaluation Mode | Model | Modelfile | Ollama Tag | Config File |
|----------------|-------|-----------|------------|-------------|
| Base Baseline | Base model | N/A | `qwen2.5:0.5b-instruct` | `rag_baseline_ollama.yaml` |
| RAG Baseline | Base model | N/A | `qwen2.5:0.5b-instruct` | `rag_baseline_ollama.yaml` |
| FT Only | Fine-tuned (instruction-only) | `Modelfile.instruction_only` | `lora_science_0p5_instruction_only` | `lora_science_v1_instruction_only_ollama.yaml` |
| FT+RAG (instruction-only) | Fine-tuned (instruction-only) | `Modelfile.instruction_only` | `lora_science_0p5_instruction_only` | `lora_science_v1_instruction_only_ollama.yaml` |
| RAG-trained FT Only | Fine-tuned (RAG-trained) | `Modelfile.rag_trained` | `lora_science_0p5` | `lora_science_v1_rag_trained_ollama.yaml` |
| RAG-trained FT+RAG | Fine-tuned (RAG-trained) | `Modelfile.rag_trained` | `lora_science_0p5` | `lora_science_v1_rag_trained_ollama.yaml` |

**Key Point**: TWO fine-tuned models are needed for the complete 6-condition experiment:
- Instruction-only model: Used for FT-only and FT+RAG (instruction-only)
- RAG-trained model: Used for RAG-trained FT-only and RAG-trained FT+RAG (optimal)

## Notes

- Update the `FROM` path in `Modelfile.instruction_only` to point to your instruction-only GGUF file
- Update the `FROM` path in `Modelfile.rag_trained` to point to your RAG-trained GGUF file
- Both models are needed for the complete 6-condition comparison
