# Ollama Modelfiles

This directory contains Modelfiles for registering fine-tuned models with Ollama.

## Modelfiles

### `Modelfile` (Default - RAG-trained)
- **Model**: RAG-trained fine-tuned model (trained WITH RAG contexts)
- **Status**: NOT used in 0.5B experiment (kept for reference)
- **Ollama tag**: `lora_science_0p5` (or similar)
- **System prompt**: Mentions citations and contexts

### `Modelfile.rag_trained`
- Same as `Modelfile` - explicit version for clarity
- **Status**: NOT used in 0.5B experiment (kept for reference)

### `Modelfile.instruction_only`
- **Model**: Instruction-only fine-tuned model (trained WITHOUT RAG contexts)
- **Use for**: BOTH FT-only AND RAG+FT evaluation
  - FT-only: `lora_science_0p5b_ft_only` (without contexts)
  - RAG+FT: `hybrid_science_0p5b` (with contexts)
- **Ollama tag**: `lora_science_0p5_instruction_only` (or similar)
- **System prompt**: Flexible - works with or without contexts

## Usage

### Register Instruction-Only Model (Used for Both FT-only and RAG+FT)

**This is the ONLY model you need to register for the 0.5B experiment.**

```bash
# Update FROM path in Modelfile.instruction_only to point to your GGUF file
# Then register:
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
ollama push lora_science_0p5_instruction_only

# This same model is used for:
# - FT-only evaluation (without RAG contexts)
# - RAG+FT evaluation (with RAG contexts)
```

## Key Differences

| Aspect | RAG-trained | Instruction-only |
|--------|-------------|------------------|
| Training | With contexts | Without contexts |
| System Prompt | Mentions citations | Flexible (works with/without contexts) |
| Evaluation | RAG+FT mode only | FT-only AND RAG+FT modes |
| Modelfile | `Modelfile` or `Modelfile.rag_trained` | `Modelfile.instruction_only` |

**Note**: For this experiment, we only use the instruction-only model. The RAG-trained Modelfile is kept for reference but not used in the 0.5B comparison.

## Quick Reference

| Evaluation Mode | Model | Modelfile | Ollama Tag | Config File |
|----------------|-------|-----------|------------|-------------|
| RAG Only | Base model | N/A | `qwen2.5:0.5b-instruct` | `rag_baseline_ollama.yaml` |
| FT Only | Fine-tuned (instruction-only) | `Modelfile.instruction_only` | `lora_science_0p5_instruction_only` | `lora_science_v1_instruction_only_ollama.yaml` |
| RAG+FT | Same fine-tuned model | `Modelfile.instruction_only` | `lora_science_0p5_instruction_only` | `hybrid_science_v1_ollama.yaml` |

**Key Point**: Only ONE fine-tuned model is needed. It's used:
- Without RAG contexts → FT-only
- With RAG contexts → RAG+FT

## Notes

- Update the `FROM` path in Modelfile.instruction_only to point to your actual GGUF file location
- The RAG-trained Modelfile (`Modelfile`/`Modelfile.rag_trained`) is NOT used in this experiment
- Only the instruction-only model is needed for the 0.5B comparison
