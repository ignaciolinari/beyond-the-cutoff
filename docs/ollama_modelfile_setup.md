# Ollama Modelfile Setup Guide

## Overview

We have **two separate Modelfiles** to match the two fine-tuning approaches:

1. **`Modelfile`** (or `Modelfile.rag_trained`) - For RAG-trained model
2. **`Modelfile.instruction_only`** - For instruction-only model

## Key Differences

### System Prompts

**RAG-trained Modelfile:**
```
SYSTEM "You are a scientific research assistant. When the prompt includes retrieved context snippets or explicit citation markers, cite them inline like [1], [2], [3], etc. If no supporting context is supplied, answer in your own words without fabricated citations."
```

**Instruction-only Modelfile:**
```
SYSTEM "You are a scientific research assistant. Answer questions based on your knowledge. Provide accurate, evidence-based responses."
```

### Why Different System Prompts?

- **RAG-trained model**: Trained WITH contexts, expects to cite them
- **Instruction-only model**: Trained WITHOUT contexts, shouldn't expect citations

Using the wrong system prompt can confuse the model and lead to:
- RAG-trained model generating citations when none are provided
- Instruction-only model trying to cite non-existent contexts

## Setup Instructions

### 1. Prepare GGUF Files

**For RAG-trained model:**
```bash
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1/merged_full_model \
  --outfile outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.Q4_K_M.gguf \
  --data-type Q4_K_M
```

**For instruction-only model:**
```bash
python /path/to/llama.cpp/convert-hf-to-gguf.py \
  --model-dir outputs/lora_science_v1_instruction_only/merged_full_model \
  --outfile outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf \
  --data-type Q4_K_M
```

### 2. Update Modelfiles

**Update `Modelfile` (RAG-trained):**
```bash
# Edit ollama/Modelfile
# Update FROM path to point to your GGUF file:
# FROM ../outputs/lora_science_v1/merged_full_model/Qwen2.5-0.5B-lora_science_v1.Q4_K_M.gguf
```

**Update `Modelfile.instruction_only`:**
```bash
# Edit ollama/Modelfile.instruction_only
# Update FROM path to point to your GGUF file:
# FROM ../outputs/lora_science_v1_instruction_only/merged_full_model/Qwen2.5-0.5B-lora_science_v1_instruction_only.Q4_K_M.gguf
```

### 3. Register with Ollama

**RAG-trained model:**
```bash
ollama create lora_science_0p5 -f ollama/Modelfile
ollama push lora_science_0p5
```

**Instruction-only model:**
```bash
ollama create lora_science_0p5_instruction_only -f ollama/Modelfile.instruction_only
ollama push lora_science_0p5_instruction_only
```

### 4. Verify Registration

```bash
# List registered models
ollama list

# Test RAG-trained model (should mention citations)
ollama run lora_science_0p5 "What is machine learning?"

# Test instruction-only model (should NOT mention citations)
ollama run lora_science_0p5_instruction_only "What is machine learning?"
```

## Configuration Files

Make sure your config files point to the correct Ollama tags:

**`configs/hybrid_science_v1_ollama.yaml`** (RAG+FT):
```yaml
model: lora_science_0p5  # RAG-trained model
```

**`configs/lora_science_v1_instruction_only_ollama.yaml`** (FT-only):
```yaml
model: lora_science_0p5_instruction_only  # Instruction-only model
```

## Troubleshooting

### Model generates citations when it shouldn't
- **Problem**: Using RAG-trained Modelfile for instruction-only model
- **Solution**: Use `Modelfile.instruction_only` instead

### Model doesn't cite contexts when it should
- **Problem**: Using instruction-only Modelfile for RAG-trained model
- **Solution**: Use `Modelfile` or `Modelfile.rag_trained` instead

### Wrong model tag in config
- **Problem**: Config points to wrong Ollama tag
- **Solution**: Update config file to match registered Ollama tag

## File Structure

```
ollama/
├── Modelfile                    # RAG-trained model (default)
├── Modelfile.rag_trained        # Same as Modelfile (explicit)
├── Modelfile.instruction_only   # Instruction-only model
└── README.md                    # This guide
```

## Quick Reference

| Model Type | Modelfile | Ollama Tag | Config File | Evaluation Mode |
|------------|-----------|------------|-------------|----------------|
| RAG-trained | `Modelfile` | `lora_science_0p5` | `hybrid_science_v1_ollama.yaml` | RAG+FT |
| Instruction-only | `Modelfile.instruction_only` | `lora_science_0p5_instruction_only` | `lora_science_v1_instruction_only_ollama.yaml` | FT-only |
