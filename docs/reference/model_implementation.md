# Model Implementation Analysis

This document provides a comprehensive analysis of all models used in the Beyond the Cutoff project, comparing our implementation against official documentation from Hugging Face and Ollama.

**Last Updated:** November 2025
**Status:** ✓ All implementations verified and corrected

---

## Table of Contents

1. [Embedding Model: BAAI/bge-m3](#1-embedding-model-baaibge-m3)
2. [Reranker Model: BAAI/bge-reranker-v2-m3](#2-reranker-model-baaibge-reranker-v2-m3)
3. [Fine-tuning Base: Qwen/Qwen2.5-0.5B-Instruct](#3-fine-tuning-base-qwenqwen25-05b-instruct)
4. [Generation Models via Ollama](#4-generation-models-via-ollama)
   - [qwen2.5:0.5b-instruct](#41-qwen2505b-instruct)
   - [qwen2.5:7b-instruct-q4_K_M](#42-qwen257b-instruct-q4_k_m)
   - [qwen3:8b](#43-qwen38b)
   - [llama3.1:8b](#44-llama318b)
5. [Custom Fine-tuned Models](#5-custom-fine-tuned-models)
6. [Summary Table](#6-summary-table)

---

## 1. Embedding Model: BAAI/bge-m3

### Official Documentation
- **Source:** [HuggingFace - BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Paper:** [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity](https://arxiv.org/pdf/2402.03216.pdf)

### Model Specifications

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Embedding Dimension | 1024 | Auto-detected from model | ✓ |
| Max Sequence Length | 8192 tokens | `chunk_size: 800` words | ✓ Conservative |
| Normalization | L2 normalize for cosine similarity | `faiss.normalize_L2()` | ✓ |
| Index Type | Inner Product (after L2 norm = cosine) | `IndexFlatIP` | ✓ |
| Query Prefix | **None required** (unlike BGE v1.5) | No prefix used | ✓ |

### Implementation Details

**Configuration (`configs/default.yaml`):**
```yaml
retrieval:
  embedding_model: BAAI/bge-m3
  chunk_size: 800
  chunk_overlap: 120
```

**Indexing (`src/beyond_the_cutoff/retrieval/index.py`):**
```python
# Correct: L2 normalization + Inner Product = Cosine Similarity
index = faiss.IndexFlatIP(embeddings_nd.shape[1])
faiss.normalize_L2(embeddings_nd)
index.add(embeddings_nd)
```

**Query (`src/beyond_the_cutoff/retrieval/query.py`):**
```python
# Correct: Query vectors also normalized
faiss.normalize_L2(query_nd)
scores, idx = self._index.search(query_nd, top_k)
```

### Key Notes from Documentation

1. **No Instruction Prefix:** Unlike BGE v1.5, BGE-M3 does **not** require adding instructions to queries. Our implementation correctly omits any prefix.

2. **Multi-Functionality:** BGE-M3 supports dense, sparse, and ColBERT retrieval. We use only dense retrieval, which is the recommended default for most use cases.

3. **Chunk Size:** Official documentation supports up to 8192 tokens. Our conservative chunk size of 800 words (~1000 tokens) ensures we stay well within limits while maintaining semantic coherence.

### Verification Status: ✓ CORRECT

---

## 2. Reranker Model: BAAI/bge-reranker-v2-m3

### Official Documentation
- **Source:** [HuggingFace - BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Model Type:** Cross-encoder (not bi-encoder)

### Model Specifications

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Input Format | `[(query, passage), ...]` pairs | ✓ Correct format | ✓ |
| Output | Raw relevance scores (higher = better) | Used for sorting | ✓ |
| API | `CrossEncoder.predict(pairs)` | `_reranker.predict(pairs)` | ✓ |
| Normalization | Optional sigmoid for 0-1 range | Raw scores used | ✓ |

### Implementation Details

**Configuration (`configs/default.yaml`):**
```yaml
retrieval:
  reranker_model: BAAI/bge-reranker-v2-m3
```

**Usage (`src/beyond_the_cutoff/retrieval/query.py`):**
```python
# Correct: Using sentence-transformers CrossEncoder
from sentence_transformers import CrossEncoder

# Reranking implementation
pairs = [(query, chunk.text) for chunk in results]
ce_scores = self._reranker.predict(pairs)

# Sort by reranker scores (higher = more relevant)
for chunk, ce in zip(results, ce_scores):
    chunk.reranker_score = float(ce)
results.sort(key=lambda c: c.reranker_score, reverse=True)
```

### Key Notes from Documentation

1. **Cross-Encoder vs Bi-Encoder:** Unlike embedding models, rerankers process query-passage pairs together, enabling deeper semantic understanding.

2. **Score Interpretation:** Raw scores are sufficient for ranking. Sigmoid normalization is optional and not required for our use case.

3. **Fallback Handling:** Our implementation correctly falls back to dense scores if the reranker fails:
   ```python
   except Exception as exc:
       logger.warning("Cross-encoder reranker failed: %s", exc)
       # Falls back to similarity_score ordering
   ```

### Verification Status: ✓ CORRECT

---

## 3. Fine-tuning Base: Qwen/Qwen2.5-0.5B-Instruct

### Official Documentation
- **Source:** [HuggingFace - Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Blog:** [Qwen2.5 Release](https://qwenlm.github.io/blog/qwen2.5/)

### Model Specifications

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Parameters | 0.49B (0.36B non-embedding) | Used as-is | ✓ |
| Context Length | 32,768 tokens | `num_ctx: 4096` | ✓ Conservative |
| Generation Length | Up to 8K tokens | `max_new_tokens: 512` | ✓ |
| Architecture | Transformers with RoPE, SwiGLU, RMSNorm | N/A (used via transformers) | ✓ |
| Chat Template | ChatML format | ✓ Implemented | ✓ |

### Chat Template (ChatML Format)

**Official Format:**
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

**Our Modelfile Implementation (`ollama/Modelfile.instruction_only`):**
```
TEMPLATE """
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}<|im_start|>assistant
"""
```

### Stop Tokens

| Token | Purpose | Configured |
|-------|---------|------------|
| `<\|im_start\|>` | Start of message | ✓ |
| `<\|im_end\|>` | End of message | ✓ |

**Configuration (`configs/default.yaml`):**
```yaml
inference:
  stop_sequences:
    - "<|im_start|>"
    - "<|im_end|>"
```

### Verification Status: ✓ CORRECT

---

## 4. Generation Models via Ollama

### 4.1 qwen2.5:0.5b-instruct

**Documentation:** [Ollama - qwen2.5:0.5b-instruct](https://ollama.com/library/qwen2.5:0.5b-instruct)

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Template | ChatML | ✓ Stop tokens configured | ✓ |
| Context Window | 128K supported | 4096 used | ✓ |
| Stop Tokens | `<\|im_start\|>`, `<\|im_end\|>` | ✓ Configured | ✓ |

**Usage:** Primary inference model for RAG pipeline during 0.5B experiments.

**Configuration:**
```yaml
inference:
  model: qwen2.5:0.5b-instruct
  stop_sequences:
    - "<|im_start|>"
    - "<|im_end|>"
```

### 4.2 qwen2.5:7b-instruct-q4_K_M

**Documentation:** [Ollama - qwen2.5:7b](https://ollama.com/library/qwen2.5)

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Quantization | Q4_K_M (4-bit) | ✓ Correct tag | ✓ |
| Template | ChatML | ✓ Stop tokens configured | ✓ |
| Stop Tokens | `<\|im_start\|>`, `<\|im_end\|>` | ✓ Configured | ✓ |

**Usage:**
- **Dataset generation** (generator model in `configs/default.yaml`)
- WARNING:  **NOT used as judge** to avoid self-preference bias

**Why excluded from judging:**
The Qwen 2.5 7B model generates the training dataset. Using it as a judge would cause
self-preference bias where the model rates outputs from its own family more favorably.
Instead, we use Qwen3 8B or Llama 3.1 8B as judges (different model families).

**Configuration (`configs/default.yaml` - for generation):**
```yaml
dataset_generation:
  generator:
    model: qwen2.5:7b-instruct-q4_K_M
    stop_sequences:
      - "<|im_start|>"
      - "<|im_end|>"
```

### 4.3 qwen3:8b

**Documentation:** [Ollama - qwen3:8b](https://ollama.com/library/qwen3:8b)

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Template | ChatML | ✓ Stop tokens configured | ✓ |
| Thinking Mode | Enabled when temp > 0 | ✓ `temperature: 0.6` | ✓ |
| Stop Tokens | `<\|im_start\|>`, `<\|im_end\|>` | ✓ Configured | ✓ |

**Thinking Mode Details:**

Qwen3 introduces a "thinking" mode that enables chain-of-thought reasoning:

- **Activation:** Automatically enabled when `temperature > 0`
- **Output Format:** Model outputs `<think>...</think>` tags containing its reasoning before the final answer
- **Benefits:** Improved judgment quality through explicit reasoning steps
- **Our Configuration:** `temperature: 0.6` to enable thinking while maintaining focused outputs

**Usage:** Primary judge model (different from generator to avoid self-preference bias)

**Configuration (`configs/judges/pairwise_qwen3_8b.yaml`):**
```yaml
inference:
  model: qwen3:8b
  temperature: 0.6  # Enable thinking mode
  max_tokens: 2048  # Increased for thinking + response
  stop_sequences:
    - "<|im_start|>"
    - "<|im_end|>"
```

### 4.4 llama3.1:8b

**Documentation:** [Ollama - llama3.1:8b](https://ollama.com/library/llama3.1:8b)

| Specification | Official | Our Implementation | Status |
|--------------|----------|-------------------|--------|
| Template | Llama 3 Instruct format | ✓ | ✓ |
| Context Window | 128K | Default used | ✓ |
| Stop Tokens | `<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>` | ✓ Configured | ✓ |

**Template Format:**
```
<|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**Usage:** Alternative judge model for pairwise evaluation

**Configuration (`configs/judges/pairwise_llama31_8b.yaml`):**
```yaml
inference:
  model: llama3.1:8b
  stop_sequences:
    - "<|start_header_id|>"
    - "<|end_header_id|>"
    - "<|eot_id|>"
```

---

## 5. Custom Fine-tuned Models

### 5.1 lora_science_0p5_instruction_only

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Training:** Instruction-only (WITHOUT RAG contexts)
**Modelfile:** `ollama/Modelfile.instruction_only`

**Template:** ChatML (inherited from base)
```
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

TEMPLATE """
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}<|im_start|>assistant
"""

SYSTEM "You are a research paper assistant."
```

**Usage:**
- Condition 3: FT-only (no RAG at inference)
- Condition 4: RAG+FT (RAG contexts provided at inference)

### 5.2 lora_science_0p5

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Training:** RAG-trained (WITH RAG contexts)
**Modelfile:** `ollama/Modelfile.rag_trained`

**System Prompt:**
```
SYSTEM "You are a scientific research assistant who answers with concise, evidence-grounded prose and includes inline numeric citations like [1], [2], etc."
```

**Usage:**
- Condition 5: RAG-trained FT only (no RAG at inference)
- Condition 6: RAG-trained FT+RAG (optimal setup)

---

## 6. Summary Table

| Model | Role | Template | Stop Tokens | Status |
|-------|------|----------|-------------|--------|
| BAAI/bge-m3 | Embeddings | N/A | N/A | ✓ |
| BAAI/bge-reranker-v2-m3 | Reranking | N/A | N/A | ✓ |
| Qwen/Qwen2.5-0.5B-Instruct | Fine-tuning base | ChatML | `<\|im_start\|>`, `<\|im_end\|>` | ✓ |
| qwen2.5:0.5b-instruct | Inference | ChatML | `<\|im_start\|>`, `<\|im_end\|>` | ✓ |
| qwen2.5:7b-instruct-q4_K_M | **Generator** (NOT judge) | ChatML | `<\|im_start\|>`, `<\|im_end\|>` | ✓ |
| qwen3:8b | **Judge** (thinking mode) | ChatML | `<\|im_start\|>`, `<\|im_end\|>` | ✓ |
| llama3.1:8b | **Judge** (alternative) | Llama 3 Instruct | `<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>` | ✓ |

---

## Appendix: Configuration File Locations

| Config | Path | Models Configured |
|--------|------|-------------------|
| Default pipeline | `configs/default.yaml` | bge-m3, bge-reranker-v2-m3, qwen2.5:0.5b-instruct, qwen2.5:7b (generator) |
| Qwen3 Judge | `configs/judges/pairwise_qwen3_8b.yaml` | qwen3:8b |
| Qwen3 Dataset Judge | `configs/judges/dataset_quality_judge.yaml` | qwen3:8b |
| Llama 3.1 Judge | `configs/judges/pairwise_llama31_8b.yaml` | llama3.1:8b |
| Qwen 7B Generator | `configs/judges/ollama_qwen7b.yaml` | qwen2.5:7b (WARNING:  for generation only) |
| Qwen 7B Pairwise | `configs/judges/pairwise_qwen7b.yaml` | qwen2.5:7b (WARNING:  EXCLUDED from experiments) |
| Instruction-only FT | `ollama/Modelfile.instruction_only` | Custom LoRA |
| RAG-trained FT | `ollama/Modelfile.rag_trained` | Custom LoRA |

---

## References

1. [BGE-M3 Paper](https://arxiv.org/pdf/2402.03216.pdf) - Chen et al., 2024
2. [Qwen2.5 Blog](https://qwenlm.github.io/blog/qwen2.5/) - Qwen Team, 2024
3. [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/) - Qwen Team, 2025
4. [Llama 3.1 Announcement](https://ai.meta.com/blog/meta-llama-3-1/) - Meta AI, 2024
5. [Ollama Documentation](https://docs.ollama.com/) - Ollama, 2025
