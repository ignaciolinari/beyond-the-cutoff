# Model Adaptation Portfolio

_Last updated: 2025-11-11_

## Experiment Matrix
| Experiment ID | Description | Model | Retrieval | Notes |
| --- | --- | --- | --- | --- |
| `rag_baseline_0p5` | Baseline RAG without fine-tuning (0.5B) | Ollama `qwen2.5:0.5b-instruct` | FAISS (default config) | Control group for 0.5B series; rerun after each corpus refresh |
| `lora_science_0p5_ft` | LoRA fine-tune only (0.5B) | Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` | Disabled | Measures parameter-shift gains against `rag_baseline_0p5` |
| `hybrid_science_0p5` | Fine-tuned + RAG (0.5B) | Ollama `qwen2.5:0.5b-instruct` with merged LoRA | FAISS | Confirms additive benefit of retrieval for 0.5B |
| `rag_baseline_3b` | Baseline RAG without fine-tuning (3B) | Ollama `qwen2.5:3b-instruct-q4_K_M` | FAISS | Control group for 3B series |
| `lora_science_3b_ft` | LoRA fine-tune only (3B) | Hugging Face `Qwen/Qwen2.5-3B-Instruct` | Disabled | Mirrors 0.5B FT study at higher capacity |
| `hybrid_science_3b` | Fine-tuned + RAG (3B) | Ollama `qwen2.5:3b-instruct-q4_K_M` with merged LoRA | FAISS | Production candidate if > `rag_baseline_3b` |
| `cloud_eval_v1` | Cloud API comparator | GPT-4o (API) | N/A (context-stuffed) | Optional benchmark for ceiling analysis |

Run the 0.5B sequence (`rag_baseline_0p5` → `lora_science_0p5_ft` → `hybrid_science_0p5`) to completion before starting the 3B series; do not cross-compare across model sizes.

> TODO before beginning the 3B phase: duplicate `configs/lora_science_v1_ollama.yaml` and adjust the model tag to `qwen2.5:3b-instruct-q4_K_M` (or the merged LoRA alias) so both experiment groups have dedicated config files.

Acceptance threshold for promotion to production pilot: **≥ +5 percentage points** factuality over `rag_baseline_v1` AND citation grounding ≥ 0.35 mean coverage.

## Data Splits
- **Train**: 70% of offline dataset tasks, stratified by paper ID and task type.
- **Validation**: 15% for early stopping and hyperparameter tuning.
- **Test**: 15% held out for evaluation reports; never leak into training.
- Store splits under `evaluation/datasets/splits/<dataset_tag>_{train,val,test}.jsonl`.
- Record the RNG seed (`FINETUNE_SPLIT_SEED=20251101`) alongside the split files.

## LoRA Experiment Setup
- Base model: `Qwen/Qwen2.5-0.5B-Instruct` (float16) for the first pass; replicate with `Qwen/Qwen2.5-3B-Instruct` after 0.5B evaluation wraps.
- Target layers: attention projections + feed-forward down projections.
- Hyperparameter ranges:
  - `lora_rank`: {16, 32}
  - `learning_rate`: {1e-4, 5e-5}
  - `batch_size`: {64 tokens effective via gradient accumulation}
  - `epochs`: {2, 3}
- Schedule grid search via Colab or Kaggle; capture notebook path in `notebooks/finetuning/<experiment_id>.ipynb`.

## Adapter Export & Quantization
1. Save LoRA adapters (`adapter.safetensors`) plus training config JSON.
2. Merge adapters into base model locally to produce full checkpoint for GGUF conversion.
3. Use `llama.cpp` tooling to generate `qwen2.5-science-lora.Q4_K_M.gguf` (rename per experiment).
4. Register with Ollama using a `Modelfile` under `ollama/qwen2dot5-science-lora/Modelfile` or by updating the stock `qwen2.5:3b` tag locally.
5. Update `configs/default.yaml` overrides after validation.

## Compute Budget Guidelines
- Colab: Nvidia T4 GPU.
- Kaggle P100 fallback; adjust batch size to maintain throughput.
- For higher-capacity experiments (`qwen2.5:7b-instruct-q4_K_M`, Mixtral, etc.), request dedicated cluster time (document approvals in `docs/compute_requests.md`).

## Artifact Management
- Sync command template: `rclone copy gdrive:/beyond-cutoff/adapters/<experiment_id>/ outputs/adapters/<experiment_id>/`.
- Record SHA256 checksums in `outputs/adapters/<experiment_id>/checksums.txt`.
- Add `EXPERIMENT_METADATA.json` with:
  - Dataset tag
  - Split seed
  - Hyperparameters
  - Training runtime
  - Evaluation results summary

## Open Tasks
- Implement CLI wrapper for adapter download/verification.
- Author notebook template with automated logging to `weights.json`.
