# Model Adaptation Portfolio

_Last updated: 2025-11-01_

## Experiment Matrix
| Experiment ID | Description | Model | Retrieval | Notes |
| --- | --- | --- | --- | --- |
| `rag_baseline_v1` | Baseline RAG without fine-tuning | Ollama `qwen2:0.5b-instruct-q4_0` | FAISS (default config) | Control group; rerun after each corpus refresh |
| `lora_science_v1` | LoRA fine-tune only | Hugging Face `Qwen/Qwen2-0.5B-Instruct` | Disabled | Evaluate pure parameter-shift gains |
| `hybrid_science_v1` | Fine-tuned + RAG | Ollama custom tag (`qwen2-science-lora`) | FAISS | Expected production candidate |
| `cloud_eval_v1` | Cloud API comparator | GPT-4o (API) | N/A (context-stuffed) | Optional benchmark for ceiling analysis |

Acceptance threshold for promotion to production pilot: **≥ +5 percentage points** factuality over `rag_baseline_v1` AND citation grounding ≥ 0.35 mean coverage.

## Data Splits
- **Train**: 70% of offline dataset tasks, stratified by paper ID and task type.
- **Validation**: 15% for early stopping and hyperparameter tuning.
- **Test**: 15% held out for evaluation reports; never leak into training.
- Store splits under `evaluation/datasets/splits/<dataset_tag>_{train,val,test}.jsonl`.
- Record the RNG seed (`FINETUNE_SPLIT_SEED=20251101`) alongside the split files.

## LoRA Experiment Setup
- Base model: `Qwen/Qwen2-0.5B-Instruct` (float16).
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
3. Use `llama.cpp` tooling to generate `qwen2-science-lora.Q4_K_M.gguf`.
4. Register with Ollama using a `Modelfile` under `ollama/qwen2-science-lora/Modelfile`.
5. Update `configs/default.yaml` overrides after validation.

## Compute Budget Guidelines
- Colab: Nvidia T4 GPU.
- Kaggle P100 fallback; adjust batch size to maintain throughput.
- For larger baselines (Phi-3, Mistral), request dedicated cluster time (document approvals in `docs/compute_requests.md`).

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
