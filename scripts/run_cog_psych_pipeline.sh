#!/usr/bin/env bash
# Bootstrap the cognition/psychology corpus, rebuild retrieval assets, and refresh offline datasets.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

CONFIG_PATH="configs/autogen/cog_psych_2025.yaml"
RAW_DIR="data/raw/cog_psych_2025_run01"
PROCESSED_DIR="data/processed/cog_psych_2025_run01"
EXTERNAL_DIR="data/external/cog_psych_2025_run01"
DATASET_PATH="evaluation/datasets/cog_psych_offline_dataset.jsonl"
RAW_TASKS_PATH="evaluation/datasets/cog_psych_offline_tasks.jsonl"
TOTAL_PAPERS="${COG_PSYCH_TOTAL:-100}"

CONTACT_EMAIL="${1:-${ARXIV_CONTACT_EMAIL:-}}"
if [[ -z "${CONTACT_EMAIL}" ]]; then
  echo "Usage: scripts/run_cog_psych_pipeline.sh <contact-email>" >&2
  echo "       or export ARXIV_CONTACT_EMAIL before running." >&2
  exit 1
fi

rm -rf "${PROCESSED_DIR}" "${EXTERNAL_DIR}" "${DATASET_PATH}" "${RAW_TASKS_PATH}"
mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}" "${EXTERNAL_DIR}" "$(dirname "${RAW_TASKS_PATH}")"

export PYTORCH_SDP_DISABLE_FLASH_ATTN="${PYTORCH_SDP_DISABLE_FLASH_ATTN:-1}"
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT="${PYTORCH_SDP_DISABLE_MEM_EFFICIENT:-1}"
export PYTORCH_SDP_ATTENTION="${PYTORCH_SDP_ATTENTION:-math}"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BTC_TORCH_THREADS="${BTC_TORCH_THREADS:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

# Ensure judge/generator model is available locally.
if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ollama pull qwen2.5:3b-instruct-q4_K_M
else
  echo "Warning: ollama CLI not found; skipping model pull (ensure models are available)." >&2
fi

# Pre-fetch embedding and reranker checkpoints so ingestion has them cached.
python - <<'PY'
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install huggingface_hub before running this script") from exc

cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
    print(f"Downloading {repo} ...")
    snapshot_download(repo_id=repo, cache_dir=cache_dir, resume_download=True)
PY

# 1) Fetch the latest cognition/psychology corpus slice.
if [[ "${COG_PSYCH_FORCE_FETCH:-0}" == "1" || ! -d "${RAW_DIR}/papers" || ! -f "${RAW_DIR}/metadata.jsonl" ]]; then
  python scripts/fetch_arxiv_corpus.py \
    --contact-email "${CONTACT_EMAIL}" \
    --output-dir "${RAW_DIR}" \
    --total "${TOTAL_PAPERS}" \
    --oversample 1.8 \
    --max-results 400
else
  echo "Skipping arXiv fetch; using existing corpus at ${RAW_DIR}. Set COG_PSYCH_FORCE_FETCH=1 to refresh." >&2
fi

if [[ -f "${RAW_DIR}/metadata.jsonl" ]]; then
  cp "${RAW_DIR}/metadata.jsonl" "${RAW_DIR}/papers/" 2>/dev/null || true
fi
if [[ -f "${RAW_DIR}/metadata.csv" ]]; then
  cp "${RAW_DIR}/metadata.csv" "${RAW_DIR}/papers/" 2>/dev/null || true
fi

# 2) Ingest PDFs, rebuild metadata + FAISS index with upgraded retrieval models.
python scripts/ingest_and_index.py --config "${CONFIG_PATH}"

# 3) Regenerate offline tasks and dataset with the refreshed config.
python scripts/generate_offline_tasks.py --config "${CONFIG_PATH}" \
  --processed-dir "${PROCESSED_DIR}" \
  --output "${RAW_TASKS_PATH}"

python scripts/generate_offline_dataset.py --config "${CONFIG_PATH}" \
  --tasks "${RAW_TASKS_PATH}" \
  --output "${DATASET_PATH}"

cat <<EOF
Pipeline complete.
  Raw PDFs: ${RAW_DIR}/papers
  Processed text: ${PROCESSED_DIR}
  Index artifacts: ${EXTERNAL_DIR}/index
  Offline dataset: ${DATASET_PATH}
EOF
