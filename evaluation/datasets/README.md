# Evaluation datasets

This folder is reserved for **generated dataset artifacts** used by training/evaluation.

## Why the JSONLs are not in Git

The `*.jsonl` files in this directory are intentionally **not tracked** in Git:

- They are large, frequently regenerated artifacts.
- They may embed long excerpts from source documents and/or machine-generated outputs.
- Some files contain machine-specific absolute paths, which are not portable.

Only small metadata (like `.gitkeep` and `manifest.json`) is tracked.

## Expected files

When you run the pipeline locally, you may see files such as:

- `offline_tasks.jsonl`
- `offline_dataset.jsonl`
- `train_dataset*.jsonl`
- `eval_dataset*.jsonl`

## Reproducibility

If you need to confirm you have the same dataset revision used for a given run, compare SHA256 hashes in `manifest.json` (when provided/updated).
