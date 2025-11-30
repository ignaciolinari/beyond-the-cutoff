"""Minimal FastAPI app exposing a /ask endpoint for the paper assistant."""

# mypy: allow-untyped-decorators = True

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException

from ..config import load_config
from ..retrieval.query import RAGPipeline

app = FastAPI(title="Beyond the Cutoff - Paper Assistant API")


@lru_cache(maxsize=1)
def _load_pipeline() -> RAGPipeline:
    """Load the RAG pipeline once per process to avoid redundant index loads."""

    cfg = load_config()
    index_dir = cfg.paths.external_data / "index"
    index_path = index_dir / "index.faiss"
    mapping_path = index_dir / "mapping.tsv"
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            f"Index files not found at {index_dir}. Run scripts/data/ingest_and_index.py first."
        )
    return RAGPipeline(cfg, index_path=index_path, mapping_path=mapping_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question")
    if not question or not isinstance(question, str):
        raise HTTPException(status_code=400, detail="Missing 'question' in request body")
    try:
        pipeline = _load_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "index_missing",
                "message": str(exc),
            },
        ) from exc
    result = pipeline.ask(question)
    return result
