#!/usr/bin/env python3
"""Ask a question using the local RAG index and Ollama backend.

Usage:
    python scripts/ask.py "What does paper X say about Y?"
"""

from __future__ import annotations

import argparse

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.retrieval.query import RAGPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query the paper assistant")
    p.add_argument("question", help="The question to ask")
    p.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    index_dir = cfg.paths.external_data / "index"
    pipeline = RAGPipeline(
        cfg,
        index_path=index_dir / "index.faiss",
        mapping_path=index_dir / "mapping.tsv",
    )
    result = pipeline.ask(args.question)
    print("\nAnswer:\n" + result["answer"].strip())
    print("\nSources:")
    for i, src in enumerate(result["sources"], start=1):
        print(f"[{i}] {src}")


if __name__ == "__main__":
    main()
