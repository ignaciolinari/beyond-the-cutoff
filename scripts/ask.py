#!/usr/bin/env python3
"""Ask a question using the local RAG index and configured generation backend.

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
    for idx, citation in enumerate(result["citations"], start=1):
        source_path = citation.get("source_path") or "(unknown)"
        section = citation.get("section_title")
        page = citation.get("page")
        meta_bits = []
        if section:
            meta_bits.append(section)
        if page is not None:
            meta_bits.append(f"page {page}")
        suffix = f" ({', '.join(meta_bits)})" if meta_bits else ""
        print(f"[{idx}] {source_path}{suffix}")

        rendered = citation.get("rendered_context") or citation.get("excerpt")
        if rendered:
            preview = rendered.strip().splitlines()
            snippet = preview[0]
            if len(snippet) > 160:
                snippet = snippet[:157].rstrip() + "..."
            print(f"    {snippet}")


if __name__ == "__main__":
    main()
