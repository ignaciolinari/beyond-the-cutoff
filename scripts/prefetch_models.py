#!/usr/bin/env python3
"""Prefetch Hugging Face models into the local cache.

This is useful for speeding up local development or seeding a shared cache in CI
environments. By default it downloads ``HuggingFaceTB/SmolLM2-135M`` but you can
pass one or more alternative model ids on the command line.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=[DEFAULT_MODEL],
        help=(
            "Model ids to prefetch (default: %(default)s). "
            "Pass multiple ids to cache several checkpoints."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Optional directory to use as Hugging Face cache (sets HF_HOME if provided). "
            "Will be created if it doesn’t exist."
        ),
    )
    return parser.parse_args(argv)


def ensure_cache_dir(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    resolved = cache_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(resolved))
    print(f"Using Hugging Face cache at {resolved}")


def prefetch(model_ids: Iterable[str]) -> None:
    transformers = import_module("transformers")
    tokenizer_cls = cast(Any, transformers.AutoTokenizer)
    model_cls = cast(Any, transformers.AutoModelForCausalLM)

    for model_id in model_ids:
        model_id = model_id.strip()
        if not model_id:
            continue
        print(f"Prefetching tokenizer for {model_id}…", flush=True)
        tokenizer_cls.from_pretrained(model_id)
        print(f"Prefetching model weights for {model_id}…", flush=True)
        model_cls.from_pretrained(model_id)
    print("Prefetch complete.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_cache_dir(args.cache_dir)
    try:
        prefetch(args.models)
    except Exception as exc:  # pragma: no cover - surface informative errors
        print(f"Failed to prefetch models: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
