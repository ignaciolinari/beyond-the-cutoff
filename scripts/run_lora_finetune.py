#!/usr/bin/env python3
"""CLI to launch local LoRA fine-tuning runs using PEFT."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from beyond_the_cutoff.config import load_config
from beyond_the_cutoff.models.finetune import (
    SupervisedSample,
    load_supervised_samples,
    summarise_task_types,
    tokenise_samples,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA/PEFT fine-tuning against the offline dataset"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Project configuration file"
    )
    parser.add_argument("--train-dataset", required=True, help="Path to offline_dataset.jsonl")
    parser.add_argument("--output-dir", help="Override output directory for LoRA adapters")
    parser.add_argument(
        "--val-split", type=float, default=0.05, help="Fraction of data reserved for evaluation"
    )
    parser.add_argument("--max-samples", type=int, help="Optional cap on training samples")
    parser.add_argument(
        "--task-types",
        nargs="*",
        help="Optional task types to include (e.g. qa summaries citations)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length for tokenisation"
    )
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, help="Gradient accumulation steps")
    parser.add_argument(
        "--max-steps", type=int, help="Total optimisation steps (set <=0 to disable)"
    )
    parser.add_argument(
        "--num-epochs", type=float, help="Number of training epochs when max_steps is disabled"
    )
    parser.add_argument("--lora-r", type=int, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        nargs="*",
        help="Names of modules to wrap with LoRA adapters (default: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)",
    )
    parser.add_argument("--logging-steps", type=int, default=20, help="Logging interval (steps)")
    parser.add_argument(
        "--save-steps", type=int, default=100, help="Checkpoint save interval (steps)"
    )
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation interval (steps)")
    parser.add_argument("--resume-from", help="Path to checkpoint directory to resume from")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    parser.add_argument("--fp16", action="store_true", help="Enable float16 training")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--report-to",
        nargs="*",
        help="Optional list of trackers (e.g. tensorboard, wandb). Use 'none' to disable.",
    )
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom model code from the pretrained checkpoint",
    )
    return parser.parse_args()


def resolve_output_dir(base_dir: Path, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = (base_dir / f"run-{timestamp}").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def determine_target_modules(user_override: Sequence[str] | None) -> list[str]:
    if user_override:
        return [module.strip() for module in user_override if module.strip()]
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def compute_dataset_hash(samples: Sequence[SupervisedSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.prompt.encode("utf-8"))
        digest.update(sample.response.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cfg = load_config(args.config)
    ft_cfg = cfg.fine_tuning

    dataset_path = Path(args.train_dataset)
    samples: list[SupervisedSample] = load_supervised_samples(
        dataset_path, allowed_task_types=args.task_types
    )
    if args.max_samples is not None and args.max_samples > 0:
        rng = random.Random(args.seed)
        if args.max_samples < len(samples):
            samples = rng.sample(samples, k=args.max_samples)
        else:
            rng.shuffle(samples)

    if not samples:
        raise SystemExit("No training samples available after filtering.")

    task_histogram = summarise_task_types(samples)
    dataset_hash = compute_dataset_hash(samples)

    load_tokenizer = cast(
        Callable[..., "PreTrainedTokenizerBase"],
        AutoTokenizer.from_pretrained,
    )
    tokenizer = load_tokenizer(
        ft_cfg.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    tokenised_dataset, stats = tokenise_samples(
        samples,
        tokenizer,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    if tokenised_dataset.num_rows == 0:
        raise SystemExit(
            "All samples were filtered out during tokenisation."
            "Try increasing max sequence length or adjusting filters."
        )

    logger.info(
        "Loaded %s samples (%s usable). Token stats: avg prompt %.1f, avg response %.1f.",
        stats.total_samples,
        stats.usable_samples,
        stats.avg_prompt_tokens,
        stats.avg_response_tokens,
    )
    logger.info("Task histogram: %s", task_histogram)

    if not args.no_eval and args.val_split > 0 and tokenised_dataset.num_rows >= 2:
        split = tokenised_dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = tokenised_dataset
        eval_dataset = None

    output_dir = resolve_output_dir(ft_cfg.adapter_output_dir, args.output_dir)

    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            ft_cfg.base_model,
            trust_remote_code=args.trust_remote_code,
        ),
    )
    model.config.use_cache = False
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int):
        model.config.pad_token_id = pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r or ft_cfg.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=determine_target_modules(args.target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = cast(Any, get_peft_model(model, lora_config))
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    learning_rate = args.learning_rate or ft_cfg.learning_rate
    batch_size = args.batch_size or ft_cfg.batch_size
    grad_accum = args.gradient_accumulation or ft_cfg.gradient_accumulation_steps
    max_steps = args.max_steps if args.max_steps is not None else ft_cfg.max_steps
    if max_steps is not None and max_steps <= 0:
        max_steps = None
    num_epochs = args.num_epochs if args.num_epochs is not None else 1.0
    if max_steps:
        num_epochs = 1.0

    report_to: list[str] | str | None = None
    if args.report_to:
        if len(args.report_to) == 1 and args.report_to[0].lower() == "none":
            report_to = "none"
        else:
            filtered = [item for item in args.report_to if item.strip()]
            report_to = filtered or None

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": max_steps if max_steps is not None else -1,
        "num_train_epochs": num_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": report_to,
    }
    if eval_dataset and not args.no_eval:
        training_kwargs["evaluation_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps
    else:
        training_kwargs["evaluation_strategy"] = "no"
        training_kwargs["eval_steps"] = None

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not args.no_eval else None,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    summary_path = output_dir / "training_summary.json"
    payload = {
        "base_model": ft_cfg.base_model,
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_hash": dataset_hash,
        "task_histogram": task_histogram,
        "tokenisation": asdict(stats),
        "train_rows": train_dataset.num_rows,
        "eval_rows": eval_dataset.num_rows if eval_dataset else 0,
        "training_arguments": json.loads(
            json.dumps(
                cast(Callable[[], dict[str, Any]], training_args.to_dict)(),
                default=str,
            )
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Training complete. Adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
