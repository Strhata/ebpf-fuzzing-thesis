#!/usr/bin/env python3
"""
train.py — Phase 2 SFT training script.

Two runs, identical config. Only variable: invalid class distribution.

  --run curated   : resume from checkpoints/sft_fase1/checkpoint-1500
  --run baseline  : train from scratch on random-invalid dataset

Usage (from repo root):
  pixi run python ml/train.py --run curated
  pixi run python ml/train.py --run baseline

WandB: set WANDB_API_KEY env var for headless login, or run `wandb login` first.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"

RUNS = {
    "curated": {
        "dataset":    REPO_ROOT / "data" / "dataset_final_qwen.jsonl",
        "output_dir": REPO_ROOT / "checkpoints" / "curated_3ep",
        "resume":     REPO_ROOT / "checkpoints" / "sft_fase1" / "checkpoint-1500",
        "wandb_name": "curated-3ep",
    },
    "baseline": {
        "dataset":    REPO_ROOT / "data" / "dataset_baseline_qwen.jsonl",
        "output_dir": REPO_ROOT / "checkpoints" / "baseline_3ep",
        "resume":     None,
        "wandb_name": "baseline-3ep",
    },
}


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_prompt(sample, tokenizer):
    kernel_ver = sample.get("kernel_version", "unknown")
    assembly   = sample.get("verifier_log", "")
    if sample.get("is_valid", False):
        text = f"Kernel: {kernel_ver} | Status: VALID\n### ASSEMBLY:\n{assembly}{tokenizer.eos_token}"
    else:
        err  = sample.get("error_reason_clean", "Unknown error")
        text = f"Kernel: {kernel_ver} | Status: INVALID | Error: {err}\n### ASSEMBLY:\n{assembly}{tokenizer.eos_token}"
    return {"formatted_prompt": text}


def tokenize(sample, tokenizer):
    enc = tokenizer(
        sample["formatted_prompt"],
        max_length=768,
        truncation=True,
        padding="max_length",
    )
    enc["labels"] = [
        l if l != tokenizer.pad_token_id else -100
        for l in enc["input_ids"]
    ]
    return enc


def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=["curated", "baseline"])
    args = ap.parse_args()

    cfg = RUNS[args.run]

    # Sanity checks
    if not cfg["dataset"].exists():
        print(f"[!] Dataset not found: {cfg['dataset']}", file=sys.stderr)
        print(f"[!] Run `make setup` first.", file=sys.stderr)
        sys.exit(1)
    if cfg["resume"] and not cfg["resume"].exists():
        print(f"[!] Checkpoint not found: {cfg['resume']}", file=sys.stderr)
        print(f"[!] Run `make setup` first.", file=sys.stderr)
        sys.exit(1)

    wandb.init(project="ebpf-thesis", name=cfg["wandb_name"], reinit=True)

    tokenizer = load_tokenizer()

    dataset = load_dataset("json", data_files=str(cfg["dataset"]), split="train")
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    train   = split["train"].map(lambda s: build_prompt(s, tokenizer)).map(lambda s: tokenize(s, tokenizer))
    val     = split["test"].map(lambda s: build_prompt(s, tokenizer)).map(lambda s: tokenize(s, tokenizer))
    print(f"[*] Train: {len(train)}  Val: {len(val)}")

    base = load_base_model()

    if cfg["resume"]:
        print(f"[*] Resuming from {cfg['resume']}")
        model = PeftModel.from_pretrained(base, str(cfg["resume"]), is_trainable=True)
    else:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(base, lora)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(cfg["output_dir"]),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=cfg["wandb_name"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
    )

    trainer.train()
    adapter_path = cfg["output_dir"] / "adattatore_ebpf_v1"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    wandb.finish()
    print(f"[+] Model saved → {adapter_path}")


if __name__ == "__main__":
    main()
