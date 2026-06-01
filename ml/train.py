#!/usr/bin/env python3
"""
train.py — SFT phase: fine-tune Qwen2.5-Coder-1.5B from base on the enriched dataset.

Loads data/dataset_final_qwen_enriched.jsonl, produces a stratified 70/15/15 split
(by is_valid, seed=42), freezes the test split to disk before any training begins,
then runs 3-epoch QLoRA SFT with per-epoch encoder pass-rate evaluation.

Usage:
    pixi run python ml/train.py
    pixi run python ml/train.py --output-dir checkpoints/sft_retrain
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import wandb
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

REPO_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(Path(__file__).parent))
from reward import _encode_to_hex, strip_verifier_log

MODEL_NAME   = "Qwen/Qwen2.5-Coder-1.5B"
ENRICHED_DATA = REPO_ROOT / "data" / "dataset_final_qwen_enriched.jsonl"
TEST_SPLIT_PATH = REPO_ROOT / "data" / "test_split.jsonl"
DEFAULT_OUTPUT  = REPO_ROOT / "checkpoints" / "sft_retrain"


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_split(
    path: Path = ENRICHED_DATA,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load enriched JSONL and return (train, val, test) as lists of dicts.

    Applies two stratified splits (by is_valid) to produce 70/15/15.
    Skips any row with the _metadata key (first-line header from enrich_dataset.py).
    """
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_metadata" in obj:
            continue
        rows.append(obj)

    labels = [int(r.get("is_valid", False)) for r in rows]

    # Step 1: split off 15% test
    trainval, test, lv_trainval, _ = train_test_split(
        rows, labels, test_size=0.15, stratify=labels, random_state=seed
    )
    # Step 2: split trainval into 70% train / 15% val (15/85 ≈ 0.1765)
    train, val = train_test_split(
        trainval, test_size=15 / 85, stratify=lv_trainval, random_state=seed
    )
    return train, val, test


def save_test_split(test_samples: list[dict], path: Path = TEST_SPLIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in test_samples:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def build_prompt(sample: dict, tokenizer) -> dict:
    coverage = sample.get("coverage_bin", "low")
    novelty  = sample.get("novelty_bin",  "low")
    bare = strip_verifier_log(sample.get("verifier_log", ""))
    text = f"[coverage={coverage}][novelty={novelty}]\n### ASSEMBLY:\n{bare}{tokenizer.eos_token}"
    return {"formatted_prompt": text}


def tokenize(sample: dict, tokenizer) -> dict:
    enc = tokenizer(
        sample["formatted_prompt"],
        max_length=768,
        truncation=True,
        padding="max_length",
    )
    enc["labels"] = [
        tok if tok != tokenizer.pad_token_id else -100
        for tok in enc["input_ids"]
    ]
    return enc


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def make_training_args(output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,       # 2e-4 caused plateau at ~1.2; lower rate converges further
        warmup_ratio=0.03,        # ~216 warmup steps over 7224 total
        lr_scheduler_type="cosine",
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=500,
        eval_strategy="epoch",
        seed=42,
        report_to="wandb",
    )


# ---------------------------------------------------------------------------
# Per-epoch encoder pass-rate callback
# ---------------------------------------------------------------------------

class EncoderPassRateCallback(TrainerCallback):
    """After each epoch: sample n completions from val prompts, run _encode_to_hex,
    report pass-rate and val loss to WandB."""

    def __init__(self, val_samples: list[dict], tokenizer, n_samples: int = 50):
        self._val_samples = val_samples
        self._tokenizer   = tokenizer
        self._n_samples   = n_samples

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return
        rng = random.Random(state.epoch)
        pool = rng.sample(self._val_samples, min(self._n_samples, len(self._val_samples)))

        model.eval()
        passed = 0
        with torch.no_grad():
            for sample in pool:
                prompt = build_prompt(sample, self._tokenizer)["formatted_prompt"]
                ids = self._tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).input_ids.to(model.device)
                out = model.generate(ids, max_new_tokens=256, do_sample=False)
                completion = self._tokenizer.decode(
                    out[0][ids.shape[1]:], skip_special_tokens=True
                )
                if _encode_to_hex(completion) is not None:
                    passed += 1

        pass_rate = passed / len(pool)
        wandb.log({"encoder_pass_rate": pass_rate, "epoch": state.epoch})


# ---------------------------------------------------------------------------
# Post-training test-split evaluation
# ---------------------------------------------------------------------------

def evaluate_test_split(trainer, test_samples: list[dict], tokenizer) -> None:
    from datasets import Dataset

    def _fmt(s):
        return tokenize(build_prompt(s, tokenizer), tokenizer)

    test_ds = Dataset.from_list(test_samples).map(_fmt)
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    wandb.log(metrics)

    # Encoder pass-rate on test set (up to 50 samples)
    model = trainer.model
    model.eval()
    pool = test_samples[:50]
    passed = sum(
        1 for s in pool
        if _encode_to_hex(
            tokenizer.decode(
                model.generate(
                    tokenizer(
                        build_prompt(s, tokenizer)["formatted_prompt"],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).input_ids.to(model.device),
                    max_new_tokens=256,
                    do_sample=False,
                )[0],
                skip_special_tokens=True,
            )
        ) is not None
    )
    wandb.log({"test/encoder_pass_rate": passed / len(pool)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--data",       type=Path, default=ENRICHED_DATA)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"[!] Enriched dataset not found: {args.data}", file=sys.stderr)
        print(f"[!] Run: pixi run python ml/enrich_dataset.py", file=sys.stderr)
        sys.exit(1)

    # --- 1. Split (test split saved BEFORE any model is loaded) ---
    print("[*] Loading and splitting dataset ...", flush=True)
    train_samples, val_samples, test_samples = load_and_split(args.data)
    print(f"    train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}", flush=True)

    print(f"[*] Saving test split → {TEST_SPLIT_PATH}", flush=True)
    save_test_split(test_samples)

    # --- 2. Tokenizer and model ---
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,  # load weights directly to GPU; avoids 6 GB CPU copy
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # Include MLP projections: Qwen2.5 stores most learned representations
        # in gate/up/down_proj; attention-only LoRA plateaus early on code tasks.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    # --- 3. Datasets ---
    # Strip heavy string columns (bytecode_hex, verifier_log, etc.) before
    # creating HuggingFace Datasets — keeping only what build_prompt needs.
    _prompt_keys = {"verifier_log", "coverage_bin", "novelty_bin"}

    def _fmt(s):
        return tokenize(build_prompt(s, tokenizer), tokenizer)

    def _minimal(samples):
        return [{k: s.get(k, "") for k in _prompt_keys} for s in samples]

    train_ds = Dataset.from_list(_minimal(train_samples)).map(_fmt, remove_columns=list(_prompt_keys))
    val_ds   = Dataset.from_list(_minimal(val_samples)).map(_fmt,   remove_columns=list(_prompt_keys))
    for ds in (train_ds, val_ds):
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # --- 4. Trainer ---
    from transformers import Trainer

    wandb.init(project="ebpf-thesis", name="sft-retrain", reinit=True)

    training_args = make_training_args(args.output_dir)
    callbacks = [EncoderPassRateCallback(val_samples, tokenizer)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
    )

    trainer.train()

    adapter_path = args.output_dir / "sft_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[+] Adapter saved → {adapter_path}", flush=True)

    # --- 5. Post-training test evaluation ---
    print("[*] Evaluating frozen test split ...", flush=True)
    evaluate_test_split(trainer, test_samples, tokenizer)

    wandb.finish()


if __name__ == "__main__":
    main()
