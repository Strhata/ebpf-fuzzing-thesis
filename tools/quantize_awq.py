#!/usr/bin/env python3
"""
quantize_awq.py — AWQ-quantize the merged LoRA model for fast inference.

Loads the bf16 merged model produced by merge_lora.py, runs AWQ calibration
using ~128 samples from the training dataset, and saves the AWQ-quantized
model to disk.

Must run AFTER merge_lora.py.

Usage:
  pixi run python tools/quantize_awq.py
  pixi run python tools/quantize_awq.py --model checkpoints/curated_merged --output checkpoints/curated_awq

Guardrails:
  1. Pre-flight: merged model files exist
  2. Calibration data: at least 64 samples loaded with correct format
  3. Size reduction: AWQ model must be significantly smaller than merged (>= 2x)
  4. Smoke test: AWQ model generates non-empty valid output for inference prompt
"""

import argparse
import sys
import warnings
from pathlib import Path

import subprocess

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)

MODEL_PATH   = REPO_ROOT / "checkpoints" / "curated_merged"
DATASET_PATH = REPO_ROOT / "data" / "dataset_final_qwen.jsonl"
OUTPUT_PATH  = REPO_ROOT / "checkpoints" / "curated_awq"
PROBE_PROMPT = "Kernel: unknown | Status: VALID\n### ASSEMBLY:\n"
N_CALIB      = 128


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_calib_data(dataset_path: Path, tokenizer, n: int, max_seq_len: int = 512) -> list[list[int]]:
    """Load n calibration samples pre-tokenized as list[list[int]].

    autoawq's get_calib_dataset filters out any sequence longer than max_seq_len
    (default 512). Our eBPF logs are multi-thousand tokens, so we must truncate
    before passing — otherwise all samples are dropped and quantize() crashes.
    The list[list[int]] path skips re-encoding inside autoawq.
    """
    import json
    samples = []
    with open(dataset_path) as f:
        for line in f:
            if len(samples) >= n:
                break
            rec = json.loads(line)
            kernel_ver = rec.get("kernel_version", "unknown")
            assembly   = rec.get("verifier_log", "")
            if rec.get("is_valid", False):
                text = f"Kernel: {kernel_ver} | Status: VALID\n### ASSEMBLY:\n{assembly}"
            else:
                err  = rec.get("error_reason_clean", "Unknown error")
                text = f"Kernel: {kernel_ver} | Status: INVALID | Error: {err}\n### ASSEMBLY:\n{assembly}"
            ids = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
            samples.append(ids)
    return samples


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",   default=str(MODEL_PATH))
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--output",  default=str(OUTPUT_PATH))
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    args = ap.parse_args()

    model_path   = Path(args.model)
    dataset_path = Path(args.dataset)
    output_path  = Path(args.output)

    # ── guardrail 1: pre-flight ──────────────────────────────────────────────
    print("[*] Guardrail 1: pre-flight checks")
    for f in ["config.json", "generation_config.json"]:
        if not (model_path / f).exists():
            print(f"[!] Missing: {model_path / f}", file=sys.stderr)
            print(f"[!] Run merge_lora.py first.", file=sys.stderr)
            sys.exit(1)
    if not dataset_path.exists():
        print(f"[!] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[+] Merged model OK at {model_path}")
    print(f"[+] Dataset OK at {dataset_path}")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    already_quantized = (output_path / "config.json").exists()

    if already_quantized:
        print(f"[*] AWQ model already exists at {output_path} — skipping quantization.")
        print(f"    Delete {output_path} to re-quantize.")
    else:
        # ── load model ───────────────────────────────────────────────────────
        print(f"\n[*] Loading merged model for AWQ quantization...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoAWQForCausalLM.from_pretrained(str(model_path), device_map="auto")

        # ── guardrail 2: calibration data ────────────────────────────────────
        print(f"\n[*] Guardrail 2: loading {args.n_calib} calibration samples...")
        calib_data = _load_calib_data(dataset_path, tokenizer, args.n_calib)
        if len(calib_data) < 64:
            print(f"[!] Only loaded {len(calib_data)} samples — expected >= 64.", file=sys.stderr)
            sys.exit(1)
        print(f"[+] Calibration data: {len(calib_data)} samples loaded with correct prompt format.")

        # ── quantize ─────────────────────────────────────────────────────────
        print(f"\n[*] Running AWQ quantization (this takes a few minutes)...")
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

        print(f"[*] Saving AWQ model to {output_path} ...")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))

    # ── guardrail 3: size reduction ───────────────────────────────────────────
    print("\n[*] Guardrail 3: checking size reduction...")
    merged_gb = _dir_size_gb(model_path)
    awq_gb    = _dir_size_gb(output_path)
    ratio     = merged_gb / awq_gb if awq_gb > 0 else 0
    print(f"[+] Merged model: {merged_gb:.2f} GB")
    print(f"[+] AWQ model:    {awq_gb:.2f} GB")
    print(f"[+] Compression:  {ratio:.1f}x")
    if ratio < 2.0:
        print(f"[!] Compression ratio {ratio:.1f}x is lower than expected (>= 2x).", file=sys.stderr)
        print(f"[!] AWQ quantization may not have applied correctly.", file=sys.stderr)
        sys.exit(1)
    print("[+] Size check: PASS")

    # ── guardrail 4: smoke test ───────────────────────────────────────────────
    print("\n[*] Guardrail 4: smoke test — loading AWQ model and generating...")
    from awq import AutoAWQForCausalLM as AWQ
    import torch

    awq_model = AWQ.from_quantized(str(output_path), device_map="auto")
    awq_model.eval()
    awq_tok   = AutoTokenizer.from_pretrained(str(output_path))

    device = next(awq_model.parameters()).device
    inputs = awq_tok(PROBE_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = awq_model.generate(**inputs, max_new_tokens=30, do_sample=False)
    generated = awq_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if not generated.strip():
        print("[!] Smoke test FAILED — AWQ model generated empty output.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Smoke test: PASS")
    print(f"    Sample output: {generated[:100]!r}")

    print(f"\n[+] Done. AWQ model → {output_path}")
    print(f"    Next: make eval")


if __name__ == "__main__":
    main()
