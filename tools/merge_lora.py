#!/usr/bin/env python3
"""
merge_lora.py — Merge LoRA adapter into base model weights.

Loads Qwen2.5-Coder-1.5B in full bf16 precision, applies the trained LoRA
adapter, calls merge_and_unload() to bake W_new = W + A*B into the base
weights, then saves the merged model to disk.

Must run BEFORE quantize_awq.py.

Usage:
  pixi run python tools/merge_lora.py
  pixi run python tools/merge_lora.py --adapter checkpoints/curated_3ep/adattatore_ebpf_v1 --output checkpoints/curated_merged

Guardrails:
  1. Pre-flight: adapter files exist and are valid
  2. File-size sanity: merged model must be >= 2 GB on disk
  3. Smoke test: merged model generates non-empty output for the probe prompt
"""

import argparse
import sys
from pathlib import Path

import subprocess

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)

MODEL_NAME   = "Qwen/Qwen2.5-Coder-1.5B"
PROBE_PROMPT = "Kernel: unknown | Status: VALID\n### ASSEMBLY:\n"
PROBE_TOKENS = 20


# ── helpers ─────────────────────────────────────────────────────────────────


def _generate_tokens(model, tokenizer, n: int) -> list[int]:
    """Generate n tokens from PROBE_PROMPT, return token ids."""
    import torch
    inputs = tokenizer(PROBE_PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    return out[0][inputs["input_ids"].shape[1]:].tolist()


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=str(REPO_ROOT / "checkpoints" / "curated_3ep" / "adattatore_ebpf_v1"))
    ap.add_argument("--output",  default=str(REPO_ROOT / "checkpoints" / "curated_merged"))
    args = ap.parse_args()

    adapter_path = Path(args.adapter)
    output_path  = Path(args.output)

    # ── guardrail 1: pre-flight ──────────────────────────────────────────────
    print("[*] Guardrail 1: pre-flight checks")
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for f in required:
        if not (adapter_path / f).exists():
            print(f"[!] Missing: {adapter_path / f}", file=sys.stderr)
            sys.exit(1)
    print(f"[+] Adapter files OK at {adapter_path}")

    # skip if already merged
    if (output_path / "config.json").exists():
        print(f"[*] Merged model already exists at {output_path} — skipping merge.")
        print(f"    Delete {output_path} to re-run.")
        return

    # ── merge ────────────────────────────────────────────────────────────────
    print(f"\n[*] Loading base model {MODEL_NAME} in bf16 for merge...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    base  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("[*] Merging LoRA weights into base (merge_and_unload)...")
    model = model.merge_and_unload()
    model.eval()

    print(f"[*] Saving merged model to {output_path} ...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tok.save_pretrained(str(output_path))

    # ── guardrail 2: file size ───────────────────────────────────────────────
    print("\n[*] Guardrail 2: checking merged model size on disk...")
    size_gb = _dir_size_gb(output_path)
    print(f"[+] Merged model size: {size_gb:.2f} GB")
    if size_gb < 2.0:
        print(f"[!] Suspiciously small ({size_gb:.2f} GB) — expected >= 2 GB for a 1.5B bf16 model.", file=sys.stderr)
        sys.exit(1)
    print("[+] Size check: PASS")

    # ── guardrail 3: smoke test ───────────────────────────────────────────────
    print("\n[*] Guardrail 3: smoke test — generating from merged model...")
    tokens = _generate_tokens(model, tok, PROBE_TOKENS)
    decoded = tok.decode(tokens, skip_special_tokens=True)
    if not decoded.strip():
        print("[!] Smoke test FAILED — merged model generated empty output.", file=sys.stderr)
        sys.exit(1)
    print(f"[+] Smoke test: PASS")
    print(f"    Sample: {decoded[:80]!r}")

    print(f"\n[+] Done. Merged model → {output_path}")
    print(f"    Next: pixi run python tools/quantize_awq.py")


if __name__ == "__main__":
    main()
