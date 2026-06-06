#!/usr/bin/env python3
"""
generate_bytecodes.py — Stage 1 of the 60%-reconstruction: GENERATE ONLY.

Reloads the original 60%-era SFT-v1 model, generates N eBPF programs with the
exact same prompt/sampling as tools/evaluate_passrate.py, compiles each through
the clang path (compile_to_hex), and PERSISTS assembly + hex + metadata to JSONL.

No VM, no validation. Validation is a separate stage (validate_bytecodes.py) so
the generated programs become a reproducible artifact that can be re-validated
through any tool (ebpf_validator, kcov_validator) as many times as needed.

Generation is SEEDED (per-program seed = base_seed + id) so the original sin —
non-deterministic, unsaved programs — is fixed: every record can be regenerated.

Usage:
  pixi run python tools/generate_bytecodes.py \\
      --model checkpoints/curated_merged \\
      --n 100 \\
      --seed 42 \\
      --out data/reconstruction/sft_v1_$(date +%Y%m%d).jsonl
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Script lives in tools/, so its own dir is on sys.path[0] → import siblings.
from evaluate_passrate import compile_to_hex, load_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
# ml/ holds the pure-python encoder (19%-era / RL pipeline path).
sys.path.insert(0, str(_REPO_ROOT / "ml"))
from reward import _encode_to_hex, strip_verifier_log  # noqa: E402

_PROMPT = "Kernel: unknown | Status: VALID\n### ASSEMBLY:\n"


def generate_one(model, tokenizer, seed: int, temperature: float, max_new_tokens: int) -> str:
    """Generate a single program with a fixed seed → reproducible."""
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer(_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### ASSEMBLY:\n")[-1].strip()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="checkpoints/curated_merged",
                    help="Path to bf16 merged model (the 60%%-era SFT-v1).")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed; program i uses seed+i.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Generate this many programs per model.generate call. "
                         ">1 uses batched left-padded inference (faster) but seeds "
                         "once per batch, not per program (loses per-record reproducibility).")
    ap.add_argument("--out", default=None,
                    help="Output JSONL (default: data/reconstruction/sft_v1_<UTCdate>.jsonl)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "data" / "reconstruction"
        / f"sft_v1_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_label = Path(args.model).name
    print(f"[*] Generating {args.n} programs from {args.model} (seed base {args.seed})")
    model, tokenizer = load_model(args.model)

    # --- generation: per-seed (reproducible) or batched (faster) ---
    if args.batch_size > 1:
        import torch
        from benchmark_lib import generate_batch  # noqa: E402
        print(f"[*] Batched generation (batch_size={args.batch_size}); "
              f"seeded once per batch, not per program")
        torch.manual_seed(args.seed)
        res = generate_batch(
            model, tokenizer, [_PROMPT] * args.n, args.max_new_tokens,
            do_sample=True, temperature=args.temperature, batch_size=args.batch_size,
        )
        assemblies = [t.strip() for t in res.texts]
        seeds = [args.seed] * args.n  # batch-seeded, not per-program
    else:
        assemblies = [
            generate_one(model, tokenizer, args.seed + i, args.temperature, args.max_new_tokens)
            for i in range(args.n)
        ]
        seeds = [args.seed + i for i in range(args.n)]

    clang_ok = 0
    encoder_ok = 0
    with out_path.open("w") as fout, tempfile.TemporaryDirectory() as tmpdir:
        for i in range(args.n):
            seed_i = seeds[i]
            assembly = assemblies[i]
            # Path A — faithful 60%-era: verifier-log → parser → clang → objcopy.
            clang_hex = compile_to_hex(assembly, tmpdir)
            # Path B — 19%-era pure encoder: strip verifier log → infer opcodes,
            # bypasses clang (no compile gate, no parser-junk failures).
            encoder_hex = _encode_to_hex(strip_verifier_log(assembly))
            if clang_hex:
                clang_ok += 1
            if encoder_hex:
                encoder_ok += 1
            rec = {
                "id": i,
                "seed": seed_i,
                "model": model_label,
                "model_path": args.model,
                "prompt": _PROMPT,
                "gen": {
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": True,
                    "batch_size": args.batch_size,
                },
                "assembly": assembly,
                "clang_hex": clang_hex or "",
                "encoder_hex": encoder_hex or "",
                "clang_compiled": bool(clang_hex),
                "encoder_compiled": bool(encoder_hex),
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n}  (clang {clang_ok}, encoder {encoder_ok})")

    print(f"\n[+] Generated {args.n} | clang {clang_ok} "
          f"({clang_ok/args.n*100:.1f}%) | encoder {encoder_ok} "
          f"({encoder_ok/args.n*100:.1f}%)")
    print(f"[+] Saved → {out_path}")
    print(f"[*] Next: validate through KCOV with validate_bytecodes.py")


if __name__ == "__main__":
    main()
