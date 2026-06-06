#!/usr/bin/env python3
"""
diversity_sample.py — the generate→validate→measure loop for anti-clustering work.

Generates N candidate eBPF programs (batched, left-padded inference), encodes them,
validates them on the KCOV VM in batched --batch calls, then reports diversity KPIs:
total unique PCs, per-PC novelty, encode rate, and accept rate. Reuses the benchmark
analyze() / aggregate() so numbers are comparable across tools.

Generation is seeded (one seed for the whole run) so a fixed seed + batch size
reproduces the same candidate set.

Usage:
  pixi run python tools/diversity_sample.py \\
      --model checkpoints/sft_retrain/checkpoint-1500-merged \\
      --n 1000 --temperature 1.0 --top-p 0.95 --batch-size 64 \\
      --out benchmarks/diversity/run.json
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tools"))
sys.path.insert(0, str(_REPO_ROOT / "ml"))

from benchmark_lib import (  # noqa: E402
    GeneratorResult,
    ProgramRecord,
    ValidationResult,
    aggregate,
    analyze,
    generate_batch,
)

_PROMPT = "[coverage=high][novelty=high]\n### ASSEMBLY:\n"


def _batch_validate_fn(hex_list: list[str], ssh: object) -> list[dict] | None:
    """Module-level wrapper around reward._validate_batch_on_vm so tests can patch
    it without importing reward.py (which has module-level side effects)."""
    import reward  # noqa: PLC0415
    return reward._validate_batch_on_vm(hex_list, ssh)


def validations_from_batch(
    hexes: list[str | None], ssh: object, chunk_size: int = 500
) -> list[ValidationResult]:
    """Validate encoded programs via batched --batch calls and map results back by index.

    `hexes[i]` is the encoded hex, or None for an encode failure (→ ERROR, no VM).
    Programs are validated in chunks of `chunk_size`; a whole-chunk failure leaves
    those entries as ERROR rather than aborting the run.
    """
    results = [ValidationResult(verdict="ERROR", pcs=[]) for _ in hexes]
    idx = [i for i, h in enumerate(hexes) if h]
    for s in range(0, len(idx), chunk_size):
        chunk_idx = idx[s:s + chunk_size]
        chunk_hex = [hexes[i] for i in chunk_idx]
        batch = _batch_validate_fn(chunk_hex, ssh)
        if batch is None:
            continue  # whole-chunk VM failure → leave ERROR for these
        for i, res in zip(chunk_idx, batch):
            results[i] = ValidationResult(
                verdict=res.get("verdict", "ERROR"), pcs=res.get("pcs", [])
            )
    return results


def _load_model(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="Path to the merged bf16 model")
    ap.add_argument("--n", type=int, default=1000, help="Number of candidates to sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64, help="Generation batch size")
    ap.add_argument("--val-chunk", type=int, default=500, help="Programs per --batch VM call")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=_PROMPT)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=10022)
    ap.add_argument("--key", default=None, help="SSH key path (default: reward.py default)")
    ap.add_argument("--out", default=None, help="Report JSON (default: benchmarks/diversity/<UTC>.json)")
    args = ap.parse_args()

    import torch
    from reward import SSHClient, _encode_to_hex

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "benchmarks" / "diversity"
        / f"diversity_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading {args.model}")
    model, tok = _load_model(args.model)

    # --- generate (seeded → reproducible candidate set for a fixed batch size) ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    print(f"[*] Generating {args.n} candidates (temp={args.temperature}, top_p={args.top_p}, "
          f"batch={args.batch_size})")
    gen = generate_batch(
        model, tok, [args.prompt] * args.n, args.max_new_tokens,
        do_sample=True, temperature=args.temperature, top_p=args.top_p,
        batch_size=args.batch_size,
    )
    timing = GeneratorResult(
        assemblies=gen.texts,
        tokens_per_sec=gen.total_new_tokens / gen.seconds if gen.seconds > 0 else 0.0,
    )
    peak_run_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    # --- encode + batched KCOV validation ---
    hexes = [_encode_to_hex(t) or None for t in gen.texts]
    key = args.key or SSHClient.__dataclass_fields__["key"].default
    ssh = SSHClient(host=args.host, port=args.port, key=key)
    print(f"[*] Validating {sum(h is not None for h in hexes)} encoded programs via --batch")
    vres = validations_from_batch(hexes, ssh, chunk_size=args.val_chunk)

    # --- KPIs (reuse benchmark aggregator) ---
    records = [
        ProgramRecord(assembly=t, hex_str=h, complexity=analyze(t), validation=v)
        for t, h, v in zip(gen.texts, hexes, vres)
    ]
    kpis = aggregate(records, timing, peak_load_gpu_mb=0.0, peak_run_gpu_mb=peak_run_mb)
    wall = time.monotonic() - t0

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "n": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "wall_seconds": round(wall, 1),
        "kpis": asdict(kpis),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n[+] n={args.n}  encode_rate={kpis.encode_rate:.3f}  "
          f"accept_rate={kpis.pass_rate:.3f}")
    print(f"[+] total_unique_pcs={kpis.total_unique_pcs}  "
          f"novelty_score={kpis.novelty_score:.4f}")
    print(f"[+] wall={wall:.0f}s  gen_tokens_per_sec={timing.tokens_per_sec:.0f}")
    print(f"[+] Report → {out_path}")


if __name__ == "__main__":
    main()
