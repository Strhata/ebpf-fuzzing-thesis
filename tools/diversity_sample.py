#!/usr/bin/env python3
"""
diversity_sample.py — split generate→validate→measure loop for anti-clustering work.

The full test has two halves that live on different machines:

  * `generate` (GPU, e.g. Colab A100): sample N candidate eBPF programs with batched,
    left-padded inference, encode each to hex, and write a candidates JSONL artifact.
    No KCOV VM is touched here. Optionally push the artifact straight to HuggingFace.

  * `validate` (local, where the KCOV VM lives): read a candidates JSONL, validate the
    encoded programs on the VM in batched --batch calls, then report diversity KPIs
    (total unique PCs, novelty, encode rate, accept rate). Reuses the benchmark
    analyze()/aggregate() so numbers are comparable across tools.

The candidates file is a JSONL whose FIRST line is a `{"_meta": {...}}` header carrying
run-level config + generation timing, and every subsequent line is one candidate
`{"assembly": ..., "hex_str": ...}` (hex_str is null on encode failure). This lets the
validate stage rebuild the KPIs (including gen throughput) without a GPU.

Generation is seeded (one seed for the whole run) so a fixed seed + batch size
reproduces the same candidate set.

Usage:
  # On Colab (A100) — generate + push artifact to HF:
  python tools/diversity_sample.py generate \\
      --model checkpoints/sft-1epoch-v2-merged \\
      --n 1000 --temperature 1.0 --top-p 0.95 --batch-size 256 \\
      --out benchmarks/diversity/candidates/sft-v2-n1000-seed42.jsonl \\
      --hf-repo Strhata/ebpf-corpus --hf-path candidates/sft-v2-n1000-seed42.jsonl

  # On WSL (KCOV VM running) — pull then validate:
  huggingface-cli download Strhata/ebpf-corpus candidates/sft-v2-n1000-seed42.jsonl \\
      --repo-type dataset --local-dir benchmarks/diversity
  python tools/diversity_sample.py validate \\
      --candidates benchmarks/diversity/candidates/sft-v2-n1000-seed42.jsonl \\
      --out benchmarks/diversity/sft-v2-n1000-seed42.json
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
    SlotKPIs,
    ValidationResult,
    aggregate,
    analyze,
    generate_batch,
)

_PROMPT = "[coverage=high][novelty=high]\n### ASSEMBLY:\n"


# --------------------------------------------------------------------------- #
# Candidates artifact IO (the Colab→WSL handoff format)
# --------------------------------------------------------------------------- #

def write_candidates(path: Path, meta: dict, assemblies: list[str], hexes: list[str | None]) -> None:
    """Write the candidates JSONL: first line is `{"_meta": ...}`, then one record per
    candidate `{"assembly", "hex_str"}`. hex_str is null on encode failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for asm, hx in zip(assemblies, hexes):
            f.write(json.dumps({"assembly": asm, "hex_str": hx}) + "\n")


def read_candidates(path: Path) -> tuple[dict, list[str], list[str | None]]:
    """Read a candidates JSONL written by write_candidates. Returns (meta, assemblies, hexes)."""
    meta: dict = {}
    assemblies: list[str] = []
    hexes: list[str | None] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and "_meta" in obj:
                meta = obj["_meta"]
                continue
            assemblies.append(obj["assembly"])
            hexes.append(obj.get("hex_str"))
    return meta, assemblies, hexes


# --------------------------------------------------------------------------- #
# Batched KCOV validation (validate stage only)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# generate (GPU stage)
# --------------------------------------------------------------------------- #

def aggregate_streaming(
    assemblies: list[str],
    hexes: list[str | None],
    batch_validate,
    timing: GeneratorResult,
    peak_run_gpu_mb: float,
    chunk_size: int = 500,
) -> tuple[SlotKPIs, dict]:
    """Memory-bounded equivalent of building ProgramRecords + aggregate().

    Folds each program's KCOV trace into running sets/counters and discards the raw
    PC list immediately, so peak memory is O(distinct PCs) not O(total PC hits × N).
    Needed because a full N=20k run holds ~tens of thousands of PCs per program and
    OOMs a small box. `batch_validate(hex_list) -> list[dict] | None` is the chunked
    VM call (None = whole-chunk failure → those programs score as ERROR/no PCs).

    Returns (SlotKPIs, valid_only) matching aggregate() + the cmd_validate valid_only block.
    """
    n = len(assemblies)

    # --- complexity pass (no VM; independent per program) ---
    n_encoded = sum_insn = jump_total = 0
    sum_opdiv = 0.0
    for t in assemblies:
        c = analyze(t)
        if c.encoded:
            n_encoded += 1
            sum_insn += c.insn_count
            sum_opdiv += c.opcode_diversity
            jump_total += c.jump_count

    # --- validation fold (chunked; raw pcs discarded per program) ---
    all_union: set[int] = set()
    valid_union: set[int] = set()
    freq: dict[int, int] = {}
    n_accepted = sum_pcs_accepted = sum_distinct_accepted = max_pcs = 0

    idx = [i for i, h in enumerate(hexes) if h]
    for s in range(0, len(idx), chunk_size):
        chunk_hex = [hexes[i] for i in idx[s:s + chunk_size]]
        batch = batch_validate(chunk_hex)
        if batch is None:
            continue  # whole-chunk VM failure → ERROR, contributes no PCs
        for res in batch:
            pcs = res.get("pcs", [])
            distinct = set(pcs)
            all_union |= distinct
            for pc in distinct:
                freq[pc] = freq.get(pc, 0) + 1
            if len(pcs) > max_pcs:
                max_pcs = len(pcs)
            if res.get("verdict") == "ACCEPTED":
                n_accepted += 1
                valid_union |= distinct
                sum_pcs_accepted += len(pcs)
                sum_distinct_accepted += len(distinct)
            del pcs, distinct

    novelty = (
        sum(1.0 - freq[pc] / n for pc in all_union) / len(all_union)
        if all_union else 0.0
    )
    kpis = SlotKPIs(
        pass_rate=n_accepted / n if n else 0.0,
        encode_rate=n_encoded / n if n else 0.0,
        avg_pcs=sum_pcs_accepted / n_accepted if n_accepted else 0.0,
        max_pcs=max_pcs,
        total_unique_pcs=len(all_union),
        novelty_score=novelty,
        avg_insn_count=sum_insn / n_encoded if n_encoded else 0.0,
        avg_opcode_diversity=sum_opdiv / n_encoded if n_encoded else 0.0,
        avg_jump_count=jump_total / n_encoded if n_encoded else 0.0,
        tokens_per_sec=timing.tokens_per_sec,
        peak_load_gpu_mb=0.0,
        peak_run_gpu_mb=peak_run_gpu_mb,
    )
    valid_only = {
        "n_accepted": n_accepted,
        "valid_unique_pcs": len(valid_union),
        "avg_distinct_pcs_per_valid": (
            sum_distinct_accepted / n_accepted if n_accepted else 0.0
        ),
    }
    return kpis, valid_only


def _load_model(path: str, adapter: str | None = None):
    """Load a bf16 model. If `adapter` is given, load it on top of the base at `path`
    (bf16, unmerged) — identical weights to a merged model, no merge step needed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def cmd_generate(args) -> None:
    import torch
    from reward import _encode_to_hex

    if args.adapter:
        load_path, adapter = args.base, args.adapter
        model_desc = f"{args.base} + adapter:{args.adapter}"
    elif args.model:
        load_path, adapter = args.model, None
        model_desc = args.model
    else:
        sys.exit("generate: pass either --model (merged) or --base + --adapter")

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "benchmarks" / "diversity" / "candidates"
        / f"candidates_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.jsonl"
    )

    print(f"[*] Loading {model_desc}")
    model, tok = _load_model(load_path, adapter)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"[*] Generating {args.n} candidates (temp={args.temperature}, top_p={args.top_p}, "
          f"batch={args.batch_size})")
    gen = generate_batch(
        model, tok, [args.prompt] * args.n, args.max_new_tokens,
        do_sample=True, temperature=args.temperature, top_p=args.top_p,
        batch_size=args.batch_size,
    )
    peak_run_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    hexes = [_encode_to_hex(t) or None for t in gen.texts]
    n_encoded = sum(h is not None for h in hexes)

    meta = {
        "model": model_desc,
        "n": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "prompt": args.prompt,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gen_seconds": round(gen.seconds, 1),
        "total_new_tokens": gen.total_new_tokens,
        "tokens_per_sec": gen.total_new_tokens / gen.seconds if gen.seconds > 0 else 0.0,
        "peak_run_gpu_mb": peak_run_mb,
    }
    write_candidates(out_path, meta, gen.texts, hexes)
    print(f"[+] Wrote {args.n} candidates ({n_encoded} encoded) → {out_path}")
    print(f"[+] gen={gen.seconds:.0f}s  tokens_per_sec={meta['tokens_per_sec']:.0f}  "
          f"peak_run_gpu_mb={peak_run_mb:.0f}")

    if args.hf_repo:
        hf_path = args.hf_path or out_path.name
        print(f"[*] Uploading → {args.hf_repo}:{hf_path} (dataset)")
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=hf_path,
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print(f"[+] Uploaded to HF: {args.hf_repo}:{hf_path}")


# --------------------------------------------------------------------------- #
# validate (local KCOV-VM stage)
# --------------------------------------------------------------------------- #

def cmd_validate(args) -> None:
    from reward import SSHClient

    cand_path = Path(args.candidates)
    meta, assemblies, hexes = read_candidates(cand_path)
    print(f"[*] Loaded {len(assemblies)} candidates from {cand_path} "
          f"(model={meta.get('model', '?')})")

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "benchmarks" / "diversity"
        / f"diversity_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    key = args.key or SSHClient.__dataclass_fields__["key"].default
    ssh = SSHClient(host=args.host, port=args.port, key=key)
    print(f"[*] Validating {sum(h is not None for h in hexes)} encoded programs via --batch")

    # Timing/throughput come from the generate stage (recorded in _meta).
    timing = GeneratorResult(assemblies=assemblies, tokens_per_sec=meta.get("tokens_per_sec", 0.0))
    peak_run_mb = meta.get("peak_run_gpu_mb", 0.0)

    # Streaming aggregation: folds each program's KCOV trace into running sets/counters
    # and frees the raw PCs, so N=20k+ does not OOM. `valid_only` is the spine metric
    # (unique PCs from ACCEPTED programs only; total_unique_pcs unions over ALL programs).
    kpis, valid_only = aggregate_streaming(
        assemblies, hexes,
        batch_validate=lambda hx: _batch_validate_fn(hx, ssh),
        timing=timing, peak_run_gpu_mb=peak_run_mb, chunk_size=args.val_chunk,
    )
    wall = time.monotonic() - t0

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": str(cand_path),
        "meta": meta,
        "n": len(assemblies),
        "validate_wall_seconds": round(wall, 1),
        "kpis": asdict(kpis),
        "valid_only": valid_only,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n[+] n={len(assemblies)}  encode_rate={kpis.encode_rate:.3f}  "
          f"accept_rate={kpis.pass_rate:.3f}")
    print(f"[+] total_unique_pcs={kpis.total_unique_pcs} (all progs)  "
          f"VALID_unique_pcs={valid_only['valid_unique_pcs']} (n_accepted={valid_only['n_accepted']})")
    print(f"[+] novelty_score={kpis.novelty_score:.4f}")
    print(f"[+] validate_wall={wall:.0f}s  gen_tokens_per_sec={kpis.tokens_per_sec:.0f}")
    print(f"[+] Report → {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="GPU stage: sample candidates → candidates JSONL (no VM)")
    g.add_argument("--model", default=None, help="Path to a merged bf16 model")
    g.add_argument("--base", default=None, help="Base model (used with --adapter for bf16 adapter-on-base loading)")
    g.add_argument("--adapter", default=None, help="LoRA adapter dir to load on --base (bf16, no merge step)")
    g.add_argument("--n", type=int, default=1000, help="Number of candidates to sample")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--batch-size", type=int, default=256, help="Generation batch size")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--prompt", default=_PROMPT)
    g.add_argument("--out", default=None, help="Candidates JSONL (default: benchmarks/diversity/candidates/<UTC>.jsonl)")
    g.add_argument("--hf-repo", default=None, help="If set, upload the candidates JSONL to this HF dataset repo")
    g.add_argument("--hf-path", default=None, help="Path in the HF repo (default: the file's basename)")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="Local stage: validate a candidates JSONL on the KCOV VM → KPIs")
    v.add_argument("--candidates", required=True, help="Candidates JSONL from the generate stage")
    v.add_argument("--val-chunk", type=int, default=500, help="Programs per --batch VM call")
    v.add_argument("--host", default="localhost")
    v.add_argument("--port", type=int, default=10022)
    v.add_argument("--key", default=None, help="SSH key path (default: reward.py default)")
    v.add_argument("--out", default=None, help="Report JSON (default: benchmarks/diversity/<UTC>.json)")
    v.set_defaults(func=cmd_validate)

    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
