#!/usr/bin/env python3
"""
enrich_dataset.py — Enrich dataset_final_qwen.jsonl with KCOV coverage data.

For each bytecode_hex, runs kcov_validator on the VM via SSH and records:

  pc_count      — number of kernel PCs traced by the BPF verifier (raw KCOV)
  coverage_bin  — tertile of pc_count among valid-only samples (low/medium/high)
  novelty_score — mean rarity of this program's PCs across the corpus (0–1)
  novelty_bin   — tertile of novelty_score among all samples (low/medium/high)

novelty_score formula
---------------------
After collecting all programs' PC sets, build a frequency table:

    pc_freq[pc] = number of programs (with pc_count > 0) that contain pc
                  (PCs are deduplicated within each program — KCOV repeats
                   the same address many times, but we count presence, not frequency)

Then for each program i with PC set S_i:

    novelty_score_i = mean( 1 - pc_freq[pc] / N  for pc in S_i )

where N = total programs with pc_count > 0.

A PC seen in every program contributes 0; a PC unique to one program
contributes (N-1)/N ≈ 1. Programs that hit rare verifier paths score high
regardless of absolute pc_count — this is the dimension coverage_bin misses.

Implementation note — memory
-----------------------------
Storing all ~38k PCs per program in RAM simultaneously would require ~8 GB.
Instead the enrichment loop writes PCs to a binary sidecar file as each
worker thread finishes, then streams through it twice (Counter pass → score
pass). Peak extra RAM is the Counter of unique PCs (~40 MB) + one record's
worth of PCs (~300 KB). The sidecar is deleted on completion.

Sidecar binary format:
    per record: [idx: uint32][n_pcs: uint32][pcs: uint64 × n_pcs]
Records are written in thread-completion order (arbitrary idx ordering).
Both streaming passes read the entire file sequentially.

Resume note
-----------
When --output already exists (resume mode), novelty_score is set to 0.0 and
novelty_bin to "low" for the newly processed samples because the PC sets of
already-done samples are not available. Run fresh (delete the output file)
to get accurate novelty scores across the full corpus.

Output: data/dataset_final_qwen_enriched.jsonl
  Line 1: {"_metadata": true, "coverage_scale": "linear|log", "novelty_computed": true|false}
  Lines 2+: original fields + pc_count, coverage_bin, novelty_score, novelty_bin

Usage:
    pixi run python ml/enrich_dataset.py [--workers N] [--input PATH] [--output PATH]
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT  = REPO_ROOT / "data" / "dataset_final_qwen.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "dataset_final_qwen_enriched.jsonl"

_thread_local = threading.local()

sys.path.insert(0, str(Path(__file__).parent))
from reward import SSHClient


# ---------------------------------------------------------------------------
# VM validation (single sample)
# ---------------------------------------------------------------------------

def _validate_one(hex_str: str, ssh) -> tuple[int, str, list[int]]:
    """Run kcov_validator; return (pc_count, verdict, pcs)."""
    try:
        rc, stdout, _ = ssh.run(f"/mnt/corpus/kcov_validator {hex_str}", timeout=30)
    except subprocess.TimeoutExpired:
        return 0, "TIMEOUT", []
    if rc not in (0, 1):
        return 0, "ERROR", []
    try:
        result = json.loads(stdout.strip())
        pcs = result.get("pcs", [])
        return len(pcs), result.get("verdict", "ERROR"), pcs
    except (json.JSONDecodeError, ValueError):
        return 0, "ERROR", []


# ---------------------------------------------------------------------------
# Coverage tertile thresholds (valid-only pc_counts)
# ---------------------------------------------------------------------------

def compute_tertile_thresholds(pc_counts: list[int]) -> tuple[float, float, str]:
    """Return (t33, t67, scale) from a list of PC counts (valid-only).

    scale is 'log' when (p33 - p0) / (p100 - p0) < 0.05 (heavily right-skewed).
    """
    arr = np.array(pc_counts, dtype=float)
    p0   = float(np.min(arr))
    p100 = float(np.max(arr))
    p33  = float(np.percentile(arr, 33))

    denom = p100 - p0
    scale = "log" if (denom > 0 and (p33 - p0) / denom < 0.05) else "linear"

    if scale == "log":
        arr = np.log(arr + 1)

    t33 = float(np.percentile(arr, 33))
    t67 = float(np.percentile(arr, 67))
    return t33, t67, scale


def assign_bin(pc_count: int, t33: float, t67: float, scale: str) -> str:
    val = math.log(pc_count + 1) if scale == "log" else float(pc_count)
    if val < t33:
        return "low"
    elif val < t67:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Novelty thresholds and scoring
# ---------------------------------------------------------------------------

def compute_novelty_thresholds(scores: list[float]) -> tuple[float, float]:
    """Return (t33, t67) from novelty scores (linear tertiles on 0–1 range)."""
    positive = [s for s in scores if s > 0.0]
    if not positive:
        return 0.0, 0.0
    arr = np.array(positive, dtype=float)
    return float(np.percentile(arr, 33)), float(np.percentile(arr, 67))


def assign_novelty_bin(score: float, t33: float, t67: float) -> str:
    if score < t33:
        return "low"
    elif score < t67:
        return "medium"
    return "high"


def _compute_novelty_scores(sidecar_path: str, n: int) -> list[float]:
    """Stream through sidecar twice to compute per-program novelty scores.

    Pass 1: build pc_freq Counter (how many programs contain each PC).
    Pass 2: for each program, compute mean(1 - pc_freq[pc] / N for pc in pcs).
    """
    # Pass 1 — frequency table: count how many PROGRAMS contain each PC.
    # KCOV traces repeat the same address many times per run, so we deduplicate
    # within each record before updating the Counter. Without deduplication,
    # pc_freq[pc] would be total occurrences >> N, making scores go negative.
    pc_freq: Counter = Counter()
    N = 0
    with open(sidecar_path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            _, n_pcs = struct.unpack(">II", hdr)
            if n_pcs > 0:
                raw = f.read(n_pcs * 8)
                pcs = struct.unpack(f">{n_pcs}Q", raw)
                pc_freq.update(set(pcs))  # one count per program per unique PC
                N += 1

    if N == 0:
        return [0.0] * n

    # Pass 2 — per-program rarity score (unique PCs only, same reason)
    scores = [0.0] * n
    with open(sidecar_path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            idx, n_pcs = struct.unpack(">II", hdr)
            if n_pcs > 0:
                raw = f.read(n_pcs * 8)
                unique_pcs = set(struct.unpack(f">{n_pcs}Q", raw))
                scores[idx] = sum(1.0 - pc_freq[pc] / N for pc in unique_pcs) / len(unique_pcs)

    return scores


# ---------------------------------------------------------------------------
# Batch enrichment (parallelised)
# ---------------------------------------------------------------------------

def enrich(
    samples: list[dict],
    ssh_factory,
    workers: int = 4,
    progress_fn=None,
) -> tuple[list[dict], str, str]:
    """Validate all samples, compute coverage + novelty bins.

    Returns (enriched_samples, coverage_scale, novelty_scale).
    novelty_scale is always 'linear' (scores are bounded 0–1, no log needed).

    ssh_factory() is called once per worker thread.
    progress_fn(done, total) called after each completed item when provided.
    """
    n = len(samples)
    results: list[tuple[int, str] | None] = [None] * n

    # Sidecar file: binary stream of PC records written as workers complete.
    # Written out-of-order (completion order); both novelty passes read it
    # sequentially so no seeking is required.
    sidecar_fd, sidecar_path = tempfile.mkstemp(suffix=".pcbin")
    os.close(sidecar_fd)
    sidecar_lock = threading.Lock()

    def _worker(idx: int, hex_str: str) -> tuple[int, tuple[int, str]]:
        if not hasattr(_thread_local, "ssh"):
            _thread_local.ssh = ssh_factory()
        pc_count, verdict, pcs = _validate_one(hex_str, _thread_local.ssh)
        # Write PC record to sidecar (thread-safe)
        packed = struct.pack(">II", idx, len(pcs))
        if pcs:
            packed += struct.pack(f">{len(pcs)}Q", *pcs)
        with sidecar_lock:
            with open(sidecar_path, "ab") as sf:
                sf.write(packed)
        return idx, (pc_count, verdict)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, i, s["bytecode_hex"]): i
            for i, s in enumerate(samples)
        }
        done = 0
        for f in as_completed(futures):
            idx, val = f.result()
            results[idx] = val
            done += 1
            if progress_fn:
                progress_fn(done, n)

    # --- Coverage bins (from valid-only pc_counts) ---
    valid_counts = [
        results[i][0]
        for i, s in enumerate(samples)
        if s.get("is_valid", False) and results[i] is not None
    ]
    if not valid_counts:
        valid_counts = [r[0] for r in results if r is not None] or [0]

    t33_cov, t67_cov, coverage_scale = compute_tertile_thresholds(valid_counts)

    # --- Novelty scores (two streaming passes over sidecar) ---
    print("[*] Computing novelty scores (2 passes over sidecar) ...", flush=True)
    novelty_scores = _compute_novelty_scores(sidecar_path, n)
    os.unlink(sidecar_path)

    t33_nov, t67_nov = compute_novelty_thresholds(novelty_scores)

    # --- Assemble enriched rows ---
    enriched = []
    for i, s in enumerate(samples):
        pc_count, _ = results[i] if results[i] is not None else (0, "ERROR")
        nov = novelty_scores[i]
        row = dict(s)
        row["pc_count"]      = pc_count
        row["coverage_bin"]  = assign_bin(pc_count, t33_cov, t67_cov, coverage_scale)
        row["novelty_score"] = round(nov, 6)
        row["novelty_bin"]   = assign_novelty_bin(nov, t33_nov, t67_nov)
        enriched.append(row)

    return enriched, coverage_scale, "linear"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input",   type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output",  type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int,  default=4)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[!] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.output.resolve() == args.input.resolve():
        print("[!] Output must differ from input", file=sys.stderr)
        sys.exit(1)

    already_done: set[str] = set()
    if args.output.exists():
        print("[*] Resuming — scanning existing output ...", flush=True)
        with args.output.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "bytecode_hex" in row:
                        already_done.add(row["bytecode_hex"])
                except json.JSONDecodeError:
                    pass
        print(f"    {len(already_done)} already processed", flush=True)
        if already_done:
            print(
                "[!] Resume mode: novelty_score will be 0.0 for all new samples "
                "(PC sets of prior samples unavailable). Delete output and rerun "
                "for accurate novelty scores.",
                flush=True,
            )

    print(f"[*] Reading {args.input} ...", flush=True)
    all_samples = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    samples = [s for s in all_samples if s.get("bytecode_hex") not in already_done]
    print(
        f"[*] {len(samples)} samples to process ({len(already_done)} skipped). "
        f"Workers={args.workers}",
        flush=True,
    )

    def _progress(done, total):
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    enriched, coverage_scale, novelty_scale = enrich(
        samples,
        ssh_factory=lambda: SSHClient(),
        workers=args.workers,
        progress_fn=_progress,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if already_done else "w"
    with args.output.open(mode) as f:
        if not already_done:
            f.write(json.dumps({
                "_metadata": True,
                "coverage_scale": coverage_scale,
                "novelty_computed": not bool(already_done),
            }) + "\n")
        for row in enriched:
            f.write(json.dumps(row) + "\n")

    print(
        f"[+] Done. coverage_scale={coverage_scale}  novelty_scale={novelty_scale}  "
        f"rows_written={len(enriched)}  output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
