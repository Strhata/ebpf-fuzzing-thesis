#!/usr/bin/env python3
"""
build_baseline_dataset.py — Phase 2 baseline dataset construction.

Produces a dataset with the same filtering and prompt format as the curated
dataset, but NO per-category cap on invalid programs.
Only variable vs curated: invalid class distribution (dominated by top class).

Sampling:
  - All valid programs kept (mirrors curated)
  - Reservoir-sample exactly N_INVALID invalids at random (seed fixed)

Output fields match dataset_final_qwen.jsonl exactly:
  bytecode_hex, verifier_log, is_valid, error_line, error_reason, error_reason_clean

Usage (run from repo root):
  python ml/build_baseline_dataset.py

  # Or with explicit paths:
  python ml/build_baseline_dataset.py \\
      --corpus-dir  data/corpus \\
      --output-file data/dataset_baseline_qwen.jsonl \\
      [--n-invalid 14361] \\
      [--seed 42]

  Place dataset_syzkaller_*.jsonl.gz files in data/corpus/ before running.
"""

import argparse
import gzip
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


def estrai_vero_errore(verifier_log: str) -> str:
    for linea in reversed(verifier_log.strip().split("\n")):
        linea = linea.strip()
        if not linea:
            continue
        if linea.startswith("processed ") and "insns" in linea:
            continue
        if "R0=" in linea or "R1=" in linea or linea.startswith("mark_precise:"):
            continue
        return linea
    return "unknown_error"


def normalizza_errore(errore_raw: str) -> str:
    err = re.sub(r"[-+]?\b\d+\b", "<NUM>", str(errore_raw))
    err = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", err)
    return err.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir",  default="data/corpus",
                    help="Dir containing dataset_syzkaller_*.jsonl.gz (default: data/corpus)")
    ap.add_argument("--output-file", default="data/dataset_baseline_qwen.jsonl",
                    help="Output path (default: data/dataset_baseline_qwen.jsonl)")
    ap.add_argument("--n-invalid",   type=int, default=14361,
                    help="Invalid samples to reservoir-sample (default: 14361)")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    corpus_glob = os.path.join(args.corpus_dir, "dataset_syzkaller_*.jsonl.gz")
    rng = random.Random(args.seed)
    input_files = sorted(glob.glob(corpus_glob))
    if not input_files:
        print(f"[!] No files found: {corpus_glob}", file=sys.stderr)
        print(f"[!] Copy dataset_syzkaller_*.jsonl.gz into {args.corpus_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Scanning {len(input_files)} corpus files...")

    valids    = []
    reservoir = []
    invalid_seen = 0

    for file_path in input_files:
        print(f"    {os.path.basename(file_path)}")
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not sample.get("bytecode_hex"):
                    continue

                if sample.get("is_valid", False):
                    valids.append(sample)
                else:
                    vero_errore = estrai_vero_errore(sample.get("verifier_log", ""))
                    sample["error_reason_clean"] = vero_errore

                    invalid_seen += 1
                    if len(reservoir) < args.n_invalid:
                        reservoir.append(sample)
                    else:
                        j = rng.randint(0, invalid_seen - 1)
                        if j < args.n_invalid:
                            reservoir[j] = sample

    print(f"\n[*] Valid collected   : {len(valids)}")
    print(f"[*] Invalid sampled   : {len(reservoir)} / {invalid_seen} total")

    combined = valids + reservoir
    rng.shuffle(combined)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for sample in combined:
            f_out.write(json.dumps(sample) + "\n")

    error_counts: Counter = Counter()
    for s in reservoir:
        cls = normalizza_errore(s.get("error_reason_clean", "unknown"))
        error_counts[cls] += 1

    print(f"\n[+] Written {len(combined)} entries → {output_path}")
    print(f"\n[*] Baseline invalid class distribution:")
    for cls, n in error_counts.most_common():
        print(f"    {n:6d} ({n/len(reservoir)*100:5.1f}%)  {cls}")


if __name__ == "__main__":
    main()
