#!/usr/bin/env python3
"""
build_baseline_dataset.py — Phase 2 baseline dataset construction.

Produces a 27,514-entry dataset with the same filtering and prompt format
as the curated dataset, but NO per-category cap on invalid programs.
Only variable vs curated: invalid class distribution (dominated by top class).

Sampling:
  - All valid programs kept (mirrors curated)
  - Reservoir-sample exactly N_INVALID invalids at random (seed fixed)

Output fields match dataset_final_qwen.jsonl exactly:
  bytecode_hex, verifier_log, is_valid, error_line, error_reason, error_reason_clean
"""

import gzip
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


CORPUS_GLOB   = "/home/stefano-u/fuzzing_lab/shared_corpus/dataset_syzkaller_*.jsonl.gz"
OUTPUT_FILE   = "/home/stefano-u/fuzzing_lab/shared_corpus/dataset_baseline_qwen.jsonl"
N_INVALID     = 14361   # matches curated invalid count exactly
SEED          = 42


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
    rng = random.Random(SEED)
    input_files = sorted(glob.glob(CORPUS_GLOB))
    if not input_files:
        print(f"[!] No files found: {CORPUS_GLOB}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Scanning {len(input_files)} corpus files...")

    valids   = []
    # Reservoir sampling for invalids (Algorithm R)
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
                    # Add error_reason_clean (same logic as curation script)
                    vero_errore = estrai_vero_errore(sample.get("verifier_log", ""))
                    sample["error_reason_clean"] = vero_errore

                    invalid_seen += 1
                    if len(reservoir) < N_INVALID:
                        reservoir.append(sample)
                    else:
                        j = rng.randint(0, invalid_seen - 1)
                        if j < N_INVALID:
                            reservoir[j] = sample

    print(f"\n[*] Valid collected   : {len(valids)}")
    print(f"[*] Invalid sampled   : {len(reservoir)} / {invalid_seen} total")

    # Shuffle output order
    combined = valids + reservoir
    rng.shuffle(combined)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for sample in combined:
            f_out.write(json.dumps(sample) + "\n")

    # Distribution report
    error_counts: Counter = Counter()
    for s in reservoir:
        cls = normalizza_errore(s.get("error_reason_clean", "unknown"))
        error_counts[cls] += 1

    print(f"\n[+] Written {len(combined)} entries → {OUTPUT_FILE}")
    print(f"\n[*] Baseline invalid class distribution:")
    for cls, n in error_counts.most_common():
        print(f"    {n:6d} ({n/len(reservoir)*100:5.1f}%)  {cls}")


if __name__ == "__main__":
    main()
