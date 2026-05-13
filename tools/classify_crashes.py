#!/usr/bin/env python3
"""
classify_crashes.py — Phase 1: verifier error distribution.

Parses report_errori.txt produced by buzzer's corpus collection and outputs
a ranked CSV of eBPF verifier error categories.

Usage:
  python tools/classify_crashes.py \
      --verifier-report /path/to/report_errori.txt \
      [--output-dir results/]
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_verifier_report(path: Path):
    rows = []
    total_valid = total_invalid = None
    error_re = re.compile(r"\[(\d+) programmi\]\s*->\s*(.+)")
    valid_re = re.compile(r"Programmi VALIDI[^:]*:\s*(\d+)")
    invalid_re = re.compile(r"Programmi INVALIDI[^:]*:\s*(\d+)")

    for line in path.read_text(errors="replace").splitlines():
        m = error_re.search(line)
        if m:
            rows.append({"count": int(m.group(1)), "error_class": m.group(2).strip()})
        m = valid_re.search(line)
        if m:
            total_valid = int(m.group(1))
        m = invalid_re.search(line)
        if m:
            total_invalid = int(m.group(1))

    rows.sort(key=lambda r: r["count"], reverse=True)
    for row in rows:
        row["pct_of_invalid"] = f"{row['count'] / total_invalid * 100:.1f}%" if total_invalid else "N/A"
        row["pct_of_total"] = f"{row['count'] / (total_valid + total_invalid) * 100:.1f}%" if (total_valid and total_invalid) else "N/A"

    return rows, total_valid, total_invalid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verifier-report", required=True, help="Path to report_errori.txt")
    ap.add_argument("--output-dir", default="results/")
    args = ap.parse_args()

    vp = Path(args.verifier_report)
    if not vp.exists():
        print(f"[!] Not found: {vp}", file=sys.stderr)
        sys.exit(1)

    rows, total_valid, total_invalid = parse_verifier_report(vp)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / "verifier_errors.csv"
    with outfile.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["count", "pct_of_invalid", "pct_of_total", "error_class"])
        w.writeheader()
        w.writerows(rows)

    print(f"[+] {len(rows)} error classes → {outfile}")
    print(f"[*] Valid   : {total_valid:,}")
    print(f"[*] Invalid : {total_invalid:,}")
    print(f"[*] Total   : {total_valid + total_invalid:,}")
    for row in rows:
        print(f"    {row['pct_of_invalid']:>6} {row['error_class']}")


if __name__ == "__main__":
    main()
