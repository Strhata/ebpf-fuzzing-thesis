#!/usr/bin/env python3
"""
classify_crashes.py — Phase 1 crash log analysis.

Two inputs:
  1. crash_logs/*.txt   — QEMU serial console captures (one file per crash event)
  2. report_errori.txt  — verifier error distribution (optional)

Output:
  - crashes.csv         — one row per crash event
  - verifier_errors.csv — verifier error category breakdown (if input provided)

Usage:
  python tools/classify_crashes.py \
      --crash-dir fuzzing/crash_logs/ \
      [--verifier-report fuzzing/shared_corpus/report_errori.txt] \
      [--output-dir results/]
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# Patterns for KASAN / kernel panic content (present in full serial logs)
_KASAN_BUG_RE = re.compile(
    r"BUG: KASAN: (?P<type>[^\n]+)|"
    r"BUG: (?P<bug>[^\n]+)|"
    r"Oops: (?P<oops>[^\n]+)|"
    r"general protection fault",
    re.IGNORECASE,
)
_FRAME_RE = re.compile(r"\[\s*(?:[\da-f]+\s*)?\]\s+(?P<sym>\w[^\+\[\n]+)\+0x[\da-f]+/0x[\da-f]+")
_CALL_TRACE_START = re.compile(r"Call Trace:", re.IGNORECASE)
_LINUX_VERSION_RE = re.compile(r"Linux version (\S+)")
_KERNEL_TS_RE = re.compile(r"^\[\s*(\d+\.\d+)\]")


def parse_crash_log(path: Path) -> dict:
    stem = path.stem  # e.g. vm1_crash_1772604055
    parts = stem.split("_")
    vm = parts[0] if parts else "unknown"
    try:
        unix_ts = int(parts[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, IndexError):
        unix_ts = 0
        dt = "unknown"

    text = path.read_text(errors="replace")
    lines = text.splitlines()

    kernel_version = "unknown"
    m = _LINUX_VERSION_RE.search(text)
    if m:
        kernel_version = m.group(1)

    # Find last kernel timestamp in the log (approximates crash moment)
    last_kernel_ts = None
    last_kernel_msg = ""
    for line in reversed(lines):
        m = _KERNEL_TS_RE.match(line)
        if m:
            last_kernel_ts = float(m.group(1))
            last_kernel_msg = line.strip()
            break

    # Try to extract KASAN/BUG crash type
    kasan_type = None
    for line in lines:
        m = _KASAN_BUG_RE.search(line)
        if m:
            kasan_type = (m.group("type") or m.group("bug") or m.group("oops") or "general protection fault").strip()
            break

    # Extract top-3 stack frames after "Call Trace:"
    frames = []
    in_trace = False
    for line in lines:
        if _CALL_TRACE_START.search(line):
            in_trace = True
            continue
        if in_trace:
            m = _FRAME_RE.search(line)
            if m:
                frames.append(m.group("sym").strip())
                if len(frames) == 3:
                    break
            elif frames:
                break

    # Count how many complete boot sequences are in the log
    boot_count = text.count("Linux version ")

    # Detect panic trigger hint from SELinux relabeling (first-boot SELinux issue vs real crash)
    selinux_relabel = "SELinux default policy relabel" in text
    ssh_connected = "comm=\"sshd\"" in text or "sshd (" in text

    return {
        "vm": vm,
        "unix_timestamp": unix_ts,
        "datetime_utc": dt,
        "kernel_version": kernel_version,
        "kasan_type": kasan_type or ("kernel_panic_no_serial_capture" if not kasan_type else ""),
        "frame_1": frames[0] if len(frames) > 0 else "N/A",
        "frame_2": frames[1] if len(frames) > 1 else "N/A",
        "frame_3": frames[2] if len(frames) > 2 else "N/A",
        "boot_count_in_log": boot_count,
        "selinux_relabel": selinux_relabel,
        "ssh_reached": ssh_connected,
        "last_kernel_uptime_s": last_kernel_ts,
        "last_kernel_msg": last_kernel_msg[:120],
    }


def parse_verifier_report(path: Path) -> list[dict]:
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

    total = (total_valid or 0) + (total_invalid or 0)
    for row in rows:
        row["pct_of_invalid"] = f"{row['count'] / total_invalid * 100:.1f}%" if total_invalid else "N/A"
        row["pct_of_total"] = f"{row['count'] / total * 100:.1f}%" if total else "N/A"

    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows, total_valid, total_invalid


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[+] Written {len(rows)} rows → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crash-dir", default="fuzzing/crash_logs/", help="Directory with crash .txt files")
    ap.add_argument("--verifier-report", default=None, help="Path to report_errori.txt")
    ap.add_argument("--output-dir", default="results/", help="Output directory for CSVs")
    args = ap.parse_args()

    crash_dir = Path(args.crash_dir)
    output_dir = Path(args.output_dir)

    if not crash_dir.is_dir():
        print(f"[!] crash-dir not found: {crash_dir}", file=sys.stderr)
        sys.exit(1)

    logs = sorted(crash_dir.glob("*.txt"))
    if not logs:
        print(f"[!] No .txt files in {crash_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Parsing {len(logs)} crash logs from {crash_dir}")
    crash_rows = [parse_crash_log(p) for p in logs]
    crash_rows.sort(key=lambda r: r["unix_timestamp"])

    crash_fields = [
        "vm", "unix_timestamp", "datetime_utc", "kernel_version",
        "kasan_type", "frame_1", "frame_2", "frame_3",
        "boot_count_in_log", "selinux_relabel", "ssh_reached",
        "last_kernel_uptime_s", "last_kernel_msg",
    ]
    write_csv(output_dir / "crashes.csv", crash_rows, crash_fields)

    # Summary
    kasan_types = [r["kasan_type"] for r in crash_rows if r["kasan_type"] and r["kasan_type"] != "kernel_panic_no_serial_capture"]
    unique_kasan = set(kasan_types)
    print(f"[*] Total crash events : {len(crash_rows)}")
    print(f"[*] Unique KASAN types : {len(unique_kasan)} {'(none in serial logs)' if not unique_kasan else ''}")
    for t in sorted(unique_kasan):
        print(f"    - {t}")

    per_vm = {}
    for r in crash_rows:
        per_vm[r["vm"]] = per_vm.get(r["vm"], 0) + 1
    for vm, n in sorted(per_vm.items()):
        print(f"[*] {vm}: {n} crash events")

    # Optional: verifier error report
    if args.verifier_report:
        vp = Path(args.verifier_report)
        if not vp.exists():
            print(f"[!] Verifier report not found: {vp}", file=sys.stderr)
        else:
            print(f"\n[*] Parsing verifier error report: {vp}")
            verifier_rows, total_valid, total_invalid = parse_verifier_report(vp)
            verifier_fields = ["count", "pct_of_invalid", "pct_of_total", "error_class"]
            write_csv(output_dir / "verifier_errors.csv", verifier_rows, verifier_fields)
            print(f"[*] Valid programs   : {total_valid:,}")
            print(f"[*] Invalid programs : {total_invalid:,}")
            print(f"[*] Unique error classes: {len(verifier_rows)}")


if __name__ == "__main__":
    main()
