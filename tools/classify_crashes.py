#!/usr/bin/env python3
"""
classify_crashes.py — Phase 1 crash log analysis.

Inputs:
  1. crash_logs/*.txt    — QEMU serial console captures (one file per reboot event)
  2. vm_logs/vm*.txt     — continuous VM serial logs with actual crash details
  3. report_errori.txt   — verifier error distribution (optional)

Outputs:
  - crashes.csv          — one row per crash event from vm_logs
  - reboots.csv          — one row per reboot event from crash_logs
  - verifier_errors.csv  — verifier error category breakdown (if input provided)

Usage:
  python tools/classify_crashes.py \
      --crash-dir   fuzzing/crash_logs/ \
      --vm-log-dir  fuzzing/vm_logs/ \
      [--verifier-report /path/to/report_errori.txt] \
      [--output-dir results/]
"""

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


_FRAME_RE = re.compile(
    r"^\[[\s\d\.]+\]\s+(?P<questionable>\?)?\s*(?P<sym>\w[^\+\[\n<]+)\+0x[\da-f]+/0x[\da-f]+"
)
_CALL_TRACE_RE = re.compile(r"Call Trace:", re.IGNORECASE)
_TASK_END_RE = re.compile(r"</TASK>")
_KERNEL_TS_RE = re.compile(r"^\[\s*(\d+\.\d+)\]")
_LINUX_VER_RE = re.compile(r"Linux version (\S+)")

# OOM pattern
_OOM_TRIGGER_RE = re.compile(
    r"(\w[\w_]+) invoked oom-killer: gfp_mask=([^,]+)"
)
_OOM_KILL_RE = re.compile(
    r"Out of memory: Killed process \d+ \((\w[\w_]+)\) "
    r"total-vm:(\d+)kB, anon-rss:(\d+)kB"
)

# KASAN / kernel BUG
_KASAN_RE = re.compile(r"BUG: KASAN: (?P<type>[^\n]+)")
_BUG_RE = re.compile(r"BUG: (?P<type>[^\n]+)")
_OOPS_RE = re.compile(r"Oops: (?P<type>[^\n]+)")
_GENERAL_PROT_RE = re.compile(r"general protection fault")


_BOILERPLATE_FRAMES = {"dump_stack_lvl", "dump_header", "dump_stack", "___ratelimit"}


def _extract_frames(lines: list[str], start_idx: int, max_frames: int = 3) -> list[str]:
    frames = []
    for line in lines[start_idx:]:
        if _TASK_END_RE.search(line):
            break
        m = _FRAME_RE.match(line)
        if m:
            sym = m.group("sym").strip()
            if not m.group("questionable") and sym not in _BOILERPLATE_FRAMES:
                frames.append(sym)
                if len(frames) == max_frames:
                    break
    return frames


def _kernel_uptime(line: str) -> float | None:
    m = _KERNEL_TS_RE.match(line)
    return float(m.group(1)) if m else None


def parse_vm_log(path: Path) -> list[dict]:
    vm = path.stem.split("_")[0]  # vm1 / vm2 / vm3
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    kernel_version = "unknown"
    m = _LINUX_VER_RE.search(text)
    if m:
        kernel_version = m.group(1)

    events = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- OOM event ---
        m = _OOM_TRIGGER_RE.search(line)
        if m:
            trigger_proc = m.group(1)
            uptime = _kernel_uptime(line)
            # find who got killed (a few lines later)
            killed_proc = victim_vm_kb = victim_anon_kb = None
            for j in range(i, min(i + 120, len(lines))):
                km = _OOM_KILL_RE.search(lines[j])
                if km:
                    killed_proc = km.group(1)
                    victim_vm_kb = int(km.group(2))
                    victim_anon_kb = int(km.group(3))
                    break
            # extract call trace
            frames = []
            for j in range(i, min(i + 10, len(lines))):
                if _CALL_TRACE_RE.search(lines[j]):
                    frames = _extract_frames(lines, j + 1)
                    break

            events.append({
                "vm": vm,
                "kernel_version": kernel_version,
                "crash_type": f"OOM: {trigger_proc} killed",
                "trigger_process": trigger_proc,
                "killed_process": killed_proc or trigger_proc,
                "uptime_s": uptime,
                "victim_total_vm_mb": round(victim_vm_kb / 1024, 1) if victim_vm_kb else None,
                "victim_anon_rss_mb": round(victim_anon_kb / 1024, 1) if victim_anon_kb else None,
                "frame_1": frames[0] if len(frames) > 0 else "N/A",
                "frame_2": frames[1] if len(frames) > 1 else "N/A",
                "frame_3": frames[2] if len(frames) > 2 else "N/A",
            })
            i += 1
            continue

        # --- KASAN / BUG / Oops ---
        crash_type = None
        for pattern, label in [(_KASAN_RE, None), (_BUG_RE, None), (_OOPS_RE, None)]:
            m2 = pattern.search(line)
            if m2:
                crash_type = m2.group("type").strip()
                break
        if not crash_type and _GENERAL_PROT_RE.search(line):
            crash_type = "general protection fault"

        if crash_type:
            uptime = _kernel_uptime(line)
            frames = []
            for j in range(i, min(i + 30, len(lines))):
                if _CALL_TRACE_RE.search(lines[j]):
                    frames = _extract_frames(lines, j + 1)
                    break
            events.append({
                "vm": vm,
                "kernel_version": kernel_version,
                "crash_type": crash_type,
                "trigger_process": "unknown",
                "killed_process": "unknown",
                "uptime_s": uptime,
                "victim_total_vm_mb": None,
                "victim_anon_rss_mb": None,
                "frame_1": frames[0] if len(frames) > 0 else "N/A",
                "frame_2": frames[1] if len(frames) > 1 else "N/A",
                "frame_3": frames[2] if len(frames) > 2 else "N/A",
            })

        i += 1

    return events


def parse_reboot_log(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("_")
    vm = parts[0]
    try:
        unix_ts = int(parts[-1])
        dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, IndexError):
        unix_ts, dt = 0, "unknown"

    text = path.read_text(errors="replace")
    boot_count = text.count("Linux version ")
    selinux_relabel = "SELinux default policy relabel" in text
    ssh_reached = 'comm="sshd"' in text or "sshd (" in text

    last_kernel_msg = ""
    for line in reversed(text.splitlines()):
        if _KERNEL_TS_RE.match(line):
            last_kernel_msg = line.strip()[:120]
            break

    return {
        "vm": vm,
        "unix_timestamp": unix_ts,
        "datetime_utc": dt,
        "boot_sequences": boot_count,
        "selinux_relabel_reboot": selinux_relabel,
        "ssh_reached_before_crash": ssh_reached,
        "last_kernel_msg": last_kernel_msg,
    }


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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[+] {len(rows)} rows → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crash-dir",      default="fuzzing/crash_logs/")
    ap.add_argument("--vm-log-dir",     default="fuzzing/vm_logs/")
    ap.add_argument("--verifier-report", default=None)
    ap.add_argument("--output-dir",     default="results/")
    args = ap.parse_args()

    out = Path(args.output_dir)

    # --- vm_logs: actual crash events ---
    vm_log_dir = Path(args.vm_log_dir)
    crash_events = []
    if vm_log_dir.is_dir():
        vm_logs = sorted(vm_log_dir.glob("vm*.txt"))
        print(f"[*] Parsing {len(vm_logs)} vm logs from {vm_log_dir}")
        for p in vm_logs:
            events = parse_vm_log(p)
            crash_events.extend(events)
            print(f"    {p.name}: {len(events)} crash event(s)")
    else:
        print(f"[!] vm-log-dir not found: {vm_log_dir}", file=sys.stderr)

    if crash_events:
        crash_fields = [
            "vm", "kernel_version", "crash_type", "trigger_process", "killed_process",
            "uptime_s", "victim_total_vm_mb", "victim_anon_rss_mb",
            "frame_1", "frame_2", "frame_3",
        ]
        write_csv(out / "crashes.csv", crash_events, crash_fields)

        unique_types = sorted({e["crash_type"] for e in crash_events})
        print(f"\n[*] Total crash events  : {len(crash_events)}")
        print(f"[*] Unique crash types  : {len(unique_types)}")
        for t in unique_types:
            n = sum(1 for e in crash_events if e["crash_type"] == t)
            print(f"    [{n:2d}x] {t}")
        per_vm = {}
        for e in crash_events:
            per_vm[e["vm"]] = per_vm.get(e["vm"], 0) + 1
        for vm, n in sorted(per_vm.items()):
            print(f"[*] {vm}: {n} crash event(s)")
    else:
        print("[!] No crash events found in vm logs.")

    # --- crash_logs: reboot event metadata ---
    crash_dir = Path(args.crash_dir)
    if crash_dir.is_dir():
        reboot_logs = sorted(crash_dir.glob("*.txt"))
        print(f"\n[*] Parsing {len(reboot_logs)} reboot snapshots from {crash_dir}")
        reboot_rows = [parse_reboot_log(p) for p in reboot_logs]
        reboot_rows.sort(key=lambda r: r["unix_timestamp"])
        reboot_fields = [
            "vm", "unix_timestamp", "datetime_utc",
            "boot_sequences", "selinux_relabel_reboot",
            "ssh_reached_before_crash", "last_kernel_msg",
        ]
        write_csv(out / "reboots.csv", reboot_rows, reboot_fields)

    # --- verifier errors ---
    if args.verifier_report:
        vp = Path(args.verifier_report)
        if vp.exists():
            print(f"\n[*] Parsing verifier report: {vp}")
            rows, total_valid, total_invalid = parse_verifier_report(vp)
            write_csv(out / "verifier_errors.csv", rows, ["count", "pct_of_invalid", "pct_of_total", "error_class"])
            print(f"[*] Valid: {total_valid:,}  Invalid: {total_invalid:,}  Classes: {len(rows)}")
        else:
            print(f"[!] Verifier report not found: {vp}", file=sys.stderr)


if __name__ == "__main__":
    main()
