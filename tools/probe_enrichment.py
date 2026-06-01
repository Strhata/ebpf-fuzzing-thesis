#!/usr/bin/env python3
"""
probe_enrichment.py — Manual smoke-test for the enrichment pipeline.

Grabs N samples from the dataset (default 5: mix of valid/invalid),
runs each through /mnt/corpus/kcov_validator on the eval VM via SSH,
and prints raw input → output so you can verify format before enriching.

Prerequisites:
    VM must be running:  ./fuzzing/run_eval_vm.sh
    Validator present:   data/corpus/kcov_validator

Usage:
    pixi run python tools/probe_enrichment.py
    pixi run python tools/probe_enrichment.py --n 10
    pixi run python tools/probe_enrichment.py --vm-port 10022 --vm-key ~/fuzzing_lab/trixie.id_rsa
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml"))

from reward import SSHClient

DEFAULT_DATA = REPO_ROOT / "data" / "dataset_final_qwen.jsonl"
VALIDATOR    = "/mnt/corpus/kcov_validator"


def pick_samples(path: Path, n: int) -> list[dict]:
    """Return up to n/2 valid + n/2 invalid rows (best-effort)."""
    valid, invalid = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not row.get("bytecode_hex"):
            continue
        if row.get("is_valid") and len(valid) < n // 2:
            valid.append(row)
        elif not row.get("is_valid") and len(invalid) < (n - n // 2):
            invalid.append(row)
        if len(valid) + len(invalid) >= n:
            break
    return valid + invalid


def probe(samples: list[dict], ssh: SSHClient) -> None:
    # Connectivity check
    rc, out, _ = ssh.run("echo pong", timeout=5)
    if rc != 0 or "pong" not in out:
        print(f"[!] VM not reachable (rc={rc}). Run: ./fuzzing/run_eval_vm.sh", file=sys.stderr)
        sys.exit(1)
    print(f"[+] VM reachable on port {ssh.port}\n")

    for i, row in enumerate(samples):
        hex_str   = row["bytecode_hex"]
        is_valid  = row.get("is_valid", "?")
        hex_preview = hex_str[:32] + "..." if len(hex_str) > 32 else hex_str
        n_insns   = len(hex_str) // 16  # each 8-byte instruction = 16 hex chars

        print(f"--- sample {i+1} (is_valid={is_valid}, {n_insns} insns) ---")
        print(f"  INPUT hex[:32] : {hex_preview}")

        rc, stdout, stderr = ssh.run(f"{VALIDATOR} {hex_str}", timeout=30)
        print(f"  RC             : {rc}")

        if stderr.strip():
            print(f"  STDERR         : {stderr.strip()[:120]}")

        try:
            result = json.loads(stdout.strip())
            verdict  = result.get("verdict")
            pc_count = len(result.get("pcs", []))
            pcs_preview = result.get("pcs", [])[:5]
            print(f"  verdict        : {verdict}")
            print(f"  pc_count       : {pc_count}")
            print(f"  pcs[:5]        : {pcs_preview}")
        except (json.JSONDecodeError, ValueError):
            print(f"  PARSE FAIL     : {stdout.strip()[:120]}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",    type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n",       type=int,  default=5, help="Number of samples to probe (default 5)")
    ap.add_argument("--vm-host", default="localhost")
    ap.add_argument("--vm-port", type=int,  default=10022)
    ap.add_argument("--vm-key",  default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    args = ap.parse_args()

    if not args.data.exists():
        print(f"[!] Dataset not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    samples = pick_samples(args.data, args.n)
    if not samples:
        print("[!] No samples found in dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Probing {len(samples)} samples against {VALIDATOR}\n")
    ssh = SSHClient(host=args.vm_host, port=args.vm_port, key=args.vm_key)
    probe(samples, ssh)


if __name__ == "__main__":
    main()
