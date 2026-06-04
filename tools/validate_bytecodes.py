#!/usr/bin/env python3
"""
validate_bytecodes.py — Stage 3 of the 60%-reconstruction: VALIDATE.

Reads a JSONL of generated programs (from generate_bytecodes.py), uploads each
stored hex to the KCOV-instrumented kernel VM, runs kcov_validator, and records
the real verdict + kernel PC count per program — for BOTH encodings:

  clang_hex    — faithful 60%-era path (verifier-log parser → clang → objcopy)
  encoder_hex  — 19%-era pure-python encoder (bypasses clang)

This isolates the three confounds in the 60-vs-19 gap:
  model     — same programs for both encodings
  encoder   — clang_hex vs encoder_hex per program
  tool      — KCOV verdict here vs the original ebpf_validator verdict

Output: writes <input>.kcov.jsonl (input records + kcov_clang / kcov_encoder
fields) and prints a reconciliation summary (accept rates + cumulative unique PCs).

Usage:
  ./fuzzing/run_eval_vm.sh                       # VM must be up first
  pixi run python tools/validate_bytecodes.py \\
      --in data/reconstruction/sft_v1_YYYYMMDD.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml"))

from reward import SSHClient  # noqa: E402

_VALID_VERDICTS = ("ACCEPTED", "VALID")


def kcov_validate(hex_str: str, ssh: SSHClient) -> dict | None:
    """Run kcov_validator on one hex string; return {verdict, pcs} or None."""
    if not hex_str:
        return None
    try:
        rc, stdout, _ = ssh.run(f"/mnt/corpus/kcov_validator {hex_str}", timeout=30)
    except Exception as e:
        print(f"\n[!] SSH error: {e}")
        return None
    if rc in (0, 1):
        try:
            return json.loads(stdout.strip())
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="infile", required=True,
                    help="JSONL from generate_bytecodes.py")
    ap.add_argument("--out", default=None,
                    help="Output JSONL (default: <input>.kcov.jsonl)")
    ap.add_argument("--ssh-host", default="localhost")
    ap.add_argument("--ssh-port", type=int, default=10022)
    ap.add_argument("--ssh-key", default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".kcov.jsonl")
    records = [json.loads(l) for l in in_path.open() if l.strip()]
    print(f"[*] Loaded {len(records)} programs from {in_path}")

    ssh = SSHClient(host=args.ssh_host, port=args.ssh_port, key=args.ssh_key)
    rc, out, _ = ssh.run("echo ok", timeout=5)
    if rc != 0 or out.strip() != "ok":
        print(f"[!] SSH check failed (rc={rc}, out={out!r}). Is the VM up?")
        sys.exit(1)
    print(f"[*] SSH OK — {args.ssh_host}:{args.ssh_port}")

    # Counters + cumulative unique-PC frontiers, per encoding.
    stat = {
        "clang":   {"n": 0, "accepted": 0, "pcs": set()},
        "encoder": {"n": 0, "accepted": 0, "pcs": set()},
    }

    with out_path.open("w") as fout:
        for r in records:
            for which, hexfield in (("clang", "clang_hex"), ("encoder", "encoder_hex")):
                res = kcov_validate(r.get(hexfield, ""), ssh)
                if res is None:
                    r[f"kcov_{which}"] = {"verdict": "ENCODE_FAIL" if not r.get(hexfield) else "ERROR", "n_pcs": 0}
                    continue
                pcs = res.get("pcs", []) or []
                verdict = res.get("verdict", "ERROR")
                valid = verdict in _VALID_VERDICTS
                stat[which]["n"] += 1
                if valid:
                    stat[which]["accepted"] += 1
                    stat[which]["pcs"].update(pcs)
                r[f"kcov_{which}"] = {"verdict": verdict, "n_pcs": len(pcs), "valid": valid}
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            ce = r.get("kcov_clang", {}); ee = r.get("kcov_encoder", {})
            print(f"\r[{r['id']+1}/{len(records)}] "
                  f"clang={ce.get('verdict','-'):<11}({ce.get('n_pcs',0)}) "
                  f"encoder={ee.get('verdict','-'):<11}({ee.get('n_pcs',0)})   ",
                  end="", flush=True)

    n = len(records)
    print("\n\n" + "=" * 58)
    print(f" RECONSTRUCTION — {in_path.name}  (n={n})")
    print("=" * 58)
    for which in ("clang", "encoder"):
        s = stat[which]
        acc_gen = s["accepted"] / n * 100 if n else 0
        acc_run = s["accepted"] / s["n"] * 100 if s["n"] else 0
        print(f" {which:<8} | reached_kernel {s['n']:>3}/{n} | "
              f"ACCEPTED {s['accepted']:>3} "
              f"({acc_gen:.1f}% of gen, {acc_run:.1f}% of reached) | "
              f"unique PCs {len(s['pcs'])}")
    print("=" * 58)
    print(" Compare: original 60% = clang+ebpf_validator ACCEPTED/gen.")
    print("          original 19% = encoder+kcov_validator ACCEPTED/gen.")
    print(f"[+] Per-program results → {out_path}")


if __name__ == "__main__":
    main()
