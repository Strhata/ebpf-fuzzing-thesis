#!/usr/bin/env python3
"""
measure_saturation.py — F2 step 1: re-validate candidates R times on the VM.

The valid-unique-PC metric is nondeterministic (KCOV captures background kernel PCs;
borderline verdicts flip), so a single pass is a noisy point estimate. This runs R
replicates per series and caches every replicate's cumulative curve + endpoint to a
JSON, so plot_saturation.py can draw an honest band without re-touching the VM.

Needs the KCOV VM (SSH :10022). Usage:
    pixi run python tools/measure_saturation.py            # default replicate counts
    pixi run python tools/measure_saturation.py --out benchmarks/diversity/saturation_replicates.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from diversity_sample import _batch_validate_fn, read_candidates
from figures_lib import cumulative_unique_curve
from reward import SSHClient

_REPO = Path(__file__).resolve().parent.parent
_CAND = _REPO / "benchmarks/diversity/candidates"
# (label, file, n_replicates) — more replicates for the cheap small runs.
_SERIES = [
    ("SFT-2 (512 tok)", _CAND / "sft-v2-n1000-seed42.jsonl", 4),
    ("RL-2 phase-B cp200", _CAND / "rl-phaseB-cp200-n5000-seed42.jsonl", 3),
    ("SFT-2 (1024 tok)", _CAND / "sft-v2-n20000-seed42.jsonl", 3),
]


def log_points(n: int, k: int = 45) -> list[int]:
    if n <= 1:
        return [n] if n else []
    return sorted({int(round(10 ** x)) for x in
                   [i * math.log10(n) / (k - 1) for i in range(k)]} | {1, n})


def valid_pc_sets(hexes, ssh, chunk_size=500):
    sets = []
    idx = [i for i, h in enumerate(hexes) if h]
    for s in range(0, len(idx), chunk_size):
        batch = _batch_validate_fn([hexes[i] for i in idx[s:s + chunk_size]], ssh)
        if batch is None:
            continue
        for res in batch:
            if res.get("verdict") == "ACCEPTED":
                sets.append(set(res.get("pcs", [])))
    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10022)
    ap.add_argument("--out", default=str(_REPO / "benchmarks/diversity/saturation_replicates.json"))
    args = ap.parse_args()

    ssh = SSHClient(host=args.host, port=args.port,
                    key=SSHClient.__dataclass_fields__["key"].default)
    out = Path(args.out)
    result: dict[str, list] = {}

    for label, path, reps in _SERIES:
        if not path.exists():
            print(f"WARN missing {path}", flush=True)
            continue
        _, _, hexes = read_candidates(path)
        result[label] = []
        for r in range(reps):
            t0 = time.monotonic()
            sets = valid_pc_sets(hexes, ssh)
            curve = cumulative_unique_curve(sets, log_points(len(sets)))
            endpoint = curve[-1][1] if curve else 0
            result[label].append({"n_valid": len(sets), "endpoint": endpoint, "curve": curve})
            print(f"{label} rep {r + 1}/{reps}: {len(sets)} valid -> {endpoint} PCs "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
            # checkpoint after every replicate so a long run is crash-safe
            out.write_text(json.dumps(result, indent=1) + "\n")

    # summary
    print("\n=== endpoint summary ===", flush=True)
    for label, reps in result.items():
        eps = [x["endpoint"] for x in reps]
        nv = [x["n_valid"] for x in reps]
        print(f"{label}: n_valid~{round(sum(nv) / len(nv))} | endpoints {eps} "
              f"mean={sum(eps) / len(eps):.0f} min={min(eps)} max={max(eps)}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
