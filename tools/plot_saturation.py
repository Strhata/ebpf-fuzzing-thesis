#!/usr/bin/env python3
"""
plot_saturation.py — F2 step 2: draw the saturation curve from cached replicates.

Reads benchmarks/diversity/saturation_replicates.json (written by measure_saturation.py)
and plots, for the large SFT-2 run, the cumulative unique-valid-PC curve of every
replicate — the spread between replicate lines *is* the metric's nondeterminism, drawn
honestly rather than hidden behind one number. Endpoints of the other series are shown
with min–max bars. No VM, no GPU.

Usage:
    pixi run python tools/plot_saturation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

_REPO = Path(__file__).resolve().parent.parent
_MAIN = "SFT-2 (1024 tok)"  # the long run whose curve shows the wall
_COLORS = {"SFT-2 (512 tok)": "#2563EB", "RL-2 phase-B cp200": "#059669",
           "SFT-2 (1024 tok)": "#DC2626"}


def _mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(_REPO / "benchmarks/diversity/saturation_replicates.json"))
    ap.add_argument("--out", default=str(_REPO / "thesis/figures/saturation.pdf"))
    args = ap.parse_args()

    data = json.loads(Path(args.cache).read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Main series: every replicate's cumulative curve (band = nondeterminism).
    for j, rep in enumerate(data.get(_MAIN, [])):
        xs = [n for n, _ in rep["curve"]]
        ys = [u for _, u in rep["curve"]]
        ax.plot(xs, ys, color=_COLORS[_MAIN], lw=1.2, alpha=0.55,
                label=_MAIN if j == 0 else None)

    # Other series: endpoint with min–max bar across replicates.
    for label, reps in data.items():
        if label == _MAIN or not reps:
            continue
        nv = _mean([r["n_valid"] for r in reps])
        eps = [r["endpoint"] for r in reps]
        ax.errorbar([nv], [_mean(eps)], yerr=[[_mean(eps) - min(eps)], [max(eps) - _mean(eps)]],
                    fmt="o", ms=6, capsize=4, color=_COLORS.get(label, "#666"), label=label)

    ax.set_xscale("log")
    ax.set_xlabel("number of valid programs (log scale)")
    ax.set_ylabel("cumulative unique PCs from valid programs")
    ax.set_title("Valid-program coverage saturates (diversity is the wall)")
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} (+ .png)")

    # Headline range: small-vs-large endpoint ratio across replicates.
    small = next((k for k in data if k.startswith("SFT-2 (512")), None)
    if small and data.get(small) and data.get(_MAIN):
        s_eps = [r["endpoint"] for r in data[small]]
        l_eps = [r["endpoint"] for r in data[_MAIN]]
        s_nv = _mean([r["n_valid"] for r in data[small]])
        l_nv = _mean([r["n_valid"] for r in data[_MAIN]])
        lo = (min(l_eps) / max(s_eps) - 1) * 100
        hi = (max(l_eps) / min(s_eps) - 1) * 100
        ctr = (_mean(l_eps) / _mean(s_eps) - 1) * 100
        print(f"HEADLINE: {l_nv / s_nv:.1f}x more valid programs ({s_nv:.0f}->{l_nv:.0f}) "
              f"-> +{ctr:.0f}% unique PCs (range +{lo:.0f}..+{hi:.0f}%)")


if __name__ == "__main__":
    main()
