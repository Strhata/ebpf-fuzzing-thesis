#!/usr/bin/env python3
"""
plot_depth_collapse.py — F1: SFT-2 program-length distribution (mode collapse).

Reads the saved diversity candidates and plots the instruction-count histogram at
each generation budget. The point of the figure: program length is *narrow and
pinned by the token budget* — the model does not vary it. Instruction count is the
canonical bytecode count (len(hex)//16), not assembly-line counting.

No VM, no GPU: pure re-analysis of committed candidate JSONLs.

Usage:
    pixi run python tools/plot_depth_collapse.py
    pixi run python tools/plot_depth_collapse.py --out thesis/figures/depth_collapse.pdf
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figures_lib import insn_count_from_hex  # noqa: E402

# Print-friendly (this lands in a LaTeX PDF, not a dark web page).
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_SERIES = [
    ("512-token budget", _REPO / "benchmarks/diversity/candidates/sft-v2-n1000-seed42.jsonl", "#2563EB"),
    ("1024-token budget", _REPO / "benchmarks/diversity/candidates/sft-v2-n20000-seed42.jsonl", "#DC2626"),
]


def load_counts(path: Path) -> list[int]:
    """Instruction count per candidate (skips the leading _meta record)."""
    counts: list[int] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == 0:  # first line is the run _meta header
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = insn_count_from_hex(rec.get("hex_str"))
            if n > 0:
                counts.append(n)
    return counts


def describe(label: str, counts: list[int]) -> str:
    n = len(counts)
    mean, sd = st.mean(counts), st.pstdev(counts)
    cs = sorted(counts)
    return (
        f"{label}: n={n} mean={mean:.1f} sd={sd:.1f} median={int(st.median(counts))} "
        f"IQR={cs[n // 4]}-{cs[3 * n // 4]} range={cs[0]}-{cs[-1]} CV={sd / mean:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_REPO / "thesis/figures/depth_collapse.pdf"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, path, color in _DEFAULT_SERIES:
        if not path.exists():
            print(f"WARN missing {path}")
            continue
        counts = load_counts(path)
        print(describe(label, counts))
        mean, sd = st.mean(counts), st.pstdev(counts)
        # Fraction of programs (weights=1/n): the two series have very different N
        # (1k vs 20k), so raw counts would misrepresent the spread. Normalising lets
        # the shapes be compared directly — both are tight, just budget-shifted.
        weights = [1.0 / len(counts)] * len(counts)
        ax.hist(counts, bins=range(0, 105, 2), alpha=0.6, color=color, weights=weights,
                label=f"{label}: mean {mean:.0f} ±{sd:.0f}")
        ax.axvline(mean, color=color, ls="--", lw=1)

    ax.set_xlabel("program length (BPF instructions)")
    ax.set_ylabel("fraction of generated programs")
    ax.set_title("SFT-2 program length: narrow, and pinned by the token budget")
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
