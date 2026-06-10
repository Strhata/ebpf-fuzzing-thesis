#!/usr/bin/env python3
"""
plot_coverage_race_print.py — print-style (light) coverage-race figure for the thesis.

Renders cumulative unique kernel PCs vs programs submitted for the buzzer fuzzer and
the SFT-2 model, from the committed race CSVs. Unlike tools/plot_coverage_race.py
(dark slide palette), this matches the thesis figure style (white background, grid),
as in tools/plot_saturation.py.

Inputs (committed):
  data/corpus/buzzer_coverage.csv  cols: elapsed_ms, programs_submitted, valid_programs, unique_pcs
  data/corpus/model_coverage.csv   cols: ..., unique_pcs_valid, unique_pcs_all

Note on comparability: buzzer's unique_pcs is cumulative over ALL submitted programs
(its coverage table accumulates every verified program); the comparable model curve is
therefore unique_pcs_all. The model's valid-only curve is shown for context. The two
sides use different KCOV collection paths (buzzer's internal logger vs kcov_validator),
so this is a confounded cut, not a clean same-harness comparison.

Usage:
    pixi run python tools/plot_coverage_race_print.py \
        --buzzer data/corpus/buzzer_coverage.csv \
        --model  data/corpus/model_coverage.csv \
        --out    thesis/figures/coverage_race
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

_BUZZER = "#2563EB"   # blue
_MODEL_ALL = "#059669"  # green
_MODEL_VALID = "#D97706"  # amber


def _col(path: str, x: str, y: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row[x]))
            ys.append(float(row[y]))
    return xs, ys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buzzer", default="data/corpus/buzzer_coverage.csv")
    p.add_argument("--model", default="data/corpus/model_coverage.csv")
    p.add_argument("--out", default="thesis/figures/coverage_race")
    a = p.parse_args()

    bx, by = _col(a.buzzer, "programs_submitted", "unique_pcs")
    mx, ma = _col(a.model, "programs_submitted", "unique_pcs_all")
    _, mv = _col(a.model, "programs_submitted", "unique_pcs_valid")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bx, by, color=_BUZZER, lw=1.8,
            label=f"buzzer (all programs) — {by[-1]:.0f} PCs at {int(bx[-1])} submitted")
    ax.plot(mx, ma, color=_MODEL_ALL, lw=1.8,
            label=f"SFT-2 (all programs) — {ma[-1]:.0f} PCs at {int(mx[-1])} submitted")
    ax.plot(mx, mv, color=_MODEL_VALID, lw=1.5, ls="--",
            label=f"SFT-2 (valid only) — {mv[-1]:.0f} PCs")

    ax.set_xlabel("programs submitted")
    ax.set_ylabel("cumulative unique kernel PCs")
    ax.set_title("Coverage race: buzzer vs. SFT-2 (confounded — see text)")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"[*] Saved {out}.{ext}")


if __name__ == "__main__":
    main()
