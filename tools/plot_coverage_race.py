#!/usr/bin/env python3
"""
plot_coverage_race.py — Plot buzzer vs model cumulative unique PC curves.

Reads two CSVs produced by the coverage race:
  - buzzer:  /mnt/corpus/buzzer_coverage.csv   (from coverage_based.go logger)
  - model:   /mnt/corpus/model_coverage.csv    (from coverage_race.py)

Both CSVs share the same schema:
    elapsed_ms, programs_submitted, valid_programs, unique_pcs

Produces two plots saved as PNG:
  1. unique_pcs vs programs_submitted  (efficiency — normalises out throughput)
  2. unique_pcs vs elapsed_ms          (real-time — shows throughput difference)

Usage:
    pixi run python tools/plot_coverage_race.py
    pixi run python tools/plot_coverage_race.py \
        --buzzer /mnt/corpus/buzzer_coverage.csv \
        --model  /mnt/corpus/model_coverage.csv \
        --out    results/coverage_race.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Qubital palette
# ---------------------------------------------------------------------------
_BG      = "#0A0A0A"
_CARD    = "#131316"
_FG      = "#FAFAFA"
_BODY    = "#D4D4D8"
_MUTED   = "#A1A1AA"
_EMERALD = "#34D399"   # model all programs
_BLUE    = "#60A5FA"   # buzzer
_AMBER   = "#FBBF24"   # model valid only
_GRID    = "#1F1F23"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_buzzer_csv(path: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (elapsed_ms, programs_submitted, valid_programs, unique_pcs)."""
    elapsed, submitted, valid, pcs = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed.append(int(row["elapsed_ms"]))
            submitted.append(int(row["programs_submitted"]))
            valid.append(int(row["valid_programs"]))
            pcs.append(int(row["unique_pcs"]))
    return elapsed, submitted, valid, pcs


def load_model_csv(path: str) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Return (elapsed_ms, programs_submitted, valid_programs, unique_pcs_valid, unique_pcs_all)."""
    elapsed, submitted, valid, pcs_v, pcs_a = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed.append(int(row["elapsed_ms"]))
            submitted.append(int(row["programs_submitted"]))
            valid.append(int(row["valid_programs"]))
            pcs_v.append(int(row["unique_pcs_valid"]))
            pcs_a.append(int(row["unique_pcs_all"]))
    return elapsed, submitted, valid, pcs_v, pcs_a


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_BODY, labelsize=11)
    ax.xaxis.label.set_color(_BODY)
    ax.yaxis.label.set_color(_BODY)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.8, linestyle="--", alpha=0.7)


def plot(
    buzzer_csv: str,
    model_csv: str,
    out_prefix: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    b_elapsed, b_submitted, b_valid, b_pcs          = load_buzzer_csv(buzzer_csv)
    m_elapsed, m_submitted, m_valid, m_pcs_v, m_pcs_a = load_model_csv(model_csv)

    def _annotate(ax, x, y, color):
        if x and y:
            ax.annotate(f"{y[-1]:,}", xy=(x[-1], y[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        color=color, fontsize=10, va="center")

    # -----------------------------------------------------------------------
    # Plot 1: unique PCs vs programs submitted (efficiency)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.75))
    _apply_dark_style(ax, fig)

    ax.plot(b_submitted, b_pcs,    color=_BLUE,    linewidth=2.0, label="Buzzer — valid programs only")
    ax.plot(m_submitted, m_pcs_v,  color=_AMBER,   linewidth=2.0, label="LLM — valid programs only")
    ax.plot(m_submitted, m_pcs_a,  color=_EMERALD, linewidth=2.0, label="LLM — all programs (valid + rejected)", linestyle="--")

    ax.set_xlabel("Programs submitted to verifier", fontsize=13)
    ax.set_ylabel("Cumulative unique kernel PCs", fontsize=13)
    ax.set_title("Coverage race — unique verifier PCs per program", fontsize=15, pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _annotate(ax, b_submitted, b_pcs,   _BLUE)
    _annotate(ax, m_submitted, m_pcs_v, _AMBER)
    _annotate(ax, m_submitted, m_pcs_a, _EMERALD)

    ax.legend(fontsize=12, facecolor=_CARD, edgecolor=_GRID, labelcolor=_FG)
    fig.tight_layout()

    out1 = f"{out_prefix}_per_program.png"
    fig.savefig(out1, dpi=150, facecolor=_BG)
    print(f"[*] Saved {out1}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: unique PCs vs elapsed time (real-time throughput)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.75))
    _apply_dark_style(ax, fig)

    b_minutes = [ms / 60_000 for ms in b_elapsed]
    m_minutes = [ms / 60_000 for ms in m_elapsed]

    ax.plot(b_minutes, b_pcs,   color=_BLUE,    linewidth=2.0, label="Buzzer — valid programs only")
    ax.plot(m_minutes, m_pcs_v, color=_AMBER,   linewidth=2.0, label="LLM — valid programs only")
    ax.plot(m_minutes, m_pcs_a, color=_EMERALD, linewidth=2.0, label="LLM — all programs (valid + rejected)", linestyle="--")

    ax.set_xlabel("Elapsed time (minutes)", fontsize=13)
    ax.set_ylabel("Cumulative unique kernel PCs", fontsize=13)
    ax.set_title("Coverage race — unique verifier PCs over time", fontsize=15, pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _annotate(ax, b_minutes, b_pcs,   _BLUE)
    _annotate(ax, m_minutes, m_pcs_v, _AMBER)
    _annotate(ax, m_minutes, m_pcs_a, _EMERALD)

    ax.legend(fontsize=12, facecolor=_CARD, edgecolor=_GRID, labelcolor=_FG)
    fig.tight_layout()

    out2 = f"{out_prefix}_over_time.png"
    fig.savefig(out2, dpi=150, facecolor=_BG)
    print(f"[*] Saved {out2}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    print("\n--- Summary ---")
    if b_pcs and b_submitted:
        print(f"Buzzer      : {b_pcs[-1]:>7,} PCs (valid only) | {b_submitted[-1]} programs | {b_valid[-1]} valid | {b_elapsed[-1]/1000:.0f}s")
    if m_pcs_v and m_submitted:
        print(f"Model valid : {m_pcs_v[-1]:>7,} PCs (valid only) | {m_submitted[-1]} programs | {m_valid[-1]} valid | {m_elapsed[-1]/1000:.0f}s")
    if m_pcs_a and m_submitted:
        print(f"Model all   : {m_pcs_a[-1]:>7,} PCs (all progs)  | {m_submitted[-1]} programs | {m_elapsed[-1]/1000:.0f}s")
    if b_pcs and m_pcs_v and b_submitted and m_submitted:
        b_per = b_pcs[-1] / b_submitted[-1] if b_submitted[-1] else 0
        m_per = m_pcs_v[-1] / m_submitted[-1] if m_submitted[-1] else 0
        print(f"\nPCs/program (valid) — Buzzer: {b_per:.1f}  Model: {m_per:.1f}  ratio: {m_per/b_per:.2f}x" if b_per else "")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coverage race comparison")
    parser.add_argument("--buzzer", default="/mnt/corpus/buzzer_coverage.csv",
                        help="Buzzer CSV path")
    parser.add_argument("--model",  default="/mnt/corpus/model_coverage.csv",
                        help="Model CSV path")
    parser.add_argument("--out",    default="results/coverage_race",
                        help="Output path prefix (two PNGs: _per_program.png and _over_time.png)")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(args.buzzer, args.model, args.out)


if __name__ == "__main__":
    main()
