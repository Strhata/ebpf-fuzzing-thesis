#!/usr/bin/env python3
"""
plot_rl_trajectories.py — F3: the two RL runs, side by side.

Two figures, each contrasting RL-1 (the starvation) with RL-2 phase-B (the fix):

  rl_reward_std.*  reward/std per step — RL-1 collapses to 0 (GRPO gradient
                   starvation); RL-2's validity-gated soft floor keeps it > 0.
  rl_coverage.*    cumulative unique PCs per step — RL-1 plateaus (+137 total);
                   RL-2 climbs to ~4.8k but decelerating. NB this is *total* PCs
                   (incl. invalid programs' verifier-walk PCs), not valid-unique.

Pulls history from WandB (needs auth; runs live in the `huggingface` project):
  RL-1        = grpo-rl-v1        (hapj9sah, 8,373 steps)
  RL-2 phaseB = grpo-rlv2-phaseB  (bz5ymfzl, 207 steps)

Usage:
    pixi run python tools/plot_rl_trajectories.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

_REPO = Path(__file__).resolve().parent.parent
_ENTITY = "stefano-raheli-universit-del-salento"
_PROJECT = "huggingface"
_RL1 = ("hapj9sah", "RL-1 (grpo-rl-v1, 8,373 steps)", "#DC2626")
_RL2 = ("bz5ymfzl", "RL-2 phase-B (207 steps)", "#2563EB")
_STEP = "train/global_step"


def fetch(run_id: str, cols: list[str]):
    """Return a step-sorted list of (step, {col: val}) for the given run."""
    api = wandb.Api()
    run = api.run(f"{_ENTITY}/{_PROJECT}/{run_id}")
    h = run.history(samples=10_000, pandas=True)
    xcol = _STEP if _STEP in h.columns else "_step"
    out = {}
    for c in cols:
        if c in h.columns:
            s = h[[xcol, c]].dropna()
            out[c] = (s[xcol].tolist(), s[c].tolist())
    return out


def two_panel(metrics: list[str], ylabel: str, title: str, out: Path, *, logy: bool = False) -> None:
    """`metrics` = candidate column names, first present wins (logging keys drifted)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (rid, label, color) in zip(axes, (_RL1, _RL2)):
        data = fetch(rid, metrics)
        present = next((m for m in metrics if m in data), None)
        if present is None:
            print(f"WARN {rid} has none of {metrics}")
            continue
        xs, ys = data[present]
        ax.plot(xs, ys, color=color, lw=1.0)
        ax.set_title(label)
        ax.set_xlabel("training step")
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} (+ .png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(_REPO / "thesis/figures"))
    args = ap.parse_args()
    d = Path(args.outdir)

    two_panel(
        ["reward/std", "train/reward_std"], "reward / std",
        "Reward variance: RL-1 starves (std → 0), RL-2's soft floor keeps it alive",
        d / "rl_reward_std.pdf",
    )
    two_panel(
        ["cumulative_pcs"], "cumulative unique PCs (total, incl. invalid)",
        "Coverage trajectory: RL-1 plateaus; RL-2 climbs but decelerates",
        d / "rl_coverage.pdf",
    )


if __name__ == "__main__":
    main()
