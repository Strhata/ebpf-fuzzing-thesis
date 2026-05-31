#!/usr/bin/env python3
"""
analyze_rl_run.py — Parse grpo_completions.log and produce per-tier CSVs + plots.

Outputs written to results/:
    rl_analysis_tiers.csv   — per-batch tier counts + reward stats, per segment
    rl_analysis_pcs.csv     — per-batch cumulative PC count, per segment
    rl_tier_plot.png        — stacked area chart of tier counts
    rl_pc_curve.png         — cumulative PC count over batches
    rl_reward_plot.png      — per-batch mean reward and reward std

Usage:
    pixi run python tools/analyze_rl_run.py
    pixi run python tools/analyze_rl_run.py --log results/grpo_completions.log --out results/
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).parent.parent

_LINE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+"
    r"batch=(\d+)\s+i=\d+\s+"
    r"(ACCEPTED|REJECTED|ENCODE_FAIL|COMPILE_FAIL|SSH_TIMEOUT)"
    r"(?:\s+new_pcs=(\d+))?"
)

_TIERS = ("new_pcs", "valid", "rejected", "encode_fail", "crash")
_TIER_COLORS = {
    "new_pcs":    "#2ecc71",
    "valid":      "#3498db",
    "rejected":   "#e74c3c",
    "encode_fail":"#95a5a6",
    "crash":      "#f39c12",
}


def _tier(verdict: str, new_pcs: int) -> str:
    if verdict == "ACCEPTED":
        return "new_pcs" if new_pcs > 0 else "valid"
    if verdict == "REJECTED":
        return "rejected"
    if verdict in ("ENCODE_FAIL", "COMPILE_FAIL"):
        return "encode_fail"
    return "crash"  # SSH_TIMEOUT


def _reward(verdict: str, new_pcs: int) -> float:
    if verdict == "ACCEPTED":
        return 1.0 if new_pcs > 0 else 0.1
    if verdict == "REJECTED":
        return 0.1
    if verdict in ("ENCODE_FAIL", "COMPILE_FAIL"):
        return 0.0
    return 2.0  # SSH_TIMEOUT


def parse_log(log_path: Path) -> list[list[dict]]:
    """Parse log into segments. A new segment begins when batch=0 reappears after batch>0."""
    records = []
    for line in log_path.read_text().splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        batch_s, verdict, new_pcs_s = m.groups()
        new_pcs = int(new_pcs_s) if new_pcs_s else 0
        records.append({"batch": int(batch_s), "verdict": verdict, "new_pcs": new_pcs})

    segments: list[list[dict]] = []
    current: list[dict] = []
    max_batch = -1
    for r in records:
        if r["batch"] == 0 and max_batch > 0:
            segments.append(current)
            current = []
            max_batch = -1
        max_batch = max(max_batch, r["batch"])
        current.append(r)
    if current:
        segments.append(current)
    return segments


def aggregate(segment: list[dict]) -> list[dict]:
    """Return sorted per-batch rows with tier counts, reward stats, cumulative_pcs."""
    batches: dict[int, dict] = {}
    for r in segment:
        bn = r["batch"]
        if bn not in batches:
            batches[bn] = {t: 0 for t in _TIERS}
            batches[bn]["_rewards"] = []
            batches[bn]["_sum_new_pcs"] = 0
        batches[bn][_tier(r["verdict"], r["new_pcs"])] += 1
        batches[bn]["_rewards"].append(_reward(r["verdict"], r["new_pcs"]))
        batches[bn]["_sum_new_pcs"] += r["new_pcs"]

    rows = []
    cumulative_pcs = 0
    for bn in sorted(batches):
        b = batches[bn]
        rwds = b["_rewards"]
        mean_r = sum(rwds) / len(rwds) if rwds else 0.0
        std_r = statistics.stdev(rwds) if len(rwds) > 1 else 0.0
        cumulative_pcs += b["_sum_new_pcs"]
        rows.append({
            "batch": bn,
            **{t: b[t] for t in _TIERS},
            "reward_mean": round(mean_r, 4),
            "reward_std": round(std_r, 4),
            "cumulative_pcs": cumulative_pcs,
        })
    return rows


def write_tiers_csv(segments_rows: list[list[dict]], path: Path) -> None:
    fieldnames = ["segment", "batch", *_TIERS, "reward_mean", "reward_std"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for seg_idx, rows in enumerate(segments_rows):
            for row in rows:
                w.writerow({"segment": seg_idx, **row})


def write_pcs_csv(segments_rows: list[list[dict]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["segment", "batch", "cumulative_pcs"])
        w.writeheader()
        for seg_idx, rows in enumerate(segments_rows):
            for row in rows:
                w.writerow({"segment": seg_idx, "batch": row["batch"],
                            "cumulative_pcs": row["cumulative_pcs"]})


def plot_tiers(segments_rows: list[list[dict]], path: Path) -> None:
    n = len(segments_rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    for seg_idx, (rows, ax) in enumerate(zip(segments_rows, axes[:, 0])):
        xs = [r["batch"] for r in rows]
        ys = [[r[t] for r in rows] for t in _TIERS]
        colors = [_TIER_COLORS[t] for t in _TIERS]
        ax.stackplot(xs, ys, labels=list(_TIERS), colors=colors, alpha=0.8)
        ax.set_title(f"Segment {seg_idx}: tier breakdown per batch")
        ax.set_xlabel("batch")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pcs(segments_rows: list[list[dict]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for seg_idx, rows in enumerate(segments_rows):
        xs = [r["batch"] for r in rows]
        ys = [r["cumulative_pcs"] for r in rows]
        ax.plot(xs, ys, label=f"segment {seg_idx}")
    ax.set_title("Cumulative new PCs over training batches")
    ax.set_xlabel("batch")
    ax.set_ylabel("cumulative PCs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rewards(segments_rows: list[list[dict]], path: Path) -> None:
    n = len(segments_rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    for seg_idx, (rows, ax) in enumerate(zip(segments_rows, axes[:, 0])):
        xs = [r["batch"] for r in rows]
        means = [r["reward_mean"] for r in rows]
        stds = [r["reward_std"] for r in rows]
        ax.plot(xs, means, label="reward mean", color="#2980b9")
        ax.fill_between(xs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color="#2980b9", label="±std")
        ax.set_title(f"Segment {seg_idx}: per-batch reward")
        ax.set_xlabel("batch")
        ax.set_ylabel("reward")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze(log_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = parse_log(log_path)
    total_records = sum(len(s) for s in segments)
    print(f"[*] Parsed {total_records} records across {len(segments)} segment(s)")

    segments_rows = [aggregate(s) for s in segments]
    for i, rows in enumerate(segments_rows):
        total_pcs = rows[-1]["cumulative_pcs"] if rows else 0
        print(f"    segment {i}: {len(rows)} batches, {total_pcs} cumulative PCs")

    write_tiers_csv(segments_rows, output_dir / "rl_analysis_tiers.csv")
    write_pcs_csv(segments_rows, output_dir / "rl_analysis_pcs.csv")
    plot_tiers(segments_rows, output_dir / "rl_tier_plot.png")
    plot_pcs(segments_rows, output_dir / "rl_pc_curve.png")
    plot_rewards(segments_rows, output_dir / "rl_reward_plot.png")

    print(f"[+] Outputs written to {output_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(_REPO_ROOT / "results" / "grpo_completions.log"),
                    help="Path to grpo_completions.log")
    ap.add_argument("--out", default=str(_REPO_ROOT / "results"),
                    help="Output directory for CSVs and plots")
    args = ap.parse_args()
    analyze(Path(args.log), Path(args.out))


if __name__ == "__main__":
    main()
