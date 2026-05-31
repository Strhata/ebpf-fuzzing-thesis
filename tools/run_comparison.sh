#!/bin/bash
# run_comparison.sh — 600 vs 800 token comparison
# Runs Config A (600), then Config B (800), generates results/comparison_summary.txt
# Safe to interrupt: each config writes its own log before the next starts.
# NOTE: no set -e — we capture exit codes explicitly so OOM in Config B is handled gracefully.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO/results"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS/comparison_run.log"; }

# ---------------------------------------------------------------------------
# 0. Pre-flight
# ---------------------------------------------------------------------------
log "=== Comparison test starting ==="

pkill -f rl_grpo.py 2>/dev/null && log "Killed existing training run" || true
sleep 3

# Rotate any existing completions log
if [ -f "$RESULTS/grpo_completions.log" ]; then
    mv "$RESULTS/grpo_completions.log" "$RESULTS/grpo_completions_pre_cmp.log"
    log "Rotated existing grpo_completions.log"
fi

# Snapshot current PC set as the shared baseline for both configs
cp "$RESULTS/rl_pc_set.json" "$RESULTS/rl_pc_set_cmp_baseline.json"
BASELINE_PCS=$(python3 -c "import json; print(len(json.load(open('$RESULTS/rl_pc_set_cmp_baseline.json'))))")
log "PC baseline: $BASELINE_PCS unique PCs"

# ---------------------------------------------------------------------------
# 1. Config A — G=4, max_completion_length=600
# ---------------------------------------------------------------------------
log "--- Config A: G=4 max=600 beta=0.01 (100 steps) ---"

cp "$RESULTS/rl_pc_set_cmp_baseline.json" "$RESULTS/rl_pc_set.json"

cd "$REPO"
if pixi run python ml/rl_grpo.py \
    --num-generations 4 \
    --max-completion-length 600 \
    --beta 0.01 \
    --max-steps 100 \
    --output-dir checkpoints/rl_grpo_cmp600 \
    > "$RESULTS/rl_grpo_cmp600.log" 2>&1; then
    EXIT_A=0
else
    EXIT_A=$?
fi

if [ $EXIT_A -ne 0 ]; then
    log "ERROR: Config A exited with code $EXIT_A — check $RESULTS/rl_grpo_cmp600.log"
    exit 1
fi

# Save Config A's completions log before it gets overwritten
mv "$RESULTS/grpo_completions.log" "$RESULTS/grpo_completions_cmpA.log"
log "Config A done. Completions log saved."

# ---------------------------------------------------------------------------
# 2. Config B — G=4, max_completion_length=800
# ---------------------------------------------------------------------------
log "--- Config B: G=4 max=800 beta=0.01 (100 steps) ---"

# Restore baseline so both configs start from the same known PCs
cp "$RESULTS/rl_pc_set_cmp_baseline.json" "$RESULTS/rl_pc_set.json"

if pixi run python ml/rl_grpo.py \
    --num-generations 4 \
    --max-completion-length 800 \
    --beta 0.01 \
    --max-steps 100 \
    --output-dir checkpoints/rl_grpo_cmp800 \
    > "$RESULTS/rl_grpo_cmp800.log" 2>&1; then
    EXIT_B=0
else
    EXIT_B=$?
fi

if [ $EXIT_B -ne 0 ]; then
    log "ERROR: Config B exited with code $EXIT_B (may be OOM) — check $RESULTS/rl_grpo_cmp800.log"
    log "Decision: use max_completion_length=600 (Config B failed)"
    DECISION="600 (Config B failed — OOM or crash)"
    python3 - "$RESULTS" "$BASELINE_PCS" "$DECISION" << 'PYEOF'
import sys, json, re
from pathlib import Path

results_dir = Path(sys.argv[1])
baseline_pcs = int(sys.argv[2])
decision = sys.argv[3]

def parse_log(path):
    if not path.exists():
        return {"acc": 0, "cf": 0, "rif": 0, "new_pcs": 0, "total": 1,
                "acc_pct": 0, "cf_pct": 0}
    pat = re.compile(r'(ACCEPTED|COMPILE_FAIL|REJECTED)')
    npc = re.compile(r'new_pcs=(\d+)')
    counts = {"ACCEPTED": 0, "COMPILE_FAIL": 0, "REJECTED": 0}
    new_pcs = 0
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            counts[m.group(1)] += 1
        n = npc.search(line)
        if n and int(n.group(1)) > 0:
            new_pcs += int(n.group(1))
    total = sum(counts.values()) or 1
    return {
        "acc": counts["ACCEPTED"],
        "cf": counts["COMPILE_FAIL"],
        "rif": counts["REJECTED"],
        "total": total,
        "acc_pct": 100 * counts["ACCEPTED"] / total,
        "cf_pct": 100 * counts["COMPILE_FAIL"] / total,
        "new_pcs": new_pcs,
    }

a = parse_log(results_dir / "grpo_completions_cmpA.log")

report = f"""# GRPO 600 vs 800 Token Comparison
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

## Baseline
PC set at start of comparison: {baseline_pcs} unique PCs

## Config A — G=4, max=600, beta=0.01
Steps: 100
Total completions: {a['total']}
ACCEPTED:    {a['acc']:5d}  ({a['acc_pct']:.1f}%)
COMPILE_FAIL: {a['cf']:5d}  ({a['cf_pct']:.1f}%)
REJECTED:    {a['rif']:5d}  ({100*a['rif']/(a['total'] or 1):.1f}%)
New PCs:      {a['new_pcs']}

## Config B — G=4, max=800, beta=0.01
FAILED (OOM or crash) — see rl_grpo_cmp800.log

## Decision
{decision}

## Full run command
pixi run python ml/rl_grpo.py \\
  --num-generations 4 \\
  --max-completion-length 600 \\
  --beta 0.01 \\
  --output-dir checkpoints/rl_grpo_v2
"""
(results_dir / "comparison_summary.txt").write_text(report)
print(report)
PYEOF
    exit 0
fi

mv "$RESULTS/grpo_completions.log" "$RESULTS/grpo_completions_cmpB.log"
log "Config B done. Completions log saved."

# ---------------------------------------------------------------------------
# 3. Analysis + report
# ---------------------------------------------------------------------------
log "--- Generating comparison report ---"

python3 - "$RESULTS" "$BASELINE_PCS" << 'PYEOF'
import sys, json, re
from pathlib import Path

results_dir = Path(sys.argv[1])
baseline_pcs = int(sys.argv[2])

def parse_log(path):
    if not path.exists():
        return {"acc": 0, "cf": 0, "rif": 0, "new_pcs": 0, "total": 1,
                "acc_pct": 0, "cf_pct": 0}
    pat = re.compile(r'(ACCEPTED|COMPILE_FAIL|REJECTED)')
    npc = re.compile(r'new_pcs=(\d+)')
    counts = {"ACCEPTED": 0, "COMPILE_FAIL": 0, "REJECTED": 0}
    new_pcs = 0
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            counts[m.group(1)] += 1
        n = npc.search(line)
        if n and int(n.group(1)) > 0:
            new_pcs += int(n.group(1))
    total = sum(counts.values()) or 1
    return {
        "acc": counts["ACCEPTED"], "cf": counts["COMPILE_FAIL"],
        "rif": counts["REJECTED"], "total": total,
        "acc_pct": 100 * counts["ACCEPTED"] / total,
        "cf_pct": 100 * counts["COMPILE_FAIL"] / total,
        "new_pcs": new_pcs,
    }

a = parse_log(results_dir / "grpo_completions_cmpA.log")
b = parse_log(results_dir / "grpo_completions_cmpB.log")

cf_delta = a["cf_pct"] - b["cf_pct"]   # positive = B is better
npc_delta = b["new_pcs"] - a["new_pcs"]  # positive = B found more

THRESHOLD_CF = 10.0  # percentage points
if cf_delta >= THRESHOLD_CF:
    decision = f"800 tokens (COMPILE_FAIL reduced by {cf_delta:.1f}pp)"
    chosen_len = 800
elif npc_delta > 0:
    decision = f"800 tokens (more new_pcs: +{npc_delta})"
    chosen_len = 800
else:
    decision = f"600 tokens (800 showed no clear advantage)"
    chosen_len = 600

report = f"""# GRPO 600 vs 800 Token Comparison
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

## Baseline
PC set at start of comparison: {baseline_pcs} unique PCs

## Results

|                | Config A (max=600) | Config B (max=800) | Delta (B-A) |
|----------------|-------------------|---------------------|-------------|
| Steps          | 100               | 100                 | —           |
| Total completions | {a['total']:16d} | {b['total']:19d} | —           |
| ACCEPTED%     | {a['acc_pct']:17.1f}% | {b['acc_pct']:18.1f}% | {b['acc_pct']-a['acc_pct']:+.1f}pp  |
| COMPILE_FAIL%  | {a['cf_pct']:17.1f}% | {b['cf_pct']:18.1f}% | {b['cf_pct']-a['cf_pct']:+.1f}pp  |
| New PCs        | {a['new_pcs']:17d} | {b['new_pcs']:19d} | {npc_delta:+d}         |

## Decision criterion
≥10pp lower COMPILE_FAIL in B → choose 800
More new_pcs in B → choose 800
Otherwise → choose 600

COMPILE_FAIL delta (A - B): {cf_delta:+.1f}pp
New PCs delta (B - A): {npc_delta:+d}

## Decision
>>> {decision} <<<

## Full run command
pixi run python ml/rl_grpo.py \\
  --num-generations 4 \\
  --max-completion-length {chosen_len} \\
  --beta 0.01 \\
  --output-dir checkpoints/rl_grpo_v2 \\
  > results/rl_grpo_v2.log 2>&1
"""
out = results_dir / "comparison_summary.txt"
out.write_text(report)
print(report)
PYEOF

log "=== Done. Report: $RESULTS/comparison_summary.txt ==="
