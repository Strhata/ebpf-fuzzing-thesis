# Buzzer — Coverage Race Patch

Two patched files to apply on top of the upstream buzzer repo at
`/home/stefano-u/tesi/buzzer/` before building for the coverage race experiment.

The changes have already been applied to that repo. This directory just keeps a
copy in the thesis for reproducibility.

## What changed and why

### `main.go`
Removed the `MetricsUnit`, `CoverageManager`, and `addr2line` initialization.
The HTTP dashboard and its background `addr2line` goroutines burn CPU/RAM
resolving kernel PCs to source lines — this doesn't work in WSL and is pure
overhead when the goal is raw PC count tracking, not source visualization.

### `pkg/strategies/coverage_based.go`
Added two CLI flags and a CSV logger:
- `-coverage_log <path>` — where to write the CSV (default: `/mnt/corpus/buzzer_coverage.csv`)
- `-max_programs <N>`    — stop after N programs submitted (0 = run forever)

After each **valid** program, appends one row to the CSV:
```
elapsed_ms, programs_submitted, valid_programs, unique_pcs
```
This is the buzzer-side time series for the coverage race comparison plot.

## How to build and run

Build on WSL with bazel, then drop the binary into `data/corpus/` (the virtio-9p
shared dir mounted as `/mnt/corpus` inside the VM). No scp needed.

```bash
# 1. Build on WSL
cd /home/stefano-u/tesi/buzzer/
export CC=clang && export CXX=clang++
bazel build :buzzer

# 2. Drop into shared dir
cp bazel-bin/buzzer_/buzzer \
   /home/stefano-u/tesi/ebpf-fuzzing-thesis/data/corpus/buzzer

# 3. Run inside VM
/mnt/corpus/buzzer -strategy=coverage_based \
                   -max_programs=500 \
                   -coverage_log=/mnt/corpus/buzzer_coverage.csv

# 4. CSV appears at data/corpus/buzzer_coverage.csv on the host automatically
```

## Coverage race workflow (full picture)

```
buzzer run (inside VM)  →  data/corpus/buzzer_coverage.csv
model run  (on host)    →  data/corpus/model_coverage.csv   (tools/coverage_race.py)
plot       (on host)    →  results/coverage_race_*.png       (tools/plot_coverage_race.py)
```
