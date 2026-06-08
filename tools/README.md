# tools/ — what each script is, and which ones matter

Not all of these are equal. Most of `tools/` is **core pipeline**; a handful are **one-off
exploratory scripts** kept for reproducibility but not part of the main flow. Start with the core
list. Concepts/why live in [`../docs/FACTS.md`](../docs/FACTS.md) and
[`../docs/DECISIONS.md`](../docs/DECISIONS.md); how to run them in [`../docs/ops/`](../docs/ops/).

## ★ Core pipeline (the live system)

| Tool | Does |
|---|---|
| `kcov_validator/` (Go) | Loads a BPF program (`BPF_PROG_LOAD`) into the KCOV kernel, returns `{verdict, pcs[]}` JSON. The validator the whole project is built on. |
| `reward_server.py` | FastAPI server exposing `reward.compute_rewards()` over HTTP (API-key, `RL_W_*` weights). The RL-2 reward bridge. |
| `benchmark.py` + `benchmark_lib.py` | `prepare` (merge LoRA → fp16) / `run` / `report` — the model-comparison harness. |
| `diversity_sample.py` | generate → validate → measure: the saturation experiment (valid-unique PCs vs program count). |
| `coverage_race.py` + `plot_coverage_race.py` | LLM side of the buzzer-vs-LLM coverage race + its 3-curve plot. |
| `generate_bytecodes.py` + `validate_bytecodes.py` | Seeded, reproducible generation → KCOV validation (the "60 %" reconstruction; dual clang/encoder). |

## ◐ Supporting (live, auxiliary)

| Tool | Does |
|---|---|
| `merge_lora.py` | Standalone LoRA-adapter → bf16 merge (`make` target; `benchmark.py prepare` also does this inline). |
| `evaluate_passrate.py` + `ebpf_validator/` (Go) | **Legacy** clang + `ebpf_validator` pass-rate path (the 60 %-era pipeline). Kept as a diagnostic; superseded by the encoder + `kcov_validator` path. |
| `vm_watchdog.sh` | Restarts the eval VM if SSH times out during a run (called by `ml/reward.py`). |
| `analyze_rl_run.py` | Parses RL-1's `results/grpo_completions.log` → tier CSVs + reward/PC plots. RL-1-specific; has tests. |

## ○ One-off / exploratory (kept for reproducibility — not the main flow)

These have no callers; they were run by hand at a point in time. Safe to ignore on a first read.

| Tool | Was for |
|---|---|
| `classify_crashes.py` | Early (Phase-1) verifier-error distribution from buzzer's `report_errori.txt`. |
| `generate_corpus.py` | One-at-a-time inference corpus builder; superseded by `diversity_sample.py`. |
| `probe_enrichment.py` | Manual smoke-test of the enrichment format (raw input→output through the validator). |
| `run_comparison.sh` | RL-1-era 600-vs-800 token completion-length comparison; superseded by `benchmark.py`. |
| `quantize_awq.py` | AWQ-quantize the merged model → `curated_awq`. An experiment; the output was not used downstream. |
