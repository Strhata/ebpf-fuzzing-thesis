# Thesis Roadmap — eBPF Fuzzing + ML
**Target:** July graduation (45-day window). Extend to October if needed.
**Last updated:** 2026-05-31

> ⚠️ **SUPERSEDED by [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) + [`RL_V2.md`](RL_V2.md).** Several
> entries below are *true-then, false-now*: the experimental phase was **not** closed (SFT-v2 +
> RL-v2 followed); RL-v2 **ran** (200 steps) with a **validity-gated** reward, not the verdict-blind
> depth reward; the reward server tunnel is **ngrok**, not the Cloudflare Quick Tunnel named here.
> Kept as the historical decisions log; do not cite its status claims.

---

## Decisions Log

| Topic | Decision |
|---|---|
| Repo visibility | Private GitHub, professors as collaborators. Publish clean public version post-defense. |
| Repo layout | Single monorepo `ebpf-fuzzing-thesis` with `fuzzing/` and `ml/` top-level dirs |
| Large artifacts | Models + corpus on HuggingFace. Only `adattatore_ebpf_v1` (final adapter) committed to git. |
| Model format | Assembly (verifier-log format) only. Hex approach documented as abandoned. |
| Training duration | 3 epochs (`num_train_epochs=3`), replacing `max_steps=1500` which only covered 48% of data. |
| WandB | `report_to="wandb"`, loss curves during training only. Pass-rate evaluated separately post-training. |
| Pass-rate eval | Standalone script: generate 100 programs → clang → llvm-objcopy → SSH → validator → table. |
| Crash log analysis | Python script: parse logs in `results/`, extract KASAN error type + top-3 stack frames → table for thesis. |
| KCOV mode | `KCOV_TRACE_PC` (value=0). Confirmed 2026-05-17 via `linux/include/uapi/linux/kcov.h`. Flat uint64 PC array, 1 word per entry. |
| TRL version | 0.14.0 (pixi-managed, `<0.15` pin). `GRPOTrainer` importable with workaround: patch `trl.import_utils.is_vllm_available = lambda: False` before import, and set `use_vllm=False` in `GRPOConfig`. Root cause: `is_vllm_available()` returns tuple `(False, None)` which is truthy in Python. |
| BPF encoder | Pure-Python encoder bypasses clang: extracts opcode byte from verifier-log format, packs dst/src/off/imm directly. Programs that would fail clang still reach the verifier. |
| RL reward tiers (run 1) | crash→2.0, new_pcs→1.0, valid→0.1, rejected→0.1, encode_fail→0.0. Valid and rejected share value 0.1 but are distinguished by tier label in the log. |
| RL reward redesign | Depth-based verdict-blind formula after KCOV bug discovery: `depth=min(0.5, pcs/max_pcs_seen*0.5) + 1.0 if new_pcs else 0.0`. REJECTED no longer penalised — only coverage depth matters. Implemented in `ml/reward.py`. |
| RL training run | `rl_grpo_v2`: beta=0.01, G=4, max_completion_length=600, ran 8370 steps total; plateau at step ~1300 (137 new PCs / 1638 total unique PCs), run manually stopped after 8370 steps. Root cause: `kcov_validator` returned empty PC trace for REJECTED → reward_std=0 → zero GRPO gradient. Fixed in commit c8a9d02. |
| Second RL run | Pipeline fully implemented (reward server, Colab notebook, auto-resume). **Not executed** — experimental phase closed. Documented as future work. |
| Remote training platform | Colab Pro chosen over Modal — Modal was scoped and abandoned (issues #2–7, closed as not-planned). Colab Pro requires manual ~24h restart but avoids Modal billing complexity. |

---

## Work Phases

### Phase 0 — Repo setup ✅
- Created `ebpf-fuzzing-thesis` private GitHub repo
- Structure: `fuzzing/`, `ml/`, `tools/`, `docs/`, `tests/`
- Upload final adapter + curated dataset to HuggingFace, add links to README

### Phase 1 — Data collection ✅
- Modified `fuzzing/buzzer/pkg/units/ffi.go` (~50 lines) to intercept `ValidateEbpfProgram`
- Dumps JSONL at kernel FFI boundary: `bytecode_hex`, `verifier_log`, `is_valid`, `error_line`, `error_reason`
- Dockerized single-VM setup (KASAN-only kernel, virtio-9p corpus share)
- Collected ~2M entries
- Crash logs from early swarm exploration in `results/`; crash analysis in `tools/classify_crashes.py`

### Phase 2 — Dataset curation + SFT ✅
- Analyzed error class distribution across ~2M raw entries
- Balanced to 27k samples (cap 2000/class) to avoid top-error-class domination
- QLoRA fine-tune: Qwen2.5-Coder-1.5B, rank-16, Q/K/V/O projections, 3 epochs
- Merged adapter published to HuggingFace (`Strhata/ebpf-checkpoints/curated_merged`)

### Phase 3 — Pass-rate evaluation ✅
- Script: `tools/evaluate_passrate.py`
- Pipeline: load adapter → generate programs → BPF encoder → SSH to VM → `kcov_validator` → CSV
- Results in `results/passrate_*.csv`; curated model outperforms baseline

### Phase 4 — RL training ✅ (local run complete)
- Script: `ml/rl_grpo.py` — GRPO with KCOV-based reward
- Run `rl_grpo_v2`: beta=0.01, G=4, T=600 — plateaued at ~138 cumulative PCs; run stopped
- Root cause identified: `kcov_validator` discarded KCOV trace for REJECTED → reward_std=0 → zero gradient
- Bug fixed (commit c8a9d02); reward redesigned to depth-based verdict-blind formula
- Analysis: `tools/analyze_rl_run.py` → tier CSVs + plots from `results/grpo_completions.log`

### Phase 5 — Colab Pro training pipeline ✅ (implemented, not executed — future work)
- `kcov_validator` fix: returns PCs for REJECTED programs (commit c8a9d02)
- `ml/reward.py`: depth-based verdict-blind reward + `max_pcs_seen` persistence
- `tools/reward_server.py`: FastAPI server exposing reward over HTTP, API key auth
- `ml/rl_grpo.py`: `--remote-reward-url` flag, exponential-backoff retry, `--resume` auto-detect
- `ml/train_grpo_colab.ipynb`: 5-cell Colab launcher, "Run All" is idempotent
- Cloudflare Quick Tunnel for stable HTTPS URL from Colab to local reward server
- Full pipeline ready; no training run executed before experimental phase closed

---

## Repo Structure (Current)

```
ebpf-fuzzing-thesis/
├── README.md
├── ml/
│   ├── reward.py                    # KCOV reward function (depth-based verdict-blind; see redesign decision)
│   ├── rl_grpo.py                   # GRPO RL training script
│   ├── train.py                     # SFT training script
├── tools/
│   ├── analyze_rl_run.py            # Parse grpo_completions.log → tier CSVs + plots
│   ├── classify_crashes.py
│   ├── evaluate_passrate.py
│   ├── vm_watchdog.sh
│   └── kcov_validator/              # Go binary: BPF_PROG_LOAD + KCOV → {verdict, pcs:[]} JSON
├── tests/
│   ├── test_reward.py               # Reward tier + PC set persistence tests
│   ├── test_reward_encoder.py       # BPF encoder unit tests
│   ├── test_sanity_callback.py      # SanityCheckCallback tests
│   └── test_analyze_rl_run.py       # RL log analysis tests
├── data/
│   └── dataset_final_qwen.jsonl     # Curated 27k training set (also on HuggingFace)
├── checkpoints/
│   └── curated_merged/              # Merged SFT model (base for RL)
├── results/
│   ├── grpo_completions.log         # Authoritative RL reward log (rl_grpo_v2 + warm-up)
│   ├── rl_pc_set.json               # Cumulative PC set for local RL run
│   ├── rl_analysis_tiers.csv        # Per-batch tier counts (generated by analyze_rl_run.py)
│   └── ...                          # Pass-rate CSVs, comparison logs
├── docs/
│   ├── ROADMAP.md                   # This file
│   ├── colab_restart_guide.md       # Manual restart procedure for Colab Pro session
│   ├── pipeline_spec.md             # Formal pipeline spec (for advisor review)
│   ├── training_log.md              # Checkpoint state tracking
│   └── tesi_recap.md                # Thesis chapter notes
└── fuzzing/
    ├── buzzer/                      # Modified buzzer fork (ffi.go data extraction)
    └── ...
```

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Pass-rate too low on both models | Report absolute numbers honestly; frame as baseline for future RL work |
| RL plateau persists at beta=0.1 | Expected — reward_std=0 is a known GRPO signal-collapse under homogeneous reward. Document as thesis finding. |
| Colab Pro session dies mid-run | Manual restart per `docs/colab_restart_guide.md`; auto-resume picks up from last checkpoint |
| July deadline too tight | Phase 3 + Phase 4 (local run only) is the minimum deliverable; Colab Pro run is additive |
