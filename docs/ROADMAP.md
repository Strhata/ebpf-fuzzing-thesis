# Thesis Roadmap — eBPF Fuzzing + ML
**Target:** July graduation (45-day window). Extend to October if needed.
**Last updated:** 2026-05-21

---

## Decisions Log

| Topic | Decision |
|---|---|
| Repo visibility | Private GitHub, professors as collaborators. Publish clean public version post-defense. |
| Repo layout | Single monorepo `ebpf-fuzzing-thesis` with `fuzzing/` and `ml/` top-level dirs |
| Large artifacts | Models + corpus on HuggingFace. Only `adattatore_ebpf_v1` (final adapter) committed to git. |
| Model format | Assembly (verifier-log format) only. Hex approach documented as abandoned. |
| Comparison design | Two models, same architecture (Qwen2.5-Coder-1.5B QLoRA r=16), same format, same size (27k). Variable: curation only. |
| Baseline dataset | Same valid-byte filter applied, no per-category cap. Random 27k sample → dominated by top error class (~34% `math between map_value pointer and <NUM>`). Proves balancing matters. |
| Training duration | 3 epochs (`num_train_epochs=3`), replacing `max_steps=1500` which only covered 48% of data. |
| WandB | `report_to="wandb"`, loss curves during training only. Pass-rate evaluated separately post-training. |
| Pass-rate eval | Standalone script: generate 100 programs → clang → llvm-objcopy → SSH → validator → table. |
| Crash log analysis | Python script: parse logs in `results/`, extract KASAN error type + top-3 stack frames → table for thesis. |
| KCOV mode | `KCOV_TRACE_PC` (value=0). Confirmed 2026-05-17 via `linux/include/uapi/linux/kcov.h`. Flat uint64 PC array, 1 word per entry. |
| TRL version | 0.14.0 (pixi-managed, `<0.15` pin). `GRPOTrainer` importable with workaround: patch `trl.import_utils.is_vllm_available = lambda: False` before import, and set `use_vllm=False` in `GRPOConfig`. Root cause: `is_vllm_available()` returns tuple `(False, None)` which is truthy in Python. |
| BPF encoder | Pure-Python encoder bypasses clang: extracts opcode byte from verifier-log format, packs dst/src/off/imm directly. Programs that would fail clang still reach the verifier. |
| RL reward tiers | crash→2.0, new_pcs→1.0, valid→0.2, rejected→0.1, encode_fail→0.0. `valid` and `rejected` use distinct values so WandB `rewards.count()` can distinguish them. |
| RL training run | `rl_grpo_v2`: beta=0.01, G=4, max_completion_length=400, running locally on 8GB GPU. Reached plateau at ~137 cumulative PCs (reward_std=0, all batches returning 0.1). |
| Second RL run | Planned at beta=0.1 on Colab Pro (T4 GPU). Reward function stays local; exposed via FastAPI + Cloudflare Tunnel. **Not yet implemented.** |
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
- Baseline dataset: same 27k, no cap, dominated by top class — control for curation value
- QLoRA fine-tune: Qwen2.5-Coder-1.5B, rank-16, Q/K/V/O projections, 3 epochs
- Merged adapter published to HuggingFace (`Strhata/ebpf-checkpoints/curated_merged`)

### Phase 3 — Pass-rate evaluation ✅
- Script: `tools/evaluate_passrate.py`
- Pipeline: load adapter → generate programs → BPF encoder → SSH to VM → `kcov_validator` → CSV
- Results in `results/passrate_*.csv`; curated model outperforms baseline

### Phase 4 — RL training (in progress)
- Script: `ml/rl_grpo.py` — GRPO with KCOV-based reward, running locally
- Run `rl_grpo_v2`: beta=0.01, G=4, started 2026-05-18, at batch ~6665 as of 2026-05-21
- Plateau reached at ~137 cumulative PCs; reward_std=0 indicates GRPO signal collapse
- Analysis: `tools/analyze_rl_run.py` → produces tier CSVs + plots from `results/grpo_completions.log`
- **Planned:** second run at beta=0.1 on Colab Pro (pending pipeline implementation)

### Phase 5 — Colab Pro training pipeline (pending)
- Local reward server (`tools/reward_server.py`) — FastAPI wrapper around `reward.py`, auth via API key
- Remote reward mode in `ml/rl_grpo.py` — `--remote-reward-url` flag, HTTP calls with retry
- Auto-resume — automatic checkpoint detection + WandB run ID continuity via `wandb_run_id.txt`
- Colab notebook (`ml/train_grpo_colab.ipynb`) — mounts Drive, calls `rl_grpo.py` with remote flag
- See `docs/colab_restart_guide.md` for manual restart procedure

---

## Repo Structure (Current)

```
ebpf-fuzzing-thesis/
├── README.md
├── ml/
│   ├── reward.py                    # KCOV reward function (tiers: crash/new_pcs/valid/rejected/encode_fail)
│   ├── rl_grpo.py                   # GRPO RL training script
│   ├── train.py                     # SFT training script
│   └── build_baseline_dataset.py
├── tools/
│   ├── analyze_rl_run.py            # Parse grpo_completions.log → tier CSVs + plots
│   ├── classify_crashes.py
│   ├── evaluate_passrate.py
│   ├── vm_watchdog.sh
│   └── kcov_validator/              # C binary: BPF_PROG_LOAD + KCOV → {verdict, pcs:[]} JSON
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
| Baseline model performs equally to curated | Still publishable: shows robustness of balancing; adjust claim in thesis |
| Pass-rate too low on both models | Report absolute numbers honestly; frame as baseline for future RL work |
| RL plateau persists at beta=0.1 | Expected — reward_std=0 is a known GRPO signal-collapse under homogeneous reward. Document as thesis finding. |
| Colab Pro session dies mid-run | Manual restart per `docs/colab_restart_guide.md`; auto-resume picks up from last checkpoint |
| July deadline too tight | Phase 3 + Phase 4 (local run only) is the minimum deliverable; Colab Pro run is additive |
