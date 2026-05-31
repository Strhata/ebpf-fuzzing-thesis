# Training Log

Dataset: `data/dataset_final_qwen.jsonl` — 24,762 train / 2,752 val (90/10 split, seed 42)  
Base model: Qwen/Qwen2.5-Coder-1.5B  
Config: QLoRA 4-bit NF4, rank-16, Q/K/V/O, lr=2e-4, batch=8 (grad_accum=8), bf16

Steps per epoch: ~3,095 | 3 epochs total: ~9,285 steps

---

## Phase 1 — sft_fase1_ass

| Field | Value |
|---|---|
| Location | `fuzzing_ml_env/modello_ebpf_sft_fase1_ass/` (external) |
| Copied to repo | `checkpoints/sft_fase1/adattatore_ebpf_v1` |
| Steps | 1,500 (`max_steps=1500` — truncated) |
| Epoch | 0.485 |
| Best eval_loss | 0.6210 |
| Resumed from | scratch (base Qwen weights) |
| Notes | Run from SFT_tesi.ipynb in fuzzing_ml_env. Same dataset, assembly format. |

---

## Phase 2 — curated_3ep warm start (2026-05-14)

| Field | Value |
|---|---|
| Location | `checkpoints/curated_3ep/` |
| Resume checkpoint | `checkpoints/curated_3ep/checkpoint-1500` |
| Steps | 1,500 warm-start steps (`max_steps=1500` — bug, now fixed) |
| Epoch | 0.485 |
| Best eval_loss | 0.6061 |
| Resumed from | `checkpoints/sft_fase1/adattatore_ebpf_v1` |
| WandB run | `curated-3ep` — https://wandb.ai/stefano-raheli-universit-del-salento/ebpf-thesis/runs/jeaaxzxr |

---

## Phase 3 — curated_3ep full run ✅ (2026-05-14, completed)

| Field | Value |
|---|---|
| Location | `checkpoints/curated_3ep/` |
| Best adapter | `checkpoints/curated_3ep/adattatore_ebpf_v1` |
| Resumed from | `checkpoints/curated_3ep/checkpoint-1500` (warm start — adapter weights only) |
| Steps | 9,288 total (3 full epochs) |
| Epochs | 3.0 |
| Final train loss | 0.5513 |
| Final eval loss | **0.5571** |
| Train runtime | 96,550 s (~26.8 h incl. eval every 100 steps; ~4 h pure training) |
| Throughput | 0.769 samples/sec (incl. eval overhead) |
| WandB run | `curated-3ep` — https://wandb.ai/stefano-raheli-universit-del-salento/ebpf-thesis/runs/7xybam1s |
| Log | `training_curated_3ep_full.log` (repo root) |

---

## Cumulative state (as of 2026-05-17)

| Model | Steps | Epochs | Eval loss | Status |
|---|---|---|---|---|
| `checkpoints/sft_fase1/adattatore_ebpf_v1` | 1,500 | 0.485 | 0.6210 | Warm-start base |
| `checkpoints/curated_3ep/adattatore_ebpf_v1` | 9,288 | 3.0 | **0.5571** | **Current best** |

---

## Phase 4 — Pass-rate evaluation ✅ (2026-05-17)

| Model | N | Compiled | ACCEPTED | Compile rate | Pass-rate |
|---|---|---|---|---|---|
| `curated-merged` | 100 | 73 | **60** | 73.0% | **60.0%** |
| `zero-shot` (Qwen2.5-Coder-1.5B base) | 100 | 1 | 1 | 1.0% | 1.0% |

**Key result**: SFT raises pass-rate from ~0% (base model, functionally zero) to **60%**, proving that verifier-log training data teaches the model to generate BPF-verifier-accepted programs.

Logs: `results/passrate_run_curated.log`, `results/passrate_run_zeroshot.log`  
CSV: `results/passrate_summary.csv`

### Next: Phase 5 — RL with GRPO

Build `tools/kcov_validator/` (Go), then `ml/reward.py` + `ml/rl_grpo.py`.  
See: `.scratch/rl-pipeline/issues-draft.md` issues 04–08.
