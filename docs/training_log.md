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

## Phase 2 — curated_3ep (this morning, 2026-05-14)

| Field | Value |
|---|---|
| Location | `checkpoints/curated_3ep/` |
| Best adapter | `checkpoints/curated_3ep/adattatore_ebpf_v1` |
| Resume checkpoint | `checkpoints/curated_3ep/checkpoint-1500` |
| Steps | 1,500 (`max_steps=1500` — bug now fixed) |
| Epoch | 0.485 |
| Best eval_loss | **0.6061** (improved from Phase 1) |
| Resumed from | `checkpoints/sft_fase1/adattatore_ebpf_v1` |
| WandB run | `curated-3ep` — https://wandb.ai/stefano-raheli-universit-del-salento/ebpf-thesis/runs/jeaaxzxr |

---

## Phase 3 — curated_3ep completion (TODO)

| Field | Value |
|---|---|
| Command | `pixi run python ml/train.py --run curated` |
| Resumes from | `checkpoints/curated_3ep/checkpoint-1500` (warm start — adapter weights only) |
| Epochs | 3 full epochs on curated dataset (fresh optimizer from checkpoint-1500 weights) |
| Fix applied | `max_steps=1500` → `num_train_epochs=3` in `ml/train.py` |
| Estimated time | ~4.3 h at ~1.67 s/step on RTX 4070 Laptop |
| Note | Full checkpoint resume blocked by PyTorch 2.5.1 CVE-2025-32434 check on optimizer.pt. Warm start used instead. |

---

## Cumulative state

Total gradient steps applied to current best model (`curated_3ep/adattatore_ebpf_v1`):
- Phase 1: 1,500 steps on curated dataset
- Phase 2: 1,500 more steps on curated dataset  
- **Total: ~3,000 steps ≈ 0.97 epochs on curated dataset**

Phase 3 will bring this to ~9,285 steps = 3 full epochs.

---

## Evaluation (TODO — after Phase 3)

Pipeline: `tools/evaluate_passrate.py --adapter checkpoints/curated_3ep/adattatore_ebpf_v1`  
Requires: QEMU VM running `bzImage_kasan_kcov`, `ebpf_validator` at `/mnt/corpus/`  
See: `.scratch/thesis-plan/PRD.md` for full evaluation spec.
