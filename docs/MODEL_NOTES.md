# Model Notes — SFT-v2 (`sft-1epoch-v2`)

> The committed record of **how the canonical SFT-v2 model was trained** and **how it performs** on
> the diversity benchmark. Structured to mirror the thesis chapter split (Methodology →
> Experimental Results → Conclusions) so it can feed ch4/ch6.
>
> **Canonical SFT-v2** = the **full one-epoch** run `sft-1epoch-v2` (~2,408 steps), adapter at
> `checkpoints/sft-1epoch-v2/sft_adapter`. (An earlier partial probe, `sft_retrain`/checkpoint-1500
> at ~1,500 steps / 0.62 epoch, is the n=20 "19%" benchmark — same work, fewer steps, not the
> diversity model.) Base `Qwen/Qwen2.5-Coder-1.5B`. Last updated: 2026-06-07.

---

## 1. Methodology — how the model was trained

### 1.1 Data collection
- Corpus mined with a modified **buzzer** fuzzer that dumps `(bytecode_hex, verifier_log)`
  pairs as it explores. ~2M raw programs collected; ~27k curated.
- Coverage signal is **KCOV** (`KCOV_TRACE_PC`): the flat array of kernel PCs the BPF
  verifier walks while checking a program. This is the diversity metric throughout.

### 1.2 Dataset curation & enrichment
- Training file: `dataset_final_qwen_enriched.jsonl` (HF: `Strhata/ebpf-corpus`).
- Each program is enriched with a **novelty score** and binned into `coverage_bin` and
  `novelty_bin` (low/med/high tertiles) via `ml/enrich_dataset.py`.
- Split: **stratified 70/15/15** train/val/test (`seed=42`), test frozen to disk before
  training. Stratified on validity.

### 1.3 Representation (the important design choice)
- Programs are learned as **BPF assembly text**, not raw bytecode. A pure-Python
  assembler (`reward._encode_to_hex`) converts assembly → bytecode for validation.
- Prompt format (control tokens steer coverage/novelty at inference):
  ```
  [coverage={low|med|high}][novelty={low|med|high}]
  ### ASSEMBLY:
  {assembly}{eos}
  ```
- Inference for the benchmark uses `[coverage=high][novelty=high]` to request the most
  ambitious programs.

### 1.4 SFT configuration (QLoRA)
| knob | value |
|---|---|
| base | `Qwen/Qwen2.5-Coder-1.5B` |
| quantization | 4-bit **NF4**, double-quant, **bf16** compute (`BitsAndBytesConfig`) |
| LoRA | r=16, α=32, dropout=0.05 |
| LoRA targets | `q/k/v/o_proj` **+ MLP** `gate/up/down_proj` (attention-only plateaus on code) |
| optimizer | `paged_adamw_8bit` |
| learning rate | **5e-5**, cosine schedule, warmup_ratio 0.03 (2e-4 plateaued at loss ~1.2) |
| batch | per-device 1 × grad-accum 8 = **effective 8** |
| seq length | **2048** tokens, gradient checkpointing on |
| epochs | **1** (≈2408 optimizer steps) |
| seed | 42 |

> Precision note: base weights stored 4-bit NF4 but **all compute is bf16**. For
> inference/RL the adapter is loaded on the bf16 base (or merged) — never 4-bit
> (post-merge 4-bit destroyed quality in the benchmark).

---

## 2. Experimental Results — performance

### 2.1 Training dynamics (SFT-v2 run, W&B `ebpf-thesis`)
- 1 epoch, ~5.8h on A100. train/loss ≈ 1.07, eval/loss ≈ 1.08, test/loss ≈ 1.07 — no
  overfitting. Encoder pass-rate (assembly → valid bytecode) ≈ 0.90 val / 1.00 test.
- Healthy convergence; loss is **not** the bottleneck.

### 2.2 Diversity evaluation (the spine metric)
Pipeline: `tools/diversity_sample.py generate` (Colab A100, bf16 adapter-on-base) →
candidates JSONL → `validate` (local KCOV VM, batched). Spine metric = **unique KCOV PCs
from VALID (verifier-ACCEPTED) programs**. `total_unique_pcs` unions over *all* programs
incl. rejected, so it overcounts; use the valid-only figures.

| metric | n=1000 (512 tok) | n=20000 (1024 tok) |
|---|---:|---:|
| accept_rate (validity) | 7.5% | 6.6% |
| **valid programs** | 75 | 1,310 |
| **valid_unique_pcs** (spine) | 3,462 | 3,862 |
| avg distinct PCs / valid prog | 2,337 | 2,428 |
| total_unique_pcs (all progs) | 4,749 | 4,972 |
| novelty_score | 0.752 | 0.758 |
| avg_insn_count | 30.2 | 57.6 |

Artifacts: `benchmarks/diversity/sft-v2-n{1000,20000}-seed42.json`.

### 2.3 The saturation result (headline)
- **17.5× more valid programs → only +12% unique PCs** (3,462 → 3,862).
- **1,310 valid programs cover just 1.59×** what a single average valid program covers
  (3,862 / 2,428) — i.e. the programs are nearly redundant, marching the same verifier paths.
- Doubling program length (512→1024 tok, `avg_insn` 30→58) did **not** raise diversity →
  the wall is **clustering, not program length, and not loss**.

---

## 3. Conclusions / reminders
- **Diversity, not validity, is the wall** — now with a hard saturation curve (2 points;
  a 3rd intermediate N would make a cleaner thesis figure). The model is well-trained and
  generates deep programs, but they cluster.
- Validity is also low (~7%) under the `high/high` prompt — secondary issue; even the valid
  set saturates, so raising validity alone won't fix diversity.
- **Next bet:** RL-v2 to break clustering (sampling diversity / novelty reward). May fail —
  that's the open experiment.

### Reproduce
```bash
# generate (Colab A100): ml/diversity_generate_colab.ipynb  (no HF token; Drive + download)
# validate (local, VM up):
./fuzzing/run_eval_vm.sh
pixi run python tools/diversity_sample.py validate \
    --candidates benchmarks/diversity/candidates/sft-v2-n20000-seed42.jsonl \
    --out benchmarks/diversity/sft-v2-n20000-seed42.json
```

> Gotchas: validate streams PCs (memory-bounded) — needed because N=20k OOM'd the 9.7GB
> WSL box otherwise. A100 in Colab is **40GB GPU** (the 80GB is system RAM); size gen
> batches against 40GB (batch 800 × 1024 tok ≈ 33GB peak).

---

## 4. RL-v2 design (the bet to break clustering) — decided 2026-06-08

Grounded in literature: the SFT-v2 clustering is the documented **mode/diversity/entropy
collapse** of GRPO under a scalar reward. Standard fix = **reward defined over the group**
(GAPO / DiverseGRPO) + entropy/clip-higher; QD/MAP-Elites archive is the escalation.

### 4.1 The two facts that drive the design
- **The model has no validity lever.** SFT prompt conditions only on `[coverage][novelty]`;
  there is no validity token. Training data is **48% valid / 52% invalid** (13,153/27,514),
  so under any prompt it samples a ~half-invalid distribution → ~7% accept at `high/high`.
  Validity is therefore the biggest untapped RL lever (buzzer cares only about valid).
- **~7% valid + G generations** ⇒ at G=8 a group averages ~0.6 valid (most groups have
  ZERO valid → novelty never fires); at **G=16** ~1.3 valid/group → signal exists.

### 4.2 Reward ladder (in `ml/reward.py`, env-tunable `RL_W_*`)
Monotonic so valid always beats invalid:
```
encode fail / < floor insns   -> 0.0
VM ERROR / whole-batch crash   -> 0.0   (was 2.0 in RL-v1 — that rewarded SSH flakiness)
REJECTED (invalid)             -> W_REJECT_MAX * min(1, len(pcs)/max_pcs_seen)   [default 0.3]
ACCEPTED (valid)               -> W_VALID + W_NOVELTY * group_novelty(p)         [defaults 1.0, 1.0]
```
- **Soft floor** (depth walked before rejection) is deliberate: at 7% valid most groups are
  all-invalid; a hard 0-gate → identical rewards → `reward_std=0` → no gradient. That `std=0`
  gradient starvation **was the RL-v1 negative result.** The floor guarantees within-group
  variance so the model learns "get closer to valid" before it ever produces a valid one.
- **group_novelty(p)** = mean over p's PCs of `(1 - freq/n_accepted)`, freq = # ACCEPTED
  programs in THIS batch hitting that PC. Per-group ("Option A"): stateless, never starves.

### 4.3 Phasing (per-group A now; decayed-global B is the real anti-clustering run)
- **A = per-group novelty (built).** Reward novelty vs batch siblings only. Smoke-test reward:
  proves the loop runs. Fights *crowding within a batch*, not repetition across the run.
- **B = decayed global (next, NOT built).** Novelty vs a running PC-frequency map living in
  the reward server process (local, persists across `/rewards` calls — seed = existing
  `_pc_set`). Decay (pay less the more a PC was hit) avoids naive-B's reward collapse. This is
  the published anti-clustering recipe and the actual thesis experiment. QD archive = escalation.

### 4.4 Config (locked) + smoke-test gate
- **G=16** target (memory-gated on 40GB; fallback ladder: drop MAX_LEN → G=12 → G=8, *protect
  G over length*). **max_completion_length=512** — empirically justified (§2.3: 512→1024 raised
  insns 30→58 but unique PCs only +12%; length is not the diversity lever, so spend memory on G).
  beta=0.05, lr=5e-6, temperature=0.9 (unchanged).
- **Smoke test = small throwaway run** (`MAX_STEPS=20`). Pass criteria (W&B):
  **`reward/std > 0`** (the RL-v1 guard), `reward/mean` up, `valid_rate > 0`, no NaN, KL bounded.
  Its job is "does the machine turn," NOT data — don't extract science from it.

### 4.5 Architecture (unchanged bridge; reward rewrite is server-side)
Colab GRPOTrainer → ngrok HTTP `/rewards` → local FastAPI `reward_server.py` → SSH → KCOV VM.
Training knobs live in notebook **Cell 4** (Colab); **reward weights live on the local server**
(`RL_W_*` env at uvicorn launch). RL needs a **merged fp16** model — notebook **Cell 3b** merges
the SFT-v2 adapter onto the base once (never 4-bit a merge: benchmark merged_bnb4bit → 0% valid).

> Tests: `tests/test_reward.py` (32) rewritten for this ladder incl. a regression guard
> (`test_all_rejected_group_still_has_variance`) that fails if the RL-v1 `std=0` trap returns;
> encoder (`test_reward_encoder.py`, 50) and server (`test_reward_server.py`, 6) tests are separate
> files — 88 total.
