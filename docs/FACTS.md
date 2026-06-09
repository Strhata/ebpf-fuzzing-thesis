# FACTS — canonical reference (always current)

**This file is overwritten in place. It states what is true *now*, never what happened.** No dates on
facts, no "superseded" banners — when something changes, edit the line. History lives in
[`JOURNAL.md`](JOURNAL.md); rationale in [`DECISIONS.md`](DECISIONS.md); how-to in [`ops/`](ops/).

---

## 1. The objective (and how we judge a model)

Verifier bugs are found by programs that are **valid** (the verifier accepts them) **and**
**path-diverse** (they exercise many distinct verifier code paths). A rejected program proves the
verifier did its job — it is not a finding.

**The only metric that counts: unique KCOV PCs reached by *valid* programs.**

> **Pass-rate is not a result.** A model that emits one trivial 2-instruction valid program scores
> ~100% pass-rate and finds nothing. 60 %, 73 %, 100 % — all meaningless on their own. Pass-rate is at
> most a *diagnostic*; it is never a headline and never a "win." Describe a model by **what it can
> actually produce**, not by a percentage. If a percentage appears anywhere, it must first answer
> *"why is this relevant to valid-and-diverse?"*

```
Data collection → SFT → RL (GRPO) → coverage evaluation (unique PCs from valid programs)
```

**The finding so far: diversity, not validity, is the wall.** Valid programs *saturate* in coverage —
~16× more valid programs buy only ~+28 % unique PCs (replicate range +9–45 %; the metric is noisy, the
sub-linearity is robust). Raising validity does not help; the programs cluster onto the same verifier paths.

| Axis | Value |
|---|---|
| Base model | Qwen2.5-Coder-1.5B |
| Fine-tuning | QLoRA (SFT) + GRPO (RL) |
| Primary metric | **unique KCOV PCs from valid programs** |
| Secondary (diagnostic only) | per-program coverage depth; pass-rate |
| Kernel | Linux 6.8 + KCOV + KASAN, Debian trixie QEMU VM |
| Hardware | RTX 4070 Laptop 8 GB (local) / A100 40 GB (Colab) |

---

## 2. Models — *a model is its prompt + its training steps + what it can do*

A model is characterised by **(a) the prompt format it was trained on** and **(b) how many training
steps it saw** — nothing else (where it trained is irrelevant). Each is then described by **what it
can produce**, not by a score.

| Model | Prompt format | Steps | **What it can do** | Repo dir |
|---|---|---|---|---|
| **SFT-1** | `Status: VALID` + verifier-log assembly | 9,288 (3 ep) | Only **trivial 1–2 instruction programs** — the verifier-log format is ~232 tok/instruction, so the completion budget fit almost nothing. Valid but too short to explore anything. | `checkpoints/curated_3ep/` (merged: `curated_merged/`) |
| **SFT-2 (partial probe)** | `[coverage][novelty]` + stripped assembly | ~1,500 (0.62 ep) | A local "does it generate bytecode at all" probe. The n=20 "19 %" benchmark. Same work as SFT-2, fewer steps. Not a result. | `checkpoints/sft_retrain/checkpoint-1500/` |
| **SFT-2** | `[coverage][novelty]` + stripped assembly | ~2,408 (1 ep) | Generates **real, deep** programs (30–58 instructions). **Weakness: low diversity** — valid programs cluster, coverage saturates. This is the diversity-experiment and RL-2 base. | `checkpoints/sft-1epoch-v2/sft_adapter/` |
| **RL-1** | GRPO over SFT, `Status: VALID` | 8,370 | Did not learn — **reward had zero within-group variance** (`reward_std=0` from step ~1,300) → GRPO gradient starvation. **Value = the insight**, not a number. | `checkpoints/rl_grpo_v2/` (dir says v2, is run 1) |
| **RL-2** | GRPO over SFT-2, validity-gated novelty reward | ~200 (phase B) | Cleared the RL-1 `std=0` trap (reward/std 0.31) but **did not break the saturation** — RL's valid programs sit *on* the SFT-2 diversity curve. Under-trained + benchmarked out-of-regime. | local `rl_grpo_v3/` empty; `checkpoint-200` on Colab/Drive |

**Naming note:** the two SFT-2 rows are the *same* line of work at two step counts (partial probe vs
full epoch). RL directory suffixes are off by one (`rl_grpo_v2` = RL run 1). Do not rename dirs —
paths are wired into tools/WandB/benchmarks. Unqualified "SFT-2" = the full one-epoch model.

---

## 3. Prompt / IO format

```
SFT-1 prompt:           Kernel: unknown | Status: VALID
                        ### ASSEMBLY:
SFT-2 prompt:           [coverage={low|med|high}][novelty={low|med|high}]
                        ### ASSEMBLY:
completion (target):    0: (b7) r0 = 0
                        1: (95) exit            ← bare verifier-log assembly
```

Programs are learned as **BPF assembly text**, not raw bytecode. `reward._encode_to_hex()` converts
assembly → bytecode (pure Python, bypasses clang — see DECISIONS). SFT-2 control tokens
`[coverage][novelty]` are set from tertile-binned novelty scores (`ml/enrich_dataset.py`).

---

## 4. Reward (RL-2, current)

Monotonic ladder so a valid program always beats any invalid one. Verdict-**gated** (validity matters).
Weights are env vars read at reward-server launch.

```
encode fail / < 15 insns      → 0.0
VM ERROR / whole-batch crash  → 0.0
REJECTED (invalid)            → W_REJECT_MAX · min(1, len(pcs)/max_pcs_seen)     # soft floor
ACCEPTED (valid)              → W_VALID + W_NOVELTY·group_novelty + W_GLOBAL·global_novelty
```

| env | default | meaning |
|---|---:|---|
| `RL_W_VALID` | 1.0 | bonus for crossing into ACCEPTED |
| `RL_W_NOVELTY` | 1.0 | per-group novelty (phase A) |
| `RL_W_GLOBAL` | 0.0 | decayed-global novelty (phase B; set 2.0 to enable) |
| `RL_W_REJECT_MAX` | 0.3 | ceiling on the invalid soft floor (< `W_VALID` by design) |
| `RL_GLOBAL_DECAY` | 1.0 | per-batch ageing of the global PC-frequency map |

- **Soft floor** = partial credit for how far the verifier walked before rejecting (`len(pcs)`).
  Guarantees within-group variance at ~7 % valid → fixes the RL-1 `std=0` starvation.
- **group_novelty(p)** = mean over p's PCs of `(1 − freq/n_accepted)`, freq within the batch.
- **global_novelty(p)** = mean over p's PCs of `1/(1 + global_freq[pc])`, persistent across the run.

Training config: **G=16**, `max_completion_length=512`, beta=0.05, lr=5e-6, temperature=0.9,
LoRA r=16/α=32 on q/k/v/o. (Why these: DECISIONS.)

---

## 5. Numbers that matter

**The metric (unique KCOV PCs from valid programs) and its saturation — the actual result:**

> **The valid-unique-PC metric is nondeterministic.** Re-validating the *same* saved programs gives
> different counts run-to-run: KCOV records every kernel PC touched during the trace (incl. background
> interrupt/scheduler PCs that accumulate with program count), and a few borderline verdicts flip
> (timing). Measured spread over replicates: **±~1 % at 75 valid, ±~12 % at 1,300 valid.** So a single
> pass is a noisy point estimate — report mean (range), and headline the *sub-linear trend*, never a
> precise %. Replicates: `benchmarks/diversity/saturation_replicates.json`.

| | SFT-2 @512 (75 v) | SFT-2 @1024 (1,312 v) | RL-2 cp200 (433 v) |
|---|---:|---:|---:|
| valid programs | 75 | ~1,312 | 433 |
| **valid-unique PCs** — mean (range) | 3,421 (3,365–3,455) | 4,347 (3,649–4,700) | 3,685 (3,606–3,833) |
| replicates | 4 | 3 | 3 |
| distinct PCs / valid program | ~2,337 | ~2,428 | — |
| novelty_score | 0.752 | 0.758 | 0.749 |

→ **Within one n=20k run, ~16× more valid programs (≈80 → 1,300) raise unique valid PCs by only
~+28 % (replicate range +9 % to +45 %)** — i.e. coverage grows *far sub-linearly* with program count;
the generator revisits the same verifier paths. The sub-linearity is robust across every replicate
(even the most generous: 16× programs → 1.45× PCs); only the exact % is noisy. *(The earlier single-sample
"17.5× → +12 %" is superseded — +12 % was a low draw of a ±12 %-noisy metric.)* The saturation is an
**SFT-2 property** (measured on SFT-2 generations). RL-2's cp200 (3,685, range 3,606–3,833) sits in the
same band — but whether KCOV-reward RL can *exceed* the SFT-2 ceiling is **open (RQ2)**, not closed (the
200-step run is ~40× shorter than RL-1; see JOURNAL 2026-06-08). Figure: `thesis/figures/saturation.*`.

**Program length is narrow and pinned by the token budget** (canonical bytecode count `len(hex)//16`,
from `benchmarks/diversity/candidates/`; figure `thesis/figures/depth_collapse.*`, script
`tools/plot_depth_collapse.py`):

| budget | n | mean | sd | median | IQR | range |
|---|---:|---:|---:|---:|---:|---:|
| 512 tok | 1,000 | 30.2 | 3.5 | 30 | 28–33 | 17–41 |
| 1024 tok | 20,000 | 57.6 | 8.6 | 58 | 52–64 | 24–99 |

The median just scales with the budget (30→58); the model does not vary program length. *(The earlier
"929/1000 in one band" was an assembly-line-count artifact — canonical counting gives 58 % in the modal
bin, 99 % within 20–39. Do not use the 929 figure.)*

> RL-2 `cumulative_pcs`=4,836 (WandB `bz5ymfzl`, step 207) is **total** PCs incl. invalid programs'
> verifier-walk PCs (soft floor) — **not** the valid-unique metric (3,606). Don't conflate the two.

**Diagnostic-only (do not headline):** SFT-1 accept rate is ~73 % via pure-encoder+KCOV (the legacy
"60 %" was a clang-parser artefact). This number is *meaningless as a success measure* — SFT-1's valid
programs are trivial 1–2 instruction programs. It is recorded only to explain the legacy figure and to
show validity ≠ coverage (73 valid programs → only ~2,700 unique PCs).

> `avg_pcs` in benchmark JSON (the "~254k") is **raw KCOV trace length** (every PC hit, loops
> included), **not** coverage. Always use the *distinct* / *valid-unique* figures above.

---

## 6. Dataset

- Raw: ~2M `(bytecode_hex, verifier_log, is_valid, error_line, error_reason)` from modified buzzer.
- Curated: **27,514** examples, capped 2,000/error-class, 48 % valid / 52 % invalid (13,153 / 14,361).
  → `data/dataset_final_qwen.jsonl` (HF `Strhata/ebpf-corpus`).
- Enriched (SFT-2): `+ novelty_score + coverage_bin + novelty_bin`, stratified 70/15/15 (seed 42).
  → `data/dataset_final_qwen_enriched.jsonl` (an earlier variant `corpus_ml_enriched.jsonl` also exists).

---

## 7. File map

| File | What it does |
|---|---|
| `ml/train.py` | SFT training (QLoRA, `EncoderPassRateCallback`) |
| `ml/rl_grpo.py` | GRPO RL training, remote-reward bridge, `--resume` |
| `ml/reward.py` | `_encode_to_hex()`, `compute_rewards()`, the RL-2 validity-gated ladder |
| `ml/enrich_dataset.py` | novelty score + tertile binning (SFT-2) |
| `tools/kcov_validator/main.go` | Go binary: `BPF_PROG_LOAD` + KCOV → `{verdict, pcs:[]}` |
| `tools/reward_server.py` | FastAPI reward server (HTTP, API-key, env `RL_W_*`) |
| `tools/benchmark.py` / `benchmark_lib.py` | prepare / run / report model comparison |
| `tools/diversity_sample.py` | generate → validate → measure the saturation experiment |
| `tools/coverage_race.py` / `plot_coverage_race.py` | LLM side of buzzer-vs-LLM race + plot |
| `tools/evaluate_passrate.py` | legacy clang + `ebpf_validator` pass-rate eval (diagnostic) |
| `tools/generate_bytecodes.py` / `validate_bytecodes.py` | seeded reconstruction (dual-encode → KCOV) |
| `fuzzing/buzzer/pkg/units/ffi.go` | buzzer data-collection patch (JSONL at FFI boundary) |
| `fuzzing/run_eval_vm.sh` | start eval VM (SSH :10022) |
| `data/dataset_final_qwen.jsonl` | 27,514 curated SFT samples |
| `benchmarks/diversity/*.json` | saturation + RL-2 phase-B results |
| `data/reconstruction/sft_v1_*.kcov.jsonl` | seeded SFT-1 reconstruction artifacts |

---

## 8. Artifacts (HuggingFace)

| Artifact | Location |
|---|---|
| SFT dataset (27k) | `Strhata/ebpf-corpus` |
| SFT-1 adapter / merged | `Strhata/ebpf-checkpoints/curated_3ep_final`, `/curated_merged` |
| SFT-2, RL checkpoints | local / Colab-Drive only — not published |

---

## 9. Infrastructure *(deliberately terse — not the point of the work)*

Generation (Colab A100) ↔ KCOV reward (local WSL VM) over an HTTP bridge:
`GRPOTrainer → ngrok → FastAPI reward_server → SSH :10022 → KCOV VM`. KCOV mode `KCOV_TRACE_PC`,
flat uint64 PC array. RL needs a **merged fp16** model (never 4-bit a merge). Run details: `ops/`.
