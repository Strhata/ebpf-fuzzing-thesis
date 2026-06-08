# Canonical naming map

Thesis-facing material (chapters, slides, `PROJECT_HISTORY.md`) uses the canonical names
**SFT-v1 / SFT-v2 / RL-v1 / RL-v2**. Repository directories grew organically and do **not** match
one-to-one — in particular the RL directory suffixes are **off by one** (`rl_grpo_v2` is RL run 1).
This table is the single mapping. Do not rename directories (paths are wired into tools, WandB
resume files, and `benchmarks/`); use this map instead.

| Canonical | What it is | Repo location | Notes |
|---|---|---|---|
| **SFT-v1** | First usable SFT model | `checkpoints/curated_3ep/` | adapter `adattatore_ebpf_v1`; 9,288 steps (3 ep), eval_loss 0.5571 |
| SFT-v1 (merged) | bf16 merge of SFT-v1 | `checkpoints/curated_merged/` | the "60%-era" model in the reconstruction (truly **73%** valid) |
| SFT-v1 (precursor) | truncated warm-start | `checkpoints/sft_fase1/` | `max_steps=1500` bug, 48% of data; not a result |
| SFT-v1 (AWQ) | quantized SFT-v1 | `checkpoints/curated_awq/` | experiment; not used downstream |
| **SFT-v2 (partial)** | Probe retrain — did it generate at all | `checkpoints/sft_retrain/checkpoint-1500` | **1,500 steps / 0.62 epoch**; the n=20 "19%" benchmark. A short probe, **not** the diversity model |
| **SFT-v2** | Full one-epoch retrain — the canonical SFT-v2 | `checkpoints/sft-1epoch-v2/sft_adapter` | **full 1 epoch (~2,408 steps)**; novelty-aware format; the **diversity-saturation + RL-v2 base** (~7% valid, 3,8xx valid-unique PCs) |
| **RL-v1** | First (and only fully-run) GRPO run | `checkpoints/rl_grpo_v2/` | **dir says v2, it is run 1**; 8,370 steps, reward_std=0 plateau |
| RL-v1 (aborted) | early/comparison GRPO attempts | `checkpoints/rl_grpo`, `rl_grpo_cmp600/800/1200` | short max-step probes; not results |
| **RL-v2** | Validity-gated novelty reward (the bet) | `checkpoints/rl_grpo_v3/` (empty) | **ran** phase-A smoke + phase-B 200 steps → no diversity breakthrough; `checkpoint-200` lives on Colab/Drive, local dir empty |

The two SFT-v2 entries are the **same line of work at two step counts** — a partial probe (~1,500
steps) and the full one-epoch run (~2,408 steps). The distinction is *training steps*, nothing else;
where each trained is irrelevant. When a doc or chapter says "SFT-v2" unqualified it means the full
one-epoch model (`sft-1epoch-v2`).

## Era → pipeline (so numbers aren't compared across mismatched setups)

| Era | Model | Encode path | Validator | Headline number |
|---|---|---|---|---|
| 60%-era | SFT-v1 merged | verifier-log → clang → objcopy | `ebpf_validator` | 60% pass-rate (parser-deflated; really **73%** — see `data/reconstruction/REPORT.md`) |
| diversity / RL era | SFT-v2 (full epoch) | pure-Python `_encode_to_hex` | `kcov_validator` | ~7% valid; **valid-unique PCs saturate** (3,462 → 3,862 for 17.5× more programs); ~2,400 *distinct* PCs/valid program |

> The "~254k PCs" figure seen in older notes is `avg_pcs` from the benchmark JSON — the **raw KCOV
> trace length** (every PC hit, loops included), *not* coverage. The coverage figure is ~2,400
> *distinct* PCs/valid program and 3,8xx valid-unique PCs over a run. Use the distinct/unique figures.

WandB run names are set at launch (`--run-name`), decoupled from directory names, and are frozen
history — they are intentionally *not* reconciled here. For the next run, pass `--run-name rl-v2`.
