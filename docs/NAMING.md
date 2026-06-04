# Canonical naming map

Thesis-facing material (chapters, slides, `PROJECT_HISTORY.md`) uses the canonical names
**SFT-v1 / SFT-v2 / RL-v1 / RL-v2**. Repository directories grew organically and do **not** match
one-to-one — in particular the RL directory suffixes are **off by one** (`rl_grpo_v2` is RL run 1).
This table is the single mapping. Do not rename directories (paths are wired into tools, WandB
resume files, and `benchmarks/`); use this map instead.

| Canonical | What it is | Repo location | Notes |
|---|---|---|---|
| **SFT-v1** | First usable SFT model | `checkpoints/curated_3ep/` | adapter `adattatore_ebpf_v1`; 9,288 steps, eval_loss 0.5571 |
| SFT-v1 (merged) | bf16 merge of SFT-v1 | `checkpoints/curated_merged/` | the "60%-era" model in the reconstruction |
| SFT-v1 (precursor) | truncated warm-start | `checkpoints/sft_fase1/` | `max_steps=1500` bug, 48% of data; not a result |
| SFT-v1 (AWQ) | quantized SFT-v1 | `checkpoints/curated_awq/` | experiment; not used downstream |
| **SFT-v2** | Retrain: stripped format, novelty-aware | `checkpoints/sft_retrain/checkpoint-1500` | merged at `checkpoint-1500-merged`; the "19%" / deep-but-clustered model |
| **RL-v1** | First (and only executed) GRPO run | `checkpoints/rl_grpo_v2/` | **dir says v2, it is run 1**; 8,370 steps, reward_std=0 plateau |
| RL-v1 (aborted) | early/comparison GRPO attempts | `checkpoints/rl_grpo`, `rl_grpo_cmp600/800/1200` | short max-step probes; not results |
| **RL-v2** | The bet — depth+novelty reward | `checkpoints/rl_grpo_v3/` | **empty — not yet run**; future work |

## Era → pipeline (so numbers aren't compared across mismatched setups)

| Era | Model | Encode path | Validator | Headline number |
|---|---|---|---|---|
| 60%-era | SFT-v1 merged | verifier-log → clang → objcopy | `ebpf_validator` | 60% pass-rate (parser-deflated; really 73% — see `data/reconstruction/REPORT.md`) |
| RL / coverage era | SFT-v2 | pure-Python `_encode_to_hex` | `kcov_validator` | 19% valid; ~254k PCs/program; coverage saturates |

WandB run names are set at launch (`--run-name`), decoupled from directory names, and are frozen
history — they are intentionally *not* reconciled here. For the next run, pass `--run-name rl-v2`.
