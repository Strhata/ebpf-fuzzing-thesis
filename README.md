# eBPF Fuzzing + LLM Fine-Tuning — Thesis

Research project exploring LLM-guided eBPF program generation for kernel verifier coverage.
Università del Salento, 2025–2026.

## What this is

**The quest:** verifier bugs are found by *valid* programs that exercise *many distinct* verifier
paths. A rejected program means the verifier did its job — it is not a finding. So the goal is a
generator that is simultaneously **valid** and **path-diverse**, and the true metric is
**unique KCOV PCs reached by valid programs** — not pass-rate alone, and not coverage from
rejected programs.

```
Data collection → SFT → RL (GRPO) → Coverage evaluation (unique PCs from valid programs)
```

> **Docs are split by purpose** — this README is a summary; where they differ, these win:
> [`docs/FACTS.md`](docs/FACTS.md) (what is true now: models, metrics, naming) ·
> [`docs/JOURNAL.md`](docs/JOURNAL.md) (what happened, in order) ·
> [`docs/DECISIONS.md`](docs/DECISIONS.md) (why) · [`docs/ops/`](docs/ops/) (how to run).

### New here? Read in this order
- **Understand the work** → this README, then [`docs/FACTS.md`](docs/FACTS.md) §1–2, then [`docs/JOURNAL.md`](docs/JOURNAL.md) top-to-bottom for the story.
- **Run it** → [`docs/ops/running.md`](docs/ops/running.md).
- **Examine the thesis** → [`thesis/main.pdf`](thesis/).

**What is our contribution vs borrowed:** the actual work is `ml/`, the core of `tools/`
(`kcov_validator`, `reward_server.py`, `reward.py`, `benchmark.py`, `diversity_sample.py`), the
one-file buzzer patch `fuzzing/buzzer/pkg/units/ffi.go`, and `thesis/`. **Everything else under
`fuzzing/buzzer/` (≈96 files) is Google's [buzzer](https://github.com/google/buzzer), vendored
unchanged** — used only as a data generator, not part of the research contribution.

### Current state (2026-06-08)

| Phase | Status | What we learned |
|---|---|---|
| Data collection | ✅ | ~2M programs via modified buzzer → 27k curated |
| SFT-v1 (`curated_3ep`) | ✅ | Generates valid programs, but verifier-log format is ~232 tok/insn → only ~2 instructions fit the budget → too trivial to explore |
| RL-v1 (`rl_grpo_v2`) | ✅ negative result | 8,370 steps; reward had zero within-group variance → GRPO gradient starvation (`reward_std=0`, no learning). **Taught us: fix format + reward.** Not a failure — information. |
| Reward + format redesign | ✅ | Stripped format; depth-based verdict-blind reward; remote (Colab) reward server |
| SFT-v2 (`sft-1epoch-v2`, full 1-epoch) | ✅ | Generates real, **deep** programs. Weakness: **low diversity** — clusters; valid-coverage **saturates** (17.5× more valid programs → +12% unique PCs). (An earlier partial probe `sft_retrain`/cp1500 is the n=20 "19%" benchmark — same work, fewer steps.) |
| RL-v2 (validity-gated novelty reward) | 🔬 ran, no breakthrough | Phase-A smoke + phase-B 200 steps: cleared the RL-v1 `std=0` trap but RL's valid programs sit *on* the SFT saturation curve. Under-trained + benchmarked out-of-regime. See [`docs/JOURNAL.md`](docs/JOURNAL.md) (2026-06-08). |

**On the old "60% pass-rate":** it was SFT-v1 measured through a clang gate whose parser bug
deflated it; reconstructed honestly (pure encoder + KCOV) SFT-v1 is **73% valid**. But validity
isn't the story — 73 valid programs reach only ~2,700 unique PCs. **Diversity, not validity, is
the wall.** Full forensics: [`docs/JOURNAL.md`](docs/JOURNAL.md) (2026-06-04) + the seeded artifacts in `data/reconstruction/`.

---

## Pipeline

### 1 — Data collection

Modified [buzzer](https://github.com/google/buzzer) (Google's eBPF fuzzer) to dump generated
programs and their verifier outcomes as JSONL at the kernel FFI boundary. Collected ~2M entries
via a Dockerized single-VM setup, curated to 27k balanced samples.

Output: `data/dataset_final_qwen.jsonl`

### 2 — Supervised fine-tuning (SFT)

QLoRA fine-tuned `Qwen2.5-Coder-1.5B` on verifier-log assembly format for 3 epochs.
The model learns to generate syntactically and semantically valid BPF programs given
a target status header.

| Checkpoint | Steps | Eval loss | Accept-rate (honest) |
|---|---|---|---|
| `checkpoints/curated_3ep/` (SFT-v1) | 9,288 (3 ep) | 0.5571 | — |
| `checkpoints/curated_merged/` (merged bf16) | — | — | **73%** valid via encoder+KCOV (the "60%" was clang-gate deflated — see [`docs/FACTS.md`](docs/FACTS.md) §5) |

See [`docs/JOURNAL.md`](docs/JOURNAL.md) for the full, verified phase history.

### 3 — RL fine-tuning (GRPO)

GRPO (Group Relative Policy Optimization) with a KCOV-based reward signal.
The model generates BPF programs; each is validated on a live kernel VM instrumented
with KCOV (`KCOV_TRACE_PC`, flat uint64 PC array).

**Key design: pure Python BPF encoder** — the model output (verifier-log assembly format)
is encoded directly to raw BPF bytecode via `ml/reward.py:_encode_to_hex`, bypassing clang,
so unusual instruction sequences reach the verifier instead of being rejected by the assembler
(why: [`docs/DECISIONS.md`](docs/DECISIONS.md) D4).

The current RL-2 reward is a **validity-gated novelty ladder** (valid always beats invalid; invalid
gets a soft floor for verifier-walk depth; valid adds per-group + global novelty) — the fix for the
RL-1 `reward_std=0` gradient starvation. Full ladder + weights in [`docs/FACTS.md`](docs/FACTS.md) §4;
the rationale and the superseded designs in [`docs/DECISIONS.md`](docs/DECISIONS.md) D5 and
[`docs/JOURNAL.md`](docs/JOURNAL.md).

### 4 — Evaluation

A Go binary (`tools/kcov_validator`) loads a BPF program into a KCOV-instrumented kernel and
returns the set of kernel PCs hit during verification. The VM runs Debian trixie with a
custom kernel 6.8 + KCOV + KASAN.

---

## Repository structure

The full file-map is in [`docs/FACTS.md`](docs/FACTS.md) §7. The shape:

```
ml/        ★ training        train.py (SFT) · rl_grpo.py (GRPO) · reward.py (encoder + validity-gated
                             reward) · enrich_dataset.py · *_colab.ipynb launchers
tools/     ★ pipeline        kcov_validator/ (Go: BPF_PROG_LOAD+KCOV→JSON) · reward_server.py (FastAPI)
                             · benchmark.py · diversity_sample.py · coverage_race.py + plot
                             (core vs one-off: tools/README.md)
fuzzing/   ⚙ vendored        Google's buzzer (≈96 files, NOT our work) + our patch:
                             buzzer/pkg/units/ffi.go (data-collection dump). Plus VM/Docker scripts.
thesis/    ★ deliverable     LaTeX chapters (ch1–ch7), main.pdf
docs/      ★ start here      FACTS.md (now) · JOURNAL.md (history) · DECISIONS.md (why) · ops/ (how-to)
tests/       216 tests       pixi run pytest   (test_reward 32 · _encoder 50 · _server 6 · …)
data/        dataset_final_qwen.jsonl (27k) + enriched + reconstruction artifacts
benchmarks/  diversity/*.json (the saturation-curve results) + runs/ reports
checkpoints/ adapters (off-git; on HuggingFace) · results/ (CSVs, plots; off-git)
```

★ = our contribution · ⚙ = vendored dependency.

---

## Models & Dataset

| Artifact | Location |
|---|---|
| SFT dataset (`dataset_final_qwen.jsonl`) | [Strhata/ebpf-corpus](https://huggingface.co/datasets/Strhata/ebpf-corpus) |
| SFT adapter (`curated_3ep_final`) | [Strhata/ebpf-checkpoints](https://huggingface.co/Strhata/ebpf-checkpoints/tree/main/curated_3ep_final) |
| Merged SFT model (`curated_merged`) | [Strhata/ebpf-checkpoints](https://huggingface.co/Strhata/ebpf-checkpoints/tree/main/curated_merged) |
| RL checkpoint (`rl_grpo_v2`) | Local only — run stopped at plateau |

---

## Quickstart

```bash
# Install deps
pixi install

# Run tests (no VM required)
pixi run pytest tests/

# Start evaluation VM (KCOV + KASAN kernel)
./fuzzing/run_eval_vm.sh

# Run RL training (requires VM)
pixi run python ml/rl_grpo.py \
  --num-generations 4 \
  --max-completion-length 600 \
  --beta 0.01 \
  --output-dir checkpoints/rl_grpo_v3

# Auto-resume from last checkpoint (also handles fresh start)
pixi run python ml/rl_grpo.py \
  --resume \
  --output-dir checkpoints/rl_grpo_v3

# Remote reward server (for Colab training)
REWARD_API_KEY=<key> pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000
```

---

## Key technical notes

**VRAM budget (8GB GPU):** GRPO attention scales as G×T². G=4, T=600 tokens fits; G=4, T=800 OOMs at the backward pass.

**KV cache fix:** transformers 5.x + `gradient_checkpointing` + `model.train()` forces
`past_key_values=None` in Qwen2DecoderLayer, corrupting autoregressive generation.
Fixed via `_fixed_unwrap_model_for_generation` in `ml/rl_grpo.py`.

**BPF encoder:** model output is in kernel verifier dump format (`N: (XX) instruction`).
The opcode byte `(XX)` is extracted directly; dst/src/off/imm are parsed from the
instruction text. Programs with unusual operands reach the verifier instead of being
rejected by clang.

**KCOV bug (run 1):** `kcov_validator` returned `PCs: []` for REJECTED programs.
Fixed in commit c8a9d02 — `readPCs()` helper now called for both ACCEPTED and REJECTED.

**Periodic health checks:** `SanityCheckCallback` fires every hour during local training,
logging VM status, verdict breakdown, and cumulative PC count to `results/sanity_checks.log`.
Disabled automatically in remote-reward mode (`--remote-reward-url`).
