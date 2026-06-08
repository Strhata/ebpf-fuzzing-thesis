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

```
ml/
  train.py              # SFT training (QLoRA)
  rl_grpo.py            # GRPO RL training + SanityCheckCallback
  reward.py             # Reward function: BPF encoder + depth-based KCOV reward
  train_grpo_colab.ipynb  # Colab launcher notebook (auto-resume, remote reward)
  requirements_colab.txt  # Colab-side pip deps (no torch — pre-installed on Colab)
tools/
  kcov_validator/       # Go binary: loads BPF program, returns KCOV PC set as JSON
  reward_server.py      # FastAPI server: exposes reward.py over HTTP with API key auth
  evaluate_passrate.py  # SFT pass-rate evaluation (clang + ebpf_validator)
  analyze_rl_run.py     # Parse grpo_completions.log → tier CSVs + plots
  vm_watchdog.sh        # Restarts VM if SSH times out during training
fuzzing/                # Buzzer fork (modified ffi.go), VM scripts, Docker setup
data/
  dataset_final_qwen.jsonl   # 27k curated SFT samples (bytecode_hex + verifier_log)
docs/
  FACTS.md JOURNAL.md DECISIONS.md   # reference / history / rationale
  ops/                  # how to run (ngrok, colab, pipelines)
  colab_restart_guide.md  # Colab Pro restart procedure
  ngrok_tunnel_setup.md   # Local reward server tunnel (ngrok static domain)
tests/
  test_reward.py            # Depth-based reward + PC set persistence tests
  test_reward_encoder.py    # 36 unit tests for the BPF encoder
  test_reward_server.py     # FastAPI reward server tests
  test_sanity_callback.py   # 11 unit tests for the training health monitor
checkpoints/            # Model adapters — large files on HuggingFace (see below)
results/                # Pass-rate CSVs, reward logs, PC set, analysis plots
```

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
