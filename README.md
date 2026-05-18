# eBPF Fuzzing + LLM Fine-Tuning — Thesis

Research project exploring LLM-guided eBPF program generation for kernel verifier coverage.
Università del Salento, 2025–2026.

## What this is

End-to-end pipeline for training a language model to generate eBPF programs that maximise
coverage of the Linux BPF verifier, measured via kernel KCOV instrumentation.

```
Data collection → SFT → RL (GRPO) → Coverage evaluation
```

### Current state (2026-05-18)

| Phase | Status | Key result |
|---|---|---|
| Data collection | ✅ Done | ~2M programs via modified buzzer, 27k curated |
| SFT (Qwen2.5-Coder-1.5B) | ✅ Done | **60% verifier pass-rate** (vs 1% zero-shot) |
| GRPO RL training | 🟡 Running | Step ~740/30000, 9 new PCs found so far |

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

| Checkpoint | Steps | Eval loss | Pass-rate |
|---|---|---|---|
| `checkpoints/curated_3ep/` | 9,288 (3 ep) | 0.5571 | — |
| `checkpoints/curated_merged/` | merged bf16 | — | **60%** |

See `docs/training_log.md` for full phase history.

### 3 — RL fine-tuning (GRPO)

GRPO (Group Relative Policy Optimization) with a KCOV-based reward signal.
The model generates BPF programs; each is validated on a live kernel VM instrumented
with KCOV. Reward is based on how many new kernel program-counter addresses are reached.

**Key design: pure Python BPF encoder** — the model output (verifier-log assembly format)
is encoded directly to raw BPF bytecode via `ml/reward.py:_encode_to_hex`, bypassing clang.
This lets unusual instruction sequences reach the verifier instead of being rejected by
the assembler.

| Reward tier | Signal | Value |
|---|---|---|
| New PCs discovered | ACCETTATO + unseen kernel PCs | 1.0 |
| Valid, no new PCs | ACCETTATO, PCs already known | 0.1 |
| Verifier rejected | RIFIUTATO | 0.1 |
| Kernel crash / timeout | SSH timeout (VM may have crashed) | 2.0 |
| Unparseable output | ENCODE_FAIL | 0.0 |

Current run: `checkpoints/rl_grpo_v2/` — G=4, max_completion=600 tokens, beta=0.01.

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
  reward.py             # Reward function: BPF encoder + KCOV-based reward tiers
tools/
  kcov_validator/       # Go binary: loads BPF program, returns KCOV PC set as JSON
  evaluate_passrate.py  # SFT pass-rate evaluation (clang + ebpf_validator)
  run_comparison.sh     # Token-length comparison orchestration script
  vm_watchdog.sh        # Restarts VM if SSH times out during training
fuzzing/                # Buzzer fork (modified ffi.go), VM scripts, Docker setup
data/
  dataset_final_qwen.jsonl   # 27k curated SFT samples (bytecode_hex + verifier_log)
docs/
  training_log.md       # Full phase history and checkpoints
tests/
  test_reward_encoder.py    # 36 unit tests for the BPF encoder
  test_sanity_callback.py   # 11 unit tests for the training health monitor
checkpoints/            # Model adapters — large files on HuggingFace (see below)
results/                # Pass-rate CSVs, reward logs, sanity check reports
```

---

## Models & Dataset

| Artifact | Location |
|---|---|
| SFT dataset (`dataset_final_qwen.jsonl`) | HuggingFace — link TBD |
| SFT adapter (`curated_3ep`) | HuggingFace — link TBD |
| Merged SFT model (`curated_merged`) | HuggingFace — link TBD |
| RL checkpoint (`rl_grpo_v2`) | Local only (in progress) |

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
  --output-dir checkpoints/rl_grpo_v2

# Resume from checkpoint
pixi run python ml/rl_grpo.py \
  --resume-from checkpoints/rl_grpo_v2/checkpoint-1000 \
  --output-dir checkpoints/rl_grpo_v2
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

**Periodic health checks:** `SanityCheckCallback` fires every hour during training,
logging VM status, verdict breakdown, and cumulative PC count to `results/sanity_checks.log`.
