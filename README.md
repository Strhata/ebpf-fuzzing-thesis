# eBPF Fuzzing + LLM Fine-Tuning — Thesis

Research project exploring LLM-guided eBPF program generation for kernel verifier coverage.
Università del Salento, 2025–2026.

## What this is

End-to-end pipeline for training a language model to generate eBPF programs that maximise
coverage of the Linux BPF verifier, measured via kernel KCOV instrumentation.

```
Data collection → SFT → RL (GRPO) → Coverage evaluation
```

### Current state (2026-05-21)

| Phase | Status | Key result |
|---|---|---|
| Data collection | ✅ Done | ~2M programs via modified buzzer, 27k curated |
| SFT (Qwen2.5-Coder-1.5B) | ✅ Done | **60% verifier pass-rate** (vs 1% zero-shot) |
| GRPO RL run 1 (beta=0.01) | ✅ Done | **138 new verifier PCs** in 8370 steps; plateau at step ~1300 — root cause: KCOV bug (see below) |
| Reward redesign + Colab pipeline | ✅ Done | Depth-based verdict-blind reward; FastAPI server; Colab notebook with auto-resume |
| GRPO RL run 2 (depth reward) | 🔬 Future work | Pipeline built, not executed — experimental phase closed |

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
with KCOV (`KCOV_TRACE_PC`, flat uint64 PC array).

**Key design: pure Python BPF encoder** — the model output (verifier-log assembly format)
is encoded directly to raw BPF bytecode via `ml/reward.py:_encode_to_hex`, bypassing clang.
This lets unusual instruction sequences reach the verifier instead of being rejected by
the assembler.

#### Initial reward design (run 1, `rl_grpo_v2`)

| Tier | Condition | Value |
|---|---|---|
| New PCs | ACCETTATO + unseen kernel PCs | 1.0 |
| Valid, no new PCs | ACCETTATO, PCs already known | 0.1 |
| Verifier rejected | RIFIUTATO | 0.1 |
| Crash / timeout | SSH timeout | 2.0 |
| Unparseable | ENCODE_FAIL | 0.0 |

Run 1 plateaued at ~138 cumulative PCs (step ~1300, `reward_std=0`). Root cause discovered
post-run: `kcov_validator` was discarding the KCOV trace for RIFIUTATO programs (returning
`PCs: []`), making all rejected programs look identical. With GRPO this causes
`reward_std=0` within each group → zero gradient → no learning.

#### Redesigned reward (depth-based, verdict-blind)

After fixing `kcov_validator` to return PCs for RIFIUTATO programs:

```
depth_component = min(0.5, len(pcs) / max_pcs_seen * 0.5)
discovery_bonus = 1.0  if any PC not in pre-batch snapshot
reward          = depth_component + discovery_bonus

Special cases:
  encode_fail → 0.0   (no parseable instructions)
  crash       → 2.0   (SSH timeout; VM may have crashed)
```

Verdict (ACCETTATO / RIFIUTATO) does not affect the reward — only coverage depth does.
This ensures reward variance within each GRPO group even when most programs are rejected.

A pre-batch snapshot (`pc_set_snapshot = frozenset(_pc_set)`) is taken before the loop
so all G completions compete against the same frontier regardless of evaluation order.

#### Run history

| Run | Config | Steps | Result |
|---|---|---|---|
| `rl_grpo_v2` | beta=0.01, G=4, T=600, local GPU | 8370 | 138 PCs; plateaued at ~1300 due to KCOV bug |
| Run 2 (planned) | beta=0.01, G=8, T=1200, Colab Pro | — | Future work; pipeline ready |

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
  training_log.md       # Full phase history and checkpoints
  colab_restart_guide.md  # Colab Pro restart procedure
  cloudflare_tunnel_setup.md  # Local reward server tunnel setup
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

**KCOV bug (run 1):** `kcov_validator` returned `PCs: []` for RIFIUTATO programs.
Fixed in commit c8a9d02 — `readPCs()` helper now called for both ACCETTATO and RIFIUTATO.

**Periodic health checks:** `SanityCheckCallback` fires every hour during local training,
logging VM status, verdict breakdown, and cumulative PC count to `results/sanity_checks.log`.
Disabled automatically in remote-reward mode (`--remote-reward-url`).
