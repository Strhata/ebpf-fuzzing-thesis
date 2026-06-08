# Project History — eBPF Fuzzing + LLM Thesis
**Single source of truth for phases 0–10.** Last updated: 2026-06-08.
**Verified against:** git log, training logs, benchmark run JSONs, `.scratch/fact-check/`, thesis chapters, memory.

> **Superseded on two fronts by [`docs/RL_V2.md`](RL_V2.md) (2026-06-08):** the **full SFT-v2**
> model (`sft-1epoch-v2`, the diversity-saturation base — *not* the partial `sft_retrain` probe) and
> **RL-v2**, which has since **run** (phase-A smoke + phase-B 200 steps → no diversity breakthrough)
> with a **validity-gated** reward (not the verdict-blind depth reward planned below). Where this
> file and RL_V2.md differ on SFT-v2 / RL-v2, RL_V2.md wins. Canonical model↔dir map: [`NAMING.md`](NAMING.md).

---

## 1. The quest (what this thesis is actually about)

**Verifier bugs are found by *valid* programs that exercise *many distinct* verifier paths.**
A program the verifier *rejects* proves the verifier did its job — it is not a finding. So the
objective is a generator that is *simultaneously* (a) **valid** and (b) **path-diverse**. The
true success metric is **unique kernel KCOV PCs reached by valid programs** — not pass-rate
alone (a model that emits one trivial valid program scores 100% and finds nothing), and not
coverage from rejected programs (the verifier bailing early is not exploration).

**Novel contribution:** combine LLM-guided program generation with KCOV as a GRPO
reinforcement-learning reward signal to push toward valid-and-diverse — no prior system does this.

### The honest arc

The experimental phase is a search for that generator. We may not find it — and mapping the
negative space (what does *not* work, and why) is itself the result.

- **Run 1 (SFT-v1 `curated_3ep` + RL-v1 `rl_grpo_v2`) — negative results, deliberately valuable.**
  Taught what to avoid:
  1. **Raw verifier-log format is too token-intensive** (~232 tok/instruction → the completion
     budget fit only ~2 instructions → programs too trivial to explore anything).
  2. **The reward gave zero within-group variance** → GRPO gradient starvation
     (`reward_std=0` after step ~1,300, no learning regardless of step count; cf. Nie et al.,
     arXiv:2605.07689). Compounded by a KCOV bug that returned empty PCs for rejected programs.
  → Conclusion: **format and reward must change.** Run 1 was information-gathering.

- **Run 2 model (SFT-v2) — fixes confirmed.** Stripped format → real, longer programs. It generates
  valid programs that go **deep** (high PCs per program). Its remaining weakness is **diversity**: it
  re-emits structurally similar programs, so cumulative coverage **saturates**. The canonical SFT-v2
  is the **full one-epoch run** (`sft-1epoch-v2`, ~2,408 steps); the diversity benchmark shows 17.5×
  more valid programs buy only +12% unique PCs (RL_V2.md §1). (An earlier **partial probe**,
  `sft_retrain` checkpoint-1500 / 0.62 epoch, is the n=20 "19%" benchmark below — *not* the diversity
  model; the two differ only in training steps.)

- **The bet (RL-v2, validity-gated novelty reward) — has run, no breakthrough yet.** A first phase-A
  smoke + phase-B 200-step run (RL_V2.md §5) cleared the RL-v1 `std=0` trap (reward/std 0.31) but did
  **not** break the saturation: RL's valid programs land *on* the SFT diversity curve. Under-trained
  (200 steps) and benchmarked out-of-regime — the null localizes the open problem rather than closing
  it. **Reward is validity-gated** (valid always beats invalid; invalid gets a soft floor), *not* the
  verdict-blind depth reward originally planned (see Phase 5 / §3.4 for the superseded design).

**Diversity, not validity, is the wall.** The reconstruction (§Phase 10) confirms it: SFT-v1 is
73% valid yet 73 valid programs reach only ~2,700 unique PCs — coverage barely moves with
accept-count. Validity is necessary but nowhere near sufficient.

```
Data collection → SFT → RL (GRPO) → Coverage evaluation (unique PCs from valid programs)
```

| Axis | Value |
|---|---|
| Base model | Qwen2.5-Coder-1.5B |
| Fine-tuning | QLoRA (SFT) + GRPO (RL) |
| Primary metric | **Unique KCOV PCs reached by valid programs** |
| Secondary metrics | Per-program coverage depth; verifier pass-rate (diagnostic only) |
| Infrastructure | QEMU VM, Debian trixie, Linux 6.8 + KCOV + KASAN |
| Hardware | RTX 4070 Laptop, 8 GB VRAM |
| University | Università del Salento |
| Target graduation | July 2026 (October fallback) |

---

## 2. Real Timeline

### Phase 0 — VM / Kernel infrastructure (1–10 March 2026)
*Location: `~/fuzzing_lab/` — pre-repo; verified from `docs/tesi_recap.md`.*

- `create-image.sh` adapted from syzkaller → Debian Bullseye + Trixie QEMU images (~2 GB each), SSH key auth.
- Linux **6.8.0** compiled in three variants:
  - `bzImage` — standard kernel
  - `bzImage_kasan` — KASAN only, no KCOV (throughput baseline)
  - `bzImage_kasan_kcov` — KCOV + KASAN + UBSAN (coverage-guided work)
- **4–5 March:** 30 kernel panic / KASAN dumps collected across 4 VMs (`crash_logs/`).
- **5–10 March:** Swarm runs via `start_swarm.sh` (3 VMs in parallel); profiling with strace/perf.
- Key discovery: buzzer's `coverage_based` mode uses KCOV only to **display** metrics via HTTP server — not as a mutation feedback signal. This gap (no LLM+KCOV system existed) motivated the ML pivot.

### Phase 1 — Data collection (26 March → 21 April 2026)
*Location: `~/fuzzing_lab/` + `~/fuzzing_ml_env/`.*

- `fuzzing/buzzer/pkg/units/ffi.go` modified to dump every `(bytecode_hex, verifier_log, is_valid, error_line, error_reason)` as JSONL at the kernel FFI boundary.
- Dockerized single-VM setup (`Dockerfile` + `entrypoint.sh`), corpus shared host↔guest via virtio-9p.
- **~2 million** raw programs collected.
- Corpus rotated with `rotate_dataset.sh` (8–21 April): `shared_corpus/dataset_syzkaller_347…405.jsonl.gz`, ~13 GB compressed, ~600k rows each.

### Phase 2 — Dataset curation + SFT original run (10 April → 14 May 2026)
*Location: `~/fuzzing_ml_env/` then migrated to repo `checkpoints/curated_3ep/`.*

**Dataset curation** (`data_analisys.ipynb`, 10 April):
- 73 syzkaller dumps analysed: 13,153 valid + 14,361 invalid entries.
- Error classes normalised (mask numbers/addresses).
- Capped at 2,000 examples/class → **27,514 examples** in `data/dataset_final_qwen.jsonl`.

**SFT Phase 1 — warm start** (`sft_fase1`, `SFT_tesi.ipynb`, 16 April):
- QLoRA 4-bit NF4, rank-16, Q/K/V/O, lr=2e-4, batch=8 (grad_accum=8), BF16.
- `max_steps=1500` (bug — only 48% of data). eval_loss **0.6210**.
- Saved to `checkpoints/sft_fase1/adattatore_ebpf_v1`.

**SFT Phase 2 — full 3-epoch run** (`curated_3ep`, 14 May 2026):
- Resumed from `sft_fase1/adattatore_ebpf_v1`.
- Dataset: `dataset_final_qwen.jsonl` — 24,762 train / 2,752 val (90/10, seed 42).
- Config: rank-16, Q/K/V/O only, lr=2e-4, max_length=768.
- **9,288 steps / 3.0 epochs.** eval_loss **0.5571**. Runtime ~26.8 h.
- WandB: `curated-3ep` — runs/7xybam1s.
- Final adapter: `checkpoints/curated_3ep/adattatore_ebpf_v1`.
- Merged model: `checkpoints/curated_merged/` (fp16).

### Phase 3 — Pass-rate evaluation (17 May 2026)
*First commit into this repo: `5879bc1 2026-05-17`.*

Pipeline: load merged model → generate 100 programs → BPF encoder → SSH to VM → `kcov_validator` → CSV.

| Model | N | Compiled | ACCEPTED | Pass-rate |
|---|---|---|---|---|
| `curated_merged` (SFT) | 100 | 73 | **60** | **60%** |
| zero-shot (Qwen2.5-Coder-1.5B base) | 100 | 1 | 1 | **1%** |

Source: `results/passrate_summary.csv`.

### Phase 4 — RL training run 1 (18–21 May 2026)

- **17 May:** `tools/kcov_validator/` (Go binary: BPF_PROG_LOAD + KCOV → JSON `{verdict, pcs:[]}`) committed.
- **18 May:** Pure-Python BPF encoder (`ml/reward.py:_encode_to_hex`) — bypasses clang; extracts opcode byte from verifier-log format and packs dst/src/off/imm directly.
- **Initial reward design (run 1):**

| Tier | Condition | Value |
|---|---|---|
| New PCs | ACCEPTED + unseen PCs | 1.0 |
| Valid, no new PCs | ACCEPTED | 0.1 |
| Rejected | REJECTED | 0.1 |
| Crash/timeout | SSH timeout | 2.0 |
| Encode fail | unparseable | 0.0 |

- **Run `rl_grpo_v2`:** beta=0.01, G=4, max_completion_length=600, local GPU (RTX 4070).
  - **8,370 gradient steps.**
  - Plateau at step **~1,300**: reward_std=0 → zero GRPO gradient.
  - **1,638 total unique kernel PCs** explored. **137 progressively-new PCs** (per `results/rl_analysis_pcs.csv`). *The README says 138 — off-by-one from boundary-counting; 137 is authoritative.*
  - Root cause: `kcov_validator` returned `PCs: []` for REJECTED programs → all rejected programs looked identical → reward_std=0. Fixed in commit **c8a9d02** (`readPCs()` now called for both ACCEPTED and REJECTED).

### Phase 5 — Reward redesign + Colab pipeline (21 May 2026)

**Depth-based verdict-blind reward** (`ml/reward.py`):
```python
depth_component = min(0.5, len(pcs) / max_pcs_seen * 0.5)
discovery_bonus = 1.0  # if any PC not in pre-batch snapshot
reward          = depth_component + discovery_bonus
# encode_fail → 0.0, crash → 2.0
```
Verdict (ACCEPTED/REJECTED) does not affect reward — only coverage depth and novelty matter.

**Colab Pro remote pipeline:**
- `tools/reward_server.py` — FastAPI, API key auth, exposes `compute_rewards()` over HTTP.
- `ml/rl_grpo.py` — `--remote-reward-url` flag, exponential-backoff retry, `--resume` auto-detect.
- `ml/train_grpo_colab.ipynb` — 5-cell Colab launcher, idempotent Run All.
- ngrok tunnel (static domain) for stable HTTPS from Colab to local reward server.
- **Status: pipeline fully built, never executed.** Declared future work.

### Phase 6 — Thesis writing (31 May 2026)

All 7 chapters written in one session (commits `2e5d7e1`–`f0d751b`, 2026-05-31):
- ch1: Introduction, ch2: Background, ch3: Related Work, ch4: Methodology,
  ch5: Implementation, ch6: Experimental Results, ch7: Conclusions.
- **Warning:** chapters describe the **original SFT config** (max_length=768, lr=2e-4, Q/K/V/O only, 90/10 split) — see §5 for divergences with sft_retrain.

### Phase 7 — Fact-check fixes (1 June 2026)

Commit `e0fc7bb 2026-06-01` applied 5 verified fixes:
1. `verifier.c` line count: **~21,000** (not ~100,000).
2. CVE-2023-2163 root cause: **state pruning**, not pointer arithmetic.
3. Falco attribution: **Sysdig/CNCF** (not Isovalent).
4. BpfChecker description: **differential fuzzer**, not static analysis.
5. Unprivileged BPF: "**some** default Linux configurations (most major distros restrict this)".

### Phase 8 — SFT retrain (1 June 2026, in progress)

New hyperparams prompted by OOM issues and novelty-awareness goals:

| Param | Original (curated_3ep) | Retrain (sft_retrain) |
|---|---|---|
| max_length | 768 | **2048** |
| Learning rate | 2e-4 | **5e-5 + cosine warmup** |
| LoRA targets | Q/K/V/O | **Q/K/V/O + MLP** |
| Gradient checkpointing | off | **on** |
| Dataset | plain 27k | **+ novelty_score + novelty_bin** |
| Train/val/test split | 90/10 | **stratified 70/15/15** |
| Dataset enrichment | none | `ml/enrich_dataset.py` (tertile bins) |

- Commits `8809a1a`, `54de1bb`, `b4748c6`, `101dbd3` (2026-06-01).
- Status at 2026-06-03: **checkpoint-1500** (saved Jun 1, 20:28). Not yet in thesis.
- `checkpoints/rl_grpo_v3/` exists but **empty** — no run started.

### Phase 9 — Benchmark pipeline + coverage race (3 June 2026)

**Benchmark pipeline** (commit `075819e`):
- `tools/benchmark_lib.py` — VMLifecycle, load_model, generate, analyze, validate, aggregate, write, report.
- `tools/benchmark.py` — CLI: `prepare` (merge LoRA), `run`, `report`.
- `tests/test_benchmark.py` — 32 offline tests.
- Run outputs: `benchmarks/runs/<timestamp>_run.json` + `_report.md`, `benchmarks/summary.csv`.

Key benchmark results (n=20, sft_retrain_cp1500 vs curated_3ep_cp6000):

| Model | Strategy | Prompt | pass_rate | avg_pcs |
|---|---|---|---|---|
| sft_retrain_cp1500 | merged_fp16 | `[coverage=high][novelty=high]` | ~15–20% | ~254k* |
| sft_retrain_cp1500 | merged_fp16 | neutral | ~0–5% | ~180k* |
| sft_retrain_cp1500 | merged_bnb4bit | any | 0% | 0 |
| curated_3ep_cp6000 | merged_fp16 | `Status: VALID` | 0% (ERROR) | 0 |

> *`avg_pcs` (the "~254k") is the **raw KCOV trace length** — every PC hit, loops included — **not**
> coverage. The coverage figure (the spine metric) is ~2,400 *distinct* PCs/valid program and 3,8xx
> valid-unique PCs over a run (RL_V2.md §1). Do not cite avg_pcs as a coverage achievement.

- **`merged_fp16` clear winner.** `merged_bnb4bit` destroys quality post-merge.
- `[coverage=high][novelty=high]` prefix is load-bearing (model trained on it).
- `curated_3ep` programs hang VM — model outputs verifier-log header format (`func#0 @ 0x…`); root cause: model quality degraded at cp6000.

**Coverage race** (`tools/coverage_race.py`, `tools/plot_coverage_race.py`):
- Buzzer (`coverage_based`, 500 programs, ~7.5 min): 23 valid (4.6% pass rate), 4,915 unique PCs.
- Model (first 15 observed): first ACCEPTED at program 8 (vs buzzer's program 159); pcs_valid ~2,852.
- CSV output: `data/corpus/buzzer_coverage.csv`, `data/corpus/model_coverage.csv`.
- Plots: `results/coverage_race_per_program.png`, `results/coverage_race_over_time.png`.

### Phase 10 — "60%" forensic reconstruction (4 June 2026)

The original headline "60% pass-rate" was never reproducible: `evaluate_passrate.py` saved only
`id, compiled, verdict` — not the generated programs — and generation was unseeded. The programs
behind the 60% are gone. But the whole *machinery* survives (model `curated_merged`, the clang
parser, `ebpf_validator` binary, `kcov_validator`), so we reconstructed under controlled conditions.

New tooling (decoupled generate → store → validate, so programs are persisted and re-validatable):
- `tools/generate_bytecodes.py` — reload SFT-v1, generate **seeded** (seed 42), dual-encode each
  program: `clang_hex` (faithful 60%-era clang path) **and** `encoder_hex` (pure-Python encoder).
- `tools/validate_bytecodes.py` — run both encodings through `kcov_validator`, record verdict + PCs.
- Artifact: `data/reconstruction/sft_v1_20260604.{jsonl,kcov.jsonl}` (100 programs, reproducible).

**Result (same 100 SFT-v1 programs, seed 42, via KCOV):**

| encoding | reached kernel | ACCEPTED | % of generated | unique PCs |
|---|---|---|---|---|
| clang (60%-era path) | 71/100 | 51 | 51% | 2,613 |
| pure encoder (RL-era path) | 99/100 | 73 | **73%** | 2,698 |

**Three corrections this forces:**
1. **The 60% was deflated, not inflated.** The clang parser admits verifier register-state lines
   (`0: R1=ctx() R10=fp()`) as instructions → clang fails → 29 ENCODE_FAIL, of which **23 were
   actually valid**. SFT-v1's true accept rate is **73%** (pure encoder), not 60%.
2. **"60% vs 19%" is two *models*, not two pipelines.** SFT-v1 through the *exact* encoder+KCOV
   pipeline the 19% came from scores **73%**, not 19%. The 19% is SFT-v2, which trades validity
   for longer `[coverage][novelty]` programs. The gap is SFT-v1→SFT-v2, not clang→encoder.
3. **Validity ≠ coverage.** 73 valid programs → only ~2,700 unique PCs (vs clang's 51 valid →
   ~2,613). Accept-count barely moves coverage → **diversity is the wall**, not validity.

Caveat for methodology: clang and the pure encoder produce *different* bytecode for the same
generated text (≈1 instruction apart) and occasionally disagree on verdict — "same program" is
encoder-dependent.

---

## 3. Methodology

### 3.1 Data collection

Buzzer intercepts every BPF program at the kernel FFI boundary (`fuzzing/buzzer/pkg/units/ffi.go`).
Each entry: `{bytecode_hex, verifier_log, is_valid, error_line, error_reason}`.
~2M programs collected via Docker container with nested QEMU + virtio-9p corpus share.

**Curation:** 27,514 examples, capped 2,000/error-class, assembly (verifier-log) format only.
Hex encoding tried and abandoned (documented in ROADMAP Decisions Log).

### 3.2 Input/output format

```
### PROMPT (SFT input):
Kernel: unknown | Status: VALID
### ASSEMBLY:

### COMPLETION (SFT target):
0: (85) call 1            # or w/e instructions follow
1: (95) exit
```

For sft_retrain: prompt prefix is `[coverage=high][novelty=high]` or `[coverage=medium][novelty=medium]` etc., determined by tertile-binned novelty_score from `ml/enrich_dataset.py`.

### 3.3 SFT training

Script: `ml/train.py`.
- Base: `Qwen2.5-Coder-1.5B` (transformers, QLoRA via bitsandbytes + PEFT).
- LoRA rank 16, alpha 32.
- `strip_verifier_log()` strips header lines from old-format programs.
- `EncoderPassRateCallback`: fires every 200 steps, generates 5 programs, counts ACCEPTED.
- Outputs: adapter `.bin` + adapter config in checkpoint dir.

**Merge command:**
```bash
pixi run python tools/benchmark.py prepare --model checkpoints/sft_retrain/checkpoint-1500 --strategy merged_fp16
```

### 3.4 RL training (GRPO)

Script: `ml/rl_grpo.py`.
- `GRPOTrainer` from TRL 0.14.0 (pin `<0.15`).
  - Workaround: `trl.import_utils.is_vllm_available = lambda: False` before import; `use_vllm=False` in `GRPOConfig`.
  - Root cause: `is_vllm_available()` returns `(False, None)` which is truthy.
- Reward: `ml/reward.py:compute_rewards()` → SSH to VM → `tools/kcov_validator` → `{verdict, pcs:[]}`.
- KCOV mode: `KCOV_TRACE_PC` (value=0). Flat uint64 PC array, 1 word/entry.

### 3.5 BPF encoder

`ml/reward.py:_encode_to_hex()` — pure Python, bypasses clang.
Extracts opcode byte `(XX)` from verifier-log format; parses dst/src/off/imm from instruction text.
Always call `strip_verifier_log()` first — old models output log format with header lines; new models output bare assembly; `strip_verifier_log` is a no-op on bare assembly.

### 3.6 KCOV validator

`tools/kcov_validator/main.go` — standalone Go binary.
- Calls `BPF_PROG_LOAD` with the program bytecode.
- Enables KCOV on its own thread.
- Returns JSON: `{verdict: "ACCEPTED"|"REJECTED", pcs: [0x..., ...]}`.
- **Critical fix (c8a9d02):** `readPCs()` called for BOTH ACCEPTED and REJECTED — run 1 only called it for ACCEPTED, giving empty PC trace for rejected programs.

### 3.7 Dataset enrichment (sft_retrain only)

Script: `ml/enrich_dataset.py`.
- Loads `data/dataset_final_qwen.jsonl`.
- Runs each bytecode through kcov_validator → PC set.
- Computes `novelty_score` per program: fraction of its PCs not seen in any other program (uses `set(p.validation.pcs)` for per-program counting — earlier bug used total occurrences, giving negative scores).
- Assigns `novelty_bin` via tertile thresholds: `low` / `medium` / `high`.
- Output: `data/dataset_final_qwen_enriched.jsonl` (the SFT-v2 training file; an earlier variant
  `data/corpus_ml_enriched.jsonl` also exists). *(Earlier drafts of this doc named a
  `dataset_enriched.jsonl` that was never written.)*

---

## 4. How to Run Each Pipeline

### 4.1 Prerequisites

```bash
pixi install          # installs Python env
# Start eval VM (KCOV + KASAN kernel, port 10022):
./fuzzing/run_eval_vm.sh
# SSH key: ~/fuzzing_lab/trixie.id_rsa
```

### 4.2 SFT training

```bash
pixi run python ml/train.py \
  --dataset data/dataset_final_qwen.jsonl \
  --output-dir checkpoints/sft_retrain \
  --num-train-epochs 3
```

### 4.3 Merge LoRA adapter

```bash
# Merge to fp16 (recommended — never use bnb4bit post-merge):
pixi run python tools/benchmark.py prepare \
  --model checkpoints/sft_retrain/checkpoint-1500 \
  --strategy merged_fp16
```

### 4.4 Pass-rate evaluation

```bash
pixi run python tools/evaluate_passrate.py \
  --model checkpoints/curated_merged \
  --n 100
# Output: results/passrate_summary.csv
```

### 4.5 Benchmark pipeline

```bash
# Run n=20 benchmark on a model:
pixi run python tools/benchmark.py run \
  --models sft_retrain_cp1500 \
  --n 20
# Report:
pixi run python tools/benchmark.py report
# Outputs: benchmarks/runs/<ts>_run.json, <ts>_report.md, benchmarks/summary.csv
```

### 4.6 RL training (local)

```bash
pixi run python ml/rl_grpo.py \
  --num-generations 4 \
  --max-completion-length 600 \
  --beta 0.01 \
  --output-dir checkpoints/rl_grpo_v3

# Auto-resume:
pixi run python ml/rl_grpo.py \
  --resume \
  --output-dir checkpoints/rl_grpo_v3
```

### 4.7 Remote reward server (Colab pipeline)

```bash
# Host side:
REWARD_API_KEY=<key> pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000
# Then start ngrok tunnel (see docs/ngrok_tunnel_setup.md).
# Colab side: open ml/train_grpo_colab.ipynb → Run All.
```

### 4.8 Coverage race (buzzer vs LLM)

```bash
# Step 1: Inside VM — rebuild buzzer (from /home/stefano-u/tesi/buzzer/):
export CC=clang && export CXX=clang++
bazel build :buzzer

# Step 2: Run buzzer (inside VM):
./buzzer -strategy=coverage_based -max_programs=500 \
  -coverage_log=/mnt/corpus/buzzer_coverage.csv

# Step 3: Run model side (host):
pixi run python tools/coverage_race.py --max-programs 500

# Step 4: Plot:
pixi run python tools/plot_coverage_race.py
# → results/coverage_race_per_program.png, results/coverage_race_over_time.png
```

### 4.9 Tests

```bash
pixi run pytest tests/    # all offline tests, no VM required
```

---

## 5. Known Inconsistencies (Thesis vs Code)

These exist because thesis chapters were written on 2026-05-31 describing the **original** `curated_3ep` config; `sft_retrain` started 2026-06-01.

| What | Thesis (ch4) | Code / sft_retrain | Status |
|---|---|---|---|
| `max_length` | 768 tokens | **2048** | ⚠️ Thesis wrong if retrain goes in |
| Learning rate | 2e-4 | **5e-5 + cosine warmup** | ⚠️ |
| LoRA targets | Q/K/V/O | **+ MLP** | ⚠️ |
| Gradient checkpointing | not mentioned | **enabled** | ⚠️ |
| Dataset | plain 27k | **+ novelty_score + novelty_bin** | ⚠️ |
| Train/val split | 90/10 | **stratified 70/15/15** | ⚠️ |
| GRPO prompt | `Status: VALID` | `[coverage=high][novelty=high]` | ⚠️ |
| GRPO beta | 0.01 | **0.05** | ⚠️ |

**Decision required:** include sft_retrain in thesis (→ update ch4 + ch6, ~1 week) or keep original run only.

---

## 6. Remaining Fact-Check Issues

All HIGH/MEDIUM items from `.scratch/fact-check/general-knowledge-fact-check.md` were fixed in commit `e0fc7bb 2026-06-01`. LOW items still open:

| # | Where | Claim | Status |
|---|---|---|---|
| 5 | ch1 | CVE-2024-41003 "flaw in certain instruction sequences" | Loose; optional fix: "flaw in `reg_set_min_max`" |
| 6 | ch2 | BPF origin "1993" | Defensible; cite USENIX '93 proceedings |
| 7 | ch2 | "64-bit fixed-width instructions" | Missing: 16-byte wide load exception |

---

## 7. Open Items (as of 2026-06-03)

| Item | Priority | Notes |
|---|---|---|
| **Decide: sft_retrain in thesis?** | HIGH | July deadline; ~1 week to update ch4+ch6 |
| Run pass-rate eval on sft_retrain/checkpoint-1500 | HIGH | Need number for thesis if retrain goes in |
| Run coverage race model side to completion | HIGH | Coverage race result is a key thesis figure |
| Update `training_log.md` with sft_retrain | MEDIUM | Phase 5 not documented |
| Update README "current state" date (still says 2026-05-21) | MEDIUM | Stale |
| Update ROADMAP ("experimental phase closed" but retrain live) | MEDIUM | Contradicts reality |
| Resolve `benchmarks/summary.csv` git tracking | LOW | Commit or gitignore |
| Commit `thesis/references.bib` | LOW | Modified, untracked |
| Commit or gitignore `tools/generate_corpus.py` | LOW | Untracked |
| `rl_grpo_v3/` dir is empty | LOW | Either populate or delete |
| `curated_3ep` VM hang root cause | RESEARCH | `func#0 @ 0x…` header → strip_verifier_log insufficient; semantic validity broken post-strip |
| Update `rl_grpo.py` to load `merged_fp16` not `bnb4bit` | LOW | For next RL run |

---

## 8. Key File Map

| File | What it does |
|---|---|
| `ml/train.py` | SFT training (QLoRA, `EncoderPassRateCallback`) |
| `ml/rl_grpo.py` | GRPO RL training, SSH reward bridge, remote reward URL |
| `ml/reward.py` | `_encode_to_hex()`, `compute_rewards()`; **current reward = RL-v2 validity-gated ladder** (RL_V2.md §3); the depth-based verdict-blind formula was the earlier RL-v1-era design |
| `ml/enrich_dataset.py` | novelty_score + tertile binning for sft_retrain |
| `ml/train_grpo_colab.ipynb` | Colab launcher (5-cell, idempotent Run All) |
| `tools/kcov_validator/main.go` | Go binary: BPF_PROG_LOAD + KCOV → `{verdict, pcs:[]}` |
| `tools/benchmark.py` | CLI: prepare/run/report for model comparison |
| `tools/benchmark_lib.py` | VMLifecycle, generate, analyze, validate, aggregate |
| `tools/coverage_race.py` | LLM side of buzzer-vs-LLM coverage race |
| `tools/plot_coverage_race.py` | 3-curve coverage plot |
| `tools/reward_server.py` | FastAPI server exposing reward over HTTP (API key auth) |
| `tools/evaluate_passrate.py` | Pass-rate eval: load model → generate → validator → CSV |
| `tools/analyze_rl_run.py` | Parse `grpo_completions.log` → tier CSVs + plots |
| `fuzzing/buzzer/pkg/units/ffi.go` | Buzzer data-collection patch (JSONL dump at FFI boundary) |
| `fuzzing/buzzer/pkg/strategies/coverage_based.go` | CSV logger + `-coverage_log` + `-max_programs` flags |
| `fuzzing/run_eval_vm.sh` | Start eval VM (SSH port 10022, trixie.id_rsa) |
| `data/dataset_final_qwen.jsonl` | 27,514 curated SFT samples |
| `data/corpus/buzzer_coverage.csv` | Buzzer coverage race output (elapsed_ms, programs_submitted, valid_programs, unique_pcs) |
| `data/corpus/model_coverage.csv` | LLM coverage race output (+unique_pcs_valid, unique_pcs_all) |
| `results/grpo_completions.log` | Authoritative RL reward log (rl_grpo_v2 + warm-up) |
| `results/rl_pc_set.json` | Cumulative PC set, local RL run |
| `checkpoints/curated_3ep/adattatore_ebpf_v1` | Best original SFT adapter |
| `checkpoints/sft_retrain/checkpoint-1500` | Latest retrain checkpoint (Jun 1 20:28) |
| `benchmarks/runs/` | Per-run JSON + markdown reports |
| `benchmarks/summary.csv` | Aggregated benchmark KPIs |
| `docs/training_log.md` | Phase 1–4 training history (sft_retrain not yet added) |
| `docs/ROADMAP.md` | Decisions log + phase status (partially stale as of Jun 3) |

---

## 9. HuggingFace Artifacts

| Artifact | Location |
|---|---|
| SFT dataset (27k) | `Strhata/ebpf-corpus` |
| SFT adapter (curated_3ep final) | `Strhata/ebpf-checkpoints/curated_3ep_final` |
| Merged SFT model (curated_merged fp16) | `Strhata/ebpf-checkpoints/curated_merged` |
| RL checkpoint (rl_grpo_v2) | Local only — run stopped at plateau |
| sft_retrain checkpoints | Local only — not published |

---

## 10. Verified Numbers (authoritative)

| Metric | Value | Source |
|---|---|---|
| Raw programs collected | ~2M | `tesi_recap.md` §4 |
| Curated dataset size | 27,514 | `dataset_final_qwen.jsonl` |
| Train / val split (original) | 24,762 / 2,752 (90/10) | `training_log.md` |
| SFT steps (curated_3ep) | 9,288 | `training_log.md`, WandB |
| SFT eval_loss | 0.5571 | `training_log.md` |
| SFT-v1 pass-rate (60%-era, clang+ebpf_validator, N=100) | 60% vs 1% zero-shot | `results/passrate_summary.csv` |
| SFT-v1 accept-rate reconstructed (pure encoder + KCOV, N=100, seed 42) | **73%** (clang path 51%; 60%-era number was parser-deflated) | `data/reconstruction/sft_v1_20260604.kcov.jsonl` |
| SFT-v1 coverage despite 73 valid programs | ~2,698 unique PCs (validity ≠ coverage) | `data/reconstruction/sft_v1_20260604.kcov.jsonl` |
| GRPO run 1 steps | 8,370 | `results/grpo_completions.log` |
| GRPO plateau at step | ~1,300 | `results/rl_analysis_pcs.csv` |
| Total unique PCs explored (run 1) | 1,638 | `results/rl_analysis_pcs.csv` |
| Progressively-new PCs (run 1) | **137** (README says 138 — off-by-one; CSV is authoritative) | `results/rl_analysis_pcs.csv` |
| Benchmark pass-rate (sft_retrain_cp1500, merged_fp16) | ~15–20% | `benchmarks/runs/` |
| Benchmark avg raw KCOV trace length (sft_retrain_cp1500, merged_fp16) | ~254k (`avg_pcs` = raw trace, **not** coverage; distinct ≈2,400/prog) | `benchmarks/runs/` |
| SFT-v2 (full, sft-1epoch-v2) valid-unique PCs | 3,462 (n=1k) → 3,862 (n=20k) — saturates | RL_V2.md §1, `benchmarks/diversity/` |
| RL-v2 phase-B (cp200, n=5k) valid-unique PCs | 3,606 (on the SFT saturation curve — no breakthrough) | `benchmarks/diversity/rl-phaseB-cp200-n5000-seed42.json` |
| Buzzer coverage race (500 programs) | 23 valid, 4,915 unique PCs | `data/corpus/buzzer_coverage.csv` |
| LLM first ACCEPTED program | program #8 (vs buzzer #159) | `data/corpus/model_coverage.csv` |
| verifier.c line count (Linux 6.8) | ~21,000 | github.com/torvalds/linux/blob/v6.8/kernel/bpf/verifier.c |
