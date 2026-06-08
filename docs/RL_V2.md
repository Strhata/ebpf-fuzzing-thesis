# RL-v2 — Reinforcement Learning to Break Verifier-Path Clustering

Canonical, version-controlled record of the RL-v2 experiment: training the SFT-v2 generator
with GRPO under a **validity-gated, novelty-shaped** reward so that valid eBPF programs exercise
*many distinct* verifier paths, not the same cluster.

- **Status:** design settled; phase-A smoke + phase-B 200-step preliminary both in (§5). Open:
  in-regime (512/temp0.9) tiebreaker re-benchmark, and a longer phase-B run.
- **Code:** `ml/reward.py`, `tools/reward_server.py`, `ml/rl_grpo.py`, `tests/test_reward.py`,
  `ml/train_grpo_colab.ipynb`. Commits `396c7e0` (phase A), `8e1a9a3` (phase B), `9e77638` (notebook).
- **Spine:** verifier bugs are found by *valid* programs covering *many distinct* KCOV PCs.
  The metric is **unique KCOV PCs from valid programs**. Diversity, not validity, is the wall.

---

## 1. Problem & prior result

SFT-v2 (`Qwen2.5-Coder-1.5B` + QLoRA, 1 epoch) generates deep, real eBPF programs but its valid
programs **saturate** in coverage — they march the same verifier paths.

| metric | n=1,000 (512 tok) | n=20,000 (1024 tok) |
|---|---:|---:|
| accept_rate (validity) | 7.5% | 6.6% |
| valid programs | 75 | 1,310 |
| **valid unique PCs** (spine metric) | 3,462 | **3,862** |
| avg distinct PCs / valid program | 2,337 | 2,428 |
| avg instructions | 30.2 | 57.6 |

**17.5× more valid programs → only +12% unique PCs.** 1,310 valid programs cover just 1.59× a
single average program. Doubling generation length (512→1024, avg_insn 30→58) did **not** raise
diversity. ⇒ the wall is **clustering**, not validity, length, or loss. (Full SFT-v2 training
details in the gitignored `docs/MODEL_NOTES.md`.)

---

## 2. Literature framing

The SFT-v2 clustering is the documented **mode / diversity / entropy collapse** of GRPO under a
scalar reward — *"relying solely on scalar rewards, the policy collapses into the dominant mode,
ignoring equally valid but distinct strategies."* It is the default failure, not a quirk of this
setup. Two fix families converge here:

- **Reward defined over the group** — GAPO, DiverseGRPO. Reward a completion for properties of the
  *group* (coverage, diversity), not just itself. ⇒ our per-group novelty (§3, phase A).
- **Persistent novelty / Quality-Diversity** — novelty search and MAP-Elites ("illuminating search
  spaces"): don't maximise one number, fill a grid of distinct niches. Coverage-as-RL-reward is
  established in fuzzing (RLFuzz). ⇒ our decayed-global archive (§3, phase B).
- **Optimisation-level guards** — entropy bonus, clip-higher, dynamic clipping — slow the policy
  sharpening into one mode. Held in reserve as escalation.

Sources: [DiverseGRPO](https://arxiv.org/html/2512.21514) ·
[GAPO / Group-Aware RL](https://arxiv.org/html/2511.12596v1) ·
[DRA-GRPO](https://arxiv.org/pdf/2505.09655) ·
[Scaling RL / prolonged training](https://arxiv.org/pdf/2507.12507) ·
[RL-based fuzzing survey](https://www.researchgate.net/publication/342080235_Reinforcement_Learning-Based_Fuzzing_Technology) ·
[Quality-Diversity for sparse rewards](https://arxiv.org/pdf/2203.01027) ·
[GRPO++ tricks](https://cameronrwolfe.substack.com/p/grpo-tricks) ·
[TRL logging / RL metrics](https://huggingface.co/docs/trl/logging).

---

## 3. Method

### 3.1 Two facts that drive the design

1. **The model has no validity lever.** The SFT prompt conditions only on
   `[coverage=…][novelty=…]` — there is no validity token. Training data is **48% valid / 52%
   invalid** (13,153 / 27,514), so under any prompt the model samples a ~half-invalid distribution
   → ~7% accept at `high/high`. Validity is the biggest untapped lever (the fuzzer cares only about
   valid programs), and it must be *built into the reward*.
2. **Validity rate × group size.** A GRPO group of G completions contains ≈ `0.07·G` valid
   programs. At G=8 that is ~0.6 (most groups have **zero** valid → the novelty term never fires);
   at **G=16** it is ~1.3 → the signal exists per group.

### 3.2 Reward ladder (`ml/reward.py`)

Monotonic so a valid program always beats any invalid one:

```
encode fail / < 15 insns      → 0.0
VM ERROR / whole-batch crash  → 0.0        (RL-v1 paid 2.0 here — that rewarded SSH flakiness)
REJECTED (invalid)            → W_REJECT_MAX · min(1, len(pcs)/max_pcs_seen)
ACCEPTED (valid)              → W_VALID + W_NOVELTY · group_novelty(p) + W_GLOBAL · global_novelty(p)
```

- **Soft floor (invalid).** Partial credit for *how far the verifier walked before rejecting*
  (`len(pcs)`). This is the RL-v1 fix: at 7% valid most groups are all-invalid, so a hard 0-gate
  gives identical rewards → `reward_std = 0` → no gradient. The RL-v1 negative result *was* this
  `std=0` gradient starvation. The floor guarantees within-group variance, so the policy learns
  "get closer to valid" before it ever produces a valid program.
- **Per-group novelty (phase A).** `group_novelty(p) = mean over p's PCs of (1 − freq/n_accepted)`,
  freq = how many ACCEPTED programs in *this batch* hit that PC. A path every valid sibling walks
  pays 0; a path only p walks pays ~1. Stateless, never starves — but only fights crowding *within
  a batch*.
- **Decayed-global novelty (phase B).** `global_novelty(p) = mean over p's PCs of
  1/(1 + global_freq[pc])`, where `global_freq` is a persistent per-PC hit count over ACCEPTED
  programs across the **whole run** (snapshot pre-batch). A brand-new PC pays ~1.0; a PC hit 100×
  pays ~0.01. The `1/(1+freq)` form **fades** re-tread paths rather than hard-zeroing them, avoiding
  the naive-archive reward collapse. Optional `RL_GLOBAL_DECAY < 1` ages all counts each batch so
  abandoned regions slowly become rewarding again. State persists to
  `results/rl_global_pc_freq*.json`; accepted-only so it tracks the spine metric.

### 3.3 Tunable weights (env, read at server launch)

| env var | default | meaning |
|---|---:|---|
| `RL_W_VALID` | 1.0 | flat bonus for crossing into ACCEPTED |
| `RL_W_NOVELTY` | 1.0 | weight on per-group novelty (phase A) |
| `RL_W_GLOBAL` | **0.0** | weight on decayed-global novelty (**phase B**; set 2.0 to enable) |
| `RL_W_REJECT_MAX` | 0.3 | ceiling on invalid soft floor (< `W_VALID` by design) |
| `RL_GLOBAL_DECAY` | 1.0 | per-batch multiplicative ageing of `global_freq` (1.0 = off) |

`RL_W_GLOBAL=0` ⇒ phase A; `RL_W_GLOBAL=2.0` ⇒ phase B (fresh-PC valid ≈ 3.0, clustered valid
→ ~1.0 as the frontier fills).

### 3.4 Training config

- **G = 16** (group size). Memory-gated on the 40GB A100; fallback ladder **drop MAX_LEN → G=12 →
  G=8** — protect G over length.
- **max_completion_length = 512.** Empirically justified: SFT-v2 512→1024 raised avg_insn 30→58 but
  unique PCs only +12% → length is not the diversity lever; spend the saved memory on larger G.
- beta = 0.05, lr = 5e-6, temperature = 0.9, LoRA r=16/α=32 on q/k/v/o. (Unchanged from RL-v1.)

---

## 4. Experimental setup / reproduction

**Architecture (split brain).** Generation runs on the Colab A100; the KCOV reward runs on the
local WSL box (no nested KVM on Colab). The bridge is unchanged from RL-v1:

```
Colab GRPOTrainer (rl_grpo.py --remote-reward-url)
  → HTTP POST /rewards   (ngrok tunnel, X-API-Key)
     → local FastAPI tools/reward_server.py
        → reward.compute_rewards(completions, ssh)
           → SSH (port 10022) → KCOV VM → kcov_validator --batch
```

Reward weights live on the **local server** (env above); training knobs live in **notebook Cell 4**.

**Local (WSL):**
```bash
./fuzzing/run_eval_vm.sh                         # KCOV VM up (SSH :10022)
RL_W_VALID=1.0 RL_W_NOVELTY=1.0 RL_W_GLOBAL=2.0 RL_W_REJECT_MAX=0.3 RL_GLOBAL_DECAY=1.0 \
  REWARD_API_KEY=<key> \
  pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000
ngrok http 8000                                  # public URL → Colab REWARD_SERVER_URL secret
```
Restart the server after any `reward.py` change (it imports once). Use Colab-specific state files
(`results/rl_*_colab.json`); delete them to start a run with an empty frontier.

**Colab (`ml/train_grpo_colab.ipynb`):** Cell 1 mount → 2 clone/pull → 3 install → **3b merge
SFT-v2 adapter → fp16** (RL needs a merged model; never 4-bit a merge) → 4 config → 5 launch.
Secrets: `REWARD_SERVER_URL`, `REWARD_API_KEY`, `WANDB_API_KEY`, `GITHUB_TOKEN`.

**Telemetry (W&B), per reward batch:** `reward/std` (the RL-v1 guard, must be > 0), `reward/mean`,
`valid_rate`, `verdict/{accepted,rejected,encode_fail,error,crash}`, `novelty/group_mean`,
`novelty/global_mean`, `novelty/global_frontier`, plus TRL's `kl`, `grad_norm`, `loss`.

**Tests:** `tests/test_reward.py` (88 reward+encoder+server tests) incl. the RL-v1 regression guard
`test_all_rejected_group_still_has_variance` and the phase-B decay/snapshot/accepted-only tests.

---

## 5. Results

### 5.1 Smoke test — phase A (2026-06-08, 20 steps, G=16) — **PASS**

The loop turns and the RL-v1 trap is cleared:

| signal | value | reading |
|---|---:|---|
| `reward/std` | **0.31** | > 0 → gradient exists (RL-v1 sat at 0) |
| `kl` | 0.005 | bounded, healthy |
| `valid_rate` | **~27%** | ~4× the 7% SFT baseline → validity is *not* the bottleneck |
| group novelty (accepted) | **~0.05** | ≈95% of each valid program's PCs shared with siblings → clustering, quantified in the reward |

The soft floor produced within-group variance as designed; the small group-novelty figure (0.05)
confirms per-group novelty alone is a weak anti-clustering signal here — motivating phase B.

### 5.2 Phase B — decayed-global novelty — 200-step preliminary (2026-06-08)

Trained ~200 GRPO steps with `RL_W_GLOBAL=2.0` (G=16, 512 tok, temp 0.9), saved
`checkpoint-200`. Stopped early on Colab compute-unit exhaustion. The diversity benchmark was
re-run on the RL model (`sft_v2_merged` + `checkpoint-200`), matched to the SFT-v2 n=20k baseline
config (1024 tok, temp 1.0, top_p 0.95, seed 42):

| | SFT-v2 n=1k | SFT-v2 n=20k | RL phase-B cp200 n=5k |
|---|---:|---:|---:|
| accept_rate | 7.5% | 6.6% | **8.7%** |
| valid programs | 75 | 1,310 | **433** |
| **valid_unique_pcs** | 3,462 | 3,862 | **3,606** |
| novelty_score | 0.752 | 0.758 | **0.749** |

**Verdict: no diversity breakthrough at 200 steps.** Placed on the SFT saturation curve by
valid-program count — SFT (75 → 3,462) … (1,310 → 3,862) — RL's (433 → 3,606) lands *on the same
curve* (linear-interp SFT at 433 valid ≈ 3,580; the concave true curve is a touch higher). Novelty
is identical (0.749 vs 0.752). The RL model's valid programs cluster like SFT's. Validity nudged up
(6.6% → 8.7%, ~30% rel) — the gate does something, but diversity did not move.

**Two caveats before concluding (both real):**
1. **Only 200 steps** — the policy barely shifted; the in-training `global_frontier` was still
   climbing when stopped. This is the floor of the method, not its ceiling.
2. **Benchmarked out of training regime.** RL trained at 512 tok / temp 0.9; the benchmark ran
   1024 tok / temp 1.0 (to match the SFT baseline). Inference validity here (8.7%) is far below the
   ~30–49% observed *during training* → the 1024/temp-1.0 benchmark may mask RL gains. The honest
   tiebreaker is to re-benchmark BOTH models at 512 tok / temp 0.9 (RL's own regime).

Artifacts: `benchmarks/diversity/rl-phaseB-cp200-n5000-seed42.json` (candidates gitignored).

---

## 6. Conclusion / next

**Thesis-level conclusion (honest, defensible):** fine-tuning yields a generator of valid, deep
verifier programs, but the binding constraint on learned verifier fuzzing is **path diversity, not
validity** — shown by a hard saturation curve (17.5× more valid programs → +12% coverage). A
validity-gated, novelty-shaped RL objective is the natural intervention; a first 200-step phase-B
run did **not** break the saturation (RL valid programs sit on the SFT diversity curve). This
*localizes* the open problem and leaves the metric + infrastructure to attack it. The RL null
result is evidence *for* the thesis (diversity is the wall), not against the approach — it is
under-trained (200 steps) and benchmarked out of regime.

**Next, cheap → expensive:**
- **Tiebreaker (no training):** re-benchmark SFT-v2 *and* RL at 512 tok / temp 0.9 (RL's regime).
  Decides whether RL gains are real-but-regime-specific or absent.
- **More steps:** 200 is far short of reshaping the policy; the in-training frontier was still
  climbing. Needs compute beyond the current Colab budget.
- **Stronger signal:** raise `RL_W_GLOBAL` (2→6), enable `RL_GLOBAL_DECAY < 1`, add an entropy
  bonus / clip-higher, or move to an explicit QD/MAP-Elites archive over verifier-path niches.
- **Throughput for longer runs:** bump G (16→32, the A100 was half-idle at 20/40GB) and/or
  `use_vllm=True` (generation is the wall-clock bottleneck).
