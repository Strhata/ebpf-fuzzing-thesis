# RL-v2 — Reinforcement Learning to Break Verifier-Path Clustering

Canonical, version-controlled record of the RL-v2 experiment: training the SFT-v2 generator
with GRPO under a **validity-gated, novelty-shaped** reward so that valid eBPF programs exercise
*many distinct* verifier paths, not the same cluster.

- **Status:** design + smoke result settled (2026-06-08); phase-B results pending (§5.2).
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

### 5.2 Phase B — decayed-global novelty — *PENDING*

> _Stub — fill when the phase-B run lands._ The verdict metric is **`novelty/global_frontier`**
> (distinct KCOV PCs ever covered by valid programs): **climbing** = the global reward is forcing
> the policy into new paths (clustering breaking); **plateauing** = clustering wins despite the
> reward. Supporting: `novelty/global_mean` should start ~1.0 and decline as the archive fills.
>
> To capture: `global_frontier` curve vs steps, final frontier vs the SFT-v2 baseline (3,862 valid
> unique PCs), `valid_rate` stability, and the reward/KL/entropy traces.

---

## 6. Conclusion / next

- **Validity is solved enough** (~27% under RL) — the open problem is diversity, exactly the spine.
- **Phase A** proves the loop and clears the RL-v1 `std=0` trap but is a weak anti-clustering signal.
- **Phase B** (running) is the real bet: penalise re-treading the run-wide cluster.
- **If phase B plateaus**, escalate per the literature: raise `RL_W_GLOBAL`, enable
  `RL_GLOBAL_DECAY < 1`, add an entropy bonus / clip-higher, or move to an explicit QD/MAP-Elites
  archive over verifier-path niches.
