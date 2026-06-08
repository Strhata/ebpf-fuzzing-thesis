# Documentation Audit — provenance + temporal-truth ledger

**Date:** 2026-06-08. **Scope:** every project-owned doc + the thesis. **Method:** read each file,
date it via `git log --diff-filter=A`/last-commit, then classify every cross-cutting claim against
current ground truth (code, checkpoints, benchmark JSON). Vendored buzzer docs (`fuzzing/buzzer/**`)
are out of scope (Google's, not our truth).

**Truth classes:** `TF` true-then/false-now (correct when written; project moved on) · `FA`
false-always · `STALE` outdated · `CONTRA` contradicts another doc · `RND` unsupported/"random" ·
`UNITS` quantity mislabeled · `OK` verified consistent.

---

## 1. Root cause — three freeze-points

The repo's docs were frozen at three moments and never fully reconciled forward:

| Stratum | Date | Froze | Believed-true then |
|---|---|---|---|
| **S1 early** | 05-13→05-31 | ROADMAP, training_log, tesi_recap, pipeline_spec, README(v1), **all thesis chapters** | curated_3ep is the model; 60% pass-rate; RL-v2 = depth-based verdict-blind, not yet run; Cloudflare tunnel |
| **S2 consolidation** | 06-04 | PROJECT_HISTORY (declared SSOT), NAMING, REPORT, README(v2), ngrok doc | 60% was deflated→73% (SFT-v1); SFT-v2 = sft_retrain (19%); RL-v2 still not run; ngrok |
| **S3 diversity+RL-v2** | 06-07→06-08 | MODEL_NOTES (gitignored), RL_V2 | SFT-v2 = **sft-1epoch-v2** (7%); diversity is the wall (saturation curve); **RL-v2 HAS run** 200 steps → null; reward is **validity-gated**, not verdict-blind |

Every finding below is a claim from an earlier stratum that a later stratum overturned, or a
cross-stratum contradiction never reconciled. **PROJECT_HISTORY calls itself the SSOT but is itself
S2 — superseded by S3 (RL_V2/MODEL_NOTES) on the RL-v2 and SFT-v2 fronts.**

---

## 2. Canonical truths (what every doc must agree on after the fix)

1. **SFT-v1** = `checkpoints/curated_3ep` (merged `curated_merged`). 3 epochs, eval_loss 0.5571.
   Accept rate **73%** via pure-encoder+KCOV (the "60%" was clang-parser-deflated). N=100, seed 42.
2. **SFT-v2** = **`checkpoints/sft-1epoch-v2/sft_adapter`** — the **full one-epoch** run (~2,408
   steps). Diversity benchmark: 7.5% (n=1k) / 6.6% (n=20k) accept; valid_unique_pcs 3,462 / 3,862.
   **This is the diversity + RL-v2 base.**
   - `sft_retrain/checkpoint-1500` is the **partial probe** of the *same* work (~1,500 steps / 0.62
     epoch, "does it generate bytecode") — the n=20 "19%" benchmark. **The only difference between
     the two is training steps.** Where each trained (local vs Colab) and which tunnel carried the
     reward (Cloudflare vs ngrok) are **irrelevant implementation details** — do not frame the
     distinction around them, in docs or thesis.
3. **RL-v1** = `checkpoints/rl_grpo_v2` (dir says v2, it is run 1). 8,370 steps, reward_std=0
   plateau ~step 1,300, 137 progressively-new / 1,638 total unique PCs.
4. **RL-v2** = validity-gated, novelty-shaped reward. **Has run:** phase-A smoke + phase-B 200 steps
   (`checkpoint-200`, on Colab/Drive — local `checkpoints/rl_grpo_v3/` is empty). Phase-B result:
   **no diversity breakthrough** (RL valid programs sit on the SFT saturation curve).
5. **Reward (RL-v2):** monotonic ladder — encode-fail/crash→0.0, REJECTED→`W_REJECT_MAX·depth`,
   ACCEPTED→`W_VALID + novelty`. **Verdict-GATED (validity matters).** The old "depth-based
   verdict-blind" reward was the S1/S2 *plan*; it is superseded.
6. **Spine metric** = unique KCOV PCs from *valid* programs. Not pass-rate; not raw trace length.
7. **Tunnel** = ngrok static domain (migrated from Cloudflare Quick Tunnel).
8. **Diversity is the wall, not validity** — saturation curve: 17.5× more valid programs → +12% PCs.

---

## 3. Findings ledger (by theme)

### A. RL-v2 status: "not yet run" — `TF`
- **Claim:** RL-v2 not executed / future work.
- **Where:** README L31,L114 · ROADMAP L25,L62 · PROJECT_HISTORY §1,§4(Phase5),§7 · NAMING L18 ·
  thesis ch1 L142, ch4 L317, ch6 L235–241, ch7 L91 · abstract L69.
- **Truth:** ran phase-A + phase-B 200 steps (RL_V2 §5). **Fix:** update all to "ran 200-step
  preliminary → no diversity breakthrough; under-trained + benchmarked out-of-regime."

### B. RL-v2 reward = "depth-based verdict-blind" — `TF` / `CONTRA`
- **Claim:** RL-v2 reward removes the accept/reject split; `depth + discovery_bonus`.
- **Where:** README L89–104 · ROADMAP L23 · PROJECT_HISTORY §3.4/Phase5 · thesis ch1 L135, ch4
  §"Reward Redesign" L260, ch6 L197–215, ch7 L62,L91 · abstract L25,L62.
- **Truth (RL_V2 §3, MODEL_NOTES §4):** reward is **validity-GATED** — valid always beats invalid;
  invalid gets a *soft floor* (depth); valid gets `W_VALID + novelty`. Verdict now matters. **Fix:**
  replace every "verdict-blind depth-based" description with the validity-gated ladder.

### C. Crash reward = 2.0 — `TF`
- **Claim:** crash/SSH-timeout → reward 2.0.
- **Where:** README L81,L100 · ROADMAP L22,L23 · PROJECT_HISTORY Phase4/§3.4.
- **Truth:** RL_V2 §3.2 / MODEL_NOTES §4.2 set crash → **0.0** ("2.0 rewarded SSH flakiness").
  True for RL-v1 reward; false for RL-v2. **Fix:** scope the 2.0 to RL-v1; state RL-v2 = 0.0.

### D. "60% pass-rate" stated bare — `TF`
- **Claim:** SFT achieves 60% verifier pass-rate.
- **Where:** thesis ch1 L117, ch6 L54, ch7 L30, abstract L29/L65 · training_log L73–76 · tesi_recap §7.
- **Truth (REPORT.md, PROJECT_HISTORY §10):** clang-parser-deflated; true accept **73%** (pure
  encoder+KCOV), SFT-v1 only; "60% vs 19%" is two *models*, not two pipelines. README/NAMING/REPORT
  already corrected; **thesis + training_log + tesi_recap still bare.** training_log/tesi_recap carry
  SUPERSEDED banners (acceptable); **thesis must be fixed** (it's the deliverable). **Fix:** thesis
  reports 73% with the clang-deflation explanation; keep 60% only as "the original clang-gate number."

### E. SFT-v2 naming collision (sft_retrain vs sft-1epoch-v2) + 19% vs 7% — `CONTRA`/`TF`
- **Claim A (S2):** SFT-v2 = `sft_retrain/checkpoint-1500`, "the 19% model."
- **Claim B (S3):** SFT-v2 = `sft-1epoch-v2`, accept ~7%.
- **Where:** NAMING L15,L25 + PROJECT_HISTORY Phase8/9 (A) vs MODEL_NOTES §0/§1.4 + RL_V2 §1 (B).
- **Truth:** both checkpoints exist; they are different models with different configs and numbers.
  NAMING (06-04) predates `sft-1epoch-v2` (06-07). **Fix:** NAMING must list **both**, mark
  `sft-1epoch-v2` as the canonical SFT-v2 (diversity/RL base), and demote `sft_retrain` to "earlier
  retrain / n=20 19% benchmark, not the diversity model." Reconcile the "19%" wherever it implies the
  diversity model.

### F. "~254k PCs per program" — `UNITS`/`RND`
- **Claim:** SFT-v2 reaches ~254k PCs/program.
- **Where:** NAMING L25 · PROJECT_HISTORY Phase9/§10 ("avg PCs ~254k").
- **Truth:** benchmark JSON `avg_pcs=254890` is the **raw KCOV trace length** (every PC hit incl.
  loop repeats), not coverage. Distinct PCs/valid-program is **~2,400** (RL_V2 §1, MODEL_NOTES §2.2).
  Citing 254k as a coverage achievement conflates trace length with unique-PC coverage. **Fix:**
  relabel as "raw trace entries"; report ~2,400 distinct / 3,8xx valid-unique as the coverage figure.

### G. Enriched dataset filename — `FA`/`CONTRA`
- **Claim:** enrichment output = `data/dataset_enriched.jsonl`.
- **Where:** PROJECT_HISTORY §3.7.
- **Truth:** that file does not exist. Real: `dataset_final_qwen_enriched.jsonl` (MODEL_NOTES) and
  `corpus_ml_enriched.jsonl`. **Fix:** name the real file; pick one canonical enriched file and note
  the other if both are live.

### H. Tunnel = Cloudflare Quick Tunnel — `TF`
- **Claim:** Cloudflare Quick Tunnel exposes the reward server.
- **Where:** thesis ch5 L212–233, ch6 L229 · ROADMAP L68 · `tools/cloudflared_config.yml.template` ·
  `ml/rl_grpo.py` (cloudflare refs).
- **Truth:** migrated to **ngrok static domain** (ngrok_tunnel_setup.md, README, RL_V2 §4, colab
  guide). **Fix:** thesis + ROADMAP → ngrok; decide whether to delete the cloudflared template +
  rl_grpo.py cloudflare vestiges or keep as fallback (state which).

### I. "88 tests in test_reward.py" — `FA`
- **Claim:** `tests/test_reward.py` has 88 reward+encoder+server tests.
- **Where:** RL_V2 §4 · MODEL_NOTES §4.5 (implies single file).
- **Truth:** test_reward.py=32; 88 = 32 + encoder 50 + server 6 across **three** files. **Fix:** "88
  across test_reward{,_encoder,_server}.py"; verify the regression guard lives in test_reward.py.

### J. Thesis is missing the entire S3 arc — `STALE` (the big one)
- **Gap:** thesis (all 05-31) never mentions SFT-v2, the diversity saturation curve, the 73%
  reconstruction, or any RL-v2 result. The project's climax (diversity-is-the-wall, the RL null
  result that localizes it) is absent. **Fix:** this is the substance of the later chapter-grilling
  pass — ch4 (validity-gated reward, SFT-v2 config), ch6 (saturation curve, reconstruction, RL-v2
  phase-B), ch7 (diversity-is-the-wall conclusion). Tracked in `docs/THESIS_REVISION.md`.

### K. Minor / housekeeping
- `137 vs 138` PCs: 137 authoritative; thesis ch6 L113 already notes README's 138 is off-by-one.
  README now uses 137 — **OK**, but README L84 vs L113-table both 137; verify no stray 138. `OK`.
- `evaluate_passrate.py` pipeline: ROADMAP Phase3 L52 says "BPF encoder → kcov_validator"; actually
  clang + `ebpf_validator` (pipeline_spec, REPORT). `STALE` — fix ROADMAP or rely on its scope.
- colab_restart_guide: title `grpo-depth-reward-v1`, `beta=0.01`, "T4 GPU" — RL-v2 is A100 +
  validity-gated + beta=0.05. `TF` — update or scope to RL-v1 era.
- `rl_grpo_v3/` empty but checkpoint-200 exists on Colab/Drive: NAMING "empty — not yet run" `TF`.
  **Fix:** "local dir empty; phase-B `checkpoint-200` on Colab/Drive."
- Colab notebook "5-cell" (README, ROADMAP) vs 6 cells incl. 3b merge (RL_V2 §4). `STALE`.
- `ebpf_validator` in both `data/corpus/` and `tools/ebpf_validator/` — harmless dup; pick one
  canonical path in docs.
- ROADMAP self-describes "experimental phase closed" (L25,L69) — reopened by S3. `TF`.

---

## 4. Per-doc disposition

| Doc | Stratum | Disposition |
|---|---|---|
| `docs/PROJECT_HISTORY.md` | S2 | **Promote to true SSOT** but patch S3 deltas: RL-v2 ran (A,B,C), SFT-v2 = sft-1epoch-v2 (E), 254k units (F), enriched filename (G), rl_grpo_v3 note (K) |
| `docs/RL_V2.md` | S3 | Mostly authoritative; fix test-count (I); cross-link as the RL-v2/SFT-v2 source of truth |
| `docs/MODEL_NOTES.md` | S3 | gitignored; fix test-count (I); consider committing (it's the only SFT-v2 method record) |
| `docs/NAMING.md` | S2 | **Rewrite** for the SFT-v2 collision (E) + 254k (F) + rl_grpo_v3 (K) |
| `README.md` | S2 | Patch RL-v2 status/reward (A,B,C), tunnel (H), SFT-v2 (E), quickstart params |
| `docs/ROADMAP.md` | S1 | **Add SUPERSEDED banner** (like training_log/tesi_recap/pipeline_spec) or update tunnel(H)/status(A)/closed(K) |
| `docs/training_log.md` | S1 | Has banner; OK. Optionally append SFT-v2 row |
| `docs/tesi_recap.md` | S1 | Has banner; OK |
| `docs/pipeline_spec.md` | S1 | Has banner; OK |
| `docs/colab_restart_guide.md` | S2/3 | Update T4→A100, beta, run-name (K) |
| `docs/ngrok_tunnel_setup.md` | S2 | `OK` |
| `data/reconstruction/REPORT.md` | S2 | `OK` (authoritative) |
| `fuzzing/buzzer-coverage-race/README.md` | S2 | `OK` |
| `fuzzing/exploration/README.md` | S1 | `OK` |
| `thesis/**` | S1 | **Largest job** — D,B,A,H + the S3 arc (J); done in the chapter-grilling pass |

---

## 5. Recommended fix order

1. **Lock the canonical truths (§2)** — get user sign-off; everything keys off it.
2. **Fix the reference docs** (NAMING, PROJECT_HISTORY, RL_V2, MODEL_NOTES, README) — small, high-value.
3. **Banner/scope the S1 docs** (ROADMAP) — cheap.
4. **Thesis** — the substantive rewrite, per-chapter (separate effort, `THESIS_REVISION.md`).

---

## 6. Resolution log (2026-06-08)

Canonical truths signed off (SFT-v2 distinction = **training steps**, infra irrelevant). Reference
docs fixed in-place; thesis deferred to the chapter-grilling pass.

| Finding | Action taken |
|---|---|
| A RL-v2 status | README, ROADMAP (banner), PROJECT_HISTORY §1 → "ran 200 steps, no breakthrough" |
| B reward verdict-blind→gated | README (note on redesign §), PROJECT_HISTORY §1, ROADMAP banner → validity-gated |
| C crash 2.0→0.0 | covered by B's RL-v1/RL-v2 scoping in PROJECT_HISTORY §1 + RL_V2 §3 (already correct) |
| D 60%→73% | reference docs already correct; **thesis still bare** → grilling pass |
| E SFT-v2 collision | NAMING rewritten (two entries, steps not infra); PROJECT_HISTORY §1, README, MODEL_NOTES header |
| F 254k units | NAMING note, PROJECT_HISTORY Phase-9 footnote + §10 relabel |
| G enriched filename | PROJECT_HISTORY §3.7 → `dataset_final_qwen_enriched.jsonl` |
| H Cloudflare→ngrok | **deleted** `cloudflared_config.yml.template` + `rl_grpo.py` comment; ROADMAP banner; **thesis ch5/ch6** still Cloudflare → grilling pass |
| I 88 tests | RL_V2 §4 + MODEL_NOTES §4.5 → "88 across 3 files (32/50/6)" |
| K rl_grpo_v3 empty | NAMING + PROJECT_HISTORY → "checkpoint-200 on Colab/Drive, local dir empty" |
| MODEL_NOTES gitignored | **un-gitignored + committed**; header updated |

**Still open (thesis, deferred to `THESIS_REVISION.md` grilling pass):** D (60% bare), B (verdict-blind
in ch1/ch4/ch6/ch7/abstract), A (RL "future work"), H (Cloudflare subsection in ch5/ch6 — strip as
irrelevant infra), J (entire SFT-v2 / saturation / reconstruction / RL-v2 arc missing). Minor
S1-doc items (ROADMAP Phase-3 pipeline desc, colab_restart_guide T4/beta) left under their banners.
