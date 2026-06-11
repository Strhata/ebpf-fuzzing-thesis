# JOURNAL — what happened, in order (append-only)

**Entries are dated events. Never edit a past entry; append a new one.** An entry records what we
did, what we learned, and what it changed. Standing facts live in [`FACTS.md`](FACTS.md) (this file
does not restate them); rationale in [`DECISIONS.md`](DECISIONS.md).

Each entry is framed by **capability** — *what could the model do, and what did the run teach* — never
by a pass-rate "win."

---

## 2026-03, weeks 1–2 — VM + kernel infrastructure
Adapted syzkaller's `create-image.sh` → Debian QEMU images. Built Linux 6.8.0 in three variants:
plain `bzImage`, `bzImage_kasan` (throughput baseline), `bzImage_kasan_kcov` (coverage work).
Collected 30 kernel panic / KASAN dumps across parallel VMs (`start_swarm.sh`); profiled with
strace/perf. **Learned:** buzzer's `coverage_based` mode uses KCOV only to *display* metrics over an
HTTP server — it does **not** feed coverage back into mutation. That gap (no LLM+KCOV-feedback system
exists) is what motivated the ML pivot.

## 2026-03-26 → 04-21 — data collection
Patched buzzer `pkg/units/ffi.go` to dump every `(bytecode_hex, verifier_log, is_valid, error_line,
error_reason)` as JSONL at the kernel FFI boundary. Dockerised single-VM setup, corpus shared via
virtio-9p, auto-restart on kernel crash (`rotate_dataset.sh`). **Collected ~2M raw programs.**

## 2026-04-10 — dataset curation
Analysed 73 syzkaller dumps (13,153 valid + 14,361 invalid). Normalised error classes (masked
numbers/addresses), capped 2,000/class → **27,514 examples** (`dataset_final_qwen.jsonl`).

## 2026-04-16 → 05-14 — SFT-1
QLoRA on Qwen2.5-Coder-1.5B, verifier-log assembly format. Warm-start `sft_fase1` (1,500 steps, a
`max_steps` bug = 48 % of data) then the full **3-epoch run** `curated_3ep` (9,288 steps, eval_loss
0.5571, ~26.8 h). **What the model could do:** generate *valid* BPF programs — but only **trivial 1–2
instruction** ones. The verifier-log format is ~232 tokens/instruction, so the completion budget left
no room for real programs. **Learned:** the *format* is the bottleneck; this is information, not a win.

## 2026-05-17 — first capability check
Built `tools/kcov_validator` (Go). Ran the legacy clang + `ebpf_validator` pass-rate eval on
`curated_merged` vs zero-shot base. The fine-tuned model produced loadable programs where the base
produced essentially none — i.e. SFT taught BPF syntax. *(The "60 %" figure from this run is a
diagnostic of syntactic validity on trivial programs, not a result; see 2026-06-04.)*

## 2026-05-18 → 05-21 — RL-1 (GRPO run 1)
First GRPO run `rl_grpo_v2` (tiered reward: new-PCs 1.0 / valid 0.1 / rejected 0.1 / crash 2.0,
G=4). Ran 8,370 steps. **It did not learn.** Plateau at step ~1,300: `reward_std=0` → zero GRPO
gradient. Root cause: `kcov_validator` returned empty PCs for REJECTED programs → all rejected
programs looked identical → no within-group variance. **The result of RL-1 is this insight** (GRPO
reward-starvation under homogeneous reward), not the 137 PCs it happened to touch. Fixed the validator
to read PCs for both verdicts (commit `c8a9d02`).

## 2026-05-21 — reward + format redesign (intermediate)
Stripped the verbose header from the format so real programs fit the budget. Designed a depth-based
verdict-blind reward and the remote (Colab) reward pipeline (FastAPI server + tunnel + auto-resume).
*(This verdict-blind reward was later replaced — see 2026-06-08.)*

## 2026-05-31 — thesis chapters drafted
All 7 chapters written in one session, describing the SFT-1 / RL-1 state of that day. *(Now stale: no
SFT-2, no saturation result, no RL-2 — to be revised. See FACTS for current truth.)*

## 2026-06-01 — fact-check pass
Corrected 5 general-knowledge claims in the chapters (verifier.c ~21k lines; CVE-2023-2163 = state
pruning; Falco = Sysdig/CNCF; BpfChecker = differential fuzzer; unprivileged-BPF caveat).

## 2026-06-01 → 06-07 — SFT-2
Retrained with `[coverage][novelty]` control tokens, +MLP LoRA targets, lr 5e-5 cosine, 2048-token
length, stratified 70/15/15. First a **partial local probe** (`sft_retrain`, ~1,500 steps — "does it
generate at all", the n=20 "19 %" benchmark), then the canonical **full one-epoch** run
(`sft-1epoch-v2`, ~2,408 steps). **What the model could do:** generate *real, deep* programs (30–58
instructions) — a genuine step up from SFT-1's trivial outputs. **Weakness surfaced:** low diversity.

## 2026-06-03 — benchmark harness + coverage race
Built `tools/benchmark.py` (prepare/run/report) and the buzzer-vs-LLM coverage race
(`coverage_race.py`). Confirmed `merged_fp16` >> `merged_bnb4bit` (4-bit post-merge destroys quality)
and that the `[coverage=high][novelty=high]` prefix is load-bearing.

## 2026-06-04 — "60 %" reconstruction
The original 60 % was not reproducible (eval saved only id/compiled/verdict, generation unseeded).
Reconstructed SFT-1 seeded (N=100, seed 42), dual-encoded through clang and the pure encoder, both via
KCOV. **Findings:** (1) the clang path *deflated* the number — it mis-parses verifier register-state
lines as instructions; SFT-1's syntactic accept rate is ~73 % via the pure encoder, not 60 %. (2)
"60 % vs 19 %" was two *models*, not two pipelines. (3) **The real lesson: validity ≠ coverage** — 73
valid programs reach only ~2,700 unique PCs. The accept-rate number itself is meaningless; the
programs are trivial/clustered. *(Forensic detail preserved in git history of `data/reconstruction/`.)*

## 2026-06-08 — diversity saturation quantified
Ran the diversity benchmark on SFT-2 at n=1k and n=20k. **The headline result:** 17.5× more valid
programs (75 → 1,310) buy only +12 % unique PCs (3,462 → 3,862); doubling program length (512→1024
tok, 30→58 instructions) did not raise diversity either. **Diversity, not validity or length, is the
wall** — the model marches the same verifier paths.

## 2026-06-08 — RL-2 (validity-gated novelty reward)
Replaced the verdict-blind reward with a **validity-gated** ladder (valid always beats invalid;
invalid gets a soft floor for depth walked; valid adds per-group + decayed-global novelty), G=16.
Phase-A smoke (20 steps): `reward/std` 0.31 — the RL-1 `std=0` trap is **cleared**, the loop learns.
Phase-B (200 steps, `RL_W_GLOBAL=2.0`): **no diversity breakthrough** — RL's valid programs land on
the SFT-2 saturation curve (valid-unique 3,606 at 433 valid programs; novelty 0.749 ≈ SFT's 0.752).
**Reading:** the null is *evidence for* the thesis (diversity is the wall), and the run is
under-trained (200 steps) + benchmarked out of its 512/temp-0.9 training regime — it localises the
open problem rather than closing it.

## 2026-06-08 — documentation reconciliation + restructure
Forensic audit of every doc against git provenance (true-then/false-now). Fixed cross-doc
contradictions (RL-2 status, reward semantics, the SFT-2 naming collision, the `avg_pcs` units error).
Then restructured docs into this model: always-current `FACTS.md`, append-only `JOURNAL.md`,
`DECISIONS.md`, `ops/` — retiring the chronological-sediment docs that needed "superseded" banners.

## 2026-06-08 — RL-2 reading corrected *(clarifies the RL-2 phase-B entry above)*
The earlier RL-2 entry called the phase-B null "evidence for the thesis (diversity is the wall)." That
**overclaims** and is withdrawn. Corrected reading, from WandB run `bz5ymfzl` (phase-B, 207 steps):
the run is ~40× shorter than RL-1's 8,370 steps; `reward/std` stays alive — **mean 0.224** (median
0.230, range 0.01–0.44) vs **RL-1's mean 0.001 (99 % of steps exactly 0)**, so the RL-1 starvation did
**not** recur (it fluctuates, never sustained-zero) — and `cumulative_pcs` rose 4320 → 4836 but
**decelerating** (last ~50
steps +31). That is *too short to tell* whether coverage would plateau or break through. The
valid-unique metric (3606) sits on the SFT-2 saturation curve, but **the saturation/mode-collapse wall
is an SFT-2 property** (measured on SFT-2 generations); whether KCOV-reward RL can exceed it is **open
(RQ2)**, not closed. RL-2 neither confirms nor refutes that the wall is unbreakable. Note also that
`cumulative_pcs=4836` is *total* PCs (includes invalid programs' verifier-walk PCs via the soft floor)
— distinct from the valid-unique metric that counts.

## 2026-06-08 — saturation metric is nondeterministic; "+12 %" superseded by a replicate range
While building the F2 figure, re-validating the *same* saved candidates on the VM gave different
valid-unique-PC counts run-to-run (e.g. n=20k: 4,728 then 3,696). Cause: KCOV records every kernel PC
hit during the trace (background interrupt/scheduler PCs accumulate with program count) and a few
borderline verdicts flip on timing. Ran replicates (`tools/measure_saturation.py` →
`benchmarks/diversity/saturation_replicates.json`): spread ±~1 % at 75 valid, **±~12 % at 1,300 valid**.
The earlier single-sample headline "17.5× → +12 % (3,462→3,862)" is a **low draw of a noisy metric** and
is superseded: within one n=20k run, ~16× more valid programs raise unique valid PCs by **~+28 %
(replicate range +9–45 %)**. The sub-linear trend (the wall) is robust across every replicate; only the
exact % was unstable. Lesson: headline the trend, report the range, never a bare % for this metric.
Figure `thesis/figures/saturation.*` shows all replicate curves as a band. (FACTS §5 updated.)

## 2026-06-08 — thesis-revision plan drafted (grill → PRD → issues)
Ran grill-me → to-prd → to-issues on the thesis rewrite. Artifacts in gitignored
`.scratch/thesis-revision/` (`PRD.md`, `issues-draft.md`): 10 vertical slices in 3 milestones
(Evidence → Narrative → Propagation), critical path 04→05→06→07. F1 depth-collapse figure +
`tools/figures_lib.py` landed first (commit `752a509`).

## 2026-06-10 — thesis reframed to evidence-only: dropped "mode collapse" / "diversity wall" as claims
Decision (user): the thesis must claim only what a measurement directly backs. Removed the interpretive
layer — "mode collapse", "the diversity wall" as a named central finding, "the generator collapses onto a
narrow mode / below its teacher", and the length-"spread"/CV framing (the only measurable length fact is
that doubling the token budget scales the median 30→58 because the model never emits a terminating `exit`;
spread/CV carried no meaning). **Kept** as the solid findings: (1) gradient starvation (σ_r=0 → zero
gradient) — measured, diagnosed, fixed; (2) the SFT-1→SFT-2 capability jump (trivial 1–2 instr → deep
valid programs) reframed as the positive result it is. The coverage saturation and the buzzer race are now
reported as bare measurements with **no mechanism claimed** (no clean same-harness baseline was run, so
cause is undetermined). RQ2 open = RL too short (≈200 vs 8,370 steps), not "a wall".
Edited: `abstract.tex` (IT+EN), `ch1` (contribution #2 + structure), `ch4` (RQ3 + reward-redesign ref),
`ch6` (§"Coverage Versus Sample Count" replaces "The Diversity Wall"; SFT-2 §retitled; depth caption),
`ch7` (summary, contribution #2, RQ3, future work). FACTS §1 + SFT-2 row updated. Thesis builds clean.
**Also fixed a provenance bug:** ch7 had attributed the `crashes.csv` OOM kills to "corpus collection";
they are actually from the early 3-VM exploration swarm (`fuzzing/exploration/start_swarm.sh`), not the
single-VM Docker collection pipeline — corrected to "early exploration phase". (cf. data-artifact-provenance.)
Verdict on early-buzzer material: `fuzzing/exploration/` swarm scripts add nothing to the argument and stay
out of the thesis (correct as-is); vendored `fuzzing/buzzer/` is upstream Google code, only `ffi.go` (~50 lines)
is ours; `fuzzing/buzzer-coverage-race/` is used (the confounded race).

## 2026-06-11 — CovRL paragraph corrected; "diversity wall" purged; design-wager framing added
Rewrote ch3 CovRL paragraph: dropped "structurally identical". Now frames the shared paradigm + the
load-bearing difference = CovRL **mutates an explicit seed corpus** vs this work **generates from a learned
distribution** (the corpus held implicitly in the model's weights, RL standing in for mutation). Cited
CovRL's 48 bugs / 11 CVEs as the warrant the paradigm finds defects. PPO-vs-GRPO demoted to a one-clause
aside (token- vs program-level reward weighting), explicitly "not the difference that bears on the results"
— author considers the algorithm name not load-bearing right now. Added the same **corpus-in-the-weights
wager** to ch7 Limitations §"samples not searches" so ch3 and ch7 align (and ch6 forward-refs it). Renamed
the LaTeX label `sec:diversity-wall` → `sec:coverage-scaling` across ch5/ch6 (author dislikes the term);
no visible "diversity wall"/"the wall"/"structurally identical" remain. Builds clean (0 undefined refs).
Completes PRD #37 §2 (CovRL) + §3 (Limitations).

## 2026-06-11 — corrected the "narrow teacher" framing: the model UNDER-covers its corpus
The "corpus inherits its teacher's range" limitation (added earlier today) was wrong and is removed.
Evidence: buzzer — the generator that produced the SFT corpus — reaches 4,915 unique PCs with 23 valid
programs (`data/corpus/buzzer_coverage.csv`) against the model's 3,460 with 92 (`model_coverage.csv`); and
the dataset bytecode, analysed, spans more verifier PCs than the model reproduces (author's analysis). The
teacher/corpus is therefore the BROADER of the two — the narrowing is introduced by the learned generator
(a lossy compression of a broad corpus into a narrow sampling distribution), not inherited from a narrow
source. ch7 Limitations paragraph reframed to "The implicit corpus is lossier than the explicit one". This
closes the narrow-teacher confound in the model's direction. TODO: cite the exact corpus-union-PC figure
(vs the model's 3,799 over the 500-program race) in ch6 + FACTS once the author supplies/confirms it.

## 2026-06-11 — RQ2 sharpened to "exceed the corpus"; RL length is compute-bounded, not a plateau
Reframed RQ2 (ch4) around the real hypothesis: SFT only imitates the corpus; RL draws reward straight from
the verifier, so a successful loop should let the model EXCEED the coverage of the corpus it was trained on.
The measurements fix the starting point — the model currently sits BELOW its corpus (under-covers it) — but
do not answer whether RL can carry it past. Stated honestly that run~2's ~200 steps was a length set by the
available cloud-GPU (Colab) budget, not by any plateau. ch7 future-work item 1 now names the decisive
experiment: longer RL → generate at the 20k scale → test whether the RL model reaches MORE unique valid PCs
than the corpus. Softened ch7 "accounts for" → "is enough to produce" (coupon-collector sufficiency, not
sole cause). Measured artifacts the thesis rests on, for the record: the dataset/corpus; the buzzer
generation + coverage race; the benchmarks; the 20k SFT-v2 (1-epoch) saturation sample; the ~5k RL-v2
(cp200, brief) sample. Builds clean.
