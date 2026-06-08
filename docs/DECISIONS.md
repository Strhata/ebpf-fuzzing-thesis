# DECISIONS — why we chose what we chose (append-only)

**One entry per decision: the choice, the context, the reason, and what we rejected.** Append new
decisions; supersede an old one with a new dated entry that references it (never edit the original).
Standing facts live in [`FACTS.md`](FACTS.md); the chronology in [`JOURNAL.md`](JOURNAL.md).

---

### D1 — Metric is unique KCOV PCs from valid programs, not pass-rate
A model that emits one trivial valid program scores ~100 % pass-rate and finds nothing. Pass-rate
rewards triviality. The fuzzing goal is *exploration*, which only valid-and-diverse programs achieve.
**Rejected:** pass-rate as the objective (kept only as a diagnostic). See FACTS §1.

### D2 — KCOV as the GRPO reward signal *(the novel contribution)*
Buzzer (and prior eBPF fuzzers) use KCOV only to display metrics, never to guide generation. Wiring
KCOV coverage into an RL reward over an LLM generator is what no prior system does. This is the thesis.

### D3 — Assembly (verifier-log) format, not raw hex
The model learns BPF programs as verifier-log assembly text, not hex bytes. Assembly is closer to the
model's pretraining distribution and is human-debuggable. **Rejected:** hex encoding (tried, abandoned
— the model could not learn structure from raw bytes).

### D4 — Pure-Python BPF encoder, bypassing clang
`reward._encode_to_hex()` packs assembly → bytecode directly. clang as the assembler *rejects* unusual
instruction sequences before they ever reach the verifier, and it mis-parses verifier register-state
lines (this deflated the legacy "60 %"). The pure encoder lets odd programs reach the verifier.
**Rejected:** clang + llvm-objcopy (the legacy path; now only in the diagnostic `evaluate_passrate.py`).

### D5 — Validity-gated reward with a soft floor (RL-2)
At ~7 % valid, most GRPO groups contain *zero* valid programs. A hard 0/1 validity gate gives every
program in such a group the same reward → `reward_std=0` → no gradient. **This was the RL-1 failure.**
The fix: invalid programs get partial credit for how far the verifier walked before rejecting (soft
floor < `W_VALID`), so within-group variance always exists and the policy learns "get closer to valid"
before it ever produces a valid program. Valid programs then add novelty. **Rejected:** the earlier
verdict-blind depth reward (it did not address the validity lever and was never run to conclusion).

*Literature grounding (the SFT-2 clustering is GRPO's documented mode/diversity/entropy collapse under
a scalar reward; the fixes converge on group-defined reward + persistent novelty / quality-diversity):*
[DiverseGRPO](https://arxiv.org/html/2512.21514) ·
[GAPO / group-aware RL](https://arxiv.org/html/2511.12596v1) ·
[DRA-GRPO](https://arxiv.org/pdf/2505.09655) ·
[scaling/prolonged RL](https://arxiv.org/pdf/2507.12507) ·
[RL-based fuzzing survey (RLFuzz)](https://www.researchgate.net/publication/342080235) ·
[Quality-Diversity for sparse rewards / MAP-Elites](https://arxiv.org/pdf/2203.01027) ·
[GRPO tricks](https://cameronrwolfe.substack.com/p/grpo-tricks).

### D6 — G=16 group size
Per-group novelty needs ≥1 valid program per group to fire. At ~7 % valid: G=8 → ~0.6 valid/group
(most groups all-invalid, novelty never fires); **G=16 → ~1.3 valid/group**, signal exists. Memory-
gated on the 40 GB A100; fallback ladder is *drop length → G=12 → G=8* — protect G over length.

### D7 — max_completion_length = 512, not 1024
Empirically: doubling length (512→1024, avg 30→58 instructions) raised unique PCs only +12 %. Length
is **not** the diversity lever. Spend the saved memory on a larger G (D6) instead.

### D8 — Split generation (Colab) from reward (local)
Colab has no nested KVM, so it cannot run the KCOV VM. Generation runs on the Colab A100; the KCOV
reward runs on the local WSL box; an HTTP bridge connects them. Tunnel = ngrok static domain (a stable
URL across reboots; replaced an earlier Cloudflare quick tunnel). *Implementation detail, not a result.*

### D9 — Merge LoRA to fp16 for inference/RL; never 4-bit a merge
Benchmark showed `merged_bnb4bit` produces 0 % valid — 4-bit quantisation after merging destroys
quality. Always merge to fp16. (4-bit NF4 is fine for *training* memory, where base weights stay quantised.)

### D10 — Colab Pro over Modal for remote training
Modal was scoped then abandoned (issues #2–7, closed not-planned) — billing/complexity. Colab Pro
needs a manual ~24 h restart but is simpler. *Implementation detail.*

### D11 — Cap dataset at 2,000 examples per error class
The raw corpus is dominated by a few error classes. Capping prevents the top class from dominating
the 27k curated set and starving rare-but-informative classes.

### D12 — Do not rename checkpoint directories
Directory names grew organically and are off-by-one (`rl_grpo_v2` = RL run 1). They are wired into
tools, WandB resume files, and `benchmarks/`. Use the canonical↔dir map in FACTS §2; never rename.

### D13 — Novelty is anchored on the literature gap, not on buzzer's internals *(supersedes D2)*
D2 rested the novelty claim on "buzzer (and prior eBPF fuzzers) use KCOV only to display metrics,
never to guide generation." A later source scan confirms buzzer ships a `coverage_based` mode; the
precise internal behaviour of that mode is someone else's code, and we do not want the thesis's
novelty to depend on a contestable assertion about it. **We retract that comparative claim.** The
defensible and sufficient novelty is the *literature gap*: we found no published work that fine-tunes
an LLM to generate eBPF programs for verifier fuzzing — the LLM-fuzzing literature targets JS engines,
C, and syscall specs, not BPF bytecode, and coverage-as-RL-reward over an LLM eBPF generator is
unattested. Buzzer's role here is **inspiration + training-data source**, stated factually. The
KCOV-reward-guided GRPO loop remains the system contribution; it is novel by the literature gap,
independent of any claim about buzzer. *(D2's "novel contribution = KCOV as the reward signal" intent
survives; only its buzzer-deficiency rationale is withdrawn.)*
