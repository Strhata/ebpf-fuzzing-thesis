# "60% pass-rate" — forensic reconstruction

**Date:** 2026-06-04 · **Model:** SFT-v1 (`checkpoints/curated_merged`) · **N:** 100 · **Seed:** 42

## Why this exists

The original headline "**60% verifier pass-rate**" was not reproducible: the eval
(`tools/evaluate_passrate.py`) persisted only `id, compiled, verdict` — never the generated
programs — and generation was unseeded. The exact programs behind the 60% are gone.

The *machinery* survives, though: the model, the clang parser, the original `ebpf_validator`
binary, and `kcov_validator`. So we reconstructed under controlled, reproducible conditions and
decoupled the stages so programs are persisted and re-validatable forever:

```
STAGE 1  generate-only   SFT-v1 → assembly → {clang_hex, encoder_hex}   (seeded)
STAGE 2  store           data/reconstruction/sft_v1_20260604.jsonl
STAGE 3  validate        each hex → kcov_validator → verdict + PCs       → *.kcov.jsonl
```

- `tools/generate_bytecodes.py` — Stage 1 (dual-encode: clang path + pure-Python encoder).
- `tools/validate_bytecodes.py` — Stage 3 (both encodings through KCOV).

## Result (same 100 programs, via KCOV)

| encoding | reached kernel | ACCEPTED | % of generated | % of reached | unique PCs |
|---|---|---|---|---|---|
| **clang** (60%-era path) | 71/100 | 51 | 51% | 71.8% | 2,613 |
| **pure encoder** (RL-era path) | 99/100 | 73 | **73%** | 73.7% | 2,698 |

Verdict breakdown — clang: `ENCODE_FAIL 29, REJECTED 20, ACCEPTED 51`;
encoder: `ENCODE_FAIL 1, REJECTED 26, ACCEPTED 73`. **23 programs the encoder accepted, clang
could not even compile.**

## What it proves

1. **The 60% was deflated, not inflated.** The clang parser admits verifier register-state lines
   (`0: R1=ctx() R10=fp()`) as if they were instructions → `clang` errors → those programs are
   counted as compile failures. 23 of the 29 failures were *valid* programs. SFT-v1's true accept
   rate is **73%** (pure encoder), not 60%.

2. **"60% vs 19%" is two *models*, not two pipelines.** Running SFT-v1 through the *exact*
   pure-encoder + KCOV pipeline that produced the "19%" gives **73%**, not 19%. The 19% belongs to
   **SFT-v2** (`sft_retrain`), which deliberately trades validity for longer, deeper
   `[coverage=high][novelty=high]` programs. The drop is SFT-v1 → SFT-v2, *not* clang → encoder.

3. **Validity ≠ coverage — diversity is the wall.** 73 valid programs reach only ~2,700 unique
   PCs; clang's 51 valid programs reach ~2,613. 22 extra accepts buy almost no new coverage. The
   programs cluster structurally. Raising the accept rate is not the lever; **diversity is.**

## Methodology caveat

`clang_hex` and `encoder_hex` are *not byte-identical* for the same generated text (they differ
by ≈1 instruction) and occasionally disagree on the verifier verdict even when both load. "The
same program" is therefore encoder-dependent — a confound to state explicitly when comparing
accept rates across the two paths.

## Reproduce

```bash
pixi run python tools/generate_bytecodes.py --n 100 --seed 42   # → sft_v1_<date>.jsonl
./fuzzing/run_eval_vm.sh                                        # VM up (SSH 10022)
pixi run python tools/validate_bytecodes.py --in data/reconstruction/sft_v1_<date>.jsonl
```
