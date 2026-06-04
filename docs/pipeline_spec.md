# eBPF Fuzzing Pipeline — Formal Specification

> ⚠️ **SUPERSEDED by [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md).** This early spec frames success
> as *pass-rate*; the project's actual metric is **unique KCOV PCs from valid programs** (valid
> *and* diverse). Kept for historical reference only — do not cite its framing.

**Version:** 1.0  
**Status:** Draft — for advisor review  
**Scope:** ML-guided eBPF program generation and kernel validation

---

## Overview

The pipeline trains a language model to generate syntactically and semantically valid eBPF programs, then evaluates quality by submitting generated programs to the real Linux kernel BPF verifier inside an instrumented VM.

```
Dataset → Fine-tune LM → Generate programs → Compile → Kernel verify → Pass-rate metric
```

---

## Stage 0 — Prerequisites

| Asset | Location | Purpose |
|-------|----------|---------|
| Base model | `Qwen/Qwen2.5-Coder-1.5B` (HuggingFace) | Pre-trained code LM |
| Training dataset | `data/dataset_final_qwen.jsonl` | Curated eBPF verifier logs |
| Kernel image | `~/fuzzing_lab/bzImage_kasan_kcov` | KASAN+KCOV instrumented kernel |
| VM disk | `~/fuzzing_lab/trixie.img` | Debian root filesystem |
| Validator binary | `data/corpus/ebpf_validator` | Thin wrapper around kernel `bpf()` syscall |

---

## Stage 1 — Dataset Preparation

**Script:** `ml/train.py` (dataset loading section)

**Input:** `data/dataset_final_qwen.jsonl`  
Each record:
```json
{
  "kernel_version": "6.x.x",
  "is_valid": true | false,
  "verifier_log": "<eBPF assembly as emitted by kernel verifier>",
  "error_reason_clean": "<human-readable error, if invalid>"
}
```

**Transformation:** Each record → formatted prompt string:
```
# Valid program
Kernel: {kernel_version} | Status: VALID
### ASSEMBLY:
{verifier_log}<eos>

# Invalid program
Kernel: {kernel_version} | Status: INVALID | Error: {error_reason_clean}
### ASSEMBLY:
{verifier_log}<eos>
```

**Split:** 90% train / 10% validation (seed=42, random split)

**Tokenisation:** Qwen2.5-Coder tokenizer, `max_length=768`, truncated + padded. Padding tokens masked from loss (`label=-100`).

**Output:** HuggingFace `Dataset` objects (`train`, `val`)

---

## Stage 2 — Model Fine-tuning (QLoRA)

**Script:** `ml/train.py`  
**Command:** `pixi run python ml/train.py --run curated`

### Model configuration

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-Coder-1.5B` |
| Quantisation | BitsAndBytes NF4 4-bit, double quant, bf16 compute |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

### Training configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Train batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Eval batch size | 8 |
| Learning rate | 2e-4 |
| Eval strategy | Every 100 steps |
| Checkpoint strategy | Every 100 steps, keep best |
| Mixed precision | bf16 |
| Attention implementation | SDPA |

### Checkpointing

- Intermediate checkpoints: `checkpoints/curated_3ep/checkpoint-{step}/`
- `load_best_model_at_end=True` — final model = lowest eval loss checkpoint

**Output:** LoRA adapter saved to `checkpoints/curated_3ep/adattatore_ebpf_v1/`  
Contains: adapter weights (`adapter_model.safetensors`) + tokenizer files.  
**Note:** Base model weights are NOT saved — adapter only (~50MB vs ~1.5GB).

### Observed training behaviour

| Metric | Start | End (epoch 3) |
|--------|-------|--------------|
| Train loss | ~0.637 | ~0.556 |
| Eval loss | ~0.615 | ~0.563 |
| Grad norm | 0.2–0.5 | stable |
| Learning rate | 2e-4 | ~0 (cosine decay) |

Eval loss plateau observed from epoch ~2.0. No overfitting signal (eval tracks train loss).

---

## Stage 3 — VM Boot and Workspace Mount

**Script:** `fuzzing/run_eval_vm.sh`

Boots a QEMU VM with:
- Instrumented kernel (`KASAN + KCOV`) for crash detection
- `data/corpus/` shared into VM at `/mnt/corpus` via `virtio-9p`
- SSH accessible at `localhost:10022`

The `ebpf_validator` binary (pre-compiled, placed in `data/corpus/`) is the VM-side component that submits programs to the kernel verifier.

**Output:** Running VM, SSH reachable, `/mnt/corpus/ebpf_validator` executable inside VM.

---

## Stage 4 — Program Generation

**Script:** `tools/evaluate_passrate.py`  
**Function:** `generate_programs()`

**Input:** LoRA adapter path, N (number of programs to generate)

**Inference prompt (prefix only — no label):**
```
Kernel: unknown | Status: VALID
### ASSEMBLY:
```

The model completes from this prefix, generating eBPF assembly in verifier-log format.

### Generation parameters

| Parameter | Value |
|-----------|-------|
| Max new tokens | 400 |
| Sampling | do_sample=True |
| Temperature | 0.8 (configurable) |
| Stop condition | `<eos>` token |

**Model loading at inference:** Same BnB 4-bit config as training + LoRA adapter loaded via `PeftModel.from_pretrained()`.

**Output:** List of N raw assembly text strings.

---

## Stage 5 — Compilation

**Script:** `tools/evaluate_passrate.py`  
**Function:** `compile_to_hex()`

For each generated assembly string:

```
verifier log text
  → regex parse: extract instruction lines
  → write GNU assembler .s file
  → clang -target bpf -c prog.s -o prog.o     (ELF object)
  → llvm-objcopy -O binary --only-section=prog prog.o prog.bin  (raw bytecode)
  → prog.bin → hex string
```

**Failure modes:**
- Empty parse result → skip (model generated non-assembly text)
- `clang` error → skip (syntactically invalid assembly)
- `llvm-objcopy` error → skip
- Empty binary → skip

**Output:** List of hex strings (empty string `""` for failed compilations).

---

## Stage 6 — Kernel Verification

**Script:** `tools/evaluate_passrate.py`  
**Function:** `validate_batch()`

All valid hex strings written to a single file, SCPed to VM at `/tmp/eval_programs.txt`.

Inside VM, a shell loop runs:
```bash
while IFS= read -r line; do
  /mnt/corpus/ebpf_validator "$line"
done < /tmp/eval_programs.txt
```

`ebpf_validator` submits each hex-encoded BPF bytecode to the kernel via the `bpf()` syscall (BPF_PROG_LOAD). The kernel BPF verifier performs:
- Safety analysis (memory bounds, pointer arithmetic)
- Termination proof (DAG check, back-edge detection)
- Type checking

**Verdict per program:**

| Verdict | Meaning |
|---------|---------|
| `VERDICT: ACCEPTED` | Kernel verifier accepted — program is safe and loadable |
| `VERDICT: REJECTED` | Kernel verifier rejected — safety violation or invalid program |
| `VERDICT: ERROR` | Validator error (syscall failed, not a verifier decision) |

**Output:** List of verdict strings, one per compiled program.

---

## Stage 7 — Metrics and Results

**Script:** `tools/evaluate_passrate.py`

| Metric | Formula |
|--------|---------|
| Compile rate | `n_compiled / n_generated` |
| Pass rate | `n_ACCEPTED / n_generated` |

**Output files:**

| File | Content |
|------|---------|
| `results/passrate_{label}.csv` | Per-program: id, compiled (bool), verdict |
| `results/passrate_summary.csv` | Aggregate row appended per run |

---

## Data Flow Summary

```
data/dataset_final_qwen.jsonl
        │
        ▼ [Stage 1] tokenise + split
 HuggingFace Dataset (train / val)
        │
        ▼ [Stage 2] QLoRA fine-tune
 checkpoints/curated_3ep/adattatore_ebpf_v1/
        │
        ▼ [Stage 4] generate N programs
 List[assembly_text]   (N items)
        │
        ▼ [Stage 5] clang + llvm-objcopy
 List[hex_string]      (subset, compile failures → "")
        │   SCP
        ▼ [Stage 6] ebpf_validator inside VM
 List[verdict]
        │
        ▼ [Stage 7] aggregate
 results/passrate_summary.csv
```

---

## Known Limitations

1. **Generation is sequential** — programs generated one at a time; no batching. Bottleneck at scale.
2. **Eval overhead during training** — `eval_steps=100` triggers a 14-minute eval run every 100 training steps. Future runs should use `eval_strategy="epoch"`.
3. **BnB 4-bit at inference** — BitsAndBytes 4-bit quantisation is optimised for training memory, not inference speed. Post-training quantisation (AWQ) would improve throughput.
4. **No LoRA merge before inference** — adapter applied dynamically each forward pass. Merging into base weights (`merge_and_unload()`) eliminates this overhead.
5. **Single-sample pass-rate** — current metric counts accepted programs but does not measure coverage (new kernel code paths reached), which is the ultimate fuzzing objective.
