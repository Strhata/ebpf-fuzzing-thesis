# eBPF Fuzzing + LLM Fine-Tuning — Thesis

Research project exploring LLM-guided eBPF program generation, evaluated via kernel verifier pass-rate.

## What this is

End-to-end pipeline:
1. **Data collection** — modified [buzzer](https://github.com/google/buzzer) (Google's eBPF fuzzer) to dump generated programs + verifier outcomes as JSONL at the kernel FFI boundary. Collected ~2M entries via a Dockerized single-VM setup.
2. **Dataset curation** — analyzed error class distribution, balanced to 27k samples (cap 2000/class) to avoid domination by top error category.
3. **Fine-tuning** — QLoRA fine-tuned Qwen2.5-Coder-1.5B on verifier-log assembly format. Model learns to generate valid eBPF programs given a target status/error class.
4. **Evaluation** — automated pipeline: generate → parse assembly → compile with clang BPF target → validate in kernel via `ebpf_validator` → pass-rate.

## Structure

```
fuzzing/        # Buzzer fork (modified ffi.go for data extraction), VM scripts, Docker setup
ml/             # Training script, dataset builder, notebooks
tools/          # evaluate_passrate.py, ebpf_validator (Go), classify_crashes.py
docs/           # Thesis notes, training log, roadmap
data/           # Curated dataset (dataset_final_qwen.jsonl) — large files on HuggingFace
checkpoints/    # Model adapters — large files on HuggingFace
```

## Models & Dataset

| Artifact | Location |
|---|---|
| Phase 1 adapter (`sft_fase1/adattatore_ebpf_v1`) | HuggingFace — link TBD |
| Curated training dataset (`dataset_final_qwen.jsonl`) | HuggingFace — link TBD |

## Quickstart (new machine)

```bash
make setup        # install deps, download dataset + checkpoint, build kernels + tools
```

See `make help` for individual build steps.

## Training

```bash
# Resume / run curated SFT training (3 epochs, auto-resumes from latest checkpoint)
pixi run python ml/train.py --run curated
```

See `docs/training_log.md` for current training state and phase history.

## Evaluation

```bash
# 1. Build ebpf_validator binary
make build-validator

# 2. Start evaluation VM (KASAN + KCOV kernel)
./fuzzing/run_eval_vm.sh

# 3. Run pass-rate evaluation
pixi run python tools/evaluate_passrate.py \
    --adapter checkpoints/curated_3ep/adattatore_ebpf_v1 \
    --label curated-3ep \
    --n 100
```

Results are saved to `results/passrate_*.csv`.
