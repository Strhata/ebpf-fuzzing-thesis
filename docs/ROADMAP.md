# Thesis Roadmap — eBPF Fuzzing + ML
**Target:** July graduation (45-day window). Extend to October if needed.
**Last updated:** 2026-05-17

---

## Decisions Log

| Topic | Decision |
|---|---|
| Repo visibility | Private GitHub, professors as collaborators. Publish clean public version post-defense. |
| Repo layout | Single monorepo `ebpf-fuzzing-thesis` with `fuzzing/` and `ml/` top-level dirs |
| Large artifacts | Models + corpus on HuggingFace. Only `adattatore_ebpf_v1` (final adapter) committed to git. |
| Model format | Assembly (verifier-log format) only. Hex approach documented as abandoned. |
| Comparison design | Two models, same architecture (Qwen2.5-Coder-1.5B QLoRA r=16), same format, same size (27k). Variable: curation only. |
| Baseline dataset | Same valid-byte filter applied, no per-category cap. Random 27k sample → dominated by top error class (~34% `math between map_value pointer and <NUM>`). Proves balancing matters. |
| Training duration | 3 epochs (`num_train_epochs=3`), replacing `max_steps=1500` which only covered 48% of data. |
| WandB | `report_to="wandb"`, loss curves during training only. Pass-rate evaluated separately post-training. |
| Pass-rate eval | Standalone script: generate 100 programs → clang → llvm-objcopy → SSH → validator → table. |
| Crash log analysis | Python script: parse logs in `results/`, extract KASAN error type + top-3 stack frames → table for thesis. |
| KCOV mode | `KCOV_TRACE_PC` (value=0). Confirmed 2026-05-17 via `linux/include/uapi/linux/kcov.h`. Flat uint64 PC array, 1 word per entry. |
| TRL version | 0.14.0 (pixi-managed, `<0.15` pin). `GRPOTrainer` importable with workaround: patch `trl.import_utils.is_vllm_available = lambda: False` before import, and set `use_vllm=False` in `GRPOConfig`. Root cause: `is_vllm_available()` returns tuple `(False, None)` which is truthy in Python. |
| Register annotation stripping | Baseline experiment: strip `; r0_w=0` style suffixes from verifier log during training. Tests whether removing register state noise improves pass-rate. Anecdotal hypothesis only — no prior data. |
| RL scope | **Out of scope for this thesis.** GRPO + kcov_validator + QEMU snapshots deferred to future work. See *Future Work* section below. |

---

## Work Phases (In Scope)

### Phase 0 — Repo setup ✅
- Created `ebpf-fuzzing-thesis` private GitHub repo
- Structure: `fuzzing/`, `ml/`, `tools/`, `docs/`
- Upload final adapter + curated dataset to HuggingFace, add links to README

### Phase 1 — Data collection ✅
- Modified `fuzzing/buzzer/pkg/units/ffi.go` (~50 lines) to intercept `ValidateEbpfProgram`
- Dumps JSONL at kernel FFI boundary: `bytecode_hex`, `verifier_log`, `is_valid`, `error_line`, `error_reason`
- Dockerized single-VM setup (KASAN-only kernel, virtio-9p corpus share)
- Collected ~2M entries
- Crash logs from early swarm exploration in `results/`; crash analysis in `tools/classify_crashes.py`

### Phase 2 — Dataset curation + SFT ✅ (training in progress)
- Analyzed error class distribution across ~2M raw entries
- Balanced to 27k samples (cap 2000/class) to avoid top-error-class domination
- Baseline dataset: same 27k, no cap, dominated by top class — control for curation value
- Format: assembly (verifier-log style) only — hex abandoned (LLMs can't generate valid hex reliably)
- QLoRA fine-tune: Qwen2.5-Coder-1.5B, rank-16, Q/K/V/O projections, 3 epochs
- See `docs/training_log.md` for current checkpoint state

### Phase 3 — Pass-rate evaluation (pending training completion)
- Script: `tools/evaluate_passrate.py`
- Pipeline: load adapter → generate 100 programs → clang BPF compile → llvm-objcopy → SSH to VM → `ebpf_validator` loop → CSV output
- Evaluation VM: KASAN+KCOV kernel (`bzImage_kasan_kcov`) via `./fuzzing/run_eval_vm.sh`
- Run on curated model and baseline model, produce comparison table
- Expected: curated model higher pass-rate, especially on rare error categories

---

## Future Work (Out of Scope)

These were considered and deferred. Documented here so the thesis can reference them honestly.

- **RL with GRPO**: generate assembly → compile → run in VM → coverage-aware reward → policy update. Infrastructure exists (KCOV-enabled kernel, evaluation VM) but training loop not implemented.
- **kcov_validator**: C tool inside VM that wraps `bpf(BPF_PROG_LOAD)` with KCOV mmap and returns `{verdict, pcs:[]}` JSON. Would replace `ebpf_validator` for RL reward computation.
- **QEMU snapshot isolation**: `savevm`/`loadvm` for crash recovery between RL steps. Needed for safe RL training but not required for SFT evaluation.
- **Register annotation stripping experiment**: strip `; r0_w=0` style suffixes from verifier log and test effect on pass-rate. Low-cost experiment if time allows after Phase 3.

---

## Repo Structure (Current)

```
ebpf-fuzzing-thesis/
├── README.md
├── fuzzing/
│   ├── buzzer/                      # Modified buzzer fork (ffi.go data extraction)
│   ├── exploration/                 # Early swarm/KCOV exploration scripts (historical)
│   ├── Dockerfile + entrypoint.sh  # Data collection container
│   ├── run_eval_vm.sh              # Evaluation VM launcher (KASAN+KCOV)
│   ├── bzImage_kasan               # KASAN-only kernel (data collection)
│   └── bzImage_kasan_kcov          # KASAN+KCOV kernel (evaluation + future RL)
├── ml/
│   ├── train.py                    # SFT training script (curated + baseline runs)
│   └── build_baseline_dataset.py
├── tools/
│   ├── classify_crashes.py
│   ├── evaluate_passrate.py        # Phase 3 pass-rate pipeline
│   └── ebpf_validator/             # Go binary for kernel verifier validation
├── data/
│   └── dataset_final_qwen.jsonl    # Curated 27k training set (also on HuggingFace)
├── checkpoints/
│   ├── sft_fase1/                  # Phase 1 adapter (warm-start base)
│   └── curated_3ep/                # Phase 2 curated adapter (in training)
├── results/                        # Crash logs, pass-rate CSVs
└── docs/
    ├── training_log.md             # Checkpoint state tracking
    └── tesi_recap.md               # Thesis chapter notes
```

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Baseline model performs equally to curated | Still publishable: shows robustness of balancing; adjust claim in thesis |
| Pass-rate too low on both models | Report absolute numbers honestly; frame as baseline for future RL work |
| Training diverges in Phase 2 warm start (lost optimizer state) | Monitor eval loss; if unstable, reduce LR and re-run from sft_fase1 adapter |
| July deadline too tight | Phase 3 is the minimum deliverable; RL section becomes "future work" in thesis |
