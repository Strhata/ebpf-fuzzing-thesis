# Thesis Roadmap — eBPF Fuzzing + ML
**Target:** July graduation (45-day window). Extend to October if needed.
**Decision date:** 2026-05-13

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
| Crash log analysis | Python script: parse 30 logs in `crash_logs/`, extract KASAN error type + top-3 stack frames → table for thesis. |
| RL scope | In thesis. GRPO algorithm. "Framework + preliminary results" framing if time runs short. |
| RL pipeline | generate assembly → `clang -target bpf -c` → `llvm-objcopy -O binary` (host) → SSH → `kcov_validator` (VM) → JSON reward |
| kcov_validator | New C tool inside VM. Input: raw hex bytes. Does: KCOV mmap setup → BPF_PROG_LOAD syscall → KCOV read → stdout JSON `{verdict, pcs:[]}` |
| Crash isolation | QEMU `savevm`/`loadvm` snapshots. Load clean snapshot before each program. Crash detected by SSH disconnect → reward assigned → restore snapshot. |
| Coverage tracking | Cumulative global set of KCOV PCs across entire RL training run. New PCs = reward. |
| Reward scale | crash=2.0 / new KCOV paths=1.0 / valid no new paths=0.4 / compiles+verifier rejects=0.1 / clang fails=0.0 |
| Reward config | `reward_scale` parameter in config file. Log reward tier breakdown per step via WandB. |

---

## Work Phases

### Phase 0 — Repo setup (Day 1)
- Create `ebpf-fuzzing-thesis` private GitHub repo
- Structure: `fuzzing/`, `ml/`, `tools/`, `docs/`
- Copy code (not models, not corpus, not VM images)
- `.gitignore`: `*.qcow2`, `*.img`, `modello_*/`, `shared_corpus/`, `.pixi/`
- Add professors as collaborators
- Upload final adapter + curated dataset to HuggingFace, add links to README

### Phase 1 — Crash log analysis (Day 1–2)
- Write `tools/classify_crashes.py`
- Input: `fuzzing_lab/crash_logs/*.txt`
- Output: CSV table (VM, timestamp, KASAN type, top-3 stack frames)
- Use table in thesis section: "buzzer triggered N unique crash categories"

### Phase 2 — SFT retrain + comparison (Day 2–10)
1. Build baseline dataset: random 27k from valid-byte-filtered corpus (no category cap)
   - Script: `ml/build_baseline_dataset.py`
2. Add WandB to `SFT_tesi.ipynb`:
   - `pip install wandb`
   - `wandb.init(project="ebpf-thesis", name="curated-3ep")`
   - Change `report_to="none"` → `report_to="wandb"`
   - Change `max_steps=1500` → `num_train_epochs=3`
3. Train curated model (overnight run 1)
4. Train baseline model, same config, `wandb name="baseline-3ep"` (overnight run 2)
5. Compare WandB loss curves — expected: curated model lower eval loss, faster convergence

### Phase 3 — Pass-rate evaluation pipeline (Day 8–12)
- Script: `tools/evaluate_passrate.py`
- Steps: load model → generate 100 programs → clang compile → llvm-objcopy → SSH into VM → run `ebpf_validator` in loop → count ACCETTATO → report pass-rate
- Run on both models, produce comparison table
- Expected: curated model higher pass-rate, especially on rare error categories

### Phase 4 — kcov_validator (Day 10–18)
- Write `tools/kcov_validator.c` inside VM (or cross-compile for eBPF target)
- Interface: `./kcov_validator <hex_bytes>`
- Output: `{"verdict":"ACCETTATO","pcs":[0x1234,0x5678,...]}`
- Logic:
  1. Open `/sys/kernel/debug/kcov`
  2. `ioctl(KCOV_INIT_TRACE3)` + mmap
  3. `ioctl(KCOV_ENABLE)`
  4. Call `bpf(BPF_PROG_LOAD, ...)` syscall with provided bytes
  5. `ioctl(KCOV_DISABLE)`
  6. Read PC array, output JSON
- Test: compare verdict output with existing `ebpf_validator` on same inputs

### Phase 5 — QEMU snapshot setup (Day 15–20)
- Boot VM with `bzImage_smart` (KCOV+KASAN enabled)
- Verify `kcov_validator` works inside VM
- Save clean snapshot: `(qemu) savevm clean_state`
- Write `tools/vm_manager.py`:
  - `restore_snapshot()`: send `loadvm clean_state` to QEMU monitor socket
  - `run_program(hex_bytes)`: SSH → run `kcov_validator` → parse JSON → detect crash (SSH timeout/disconnect)
  - `compute_reward(verdict, new_pcs, global_pc_set)`: implement reward scale

### Phase 6 — RL training loop (Day 18–35)
- Algorithm: GRPO (Group Relative Policy Optimization)
- Notebook: `ml/rl_grpo.ipynb`
- Loop:
  1. Sample prompt (target: VALID / specific error class)
  2. Generate G=8 completions (model in 8-bit for inference speed)
  3. For each completion: clang → objcopy → `vm_manager.run_program()` → reward
  4. GRPO update on reward-ranked group
  5. Log reward tier counts to WandB
  6. Persist global KCOV PC set to disk (checkpoint)
- Early stopping: if pass-rate plateaus for 5 consecutive eval rounds

### Phase 7 — Results + thesis writing (Day 30–45)
- Thesis figures:
  - WandB loss curves (curated vs baseline)
  - Pass-rate table (curated vs baseline, per error category)
  - Crash classification table (30 logs)
  - RL reward progression over training steps
  - KCOV new-paths-per-step graph
- `[DA COMPLETARE]` sections in `tesi_recap.md`:
  - Section 6: write motivation for LLM pivot
  - Section 7: fill pass-rate table with actual numbers from `risultati_checkpoint-*.txt`

---

## Repo Structure

```
ebpf-fuzzing-thesis/
├── README.md                    # Project overview, HuggingFace links
├── fuzzing/                     # Phase 1-2 infrastructure
│   ├── create-image.sh
│   ├── start_swarm.sh
│   ├── run_node.sh / run_smart.sh
│   ├── Dockerfile + entrypoint.sh
│   └── crash_logs/              # 30 crash dumps
├── ml/                          # ML work
│   ├── data_analisys.ipynb
│   ├── SFT_tesi.ipynb           # Retrained with WandB + 3 epochs
│   ├── rl_grpo.ipynb            # RL training
│   └── build_baseline_dataset.py
├── tools/                       # Pipeline scripts
│   ├── classify_crashes.py      # Phase 1
│   ├── evaluate_passrate.py     # Phase 3
│   ├── kcov_validator.c         # Phase 4 (compiled inside VM)
│   └── vm_manager.py            # Phase 5 QEMU/SSH wrapper
└── docs/
    └── tesi_recap.md            # Updated recap
```

---

## Risk Register

| Risk | Mitigation |
|---|---|
| RL doesn't converge in 45 days | Report framework + preliminary results; frame as future work |
| KCOV interface differs on Linux 6.8.0 | Check kernel headers before writing kcov_validator |
| QEMU savevm slow on qcow2 | Benchmark first; fallback: re-boot from base image (slower but same isolation) |
| Baseline model performs equally to curated | Still publishable: shows robustness of balancing; adjust claim in thesis |
| GPU OOM during GRPO | Use 8-bit inference for generation, reduce G from 8 to 4 |
