# Thesis Recap — eBPF Verifier Fuzzing

> ⚠️ **SUPERSEDED by [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md)** as the project's source of truth.
> Kept as a personal narrative of the early learning curve; numbers and framing here predate the
> reconstruction and the valid-and-diverse metric.

> Document reconstructed from files in `/home/stefano-u/fuzzing_lab/` and `/home/stefano-u/fuzzing_ml_env/`
> and from annotations in `note.txt`. Used as a personal recap and as a reference for writing the thesis.

---

## 1. Initial Goal

The relatore asked to fuzz the **Linux kernel eBPF verifier**.
This is my first research project: part of the narrative below describes the learning curve —
choices I would make differently in hindsight, and pivots motivated by understanding what
I was actually working with along the way.

---

## 2. Buzzer — what it is and how it was used

- **What it is:** [buzzer](https://github.com/google/buzzer) is a **standalone** eBPF fuzzer
  written in Go by Google, with its own strategies (`--strategy=pointer_arithmetic`,
  `--strategy=coverage_based`). It is not an AFL wrapper.
- **Smart mode and KCOV:** buzzer's `coverage_based` mode uses KCOV, but only
  to collect human-readable metrics exposed via an HTTP server — **not** as a
  feedback signal to drive mutation. It is not a coverage-guided loop in the AFL sense.
- **Key discovery:** this limitation (KCOV used only for display, not for feedback)
  was the trigger for the pivot to the LLM+GRPO approach: if buzzer does not use
  coverage as a mutation signal, building an RL loop with KCOV as the reward is
  a genuinely new contribution.
- **Role in the project:** buzzer was used as a **data generator** — modified
  in `ffi.go` to dump every `(bytecode_hex, verifier_log)` pair — not as the
  primary fuzzer.

---

## 3. Phase 1 — VM/kernel infrastructure (1–3 March)

- [`create-image.sh`](fuzzing_lab/create-image.sh) adapted from syzkaller → Debian
  images `bullseye.img` and `trixie.img` (~2 GB each) with SSH keys for passwordless root.
- **Linux 6.8.0** sources in `fuzzing_lab/linux/`, compiled in three variants:
  - `bzImage` — standard kernel
  - `bzImage_kasan` — **KCOV disabled**, fuzzer throughput baseline
  - `bzImage_kasan_kcov` — **KCOV + KASAN + UBSAN** enabled, coverage-guided
- [`note.txt`](fuzzing_lab/note.txt) contains the QEMU command lines tried,
  including `virtfs` 9p mounting for sharing the kernel and corpus with the guest.

---

## 4. Phase 2 — Buzzer and swarm runs (5–10 March)

Runner scripts in `fuzzing_lab/`:

| Script | Role |
|--------|------|
| [`start_swarm.sh`](fuzzing_lab/start_swarm.sh) | Launches 3 QEMU VMs in parallel with auto-restart on crash |
| [`run_node.sh`](fuzzing_lab/run_node.sh) | Runner with `-b` (blind) / `-s` (smart) flag; copies `buzzer` and `vmlinux` into the guest |
| [`run_smart.sh`](fuzzing_lab/run_smart.sh) | Smart mode with metrics UI on port `8080 + ID` |
| [`run_fake.sh`](fuzzing_lab/run_fake.sh) | Sanity check with a dummy `vmlinux`, to isolate coverage problems |

Profiling attempts (to find where cycles were being spent):

- [`diagnose1.sh`](fuzzing_lab/diagnose1.sh), [`diagnose2.sh`](fuzzing_lab/diagnose2.sh),
  [`diagnose3.sh`](fuzzing_lab/diagnose3.sh), [`diagnose_bottleneck.sh`](fuzzing_lab/diagnose_bottleneck.sh)
  → `strace`, `perf`, `GODEBUG=gctrace,schedtrace`.
- Output collected in `fuzzing_lab/diagnostic_data/`.

Results collected:

- **`crash_logs/`** — 30 kernel panic / KASAN dumps (4–5 March), spread
  across the 4 active VMs (`vm1…vm4_log.txt`).

---

## 5. Phase 3 — Containerisation (26–29 March)

To make the setup reproducible and scalable:

- [`Dockerfile`](fuzzing_lab/Dockerfile) + [`entrypoint.sh`](fuzzing_lab/entrypoint.sh) →
  container that launches **nested** QEMU (`--device /dev/kvm`), waits for SSH, then runs
  `/mnt/corpus/buzzer --strategy=pointer_arithmetic`.
- Corpus shared host ↔ guest via **virtio-9p**.
- The container is `fuzzer_node_1` (name referenced in `rotate_log.txt` and `note.txt`).

---

## 6. Phase 4 — Pivot: eBPF program generation with an LLM (27 March → 16 April)

All ML work lives in `fuzzing_ml_env/`.

### Pivot motivation

The decisive discovery was understanding how buzzer uses KCOV. The `coverage_based` mode
exposes coverage metrics via an internal HTTP server — the data is human-readable but
**is not used to guide mutation**. Buzzer generates programs based on hard-coded Go
strategies, not in response to which kernel code it is covering.

This revealed a real gap: no existing system combined LLM-guided eBPF program generation
with KCOV as an RL reward signal. Using buzzer as a data collector (2M entries of
`bytecode_hex + verifier_log`) and then training an LLM with GRPO and a KCOV-based
reward was a novel approach worth investigating.

### Training setup

- **Base model:** Qwen2.5-Coder-1.5B.
- **Technique:** **QLoRA** (4-bit NF4), rank-16 on `Q/K/V/O`; ~0.14% parameters
  trained (≈ 2.1 M out of 1.5 B).
- **Hyperparameters:** effective batch 8 (gradient accumulation), `max_seq_len` 768,
  learning rate 1–2e-4, AdamW 8-bit.
- **Hardware:** RTX 4070 Laptop, 8.59 GB VRAM, compute 8.9, BF16 ok
  (verified in [`gpu_check.ipynb`](fuzzing_ml_env/gpu_check.ipynb)).
- **Environment:** [`pixi.toml`](fuzzing_ml_env/pixi.toml) — Python 3.11, PyTorch 2.5.1 + CUDA 12.1,
  Transformers ≥ 5.4, PEFT 0.18.1, BitsAndBytes, SentencePiece.

### Notebooks

| Notebook | Date | Contents |
|----------|------|----------|
| [`Untitled.ipynb`](fuzzing_ml_env/Untitled.ipynb) | 29 Mar | Scratch pad. Universal prompt: `Status: VALID \| Complexity: N insns` for valid, `Status: INVALID \| Error: … \| Instr: …` for invalid. 2000 steps, transfer learning from `modello_ebpf_produzione/checkpoint-1000`. |
| [`data_analisys.ipynb`](fuzzing_ml_env/data_analisys.ipynb) | 10 Apr | Analysis of 73 syzkaller dumps (27 Mar → 9 Apr): 13 153 valid + 14 361 invalid. Verifier error normalisation (mask numbers and addresses). Cap 2 000 examples per error class → `dataset_final_qwen.jsonl` (27 514 examples). |
| [`qwen_fine_tuning.ipynb`](fuzzing_ml_env/qwen_fine_tuning.ipynb) | 16 Apr | Intermediate fine-tuning version (fp16). |
| [`SFT_tesi.ipynb`](fuzzing_ml_env/SFT_tesi.ipynb) | 16 Apr | Thesis version: BF16 + FlashAttention-2, 1500 steps, 90/10 split, best-on-val checkpoint. |
| [`gpu_check.ipynb`](fuzzing_ml_env/gpu_check.ipynb) | 16 Apr | Hardware diagnostics. |

### Model directories (chronological)

| Directory | Role |
|-----------|------|
| [`modello_ebpf_lora/`](fuzzing_ml_env/modello_ebpf_lora/) | First LoRA attempts (checkpoints 100–500) |
| [`modello_ebpf_finetuned/`](fuzzing_ml_env/modello_ebpf_finetuned/) | Initial SFT baseline |
| [`modello_ebpf_fase2/`](fuzzing_ml_env/modello_ebpf_fase2/) | "Phase 2" run |
| [`modello_ebpf_produzione/`](fuzzing_ml_env/modello_ebpf_produzione/) | `checkpoint-1000` used as base for transfer learning |
| [`modello_ebpf_3000/`](fuzzing_ml_env/modello_ebpf_3000/) | Full 2000-step run, checkpoints 500–2000 |
| [`modello_ebpf_definitivo/`](fuzzing_ml_env/modello_ebpf_definitivo/) | Final LoRA adapter (~8.7 MB) |
| [`adattatore_ebpf_finale/`](fuzzing_ml_env/adattatore_ebpf_finale/) | Copy / finalisation of the adapter |

### Generated outputs

Files [`risultati_checkpoint-*.txt`](fuzzing_ml_env/) contain eBPF bytecode
generated by the various checkpoints, in hexadecimal format, terminated by
`9500000000000000` (`BPF_EXIT` instruction).

---

## 7. Phase 5 — Closed-loop evaluation (documented in `note.txt`)

Pipeline built by hand to measure generation quality:

1. Generate bytecode with the fine-tuned checkpoint (from notebooks above).
2. Inside the QEMU VM, mount the corpus via 9p at `/mnt/corpus`.
3. Feed the `risultati_checkpoint-*.txt` file line by line to
   **`ebpf_validator`** (written by me), which calls the verifier and prints
   `VERDICT: ACCEPTED` / `VERDICT: REJECTED`.
4. Count accepted programs and compute pass-rate per checkpoint.
5. For the first 25 programs, full verifier log dump to classify rejection causes.

The full loop is in `note.txt` (lines 56–87), including the `docker exec`
version that invokes `ebpf_validator` on container `fuzzer_node_1`.

### Measured pass-rates (final results)

Formal evaluation on `curated_merged` (final SFT model, 3 complete epochs)
vs `zero-shot` (Qwen2.5-Coder-1.5B base with no fine-tuning). N=100 programs each.

| Model | N | Compiled | ACCEPTED | Compile rate | Pass-rate |
|-------|---|----------|-----------|--------------|-----------|
| `curated-merged` (SFT) | 100 | 73 | **60** | 73.0% | **60.0%** |
| `zero-shot` (base) | 100 | 1 | 1 | 1.0% | **1.0%** |

Source: `results/passrate_summary.csv`, logs in `results/passrate_run_curated.log` and
`results/passrate_run_zeroshot.log`.

**Key result:** fine-tuning raises the pass-rate from ~0% (base, functionally zero)
to 60%, demonstrating that the model learned BPF verifier syntax from the training data.

---

## 8. Phase 6 — Corpus rotation (8–21 April)

- [`rotate_dataset.sh`](fuzzing_lab/rotate_dataset.sh) — pauses `fuzzer_node_1`,
  compresses the JSONL with gzip + timestamp, restarts the container.
- The container *can* crash (that is the point of kernel fuzzing): hence the
  automatic restart. Lines `[!] fuzzer_node_1 is not running` in
  [`rotate_log.txt`](fuzzing_lab/rotate_log.txt) are the rotation script encountering
  the node between a crash and restart — expected behaviour, not a bug.
- [`shared_corpus/`](fuzzing_lab/shared_corpus/) — ~13 GB compressed,
  `dataset_syzkaller_347…405.jsonl.gz` (8–9 April), ~600k rows each, field
  `bytecode_hex`.

---

## 9. What was NOT done

- **RL run 2 not executed.** The pipeline with depth-based verdict-blind reward was
  implemented and validated locally, but no full training run was executed.
  Documented as future work.
- **No quantitative comparison** between buzzer runs and bytecode generated by
  fine-tuned Qwen.

---

## 10. Next steps (future work)

1. **RL run 2** with depth-based verdict-blind reward (already implemented in `ml/reward.py`)
   and a neutral prompt (without `Status: VALID`) — would isolate the prompt bias variable
   from the KCOV bug.
2. **SFT with a different prompt** — train toward programs that stress the verifier
   instead of generating valid programs; evaluate whether this changes RL behaviour.
3. Systematic comparison between coverage-guided buzzer and the fine-tuned model
   at equal wall-clock time, measured in unique verifier PCs.

---

## Appendix A — Key file timeline

| Date | File / directory | Description |
|------|-----------------|-------------|
| 1 Mar | `bullseye/`, `bullseye.id_rsa` | Debian bullseye image + SSH key |
| 1 Mar | `trixie/`, `trixie.id_rsa` | Debian trixie image + SSH key |
| 1 Mar | `create-image.sh` | Image builder (adapted from syzkaller) |
| 3 Mar | `buzzer_bin` | Compiled buzzer binary |
| 4–5 Mar | `crash_logs/` | 30 kernel panic / KASAN dumps |
| 5 Mar | `start_swarm.sh`, `run_node.sh` | 3-VM orchestration + node runner |
| 6 Mar | `run_smart.sh`, `run_fake.sh` | Smart mode / sanity check |
| 7–10 Mar | `diagnose*.sh`, `diagnostic_data/` | Fuzzer profiling |
| 10 Mar | `linux/` (last build) | Linux 6.8.0 with 3 `bzImage` variants (standard/blind/smart) |
| 26 Mar | `bullseye.img` | Final bullseye image |
| 27 Mar | `Dockerfile`, `entrypoint.sh` | Containerisation |
| 28 Mar | `pixi.toml`, first LoRA runs | ML environment setup + `modello_ebpf_lora/` |
| 28 Mar | `modello_ebpf_fase2/`, `modello_ebpf_produzione/` | Phase 2 + production checkpoint |
| 28–29 Mar | `risultati_checkpoint-500/1000/2000/3000.txt` | Checkpoint generations |
| 29 Mar | `Untitled.ipynb`, `trixie.img` | Training scratch pad + final trixie image |
| 8 Apr | `rotate_dataset.sh`, `note.txt` (updated) | Corpus rotation + validation pipeline |
| 8–9 Apr | `shared_corpus/dataset_syzkaller_347…405.jsonl.gz` | ~13 GB rotated corpus |
| 10 Apr | `data_analisys.ipynb` | Final dataset: 27 514 examples |
| 16 Apr | `SFT_tesi.ipynb`, `qwen_fine_tuning.ipynb`, `gpu_check.ipynb` | Thesis training version |
| 21 Apr | `rotate_log.txt` (last line) | Last recorded corpus rotation run |

---

## Appendix B — Key commands from `note.txt`

**QEMU launch with kernel source sharing via 9p**
```bash
qemu-system-x86_64 \
    -m 4G -smp 4 \
    -kernel $HOME/tesi/linux/arch/x86/boot/bzImage \
    -append "console=ttyS0 root=/dev/sda earlyprintk=serial net.ifnames=0" \
    -drive file=$HOME/vm_image/trixie.img,format=raw \
    -net user,hostfwd=tcp::10022-:22,hostfwd=tcp::10250-:10250 \
    -net nic,model=e1000 \
    -display none -pidfile vm.pid \
    -virtfs local,path=$HOME/tesi/linux,mount_tag=host_linux,security_model=none,id=linux_src \
    -daemonize
```

**Upload `vmlinux` and `buzzer` to the guest**
```bash
scp -i trixie.id_rsa -P 10022 -o "StrictHostKeyChecking no" \
    $HOME/tesi/linux/vmlinux root@localhost:/root/vmlinux
scp -i trixie.id_rsa -P 10022 -o "StrictHostKeyChecking no" \
    $HOME/tesi/buzzer/bazel-bin/buzzer_/buzzer root@localhost:/root/buzzer
```

**QEMU with `bzImage_kasan` and shared corpus**
```bash
qemu-system-x86_64 \
    -m 2G -smp 2 \
    -kernel linux/arch/x86/boot/bzImage_kasan \
    -append "console=ttyS0 root=/dev/sda rw earlyprintk=serial net.ifnames=0" \
    -drive file=trixie.img,format=raw \
    -nographic \
    -fsdev local,security_model=none,id=fsdev_corpus,path=./shared_corpus \
    -device virtio-9p-pci,id=fs_corpus,fsdev=fsdev_corpus,mount_tag=corpus_share

# inside the guest:
mkdir -p /mnt/corpus
mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 corpus_share /mnt/corpus
```

**Bytecode validation loop**
```bash
cd /mnt/corpus
counter=1
accepted=0
while read -r line; do
    res=$(./ebpf_validator "$line" | grep "VERDICT")
    if [[ $res == *"ACCEPTED"* ]]; then
        accepted=$((accepted+1))
        echo "[+] Prog $counter: OK"
    else
        echo "[-] Prog $counter: FAIL"
    fi
    counter=$((counter+1))
done < risultati_checkpoint-1000-1.txt

echo " FINAL RESULT: $accepted out of $((counter-1))"
echo " PASS RATE: $((accepted * 100 / (counter-1)))%"
```

**Single bytecode validation via `docker exec`**
```bash
docker exec -it fuzzer_node_1 \
    ssh -p 10022 -i trixie.id_rsa -o "StrictHostKeyChecking no" root@127.0.0.1 \
    "/mnt/corpus/ebpf_validator <hex_bytecode>"
```

**Run the fuzzer container**
```bash
docker run -d \
    --name fuzzer_node_1 \
    --device /dev/kvm \
    -v /home/stefano-u/fuzzing_lab/shared_corpus:/shared_corpus \
    ebpffuzzer:v1
```
