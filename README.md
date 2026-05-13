# eBPF Fuzzing + LLM Fine-Tuning — Thesis

Research project exploring LLM-guided eBPF program generation, evaluated via kernel verifier pass-rate and coverage-guided RL feedback.

## Structure

```
fuzzing/        # Infrastructure: QEMU VMs, Docker swarm, crash logs
ml/             # Notebooks: corpus analysis, SFT training, RL loop
tools/          # Pipeline scripts: crash classifier, pass-rate eval, kcov validator
docs/           # Thesis notes and roadmap
```

## Models & Dataset

| Artifact | Location |
|---|---|
| Final adapter (`adattatore_ebpf_v1`) | HuggingFace — link TBD |
| Curated training corpus | HuggingFace — link TBD |

## Setup

```bash
# Build fuzzer Docker image
docker build -t ebpffuzzer:v1 ./fuzzing

# Launch 3-VM fuzzing swarm
./fuzzing/start_swarm.sh 3

# Run a single node manually
./fuzzing/run_smart.sh 1
```

## Results snapshot

| Checkpoint | Pass rate |
|---|---|
| SFT 500 steps | see `shared_corpus/risultati_checkpoint-500.txt` |
| SFT 1000 steps | see `shared_corpus/risultati_checkpoint-1000.txt` |
| SFT 2000 steps | see `shared_corpus/risultati_checkpoint-2000.txt` |
| SFT 3000 steps | see `shared_corpus/risultati_checkpoint_3000.txt` |

Models and corpus are on HuggingFace (too large for git).
