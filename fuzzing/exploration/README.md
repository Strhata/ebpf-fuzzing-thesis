# Exploration Scripts

Scripts from the early exploration phase (March 2026) before the Docker-based data collection pipeline was built.

These were used to understand buzzer's behaviour, test KCOV performance, and collect the crash logs in `results/`.

| Script | What it did |
|---|---|
| `start_swarm.sh` | Launched 3 bare QEMU VMs in parallel with auto-restart on crash. Produced the OOM entries in `results/crashes.csv`. |
| `run_node.sh` | Single VM runner with `-b` (KASAN only) / `-s` (KASAN+KCOV) mode selection. Used for manual testing. |
| `run_smart.sh` | KASAN+KCOV mode with buzzer metrics exposed on HTTP port. Used to observe KCOV overhead. |

**Not part of the data collection pipeline.** Data collection used `Dockerfile` + `entrypoint.sh` (single Docker container, single VM, virtio-9p corpus share).
