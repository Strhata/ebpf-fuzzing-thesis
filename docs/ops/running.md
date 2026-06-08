# Running the pipelines (how-to)

Current run instructions. Overwrite in place when commands change. Concepts/why live in
[`../FACTS.md`](../FACTS.md) and [`../DECISIONS.md`](../DECISIONS.md).

## Prerequisites
```bash
pixi install
./fuzzing/run_eval_vm.sh           # KCOV+KASAN eval VM, SSH :10022 (key ~/fuzzing_lab/trixie.id_rsa)
make build-validator               # builds + deploys tools/kcov_validator
```

## Tests (no VM)
```bash
pixi run pytest tests/
```

## SFT training
```bash
pixi run python ml/train.py --dataset data/dataset_final_qwen.jsonl \
  --output-dir checkpoints/<name> --num-train-epochs 3
```

## Merge LoRA → fp16 (required before RL/inference; never 4-bit a merge — see D9)
```bash
pixi run python tools/benchmark.py prepare --model checkpoints/<name>/checkpoint-NNNN --strategy merged_fp16
```

## Benchmark (model comparison)
```bash
pixi run python tools/benchmark.py run --models <model_id> --n 20
pixi run python tools/benchmark.py report      # → benchmarks/runs/<ts>_{run.json,report.md}, summary.csv
```

## Diversity / saturation experiment
```bash
# generate (Colab A100): ml/diversity_generate_colab.ipynb  → candidates JSONL
pixi run python tools/diversity_sample.py validate \
  --candidates benchmarks/diversity/candidates/<name>.jsonl \
  --out benchmarks/diversity/<name>.json
```

## RL training (GRPO)
Local reward server + tunnel (see `ngrok.md`), training on Colab (see `colab.md`):
```bash
# local host: reward server (weights via RL_W_* env — see FACTS §4)
RL_W_VALID=1.0 RL_W_NOVELTY=1.0 RL_W_GLOBAL=2.0 RL_W_REJECT_MAX=0.3 REWARD_API_KEY=<key> \
  pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000
# Colab: ml/train_grpo_colab.ipynb → Run All (Cell 3b merges SFT-2 adapter → fp16)
```
Restart the reward server after any `ml/reward.py` change (it imports once).

## "60 %" reconstruction (seeded, reproducible)
```bash
pixi run python tools/generate_bytecodes.py --n 100 --seed 42
pixi run python tools/validate_bytecodes.py --in data/reconstruction/sft_v1_<date>.jsonl
```

## Coverage race (buzzer vs LLM)
```bash
# in VM: /mnt/corpus/buzzer -strategy=coverage_based -max_programs=500 -coverage_log=/mnt/corpus/buzzer_coverage.csv
pixi run python tools/coverage_race.py --max-programs 500
pixi run python tools/plot_coverage_race.py     # → results/coverage_race_*.png
```
