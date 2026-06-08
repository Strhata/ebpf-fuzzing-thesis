#!/usr/bin/env python3
"""
rl_grpo.py — GRPO RL training for eBPF program generation.

Starts from the merged fp16 SFT model (default checkpoints/sft_retrain/checkpoint-1500-merged)
and fine-tunes with GRPO using KCOV-based rewards from the eval VM. The merged model is loaded
in bf16 (NOT 4-bit — 4-bit quantising a merged fine-tune collapses it; see --load-4bit).

Do NOT use rl_grpo_v2, rl_grpo_v3, or rl_grpo_cmp* checkpoints as starting points.

Usage:
    pixi run python ml/rl_grpo.py                              # full training (local reward)
    pixi run python ml/rl_grpo.py --smoke-test                 # 10 steps, mock reward
    pixi run python ml/rl_grpo.py --resume                     # auto-resume from last checkpoint
    pixi run python ml/rl_grpo.py --remote-reward-url URL      # remote reward server (Colab)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from statistics import pstdev

_REPO_ROOT_GRPO = Path(__file__).parent.parent
_GRPO_DEBUG_LOG = _REPO_ROOT_GRPO / "results" / "grpo_completions.log"
# results/ is gitignored → absent on a fresh clone (e.g. Colab). Create it before logging.
_GRPO_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(_GRPO_DEBUG_LOG),
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)
_log_grpo = logging.getLogger("grpo")

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ml"))

# TRL 0.14 bug: is_vllm_available() returns tuple (False, None) which is truthy.
# Monkey-patch before importing GRPOTrainer to prevent a spurious vllm import.
import trl.import_utils as _trl_utils
_trl_utils.is_vllm_available = lambda: False

import re
import datetime

import requests
import torch
from transformers import BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import wandb
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer import grpo_trainer as _grpo_mod

import reward as rw

# Root cause fix: transformers 5.x + gradient_checkpointing + train() mode causes
# Qwen2DecoderLayer to forcibly set past_key_values=None in every forward pass,
# corrupting the KV cache during autoregressive generation and producing garbage text.
# Fix: switch to eval + disable grad_ckpt for the generate() call, restore afterward.
import trl.models as _trl_models
import contextlib as _contextlib

_orig_unwrap = _trl_models.unwrap_model_for_generation.__wrapped__ if hasattr(_trl_models.unwrap_model_for_generation, '__wrapped__') else None

@_contextlib.contextmanager
def _fixed_unwrap_model_for_generation(model, accelerator, is_peft_model=False, gather_deepspeed3_params=True):
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    had_grad_ckpt = getattr(unwrapped, 'is_gradient_checkpointing', False)
    if was_training:
        unwrapped.eval()
    if had_grad_ckpt:
        unwrapped.gradient_checkpointing_disable()
    try:
        if is_peft_model:
            unwrapped.pretrained_model.disable_adapter()
        # No deepspeed zero-3 handling needed for our setup
        yield unwrapped
    finally:
        if had_grad_ckpt:
            unwrapped.gradient_checkpointing_enable()
        if was_training:
            unwrapped.train()

_trl_models.unwrap_model_for_generation = _fixed_unwrap_model_for_generation
# Also patch the reference in grpo_trainer module namespace
_grpo_mod.unwrap_model_for_generation = _fixed_unwrap_model_for_generation

_SANITY_LOG = _REPO_ROOT / "results" / "sanity_checks.log"
_VERDICT_PAT = re.compile(r'\b(ACCEPTED|REJECTED|ENCODE_FAIL|SSH_TIMEOUT)\b')


class SanityCheckCallback(TrainerCallback):
    """Periodic health check — fires every `interval_secs` seconds.

    Checks:
      - VM reachability (SSH echo)
      - Verdict breakdown for the last 200 completions
      - Cumulative PC count
    Prints a report to stdout and appends to results/sanity_checks.log.
    Prints a WARNING line if ENCODE_FAIL > 50% or VM is unreachable.
    """

    def __init__(self, ssh: rw.SSHClient, interval_secs: int = 3600):
        self._ssh = ssh
        self._interval = interval_secs
        self._next_check = time.time() + interval_secs

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if time.time() < self._next_check:
            return
        self._next_check = time.time() + self._interval
        self._run_check(state.global_step)

    def _run_check(self, step: int) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{now}] === SANITY CHECK step={step} ==="]

        # PC discovery
        lines.append(f"  cumulative_pcs : {len(rw._pc_set)}")

        # Recent verdict breakdown (last 200 completions from log)
        verdicts = self._recent_verdicts(200)
        total = sum(verdicts.values()) or 1
        breakdown = "  ".join(
            f"{k}={v}({100 * v // total}%)" for k, v in verdicts.items() if v
        )
        lines.append(f"  last_200       : {breakdown or 'no data yet'}")

        # VM connectivity
        try:
            rc, out, _ = self._ssh.run("echo pong", timeout=5)
            vm_ok = rc == 0 and "pong" in out
            vm_status = "UP" if vm_ok else f"UNEXPECTED rc={rc}"
        except Exception as exc:
            vm_ok = False
            vm_status = f"DOWN ({exc})"
        lines.append(f"  vm_status      : {vm_status}")

        # Alert conditions
        ef_rate = verdicts.get("ENCODE_FAIL", 0) / total
        if ef_rate > 0.5:
            lines.append(
                f"  WARNING: ENCODE_FAIL {ef_rate:.0%} > 50% — encoder may have regressed"
            )
        if not vm_ok:
            lines.append(
                "  WARNING: VM unreachable — rewards will time out (crash signal flood)"
            )

        report = "\n".join(lines)
        print(report, flush=True)
        with _SANITY_LOG.open("a") as f:
            f.write(report + "\n\n")

    def _recent_verdicts(self, n: int) -> dict:
        counts: dict[str, int] = {}
        try:
            text = _GRPO_DEBUG_LOG.read_text()
            matches = _VERDICT_PAT.findall(text)
            for v in matches[-n:]:
                counts[v] = counts.get(v, 0) + 1
        except FileNotFoundError:
            pass
        return counts


# Reward telemetry buffer. The reward_fn runs inside compute_loss; logging from there with
# wandb.log(commit=False) gets dropped by wandb's step counter, so custom metrics (novelty/*,
# valid_rate, verdict/*) silently never chart. Instead the reward_fn stashes them here and the
# callback below flushes them at on_log with the Trainer's explicit global_step.
_REWARD_TELEMETRY: dict = {}


class RewardTelemetryCallback(TrainerCallback):
    """Flush the latest reward telemetry into wandb at the Trainer's logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is not None and _REWARD_TELEMETRY:
            wandb.log(dict(_REWARD_TELEMETRY), step=state.global_step)


def _make_remote_reward_fn(base_url: str, api_key: str):
    """Return a reward function that calls the remote reward server over HTTP.

    Retries with exponential backoff for up to 5 minutes on any network error.
    """
    endpoint = base_url.rstrip("/") + "/rewards"
    # ngrok free tier serves a browser-warning interstitial (HTML) instead of the
    # JSON response unless this header is present → resp.json() would fail. Harmless
    # on cloudflared/other tunnels.
    headers = {"X-API-Key": api_key, "ngrok-skip-browser-warning": "true"}

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        payload = {"completions": completions}
        deadline = time.time() + 300  # 5 minutes
        delay = 5
        while True:
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if time.time() + delay > deadline:
                    raise RuntimeError(
                        f"Reward server unreachable after 5 minutes: {exc}"
                    ) from exc
                print(f"[WARN] Reward server error ({exc}), retrying in {delay}s…", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)

        rewards: list[float] = data["rewards"]

        # Read everything from the SERVER response `data` — the local rw module is empty on
        # Colab (the server runs compute_rewards), so rw._last_* / rw._global_pc_freq are 0 here.
        total_pcs = data.get("total_pcs_per_program", [])
        vc = data.get("verdict_counts", {})
        _REWARD_TELEMETRY.clear()
        _REWARD_TELEMETRY.update({
            "reward/mean": sum(rewards) / max(1, len(rewards)),
            "reward/std": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "reward/max": max(rewards) if rewards else 0.0,
            "cumulative_pcs": data.get("cumulative_pcs", 0),
            "depth/mean": sum(total_pcs) / max(1, len(total_pcs)),
            "depth/max": data.get("max_pcs_seen", 0),
            "verdict/accepted": vc.get("accepted", 0),
            "verdict/rejected": vc.get("rejected", 0),
            "verdict/encode_fail": vc.get("encode_fail", 0),
            "verdict/error": vc.get("error", 0),
            "verdict/crash": vc.get("crash", 0),
            "valid_rate": vc.get("accepted", 0) / max(1, len(rewards)),
            "novelty/group_mean": data.get("group_novelty_mean", 0.0),
            "novelty/global_mean": data.get("global_novelty_mean", 0.0),
            "novelty/global_frontier": data.get("global_frontier", 0),
        })

        return rewards

    return reward_fn


_PROMPT = "[coverage=high][novelty=high]\n### ASSEMBLY:\n"


def _build_dataset(n: int) -> Dataset:
    return Dataset.from_dict({"prompt": [_PROMPT] * n})


def _make_reward_fn(ssh: rw.SSHClient, smoke_test: bool):
    """Return reward function with signature (prompts, completions, **kw) -> list[float]."""
    _call = [0]

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        if smoke_test:
            return [0.2] * len(completions)

        call_id = _call[0]
        _call[0] += 1
        if call_id < 3:
            for j, (p, c) in enumerate(zip(prompts, completions)):
                print(f"[GRPO-PROMPT] call={call_id} j={j} prompt={p[:80]!r}", flush=True)
                print(f"[GRPO-COMPLETION] call={call_id} j={j} completion={c[:200]!r}", flush=True)

        rewards = rw.compute_rewards(completions, ssh)

        pcs = rw._last_pcs_per_program
        vc = rw._last_verdict_counts
        _REWARD_TELEMETRY.clear()
        _REWARD_TELEMETRY.update({
            "reward/mean": sum(rewards) / len(rewards),
            "reward/std": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "reward/max": max(rewards),
            "cumulative_pcs": len(rw._pc_set),
            "depth/mean": sum(pcs) / max(1, len(pcs)),
            "depth/max": rw._max_pcs_seen,
            "verdict/accepted": vc.get("accepted", 0),
            "verdict/rejected": vc.get("rejected", 0),
            "verdict/encode_fail": vc.get("encode_fail", 0),
            "verdict/error": vc.get("error", 0),
            "verdict/crash": vc.get("crash", 0),
            "valid_rate": vc.get("accepted", 0) / max(1, len(rewards)),
            "novelty/group_mean": rw._last_group_novelty_mean,
            "novelty/global_mean": rw._last_global_novelty_mean,
            "novelty/global_frontier": len(rw._global_pc_freq),
        })

        return rewards

    return reward_fn


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="10 steps, G=2, mock reward — no VM required")
    ap.add_argument("--model", default=str(_REPO_ROOT / "checkpoints" / "sft_retrain" / "checkpoint-1500-merged"),
                    help="Path to the merged fp16 SFT model (e.g. checkpoint-1500-merged). "
                         "Loaded in bf16 by default — do NOT 4-bit a merged fine-tune "
                         "(benchmark: merged_bnb4bit → 0%% valid). Use --load-4bit only with "
                         "an unmerged base + adapter.")
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "checkpoints" / "rl_grpo"))
    ap.add_argument("--resume", action="store_true",
                    help="Auto-resume from latest checkpoint in output-dir and continue WandB run")
    ap.add_argument("--run-name", default="grpo-rl-v1",
                    help="WandB run name (default: grpo-rl-v1)")
    ap.add_argument("--remote-reward-url", default=None,
                    help="Base URL of reward server (e.g. https://your-name.ngrok-free.app). "
                         "When set, reward.py and SSH are not used. "
                         "API key read from REWARD_API_KEY env var.")
    ap.add_argument("--vm-host", default="localhost")
    ap.add_argument("--vm-port", type=int, default=10022)
    ap.add_argument("--vm-key", default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    ap.add_argument("--num-generations", type=int, default=16,
                    help="G: completions per prompt per step (= GRPO group size). Target 16: "
                         "at ~7%% valid that is ~1.3 valid programs/group so the per-group "
                         "novelty signal exists; drop only if it OOMs the 40GB A100.")
    ap.add_argument("--max-completion-length", type=int, default=512,
                    help="Token budget per completion. 512 justified empirically: SFT-v2 "
                         "diversity run showed 512->1024 raised avg_insn 30->58 but unique PCs "
                         "only +12%% — length is not the diversity lever, so spend the saved "
                         "memory on larger G instead.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Stop after this many steps (-1 = run until end of dataset)")
    ap.add_argument("--beta", type=float, default=0.05,
                    help="KL penalty coefficient — 0.05 anchors policy to SFT distribution")
    ap.add_argument("--load-4bit", action="store_true",
                    help="Load base in 4-bit NF4 (QLoRA). Only for an UNMERGED base model. "
                         "Default off: a merged SFT fine-tune is loaded in bf16, because "
                         "4-bit quantising a merged model collapses its quality "
                         "(benchmark: sft_retrain merged_bnb4bit → 0%% valid, 0 PCs).")
    ap.add_argument("--sanity-interval", type=int, default=3600,
                    help="Seconds between sanity checks (default 1h; 0 = disabled)")
    args = ap.parse_args()

    remote = args.remote_reward_url
    if args.smoke_test:
        n_steps = 10
        n_generations = 2
        n_dataset = 100
        print("[smoke-test] 10 steps, G=2, mock reward — VM not required")
    else:
        n_steps = args.max_steps
        n_generations = args.num_generations
        n_dataset = 10_000

    # WandB run continuity: set env vars before GRPOTrainer initialises wandb.
    run_id_file = Path(args.output_dir) / "wandb_run_id.txt"
    if args.resume and run_id_file.exists() and not args.smoke_test:
        run_id = run_id_file.read_text().strip()
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "must"
        print(f"[*] Resuming WandB run {run_id}")

    # Checkpoint auto-detection. get_last_checkpoint() raises if the dir does not
    # exist (first run on a fresh Drive) — create it so --resume means "fresh start".
    resume_checkpoint = None
    if args.resume:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        resume_checkpoint = get_last_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"[*] Resuming from checkpoint {resume_checkpoint}")
        else:
            print(f"[*] --resume: no checkpoint found in {args.output_dir}, starting fresh")

    ssh = rw.SSHClient(host=args.vm_host, port=args.vm_port, key=args.vm_key)

    if args.load_4bit:
        print(f"[*] Loading model from {args.model} (4-bit NF4 / QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # Default: bf16. A merged SFT fine-tune must NOT be 4-bit quantised
        # (benchmark: merged_bnb4bit → 0% valid). The RL LoRA below trains on top.
        print(f"[*] Loading model from {args.model} (bf16, no quantisation)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    # Required for gradient checkpointing with PEFT LoRA
    model.enable_input_require_grads()
    # TRL 0.14 / PEFT 0.18 compat: TRL sets model.warnings_issued["estimate_tokens"] after
    # wrapping with get_peft_model; PEFT proxies __getattr__ to base model, which doesn't have
    # this attr. Pre-seeding the dict lets the proxy find it via __getattr__ and mutate it.
    model.warnings_issued = {}

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name="grpo-smoke-test" if args.smoke_test else args.run_name,
        num_generations=n_generations,
        max_prompt_length=64,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=10 if args.smoke_test else n_steps,
        learning_rate=5e-6,
        beta=args.beta,
        temperature=0.9,
        logging_steps=1,
        save_steps=10 if args.smoke_test else 200,
        report_to="none" if args.smoke_test else "wandb",
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
    )

    dataset = _build_dataset(n_dataset)

    if remote and not args.smoke_test:
        api_key = os.environ.get("REWARD_API_KEY", "")
        if not api_key:
            raise SystemExit("REWARD_API_KEY env var is required with --remote-reward-url")
        reward_fn = _make_remote_reward_fn(remote, api_key)
        print(f"[*] Remote reward: {remote}")
    else:
        reward_fn = _make_reward_fn(ssh, smoke_test=args.smoke_test)

    callbacks = []
    if not args.smoke_test:
        callbacks.append(RewardTelemetryCallback())   # flush reward metrics at the Trainer's step
    if not args.smoke_test and not remote and args.sanity_interval > 0:
        callbacks.append(SanityCheckCallback(ssh, interval_secs=args.sanity_interval))
        print(f"[*] Sanity check every {args.sanity_interval}s → results/sanity_checks.log")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )

    # Write WandB run ID for new runs so --resume can restore continuity later.
    if not args.smoke_test and wandb.run and not run_id_file.exists():
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(wandb.run.id)
        print(f"[*] WandB run ID saved to {run_id_file}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    if not remote:
        rw.save_pc_set()
    print(f"[+] Done. Checkpoints in {args.output_dir}")
    if args.smoke_test:
        print("[smoke-test] PASSED")


if __name__ == "__main__":
    main()
