#!/usr/bin/env python3
"""
rl_grpo.py — GRPO RL training for eBPF program generation.

Starts from checkpoints/curated_merged/ (merged SFT model) and fine-tunes
with GRPO using KCOV-based rewards from the eval VM.

Usage:
    pixi run python ml/rl_grpo.py                         # full training
    pixi run python ml/rl_grpo.py --smoke-test            # 10 steps, mock reward
    pixi run python ml/rl_grpo.py --resume-from checkpoints/rl_grpo/checkpoint-200
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ml"))

# TRL 0.14 bug: is_vllm_available() returns tuple (False, None) which is truthy.
# Monkey-patch before importing GRPOTrainer to prevent a spurious vllm import.
import trl.import_utils as _trl_utils
_trl_utils.is_vllm_available = lambda: False

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import reward as rw

_PROMPT = "Kernel: unknown | Status: VALID\n### ASSEMBLY:\n"


def _build_dataset(n: int) -> Dataset:
    return Dataset.from_dict({"prompt": [_PROMPT] * n})


def _make_reward_fn(ssh: rw.SSHClient, smoke_test: bool):
    """Return reward function with signature (prompts, completions, **kw) -> list[float]."""
    _state = {"prev_pc_count": len(rw._pc_set)}

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        if smoke_test:
            return [0.4] * len(completions)

        rewards = rw.compute_rewards(completions, ssh)

        if wandb.run is not None:
            n_new = len(rw._pc_set) - _state["prev_pc_count"]
            _state["prev_pc_count"] = len(rw._pc_set)
            wandb.log({
                "reward/mean": sum(rewards) / len(rewards),
                "reward/max": max(rewards),
                "new_pcs": n_new,
                "cumulative_pcs": len(rw._pc_set),
                "tier/compile_fail": rewards.count(0.0),
                "tier/rejected": rewards.count(0.1),
                "tier/valid": rewards.count(0.4),
                "tier/new_pcs": rewards.count(1.0),
                "tier/crash": rewards.count(2.0),
            }, commit=False)

        return rewards

    return reward_fn


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="10 steps, G=2, mock reward — no VM required")
    ap.add_argument("--model", default=str(_REPO_ROOT / "checkpoints" / "curated_merged"),
                    help="Path to base model (merged bf16 SFT checkpoint)")
    ap.add_argument("--output-dir", default=str(_REPO_ROOT / "checkpoints" / "rl_grpo"))
    ap.add_argument("--resume-from", default=None, help="Resume from checkpoint path")
    ap.add_argument("--vm-host", default="localhost")
    ap.add_argument("--vm-port", type=int, default=10022)
    ap.add_argument("--vm-key", default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    ap.add_argument("--num-generations", type=int, default=8,
                    help="G: completions per prompt per step (reduce if VRAM tight)")
    args = ap.parse_args()

    ssh = rw.SSHClient(host=args.vm_host, port=args.vm_port, key=args.vm_key)

    if args.smoke_test:
        n_steps = 10
        n_generations = 2
        n_dataset = 100
        print("[smoke-test] 10 steps, G=2, mock reward — VM not required")
    else:
        n_steps = -1
        n_generations = args.num_generations
        n_dataset = 10_000

    print(f"[*] Loading model from {args.model}")
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
        run_name="grpo-smoke-test" if args.smoke_test else "grpo-rl-v1",
        num_generations=n_generations,
        max_prompt_length=64,
        max_completion_length=400,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1 if args.smoke_test else 3,
        max_steps=10 if args.smoke_test else n_steps,
        learning_rate=5e-6,
        beta=0.0,
        temperature=0.9,
        logging_steps=1,
        save_steps=10 if args.smoke_test else 200,
        report_to="none" if args.smoke_test else "wandb",
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
    )

    dataset = _build_dataset(n_dataset)
    reward_fn = _make_reward_fn(ssh, smoke_test=args.smoke_test)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    if args.resume_from:
        print(f"[*] Resuming from {args.resume_from}")
        rw._load_pc_set()

    trainer.train(resume_from_checkpoint=args.resume_from)

    rw.save_pc_set()
    trainer.save_model()
    print(f"[+] Done. Checkpoints in {args.output_dir}")
    if args.smoke_test:
        print("[smoke-test] PASSED")


if __name__ == "__main__":
    main()
