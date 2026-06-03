#!/usr/bin/env python3
"""
generate_corpus.py — Run inference on the SFT model and build a bytecode corpus.

Generates eBPF programs one at a time (no batching) with max token length,
encodes each with the Python assembler from reward.py, and appends to a JSONL
corpus file.  Optionally validates on the QEMU VM to record verdict + KCOV PCs.

Usage:
    pixi run python tools/generate_corpus.py
    pixi run python tools/generate_corpus.py --n 2000 --max-tokens 4096
    pixi run python tools/generate_corpus.py --n 1000 --validate --vm-key ~/fuzzing_lab/trixie.id_rsa
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ml"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from reward import _encode_to_hex, _validate_on_vm, SSHClient

_PROMPT = "[coverage=high][novelty=high]\n### ASSEMBLY:\n"
_MIN_INSTRUCTIONS = 3  # intentionally low — we want to see what the model actually produces

_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default=str(_REPO_ROOT / "checkpoints" / "sft_retrain" / "checkpoint-1500"),
        help="Path to model or LoRA adapter (default: sft_retrain/checkpoint-1500)",
    )
    ap.add_argument(
        "--base-model",
        default=_BASE_MODEL,
        help="Base model to load adapter onto (only used if --model is a LoRA adapter)",
    )
    ap.add_argument("--n", type=int, default=1000, help="Programs to generate")
    ap.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per generation")
    ap.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    ap.add_argument(
        "--out",
        default=str(_REPO_ROOT / "data" / "corpus_ml.jsonl"),
        help="Output JSONL path (appends if exists)",
    )
    ap.add_argument("--validate", action="store_true", help="Validate each program on the QEMU VM")
    ap.add_argument("--vm-host", default="localhost")
    ap.add_argument("--vm-port", type=int, default=10022)
    ap.add_argument("--vm-key", default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    args = ap.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_path = Path(args.model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        print(f"[*] LoRA adapter detected — loading base {args.base_model} + adapter {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    else:
        print(f"[*] Loading merged model from {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    ssh = SSHClient(host=args.vm_host, port=args.vm_port, key=args.vm_key) if args.validate else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = encoded = validated = 0
    inputs = tokenizer(_PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    print(f"[*] Generating {args.n} programs → {out_path}")
    with out_path.open("a") as f:
        for i in range(args.n):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
            total += 1

            print(f"\n--- program {i} ---\n{generated[:600]}\n---", flush=True)

            hex_str = _encode_to_hex(generated)
            insn_count = len(hex_str) // 16 if hex_str else 0

            if not hex_str or insn_count < _MIN_INSTRUCTIONS:
                print(f"  [{i}/{args.n}] encode_fail insn_count={insn_count}", flush=True)
                continue

            encoded += 1
            record: dict = {
                "assembly": generated,
                "bytecode_hex": hex_str,
                "insn_count": insn_count,
            }

            if ssh:
                result = _validate_on_vm(hex_str, ssh)
                if result:
                    record["verdict"] = result.get("verdict", "ERROR")
                    record["pcs"] = result.get("pcs", [])
                    if record["verdict"] in ("ACCEPTED", "REJECTED"):
                        validated += 1

            f.write(json.dumps(record) + "\n")
            f.flush()

            if i % 100 == 0:
                print(
                    f"  [{i}/{args.n}] encoded={encoded} validated={validated} "
                    f"last_insns={insn_count}",
                    flush=True,
                )

    print(f"[+] Done. generated={total} encoded={encoded} validated={validated} → {out_path}")


if __name__ == "__main__":
    main()
