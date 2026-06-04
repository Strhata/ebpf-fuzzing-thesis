#!/usr/bin/env python3
"""
coverage_race.py — Model-side coverage race runner.

Generates eBPF programs one at a time using the SFT retrain model, validates
each via kcov_validator over SSH, and tracks cumulative unique kernel PCs.
Writes the same CSV format as buzzer's coverage_based logger so both curves
can be plotted on the same axes.

Output CSV columns:
    elapsed_ms          — wall time since script start
    programs_submitted  — total programs sent to the verifier (valid + invalid)
    valid_programs      — programs accepted by the verifier
    unique_pcs          — cumulative unique kernel PCs seen so far

Usage:
    pixi run python tools/coverage_race.py --max-programs 500
    pixi run python tools/coverage_race.py --max-programs 500 --output /mnt/corpus/model_coverage.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml"))

from reward import SSHClient, _encode_to_hex  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL_PATH = str(REPO_ROOT / "checkpoints" / "sft_retrain" / "checkpoint-1500-merged")
_PROMPT = "[coverage=high][novelty=high]\n### ASSEMBLY:\n"
_MAX_NEW_TOKENS = 2048
_TEMPERATURE = 0.9
_DEFAULT_OUTPUT = "/mnt/corpus/model_coverage.csv"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] Loading model from {_MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
    model.eval()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"[*] Model loaded — peak GPU {peak_mb:.0f} MB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-program generation
# ---------------------------------------------------------------------------

def generate_one(model, tokenizer) -> str:
    import torch

    inputs = tokenizer(_PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=_TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(max_programs: int, output: str, ssh: SSHClient) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    cumulative_pcs_valid: set[int] = set()
    cumulative_pcs_all: set[int] = set()
    programs_submitted = 0
    valid_programs = 0
    start = time.monotonic()

    with open(out_path, "w") as f:
        f.write("elapsed_ms,programs_submitted,valid_programs,unique_pcs_valid,unique_pcs_all\n")
        f.flush()

        print(f"[*] Starting race — {max_programs} programs → {output}")

        while programs_submitted < max_programs:
            # Generate
            assembly = generate_one(model, tokenizer)

            # Encode
            hex_str = _encode_to_hex(assembly)
            programs_submitted += 1

            if hex_str is None:
                elapsed = int((time.monotonic() - start) * 1000)
                print(f"\r[{programs_submitted}/{max_programs}] encode_fail  | valid={valid_programs} | pcs_valid={len(cumulative_pcs_valid)} | pcs_all={len(cumulative_pcs_all)}", end="", flush=True)
                f.write(f"{elapsed},{programs_submitted},{valid_programs},{len(cumulative_pcs_valid)},{len(cumulative_pcs_all)}\n")
                f.flush()
                continue

            # Validate on VM
            try:
                rc, stdout, _ = ssh.run(f"/mnt/corpus/kcov_validator {hex_str}", timeout=30)
            except Exception as e:
                print(f"\n[!] SSH error: {e}")
                continue

            result = None
            if rc in (0, 1):
                try:
                    result = json.loads(stdout.strip())
                except (json.JSONDecodeError, ValueError):
                    pass

            if result is not None:
                pcs = result.get("pcs", [])
                is_valid = result.get("verdict") in ("ACCEPTED", "VALID")
                if is_valid:
                    valid_programs += 1
                cumulative_pcs_valid.update(pcs) if is_valid else None
                cumulative_pcs_all.update(pcs)

            elapsed = int((time.monotonic() - start) * 1000)
            verdict = result.get("verdict", "ERROR") if result else "ERROR"
            new_pcs = len(pcs) if result else 0
            print(
                f"\r[{programs_submitted}/{max_programs}] {verdict:<8} | "
                f"valid={valid_programs} | pcs_valid={len(cumulative_pcs_valid)} | pcs_all={len(cumulative_pcs_all)} | "
                f"this_prog={new_pcs} pcs",
                end="", flush=True,
            )

            f.write(f"{elapsed},{programs_submitted},{valid_programs},{len(cumulative_pcs_valid)},{len(cumulative_pcs_all)}\n")
            f.flush()

    print(f"\n[*] Done — {programs_submitted} programs, {valid_programs} valid, pcs_valid={len(cumulative_pcs_valid)} pcs_all={len(cumulative_pcs_all)}")
    print(f"[*] Results written to {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Model-side coverage race runner")
    parser.add_argument("--max-programs", type=int, default=500,
                        help="Number of programs to generate and validate (default: 500)")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})")
    parser.add_argument("--ssh-host", default="localhost")
    parser.add_argument("--ssh-port", type=int, default=10022)
    parser.add_argument("--ssh-key", default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    args = parser.parse_args()

    ssh = SSHClient(host=args.ssh_host, port=args.ssh_port, key=args.ssh_key)

    # Quick connectivity check
    rc, out, _ = ssh.run("echo ok", timeout=5)
    if rc != 0 or out.strip() != "ok":
        print(f"[!] SSH connectivity check failed (rc={rc}, out={out!r}). Is the VM running?")
        sys.exit(1)
    print(f"[*] SSH OK — {args.ssh_host}:{args.ssh_port}")

    run(max_programs=args.max_programs, output=args.output, ssh=ssh)


if __name__ == "__main__":
    main()
