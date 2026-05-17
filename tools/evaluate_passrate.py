#!/usr/bin/env python3
"""
evaluate_passrate.py — Phase 3 pass-rate evaluation pipeline.

For a given bf16 merged model:
  1. Load model (bf16 + torch.compile for fast inference)
  2. Generate N eBPF programs (verifier-log format)
  3. Compile each: clang -target bpf → llvm-objcopy → raw hex
  4. Batch SCP hex strings to VM → run ebpf_validator in loop
  5. Report pass-rate table

Usage:
  python tools/evaluate_passrate.py \\
      --model   /path/to/curated_merged \\
      [--label  curated-merged] \\
      [--n 100] \\
      [--vm-port 10022] \\
      [--vm-key  ~/fuzzing_lab/trixie.id_rsa] \\
      [--vm-user root] \\
      [--vm-host localhost] \\
      [--output-dir results/]

VM must be running with shared_corpus mounted at /mnt/corpus:
  ./fuzzing/run_eval_vm.sh
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Assembly parsing  (ported from SFT_tesi.ipynb verifier_to_elf)
# ---------------------------------------------------------------------------

_INST_PATTERN = re.compile(r"^\s*\d+:\s*\([0-9a-fA-F]+\)\s*(.+?)(?:\s*;|$)")
_SKIP_LINES   = {"mark_precise", "<|", "func#"}


def _parse_instructions(assembly_text: str) -> list[str]:
    instructions = []
    for line in assembly_text.splitlines():
        line = line.strip()
        if not line or any(s in line for s in _SKIP_LINES):
            continue
        m = _INST_PATTERN.match(line)
        inst = m.group(1).strip() if m else (line.split(";")[0].strip() if ("=" in line or "goto" in line or "exit" in line) else "")
        if inst and ("=" in inst or any(k in inst for k in ("goto", "exit", "call"))):
            instructions.append(inst)
    return instructions


_PC_GOTO = re.compile(r'\bgoto\s+pc([+-]\d+)')


def _fix_branch(inst: str) -> str:
    """Convert verifier-log branch syntax to LLVM BPF assembler syntax.

    Verifier log:  if r4 & 0x1234 goto pc+71
    LLVM BPF asm:  if r4 & 0x1234 goto +71
    """
    return _PC_GOTO.sub(r'goto \1', inst)


def compile_to_hex(assembly_text: str, tmpdir: str) -> str | None:
    """Parse verifier log → eBPF assembly → ELF → raw hex. Returns hex string or None."""
    instructions = _parse_instructions(assembly_text)
    if not instructions:
        return None

    instructions = [_fix_branch(i) for i in instructions]

    asm_src = ".section prog\n.globl main\nmain:\n"
    asm_src += "\n".join(f"    {i}" for i in instructions)
    if not any("exit" in i.lower() for i in instructions):
        asm_src += "\n    r0 = 0\n    exit"

    asm_file = os.path.join(tmpdir, "prog.s")
    elf_file = os.path.join(tmpdir, "prog.o")
    bin_file = os.path.join(tmpdir, "prog.bin")

    with open(asm_file, "w") as f:
        f.write(asm_src)

    try:
        subprocess.run(
            ["clang", "-target", "bpf", "-c", asm_file, "-o", elf_file],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["llvm-objcopy", "--dump-section", f"prog={bin_file}", elf_file, "/dev/null"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None

    if not os.path.exists(bin_file) or os.path.getsize(bin_file) == 0:
        return None

    with open(bin_file, "rb") as f:
        raw = f.read()

    return raw.hex()


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] Loading bf16 model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    model = torch.compile(model, mode="reduce-overhead")

    # Warmup: triggers compilation so the first real generate call uses the cached graph.
    print("[*] Warming up compiled model...")
    _prompt = "Kernel: unknown | Status: VALID\n### ASSEMBLY:\n"
    _inp = tokenizer(_prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        model.generate(**_inp, max_new_tokens=5, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
    print("[+] Model ready.")
    return model, tokenizer


def generate_programs(model, tokenizer, n: int, temperature: float = 0.8) -> list[str]:
    import torch
    prompt = "Kernel: unknown | Status: VALID\n### ASSEMBLY:\n"
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    programs = []
    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=temperature,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        assembly = text.split("### ASSEMBLY:\n")[-1].strip()
        programs.append(assembly)
        if (i + 1) % 10 == 0:
            print(f"    generated {i+1}/{n}")

    return programs


# ---------------------------------------------------------------------------
# VM validation
# ---------------------------------------------------------------------------

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]


def ssh_check(host: str, port: int, user: str, key: str) -> bool:
    r = subprocess.run(
        ["ssh", "-p", str(port), "-i", key, *_SSH_OPTS, f"{user}@{host}", "echo ok"],
        capture_output=True, text=True
    )
    return r.returncode == 0 and "ok" in r.stdout


def validate_batch(hex_strings: list[str], host: str, port: int, user: str, key: str) -> list[str]:
    """SCP a hex-per-line file to VM, run ebpf_validator on each, return list of verdicts."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(hex_strings) + "\n")
        local_path = f.name

    remote_path = "/tmp/eval_programs.txt"
    try:
        subprocess.run(
            ["scp", "-P", str(port), "-i", key, *_SSH_OPTS, local_path, f"{user}@{host}:{remote_path}"],
            check=True, capture_output=True,
        )

        # Run validation loop inside VM; outputs one verdict per line
        script = (
            f"while IFS= read -r line; do "
            f"  res=$(/mnt/corpus/ebpf_validator \"$line\" 2>/dev/null | grep VERDETTO || echo 'VERDETTO: ERRORE'); "
            f"  echo \"$res\"; "
            f"done < {remote_path}"
        )
        r = subprocess.run(
            ["ssh", "-p", str(port), "-i", key, *_SSH_OPTS, f"{user}@{host}", script],
            capture_output=True, text=True, timeout=600,
        )
        verdicts = [l.strip() for l in r.stdout.splitlines() if "VERDETTO" in l]
        return verdicts
    finally:
        os.unlink(local_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",      required=True,  help="Path to bf16 merged model directory (or HuggingFace model ID for zero-shot)")
    ap.add_argument("--label",      default=None,   help="Run label for output CSV (default: basename of --model)")
    ap.add_argument("--n",          type=int, default=100)
    ap.add_argument("--vm-host",    default="localhost")
    ap.add_argument("--vm-port",    type=int, default=10022)
    ap.add_argument("--vm-user",    default="root")
    ap.add_argument("--vm-key",     default=str(Path.home() / "fuzzing_lab" / "trixie.id_rsa"))
    ap.add_argument("--output-dir", default="results/")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--skip-vm",    action="store_true", help="Skip VM validation, only compile")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.model).name

    # 1. Check VM
    if not args.skip_vm:
        print(f"[*] Checking VM SSH {args.vm_user}@{args.vm_host}:{args.vm_port} ...")
        if not ssh_check(args.vm_host, args.vm_port, args.vm_user, args.vm_key):
            print("[!] VM not reachable. Start VM or use --skip-vm.", file=sys.stderr)
            sys.exit(1)
        print("[+] VM reachable.")

    # 2. Generate
    print(f"\n[*] Generating {args.n} programs from {args.model}")
    model, tokenizer = load_model(args.model)
    programs = generate_programs(model, tokenizer, args.n, args.temperature)
    del model  # free VRAM before SSH work

    # 3. Compile
    print(f"\n[*] Compiling {args.n} programs (clang -target bpf + llvm-objcopy) ...")
    hex_strings = []
    compile_ok  = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, asm in enumerate(programs):
            h = compile_to_hex(asm, tmpdir)
            hex_strings.append(h or "")
            if h:
                compile_ok += 1

    print(f"[*] Compiled: {compile_ok}/{args.n}")

    # 4. Validate
    verdicts = ["SKIPPED"] * args.n
    if not args.skip_vm:
        compilable = [(i, h) for i, h in enumerate(hex_strings) if h]
        print(f"\n[*] Sending {len(compilable)} compiled programs to VM ...")
        if compilable:
            idxs, hexes = zip(*compilable)
            raw_verdicts = validate_batch(list(hexes), args.vm_host, args.vm_port, args.vm_user, args.vm_key)
            for i, v in zip(idxs, raw_verdicts):
                verdicts[i] = v

    # 5. Results
    accettati = sum(1 for v in verdicts if "ACCETTATO" in v)
    rifiutati = sum(1 for v in verdicts if "RIFIUTATO" in v)
    errori    = sum(1 for v in verdicts if "ERRORE" in v or v == "SKIPPED" or not v)

    print(f"\n{'='*50}")
    print(f" PASS-RATE RESULTS — {label}")
    print(f"{'='*50}")
    print(f" Generated  : {args.n}")
    print(f" Compiled   : {compile_ok}  ({compile_ok/args.n*100:.1f}%)")
    print(f" ACCETTATO  : {accettati}  ({accettati/args.n*100:.1f}%)")
    print(f" RIFIUTATO  : {rifiutati}  ({rifiutati/args.n*100:.1f}%)")
    print(f" Error/Skip : {errori}")
    print(f"{'='*50}")

    # 6. Save CSV
    import csv
    out_file = out_dir / f"passrate_{label}.csv"
    with out_file.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "compiled", "verdict"])
        for i, (h, v) in enumerate(zip(hex_strings, verdicts)):
            w.writerow([i, bool(h), v])

    print(f"[+] Results → {out_file}")

    # 7. Summary line (append to comparison table)
    summary_file = out_dir / "passrate_summary.csv"
    write_header = not summary_file.exists()
    with summary_file.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["label", "n_generated", "n_compiled", "n_accettato", "compile_rate", "pass_rate"])
        w.writerow([
            label, args.n, compile_ok, accettati,
            f"{compile_ok/args.n*100:.1f}%",
            f"{accettati/args.n*100:.1f}%",
        ])
    print(f"[+] Summary  → {summary_file}")


if __name__ == "__main__":
    main()
