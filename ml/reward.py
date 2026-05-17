"""
reward.py — RL reward function for GRPO training.

Public interface:
    compute_rewards(completions: list[str], ssh: SSHClient) -> list[float]

Reward tiers:
    crash        → 2.0  (SSH timeout; VM may have crashed, coverage frontier signal)
    new_pcs      → 1.0  (ACCETTATO + at least 1 PC not seen before)
    valid        → 0.4  (ACCETTATO, no new PCs)
    rejected     → 0.1  (RIFIUTATO by BPF verifier)
    compile_fail → 0.0  (assembly did not compile)
"""

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_PC_SET_PATH = _REPO_ROOT / "results" / "rl_pc_set.json"

_pc_set: set[int] = set()
_call_count: int = 0


def _load_pc_set() -> None:
    global _pc_set
    if _PC_SET_PATH.exists():
        with _PC_SET_PATH.open() as f:
            _pc_set = set(json.load(f))


def save_pc_set() -> None:
    _PC_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PC_SET_PATH.open("w") as f:
        json.dump(list(_pc_set), f)


_load_pc_set()


# ---------------------------------------------------------------------------
# SSH client
# ---------------------------------------------------------------------------

@dataclass
class SSHClient:
    host: str = "localhost"
    port: int = 10022
    user: str = "root"
    key: str = str(Path.home() / "fuzzing_lab" / "trixie.id_rsa")

    _SSH_OPTS: list = field(default_factory=lambda: [
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
    ])

    def run(self, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
        result = subprocess.run(
            ["ssh", "-p", str(self.port), "-i", self.key,
             "-o", f"ConnectTimeout={timeout}",
             *self._SSH_OPTS,
             f"{self.user}@{self.host}", cmd],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Assembly → BPF hex compiler (mirrors tools/evaluate_passrate.py)
# ---------------------------------------------------------------------------

_INST_PATTERN = re.compile(r"^\s*\d+:\s*\([0-9a-fA-F]+\)\s*(.+?)(?:\s*;|$)")
_SKIP_LINES = {"mark_precise", "<|", "func#"}
_PC_GOTO = re.compile(r'\bgoto\s+pc([+-]\d+)')


def _parse_instructions(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or any(s in line for s in _SKIP_LINES):
            continue
        m = _INST_PATTERN.match(line)
        inst = (m.group(1).strip() if m
                else (line.split(";")[0].strip()
                      if ("=" in line or "goto" in line or "exit" in line) else ""))
        if inst and ("=" in inst or any(k in inst for k in ("goto", "exit", "call"))):
            out.append(_PC_GOTO.sub(r'goto \1', inst))
    return out


def _compile_to_hex(assembly: str, tmpdir: str) -> str | None:
    insns = _parse_instructions(assembly)
    if not insns:
        return None

    src = ".section prog\n.globl main\nmain:\n" + "\n".join(f"    {i}" for i in insns)
    if not any("exit" in i.lower() for i in insns):
        src += "\n    r0 = 0\n    exit"

    asm_f = os.path.join(tmpdir, "prog.s")
    elf_f = os.path.join(tmpdir, "prog.o")
    bin_f = os.path.join(tmpdir, "prog.bin")

    with open(asm_f, "w") as f:
        f.write(src)
    try:
        subprocess.run(["clang", "-target", "bpf", "-c", asm_f, "-o", elf_f],
                       check=True, capture_output=True)
        subprocess.run(["llvm-objcopy", "--dump-section", f"prog={bin_f}", elf_f, "/dev/null"],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None

    if not os.path.exists(bin_f) or os.path.getsize(bin_f) == 0:
        return None
    with open(bin_f, "rb") as f:
        return f.read().hex()


# ---------------------------------------------------------------------------
# VM validation
# ---------------------------------------------------------------------------

_WATCHDOG = str(_REPO_ROOT / "tools" / "vm_watchdog.sh")


def _watchdog(ssh: SSHClient) -> None:
    subprocess.run([_WATCHDOG, ssh.host, str(ssh.port), ssh.key], check=False)


def _validate_on_vm(hex_str: str, ssh: SSHClient) -> dict | None:
    """Run kcov_validator on VM. Returns parsed JSON dict, or None on error/timeout."""
    try:
        rc, stdout, _ = ssh.run(f"/mnt/corpus/kcov_validator {hex_str}", timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if rc not in (0, 1):
        return None
    try:
        return json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public reward function
# ---------------------------------------------------------------------------

def compute_rewards(completions: list[str], ssh: SSHClient) -> list[float]:
    """Compute KCOV-based RL rewards for a batch of generated assembly programs.

    Updates the global PC set in place; writes it to disk every 100 calls.
    """
    global _pc_set, _call_count

    rewards: list[float] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for assembly in completions:
            hex_str = _compile_to_hex(assembly, tmpdir)

            if not hex_str:
                rewards.append(0.0)
                continue

            result = _validate_on_vm(hex_str, ssh)

            if result is None:
                _watchdog(ssh)
                rewards.append(2.0)
                continue

            verdict = result.get("verdict", "ERRORE")

            if verdict == "RIFIUTATO":
                rewards.append(0.1)
            elif verdict == "ACCETTATO":
                new_pcs = set(result.get("pcs", [])) - _pc_set
                _pc_set.update(new_pcs)
                rewards.append(1.0 if new_pcs else 0.4)
            else:
                rewards.append(0.0)

    _call_count += 1
    if _call_count % 100 == 0:
        save_pc_set()

    return rewards
