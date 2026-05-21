"""
reward.py — RL reward function for GRPO training.

Public interface:
    compute_rewards(completions: list[str], ssh: SSHClient) -> list[float]

Reward formula (verdict-blind, depth-based):
    depth_component = min(0.5, len(pcs) / max_pcs_seen * 0.5)
    discovery_bonus = 1.0 if any PC not in pre-batch global set snapshot
    reward          = depth_component + discovery_bonus

Special cases:
    encode_fail  → 0.0  (no parseable instructions)
    crash        → 2.0  (SSH timeout; VM may have crashed)
"""

import json
import logging
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_PC_SET_PATH = _REPO_ROOT / "results" / "rl_pc_set.json"
_MAX_PCS_SEEN_PATH = _REPO_ROOT / "results" / "rl_max_pcs_seen.json"
_DEBUG_LOG = _REPO_ROOT / "results" / "reward_debug.log"

logging.basicConfig(
    filename=str(_DEBUG_LOG),
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)
_log = logging.getLogger("reward")

_pc_set: set[int] = set()
_max_pcs_seen: int = 1
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


def _load_max_pcs_seen() -> None:
    global _max_pcs_seen
    if _MAX_PCS_SEEN_PATH.exists():
        with _MAX_PCS_SEEN_PATH.open() as f:
            _max_pcs_seen = max(1, int(json.load(f)))


def _save_max_pcs_seen() -> None:
    _MAX_PCS_SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MAX_PCS_SEEN_PATH.open("w") as f:
        json.dump(_max_pcs_seen, f)


_load_pc_set()
_load_max_pcs_seen()


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
# BPF bytecode encoder — bypasses clang, sends raw instructions to verifier
#
# Model output is in kernel verifier dump format: "N: (XX) instruction [; comment]"
# The opcode byte (XX) is extracted directly; dst/src/off/imm are parsed from text.
# Programs with weird operands reach the verifier instead of being rejected by clang.
# ---------------------------------------------------------------------------

_SKIP_LINES = {"mark_precise", "<|", "func#"}
_VERIFIER_STATE = re.compile(r"^\s*\d+:\s*R\d+=")   # e.g. "0: R1=ctx() R10=fp0"
_OPCODE_LINE = re.compile(r"^\s*\d+:\s*\(([0-9a-fA-F]{2})\)\s*(.+?)(?:\s*;|$)")
_REG_RE = re.compile(r'[rw](\d+)')
_INT_RE = re.compile(r'(-?\d+)')

# Keep for debug logging (instruction count)
_INST_PATTERN = re.compile(r"^\s*\d+:\s*\([0-9a-fA-F]+\)\s*(.+?)(?:\s*;|$)")
_PC_GOTO = re.compile(r'\bgoto\s+pc([+-]\d+)')


def _parse_instructions(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or any(s in line for s in _SKIP_LINES):
            continue
        if _VERIFIER_STATE.match(line):
            continue
        m = _INST_PATTERN.match(line)
        inst = (m.group(1).strip() if m
                else (line.split(";")[0].strip()
                      if ("=" in line or "goto" in line or "exit" in line) else ""))
        if inst and ("=" in inst or any(k in inst for k in ("goto", "exit", "call"))):
            out.append(_PC_GOTO.sub(r'goto \1', inst))
    return out


def _pack_insn(code: int, dst: int, src: int, off: int, imm: int) -> bytes:
    dst = dst & 0xf
    src = src & 0xf
    off = max(-32768, min(32767, off))
    imm = max(-(2**31), min(2**31 - 1, imm))
    return struct.pack('<BBhi', code, (src << 4) | dst, off, imm)


def _reg(s: str) -> int:
    m = _REG_RE.search(s)
    return int(m.group(1)) if m else 0


def _encode_insn(opcode: int, text: str) -> bytes | None:
    text = text.strip()

    if text == 'exit':
        return _pack_insn(0x95, 0, 0, 0, 0)

    # goto ±N
    m = re.match(r'^goto\s+([+-]?\d+)$', text)
    if m:
        return _pack_insn(opcode, 0, 0, int(m.group(1)), 0)

    # call N  or  call bpf_helper_name
    m = re.match(r'^call\s+(.+)$', text)
    if m:
        arg = m.group(1).strip()
        fn_id = int(arg) if re.match(r'^\d+$', arg) else 1
        return _pack_insn(0x85, 0, 0, 0, fn_id)

    # rD = *(uXX *)(rS ± OFF)   — load
    m = re.match(r'^([rw]\d+)\s*=\s*\*\([^)]*\)\s*\(([rw]\d+)\s*([+-]\s*\d+)?\)$', text)
    if m:
        dst = _reg(m.group(1))
        src = _reg(m.group(2))
        off = int(m.group(3).replace(' ', '')) if m.group(3) else 0
        return _pack_insn(opcode, dst, src, off, 0)

    # *(uXX *)(rD ± OFF) = rS/IMM   — store
    m = re.match(r'^\*\([^)]*\)\s*\(([rw]\d+)\s*([+-]\s*\d+)?\)\s*=\s*(.+)$', text)
    if m:
        dst = _reg(m.group(1))
        off = int(m.group(2).replace(' ', '')) if m.group(2) else 0
        rhs = m.group(3).strip()
        if _REG_RE.match(rhs):
            return _pack_insn(opcode, dst, _reg(rhs), off, 0)
        imm_m = _INT_RE.search(rhs)
        return _pack_insn(opcode, dst, 0, off, int(imm_m.group(1)) if imm_m else 0)

    # if rD OP rS/IMM goto ±OFF   — conditional jump
    m = re.match(r'^if\s+([rw]\d+)\s*[=!<>s]+\s*([rw]\d+|-?\d+)\s+goto\s+([+-]?\d+)$', text)
    if m:
        dst = _reg(m.group(1))
        rhs = m.group(2)
        off = int(m.group(3))
        if _REG_RE.match(rhs):
            return _pack_insn(opcode, dst, _reg(rhs), off, 0)
        return _pack_insn(opcode, dst, 0, off, int(rhs))

    # rD = rS   — register copy
    m = re.match(r'^([rw]\d+)\s*=\s*([rw]\d+)$', text)
    if m:
        return _pack_insn(opcode, _reg(m.group(1)), _reg(m.group(2)), 0, 0)

    # rD = IMM  or  rD = IMM ll   — immediate (treat wide as truncated 32-bit)
    m = re.match(r'^([rw]\d+)\s*=\s*(-?\d+)', text)
    if m:
        return _pack_insn(opcode, _reg(m.group(1)), 0, 0, int(m.group(2)))

    # rD OP= rS   — compound with register
    m = re.match(r'^([rw]\d+)\s*(?:[+\-*/%|&^]|<<|>>|s>>)=\s*([rw]\d+)$', text)
    if m:
        return _pack_insn(opcode, _reg(m.group(1)), _reg(m.group(2)), 0, 0)

    # rD OP= IMM   — compound with immediate
    m = re.match(r'^([rw]\d+)\s*(?:[+\-*/%|&^]|<<|>>|s>>)=\s*(-?\d+)$', text)
    if m:
        return _pack_insn(opcode, _reg(m.group(1)), 0, 0, int(m.group(2)))

    # rD = -rD   — negate
    m = re.match(r'^([rw]\d+)\s*=\s*-([rw]\d+)$', text)
    if m:
        return _pack_insn(opcode, _reg(m.group(1)), 0, 0, 0)

    return None


_EXIT_INSN = _pack_insn(0x95, 0, 0, 0, 0)


def _encode_to_hex(assembly: str) -> str | None:
    """Encode BPF assembly to raw bytecode hex without clang.

    Trusts the opcode byte the model wrote; packs dst/src/off/imm directly.
    Programs the assembler would reject still reach the verifier.
    """
    insns: list[bytes] = []

    for line in assembly.splitlines():
        line = line.strip()
        if not line or any(s in line for s in _SKIP_LINES):
            continue
        if _VERIFIER_STATE.match(line):
            continue

        m = _OPCODE_LINE.match(line)
        if not m:
            # Handle bare "exit" with no opcode annotation
            if line.split(";")[0].strip() == "exit":
                insns.append(_EXIT_INSN)
            continue

        opcode = int(m.group(1), 16)
        encoded = _encode_insn(opcode, m.group(2))
        if encoded is not None:
            insns.append(encoded)
        # Permissive: unparseable instructions are skipped, not fatal

    if not insns:
        return None

    if insns[-1] != _EXIT_INSN:
        insns.append(_EXIT_INSN)

    return b"".join(insns).hex()


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

    Verdict-blind: RIFIUTATO and ACCETTATO programs are both rewarded by depth.
    Snapshot of _pc_set is taken before the batch so all completions compare
    against the same frontier regardless of evaluation order within the batch.
    """
    global _pc_set, _max_pcs_seen, _call_count

    rewards: list[float] = []
    batch_id = _call_count
    pc_set_snapshot = frozenset(_pc_set)

    for i, assembly in enumerate(completions):
        preview = assembly[:120].replace("\n", "\\n")
        if batch_id < 3:
            print(f"[REWARD DEBUG] batch={batch_id} i={i} raw={assembly[:200]!r}", flush=True)
        insns = _parse_instructions(assembly)
        hex_str = _encode_to_hex(assembly)

        if not hex_str:
            _log.debug("batch=%d i=%d ENCODE_FAIL insns=%d preview=%r",
                       batch_id, i, len(insns), preview)
            rewards.append(0.0)
            continue

        result = _validate_on_vm(hex_str, ssh)

        if result is None:
            _log.debug("batch=%d i=%d SSH_TIMEOUT hex_len=%d", batch_id, i, len(hex_str))
            _watchdog(ssh)
            rewards.append(2.0)
            continue

        verdict = result.get("verdict", "ERRORE")
        if verdict == "ERRORE":
            _log.debug("batch=%d i=%d ERRORE insns=%d preview=%r",
                       batch_id, i, len(insns), preview)
            rewards.append(0.0)
            continue

        total_pcs = set(result.get("pcs", []))
        new_pcs = total_pcs - pc_set_snapshot
        _pc_set.update(total_pcs)

        depth_component = min(0.5, len(total_pcs) / _max_pcs_seen * 0.5)
        discovery_bonus = 1.0 if new_pcs else 0.0
        r = depth_component + discovery_bonus

        if len(total_pcs) > _max_pcs_seen:
            _max_pcs_seen = len(total_pcs)

        _log.debug("batch=%d i=%d %s pcs=%d new=%d depth_r=%.3f reward=%.3f",
                   batch_id, i, verdict, len(total_pcs), len(new_pcs), depth_component, r)
        rewards.append(r)

    _call_count += 1
    if _call_count % 100 == 0:
        save_pc_set()
        _save_max_pcs_seen()

    return rewards
