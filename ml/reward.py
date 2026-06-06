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

# results/ is gitignored -> absent in a fresh clone (Colab). Create before the FileHandler opens.
_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(_DEBUG_LOG),
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)
_log = logging.getLogger("reward")

_pc_set: set[int] = set()
_max_pcs_seen: int = 1
_call_count: int = 0
_last_pcs_per_program: list[int] = []  # populated by compute_rewards; read by reward_server


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
# BPF bytecode encoder — pattern-matched, bypasses clang
#
# Input: bare instruction text (minimal format, no N: prefix, no (XX) opcode byte).
# Opcodes are inferred from instruction syntax via a priority-ordered pattern table.
# lddw (hex-address wide immediates) and unrecognised lines are silently skipped.
# ---------------------------------------------------------------------------

_SKIP_LINES = {"mark_precise", "<|", "func#"}
_VERIFIER_STATE = re.compile(r"^\s*\d+:\s*R\d+=")   # e.g. "0: R1=ctx() R10=fp0"
_REG_RE = re.compile(r'[rw](\d+)')
_INT_RE = re.compile(r'(-?\d+)')

# memory size (bits) → (LDX opcode, STX opcode, ST opcode)
_MEM_OPCODES: dict[int, tuple[int, int, int]] = {
    64: (0x79, 0x7b, 0x7a),
    32: (0x61, 0x63, 0x62),
    16: (0x69, 0x6b, 0x6a),
    8:  (0x71, 0x73, 0x72),
}

# conditional jump operator → (REG opcode, IMM opcode)
_JMP_OPCODES: dict[str, tuple[int, int]] = {
    '==':  (0x1d, 0x15),
    '!=':  (0x5d, 0x55),
    '>':   (0x2d, 0x25),
    '>=':  (0x3d, 0x35),
    '<':   (0xad, 0xa5),
    '<=':  (0xbd, 0xb5),
    's>':  (0x6d, 0x65),
    's>=': (0x7d, 0x75),
    's<':  (0xcd, 0xc5),
    's<=': (0xdd, 0xd5),
    '&':   (0x4d, 0x45),
}

# ALU operator → (ALU64_REG, ALU64_IMM, ALU32_REG, ALU32_IMM)
_ALU_OPCODES: dict[str, tuple[int, int, int, int]] = {
    '+=':   (0x0f, 0x07, 0x0c, 0x04),
    '-=':   (0x1f, 0x17, 0x1c, 0x14),
    '*=':   (0x2f, 0x27, 0x2c, 0x24),
    '/=':   (0x3f, 0x37, 0x3c, 0x34),
    '|=':   (0x4f, 0x47, 0x4c, 0x44),
    '&=':   (0x5f, 0x57, 0x5c, 0x54),
    '<<=':  (0x6f, 0x67, 0x6c, 0x64),
    '>>=':  (0x7f, 0x77, 0x7c, 0x74),
    '%=':   (0x9f, 0x97, 0x9c, 0x94),
    '^=':   (0xaf, 0xa7, 0xac, 0xa4),
    's>>=': (0xcf, 0xc7, 0xcc, 0xc4),
}


def _pack_insn(code: int, dst: int, src: int, off: int, imm: int) -> bytes:
    dst = dst & 0xf
    src = src & 0xf
    off = max(-32768, min(32767, off))
    imm = max(-(2**31), min(2**31 - 1, imm))
    return struct.pack('<BBhi', code, (src << 4) | dst, off, imm)


def _reg(s: str) -> int:
    m = _REG_RE.search(s)
    return int(m.group(1)) if m else 0


def _encode_insn(text: str) -> bytes | None:
    """Encode one bare BPF instruction to 8 bytes. Returns None if unrecognised or lddw."""
    text = text.strip()
    if not text:
        return None

    # exit
    if text == 'exit':
        return _pack_insn(0x95, 0, 0, 0, 0)

    # goto ±N
    m = re.match(r'^goto\s+([+-]?\d+)$', text)
    if m:
        return _pack_insn(0x05, 0, 0, int(m.group(1)), 0)

    # call N  or  call helper_name
    m = re.match(r'^call\s+(.+)$', text)
    if m:
        arg = m.group(1).strip()
        fn_id = int(arg) if re.match(r'^\d+$', arg) else 1
        return _pack_insn(0x85, 0, 0, 0, fn_id)

    # lddw: rD = 0x<hex> — map-pointer loads; silently skip (not portable across boots)
    if re.match(r'^[rw]\d+\s*=\s*0x[0-9a-fA-F]+$', text):
        return None

    # load: rD = *(uXX *)(rS ± off)
    m = re.match(r'^([rw]\d+)\s*=\s*\*\(u(\d+)\s*\*\)\s*\(([rw]\d+)\s*([+-]\s*\d+)?\)$', text)
    if m:
        dst = _reg(m.group(1))
        size = int(m.group(2))
        src = _reg(m.group(3))
        off = int(m.group(4).replace(' ', '')) if m.group(4) else 0
        opc = _MEM_OPCODES.get(size, _MEM_OPCODES[64])[0]
        return _pack_insn(opc, dst, src, off, 0)

    # store: *(uXX *)(rD ± off) = rS  or  *(uXX *)(rD ± off) = IMM
    m = re.match(r'^\*\(u(\d+)\s*\*\)\s*\(([rw]\d+)\s*([+-]\s*\d+)?\)\s*=\s*(.+)$', text)
    if m:
        size = int(m.group(1))
        dst = _reg(m.group(2))
        off = int(m.group(3).replace(' ', '')) if m.group(3) else 0
        rhs = m.group(4).strip()
        opcodes = _MEM_OPCODES.get(size, _MEM_OPCODES[64])
        if _REG_RE.match(rhs):
            return _pack_insn(opcodes[1], dst, _reg(rhs), off, 0)  # STX
        imm_m = _INT_RE.search(rhs)
        return _pack_insn(opcodes[2], dst, 0, off, int(imm_m.group(1)) if imm_m else 0)  # ST

    # conditional jump: if rD OP rS/IMM goto ±off
    m = re.match(
        r'^if\s+([rw]\d+)\s*(s?>=|s?<=|s?>|s?<|==|!=|&)\s*([rw]\d+|-?\d+)\s+goto\s+([+-]?\d+)$',
        text,
    )
    if m:
        dst = _reg(m.group(1))
        op = m.group(2)
        rhs = m.group(3)
        off = int(m.group(4))
        opcodes = _JMP_OPCODES.get(op)
        if opcodes is None:
            return None
        if _REG_RE.match(rhs):
            return _pack_insn(opcodes[0], dst, _reg(rhs), off, 0)
        return _pack_insn(opcodes[1], dst, 0, off, int(rhs))

    # negate: rD = -rD  (before reg-copy to avoid confusion)
    m = re.match(r'^([rw]\d+)\s*=\s*-([rw]\d+)$', text)
    if m:
        opc = 0x87 if m.group(1).startswith('r') else 0x84
        return _pack_insn(opc, _reg(m.group(1)), 0, 0, 0)

    # register copy: rD = rS
    m = re.match(r'^([rw]\d+)\s*=\s*([rw]\d+)$', text)
    if m:
        opc = 0xbf if m.group(1).startswith('r') else 0xbc
        return _pack_insn(opc, _reg(m.group(1)), _reg(m.group(2)), 0, 0)

    # immediate: rD = IMM (decimal only — hex is handled by lddw check above)
    m = re.match(r'^([rw]\d+)\s*=\s*(-?\d+)$', text)
    if m:
        opc = 0xb7 if m.group(1).startswith('r') else 0xb4
        return _pack_insn(opc, _reg(m.group(1)), 0, 0, int(m.group(2)))

    # ALU compound: rD OP= rS  or  rD OP= IMM
    m = re.match(r'^([rw]\d+)\s*(s>>=|<<=|>>=|[+\-*/%|&^]=)\s*([rw]\d+|-?\d+)$', text)
    if m:
        dst_s = m.group(1)
        op = m.group(2)
        rhs = m.group(3)
        is64 = dst_s.startswith('r')
        opcodes = _ALU_OPCODES.get(op)
        if opcodes is None:
            return None
        if _REG_RE.match(rhs):
            return _pack_insn(opcodes[0] if is64 else opcodes[2], _reg(dst_s), _reg(rhs), 0, 0)
        return _pack_insn(opcodes[1] if is64 else opcodes[3], _reg(dst_s), 0, 0, int(rhs))

    return None


_EXIT_INSN = _pack_insn(0x95, 0, 0, 0, 0)

# ---------------------------------------------------------------------------
# Verifier-log format stripper
# ---------------------------------------------------------------------------

_STRIP_FROM = re.compile(r'^from\s+\d+\s+to\s+\d+:')
_STRIP_NUM = re.compile(r'^\s*\d+:\s*')
_STRIP_OPCODE_BYTE = re.compile(r'\([0-9a-fA-F]{2}\)\s*')
_STRIP_COMMENT = re.compile(r'\s*;.*$')
_STRIP_GOTO_PC = re.compile(r'\bgoto\s+pc([+-]\d+)')


def strip_verifier_log(text: str) -> str:
    """Convert a full verifier-log string to minimal bare assembly format.

    Removes N: line numbers, (XX) opcode bytes, register-state lines,
    from-to transition headers, and ; comment suffixes.
    Normalises 'goto pc±N' to 'goto ±N'.
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(s in line for s in _SKIP_LINES):
            continue
        if _VERIFIER_STATE.match(line) or _STRIP_FROM.match(line):
            continue
        line = _STRIP_NUM.sub('', line)
        line = _STRIP_OPCODE_BYTE.sub('', line)
        line = _STRIP_COMMENT.sub('', line)
        line = _STRIP_GOTO_PC.sub(r'goto \1', line)
        line = line.strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


def _encode_to_hex(assembly: str) -> str | None:
    """Encode minimal-format BPF assembly to raw bytecode hex without clang.

    Input is bare instruction text (no N: line numbers, no (XX) opcode bytes).
    Opcodes are inferred from instruction syntax. Unrecognised lines are skipped.
    Programs that clang or an assembler would reject still reach the verifier.
    """
    insns: list[bytes] = []

    for line in assembly.splitlines():
        line = line.strip()
        if not line or any(s in line for s in _SKIP_LINES):
            continue
        if _VERIFIER_STATE.match(line):
            continue

        encoded = _encode_insn(line)
        if encoded is not None:
            insns.append(encoded)
        # Permissive: unrecognised lines silently skipped

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

# Programs with fewer than this many encoded instructions receive reward=0.0
# regardless of VM result.  Set below the dataset minimum (~28 instructions)
# to block only the trivial 2-instruction attractor from rl_grpo_v2.
_MIN_INSTRUCTION_COUNT = 15


def compute_rewards(completions: list[str], ssh: SSHClient) -> list[float]:
    """Compute KCOV-based RL rewards for a batch of generated assembly programs.

    Verdict-blind: REJECTED and ACCEPTED programs are both rewarded by depth.
    Snapshot of _pc_set is taken before the batch so all completions compare
    against the same frontier regardless of evaluation order within the batch.
    """
    global _pc_set, _max_pcs_seen, _call_count, _last_pcs_per_program

    rewards: list[float] = []
    pcs_per_program: list[int] = []
    batch_id = _call_count
    pc_set_snapshot = frozenset(_pc_set)

    for i, assembly in enumerate(completions):
        preview = assembly[:120].replace("\n", "\\n")
        if batch_id < 3:
            print(f"[REWARD DEBUG] batch={batch_id} i={i} raw={assembly[:200]!r}", flush=True)
        hex_str = _encode_to_hex(assembly)

        if not hex_str:
            _log.debug("batch=%d i=%d ENCODE_FAIL preview=%r", batch_id, i, preview)
            rewards.append(0.0)
            pcs_per_program.append(0)
            continue

        insn_count = len(hex_str) // 16
        if insn_count < _MIN_INSTRUCTION_COUNT:
            _log.debug("batch=%d i=%d FLOOR insn_count=%d preview=%r",
                       batch_id, i, insn_count, preview)
            rewards.append(0.0)
            pcs_per_program.append(0)
            continue

        result = _validate_on_vm(hex_str, ssh)

        if result is None:
            _log.debug("batch=%d i=%d SSH_TIMEOUT hex_len=%d", batch_id, i, len(hex_str))
            _watchdog(ssh)
            rewards.append(2.0)
            pcs_per_program.append(0)
            continue

        verdict = result.get("verdict", "ERROR")
        if verdict == "ERROR":
            _log.debug("batch=%d i=%d ERROR preview=%r", batch_id, i, preview)
            rewards.append(0.0)
            pcs_per_program.append(0)
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
        pcs_per_program.append(len(total_pcs))

    _last_pcs_per_program = pcs_per_program
    _call_count += 1
    if _call_count % 100 == 0:
        save_pc_set()
        _save_max_pcs_seen()

    return rewards
