"""
reward.py — RL reward function for GRPO training (RL-v2: validity-gated, per-group novelty).

Public interface:
    compute_rewards(completions: list[str], ssh: SSHClient) -> list[float]

Reward ladder (per completion) — monotonic so valid always beats invalid:
    encode fail / < floor insns   → 0.0
    VM ERROR / whole-batch crash  → 0.0   (infra noise, NOT a reward — old code paid 2.0,
                                            which made SSH flakiness the top reward)
    REJECTED (invalid)            → soft floor: W_REJECT_MAX * min(1, len(pcs)/max_pcs_seen)
                                    partial credit for how far the verifier walked before
                                    rejecting. Keeps WITHIN-GROUP reward variance alive while
                                    valid programs are rare (~7%) → avoids the RL-v1
                                    reward_std=0 gradient starvation.
    ACCEPTED (valid)              → W_VALID
                                    + W_NOVELTY * group_novelty(p)     [phase A]
                                    + W_GLOBAL  * global_novelty(p)    [phase B, off by default]

group_novelty(p) = mean over p's PCs of (1 - freq/n_accepted), where freq = how many ACCEPTED
programs in THIS batch hit that PC. A path every valid sibling walks pays 0; a path only p
walks pays ~1. Per-group ("Option A"): stateless, never starves, but only fights crowding
*within a batch* — weak against a model that re-treads the same global cluster every step.

global_novelty(p) = mean over p's PCs of 1/(1 + global_freq[pc]), where global_freq is a
persistent count of how many times each PC has been hit by an ACCEPTED program across the WHOLE
run (snapshot pre-batch). A brand-new PC pays ~1.0; a PC hit 100× pays ~0.01. This is the
decayed-global ("Option B") anti-clustering signal: re-treading the run-wide cluster stops
paying, forcing the policy outward. The 1/(1+freq) decay means it *fades* rather than hard-zeros,
avoiding the naive-archive reward collapse. Optional RL_GLOBAL_DECAY (<1) ages counts each batch
so abandoned regions slowly become rewarding again.

Weights are read from env so the reward can be tuned where the server is launched
(local WSL box), independent of the Colab training cell:
    RL_W_VALID       (default 1.0)   flat bonus for crossing into ACCEPTED
    RL_W_NOVELTY     (default 1.0)   weight on per-group novelty (phase A; ACCEPTED only)
    RL_W_GLOBAL      (default 0.0)   weight on decayed-global novelty (phase B; set >0 to enable)
    RL_W_REJECT_MAX  (default 0.3)   ceiling on the invalid soft floor (< W_VALID by design)
    RL_GLOBAL_DECAY  (default 1.0)   per-batch multiplicative ageing of global_freq (1.0 = off)
"""

import json
import logging
import os
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_PC_SET_PATH = _REPO_ROOT / "results" / "rl_pc_set.json"
_MAX_PCS_SEEN_PATH = _REPO_ROOT / "results" / "rl_max_pcs_seen.json"
_GLOBAL_FREQ_PATH = _REPO_ROOT / "results" / "rl_global_pc_freq.json"
_DEBUG_LOG = _REPO_ROOT / "results" / "reward_debug.log"

# results/ is gitignored -> absent in a fresh clone (Colab). Create before the FileHandler opens.
_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(_DEBUG_LOG),
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)
_log = logging.getLogger("reward")

# Reward weights (tunable via env at server launch; see module docstring).
W_VALID: float = float(os.environ.get("RL_W_VALID", "1.0"))
W_NOVELTY: float = float(os.environ.get("RL_W_NOVELTY", "1.0"))
W_GLOBAL: float = float(os.environ.get("RL_W_GLOBAL", "0.0"))      # phase B; 0 = off
W_REJECT_MAX: float = float(os.environ.get("RL_W_REJECT_MAX", "0.3"))
GLOBAL_DECAY: float = float(os.environ.get("RL_GLOBAL_DECAY", "1.0"))

_pc_set: set[int] = set()
_max_pcs_seen: int = 1
_global_pc_freq: dict[int, float] = {}   # phase-B state: PC → times hit by an ACCEPTED program
_call_count: int = 0
_last_pcs_per_program: list[int] = []  # populated by compute_rewards; read by reward_server
# Verdict tally for the most recent batch — read by reward_server for telemetry.
_last_verdict_counts: dict[str, int] = {
    "accepted": 0, "rejected": 0, "encode_fail": 0, "error": 0, "crash": 0,
}
# Mean novelty terms over the most recent batch's ACCEPTED programs (telemetry).
_last_group_novelty_mean: float = 0.0
_last_global_novelty_mean: float = 0.0


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


def _load_global_freq() -> None:
    global _global_pc_freq
    if _GLOBAL_FREQ_PATH.exists():
        with _GLOBAL_FREQ_PATH.open() as f:
            _global_pc_freq = {int(k): float(v) for k, v in json.load(f).items()}


def save_global_freq() -> None:
    _GLOBAL_FREQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _GLOBAL_FREQ_PATH.open("w") as f:
        json.dump({str(k): v for k, v in _global_pc_freq.items()}, f)


_load_pc_set()
_load_max_pcs_seen()
_load_global_freq()


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

    def run(self, cmd: str, timeout: int = 30, input: str | None = None) -> tuple[int, str, str]:
        result = subprocess.run(
            ["ssh", "-p", str(self.port), "-i", self.key,
             "-o", f"ConnectTimeout={timeout}",
             *self._SSH_OPTS,
             f"{self.user}@{self.host}", cmd],
            capture_output=True, text=True, timeout=timeout + 5,
            input=input,
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


def _validate_batch_on_vm(hex_list: list[str], ssh: SSHClient) -> list[dict] | None:
    """Validate many programs in ONE kcov_validator --batch call (hex piped via stdin).

    Returns a list of result dicts in input order, or None if the whole batch
    failed (SSH timeout / non-zero exit / unparseable output) so the caller can
    apply the crash reward uniformly. The validator preserves input order, so
    result[i] corresponds to hex_list[i].
    """
    if not hex_list:
        return []
    payload = "\n".join(hex_list) + "\n"
    # Per-program work is sub-ms; scale the ceiling with batch size for bulk runs.
    timeout = max(30, len(hex_list) // 10)
    try:
        rc, stdout, _ = ssh.run("/mnt/corpus/kcov_validator --batch", timeout=timeout, input=payload)
    except subprocess.TimeoutExpired:
        return None
    if rc != 0:
        return None
    try:
        parsed = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list) or len(parsed) != len(hex_list):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Public reward function
# ---------------------------------------------------------------------------

# Programs with fewer than this many encoded instructions receive reward=0.0
# regardless of VM result.  Set below the dataset minimum (~28 instructions)
# to block only the trivial 2-instruction attractor from rl_grpo_v2.
_MIN_INSTRUCTION_COUNT = 15


def compute_rewards(completions: list[str], ssh: SSHClient) -> list[float]:
    """Compute validity-gated, per-group-novelty RL rewards for a batch.

    See the module docstring for the full ladder. The batch IS one GRPO group
    (rl_grpo uses a single fixed prompt with per_device_train_batch_size=1), so
    group novelty is computed over the whole `completions` list.
    """
    global _pc_set, _max_pcs_seen, _global_pc_freq, _call_count
    global _last_pcs_per_program, _last_verdict_counts
    global _last_group_novelty_mean, _last_global_novelty_mean

    n = len(completions)
    rewards: list[float] = [0.0] * n
    pcs_per_program: list[int] = [0] * n
    counts = {"accepted": 0, "rejected": 0, "encode_fail": 0, "error": 0, "crash": 0}
    batch_id = _call_count

    # --- pass 1: encode + instruction floor; collect programs that need the VM ---
    to_validate: list[tuple[int, str]] = []   # (original index, hex)
    for i, assembly in enumerate(completions):
        preview = assembly[:120].replace("\n", "\\n")
        if batch_id < 3:
            print(f"[REWARD DEBUG] batch={batch_id} i={i} raw={assembly[:200]!r}", flush=True)
        hex_str = _encode_to_hex(assembly)

        if not hex_str:
            _log.debug("batch=%d i=%d ENCODE_FAIL preview=%r", batch_id, i, preview)
            counts["encode_fail"] += 1
            continue  # reward stays 0.0

        insn_count = len(hex_str) // 16
        if insn_count < _MIN_INSTRUCTION_COUNT:
            _log.debug("batch=%d i=%d FLOOR insn_count=%d preview=%r",
                       batch_id, i, insn_count, preview)
            counts["encode_fail"] += 1
            continue  # reward stays 0.0

        to_validate.append((i, hex_str))

    # --- one batched VM call; partition into accepted / rejected, accruing global state ---
    accepted: list[tuple[int, set]] = []   # (orig index, distinct pcs)
    rejected: list[tuple[int, set]] = []   # (orig index, distinct pcs)
    if to_validate:
        results = _validate_batch_on_vm([h for _, h in to_validate], ssh)

        if results is None:
            # whole-batch SSH timeout / VM failure → infra noise, reward 0.0 (NOT 2.0).
            _log.debug("batch=%d BATCH_TIMEOUT n=%d", batch_id, len(to_validate))
            _watchdog(ssh)
            counts["crash"] = len(to_validate)
        else:
            for (orig_i, _hex), result in zip(to_validate, results):
                verdict = result.get("verdict", "ERROR")
                pcs = set(result.get("pcs", []))
                _pc_set.update(pcs)
                if len(pcs) > _max_pcs_seen:
                    _max_pcs_seen = len(pcs)
                if verdict == "ACCEPTED":
                    accepted.append((orig_i, pcs))
                elif verdict == "REJECTED":
                    rejected.append((orig_i, pcs))
                else:
                    counts["error"] += 1  # reward stays 0.0

    # --- soft floor for invalid programs: depth walked before rejection ---
    for orig_i, pcs in rejected:
        rewards[orig_i] = W_REJECT_MAX * min(1.0, len(pcs) / _max_pcs_seen)
        pcs_per_program[orig_i] = len(pcs)
    counts["rejected"] = len(rejected)

    # --- novelty for valid programs: per-group (A) + decayed-global (B) ---
    n_acc = len(accepted)
    counts["accepted"] = n_acc
    group_nov_sum = 0.0
    global_nov_sum = 0.0
    if n_acc:
        # within-batch PC frequency for the per-group term
        freq: dict[int, int] = {}
        for _i, pcs in accepted:
            for pc in pcs:
                freq[pc] = freq.get(pc, 0) + 1
        # score against the PRE-batch global frontier so order within the batch doesn't matter
        for orig_i, pcs in accepted:
            if pcs:
                group_nov = sum(1.0 - freq[pc] / n_acc for pc in pcs) / len(pcs)
                global_nov = sum(1.0 / (1.0 + _global_pc_freq.get(pc, 0.0)) for pc in pcs) / len(pcs)
            else:
                group_nov = global_nov = 0.0
            rewards[orig_i] = W_VALID + W_NOVELTY * group_nov + W_GLOBAL * global_nov
            pcs_per_program[orig_i] = len(pcs)
            group_nov_sum += group_nov
            global_nov_sum += global_nov
            _log.debug("batch=%d i=%d ACCEPTED pcs=%d g_nov=%.3f gl_nov=%.3f reward=%.3f",
                       batch_id, orig_i, len(pcs), group_nov, global_nov, rewards[orig_i])
        # update the global frontier AFTER scoring (accepted programs only)
        if GLOBAL_DECAY < 1.0:
            for pc in list(_global_pc_freq):
                _global_pc_freq[pc] *= GLOBAL_DECAY
        for _i, pcs in accepted:
            for pc in pcs:
                _global_pc_freq[pc] = _global_pc_freq.get(pc, 0.0) + 1.0

    _last_pcs_per_program = pcs_per_program
    _last_verdict_counts = counts
    _last_group_novelty_mean = group_nov_sum / n_acc if n_acc else 0.0
    _last_global_novelty_mean = global_nov_sum / n_acc if n_acc else 0.0
    _call_count += 1
    if _call_count % 100 == 0:
        save_pc_set()
        _save_max_pcs_seen()
        save_global_freq()

    return rewards
