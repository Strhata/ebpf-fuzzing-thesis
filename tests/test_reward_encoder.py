"""
Tests for the pure-Python BPF bytecode encoder in reward.py.

All tests are offline (no VM, no SSH).  Each test verifies one behaviour of
_encode_to_hex or _encode_insn through the public _encode_to_hex interface.

Input format: bare instruction text (no N: line numbers, no (XX) opcode bytes).
Opcodes are inferred from instruction syntax.

Instruction byte layout (little-endian):
  byte 0    : opcode
  byte 1    : (src_reg << 4) | dst_reg
  bytes 2-3 : off  (signed 16-bit)
  bytes 4-7 : imm  (signed 32-bit)
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

from reward import _encode_to_hex, _pack_insn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def insns(hex_str: str) -> list[bytes]:
    """Split a hex string into 8-byte instruction chunks."""
    assert len(hex_str) % 16 == 0, f"hex length {len(hex_str)} not multiple of 16"
    return [bytes.fromhex(hex_str[i:i+16]) for i in range(0, len(hex_str), 16)]


def unpack(b: bytes) -> tuple:
    """Return (code, dst, src, off, imm) from an 8-byte instruction."""
    code, regs, off, imm = struct.unpack('<BBhi', b)
    return code, regs & 0xf, (regs >> 4) & 0xf, off, imm


# ---------------------------------------------------------------------------
# _pack_insn unit tests
# ---------------------------------------------------------------------------

def test_pack_insn_basic():
    b = _pack_insn(0xb7, 0, 0, 0, 5)
    assert len(b) == 8
    code, dst, src, off, imm = unpack(b)
    assert (code, dst, src, off, imm) == (0xb7, 0, 0, 0, 5)


def test_pack_insn_negative_imm():
    b = _pack_insn(0xb7, 0, 0, 0, -1)
    _, _, _, _, imm = unpack(b)
    assert imm == -1


def test_pack_insn_reg_encoding():
    # dst=3, src=7 → regs byte = (7<<4)|3 = 0x73
    b = _pack_insn(0x0f, 3, 7, 0, 0)
    assert b[1] == 0x73


def test_pack_insn_negative_off():
    b = _pack_insn(0x05, 0, 0, -4, 0)
    _, _, _, off, _ = unpack(b)
    assert off == -4


# ---------------------------------------------------------------------------
# empty / unparseable input
# ---------------------------------------------------------------------------

def test_empty_returns_none():
    assert _encode_to_hex("") is None


def test_only_skip_lines_returns_none():
    assert _encode_to_hex("func#0 @0\nmark_precise: frame0\n<|endoftext|>") is None


def test_only_verifier_state_returns_none():
    assert _encode_to_hex("0: R1=ctx() R10=fp0") is None


def test_unrecognised_line_skipped():
    # Unrecognised text is skipped; other valid lines in the same input still encode.
    h = _encode_to_hex("not_an_instruction\nr0 = 1\nexit")
    assert h is not None
    assert len(insns(h)) == 2  # r0=1 and exit


def test_only_unrecognised_returns_none():
    assert _encode_to_hex("not an instruction\nalso not valid") is None


# ---------------------------------------------------------------------------
# exit instruction
# ---------------------------------------------------------------------------

def test_exit_appended_when_missing():
    h = _encode_to_hex("r0 = 0")
    parts = insns(h)
    assert len(parts) == 2
    code, _, _, _, _ = unpack(parts[-1])
    assert code == 0x95


def test_exit_not_duplicated_when_present():
    h = _encode_to_hex("r0 = 0\nexit")
    parts = insns(h)
    assert len(parts) == 2
    assert unpack(parts[-1])[0] == 0x95


def test_exit_standalone():
    h = _encode_to_hex("exit")
    parts = insns(h)
    assert len(parts) == 1
    assert unpack(parts[0])[0] == 0x95


# ---------------------------------------------------------------------------
# ground truth: r0=0; exit  (byte-identical to what clang produces)
# ---------------------------------------------------------------------------

def test_minimal_program_byte_identical_to_ground_truth():
    h = _encode_to_hex("r0 = 0\nexit")
    assert h == "b7000000000000009500000000000000"


# ---------------------------------------------------------------------------
# MOV64 immediate  (b7)
# ---------------------------------------------------------------------------

def test_mov64_imm_positive():
    h = _encode_to_hex("r3 = 42\nexit")
    code, dst, src, off, imm = unpack(insns(h)[0])
    assert code == 0xb7
    assert dst == 3
    assert src == 0
    assert off == 0
    assert imm == 42


def test_mov64_imm_negative():
    h = _encode_to_hex("r0 = -772319294\nexit")
    _, _, _, _, imm = unpack(insns(h)[0])
    assert imm == -772319294


def test_mov64_imm_zero():
    h = _encode_to_hex("r5 = 0\nexit")
    code, dst, _, _, imm = unpack(insns(h)[0])
    assert code == 0xb7 and dst == 5 and imm == 0


# ---------------------------------------------------------------------------
# MOV64 register  (bf)  and  MOV32 register  (bc)
# ---------------------------------------------------------------------------

def test_mov64_reg():
    h = _encode_to_hex("r2 = r5\nexit")
    code, dst, src, off, imm = unpack(insns(h)[0])
    assert code == 0xbf
    assert dst == 2
    assert src == 5
    assert off == 0
    assert imm == 0


def test_mov64_reg_opcode():
    h = _encode_to_hex("r1 = r2\nexit")
    code, dst, src, _, _ = unpack(insns(h)[0])
    assert code == 0xbf
    assert dst == 1
    assert src == 2


def test_mov32_reg_opcode():
    h = _encode_to_hex("w1 = w2\nexit")
    code, dst, src, _, _ = unpack(insns(h)[0])
    assert code == 0xbc
    assert dst == 1
    assert src == 2


# ---------------------------------------------------------------------------
# ALU64 compound assignment — register and immediate
# ---------------------------------------------------------------------------

def test_add64_reg():
    h = _encode_to_hex("r0 = 1\nr1 = 2\nr0 += r1\nexit")
    code, dst, src, _, _ = unpack(insns(h)[2])
    assert code == 0x0f
    assert dst == 0
    assert src == 1


def test_add64_imm():
    h = _encode_to_hex("r0 = 0\nr0 += 100\nexit")
    code, dst, src, _, imm = unpack(insns(h)[1])
    assert code == 0x07
    assert dst == 0
    assert src == 0
    assert imm == 100


def test_lsh64_imm():
    h = _encode_to_hex("r0 = 1\nr0 <<= 32\nexit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0x67
    assert dst == 0
    assert imm == 32


def test_arsh64_imm():
    h = _encode_to_hex("r0 = 1\nr0 s>>= 32\nexit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0xc7
    assert dst == 0
    assert imm == 32


def test_sub64_reg():
    h = _encode_to_hex("r0 = 5\nr1 = 3\nr0 -= r1\nexit")
    code, dst, src, _, _ = unpack(insns(h)[2])
    assert code == 0x1f
    assert dst == 0
    assert src == 1


def test_mul64_imm():
    h = _encode_to_hex("r0 = 3\nr0 *= 7\nexit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0x27
    assert dst == 0
    assert imm == 7


# ---------------------------------------------------------------------------
# ALU32 (w registers)
# ---------------------------------------------------------------------------

def test_mov32_imm():
    h = _encode_to_hex("w3 = 255\nexit")
    code, dst, src, _, imm = unpack(insns(h)[0])
    assert code == 0xb4
    assert dst == 3
    assert src == 0
    assert imm == 255


def test_xor32_imm():
    h = _encode_to_hex("r0 = 0\nw0 ^= 128\nexit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0xa4
    assert dst == 0
    assert imm == 128


# ---------------------------------------------------------------------------
# NEG64 and NEG32
# ---------------------------------------------------------------------------

def test_neg64():
    h = _encode_to_hex("r1 = 5\nr1 = -r1\nexit")
    code, dst, src, _, imm = unpack(insns(h)[1])
    assert code == 0x87
    assert dst == 1
    assert src == 0
    assert imm == 0


def test_neg32():
    h = _encode_to_hex("w2 = 3\nw2 = -w2\nexit")
    code, dst, _, _, _ = unpack(insns(h)[1])
    assert code == 0x84
    assert dst == 2


# ---------------------------------------------------------------------------
# JMP — unconditional goto
# ---------------------------------------------------------------------------

def test_goto_positive():
    h = _encode_to_hex("goto +1\nr0 = 99\nexit")
    code, _, _, off, _ = unpack(insns(h)[0])
    assert code == 0x05
    assert off == 1


def test_goto_negative():
    h = _encode_to_hex("r0 = 0\ngoto -1\nexit")
    _, _, _, off, _ = unpack(insns(h)[1])
    assert off == -1


# ---------------------------------------------------------------------------
# JMP — conditional
# ---------------------------------------------------------------------------

def test_jeq_imm():
    h = _encode_to_hex("r1 = 5\nif r1 == 5 goto +2\nr0 = 0\nexit\nexit")
    code, dst, src, off, imm = unpack(insns(h)[1])
    assert code == 0x15
    assert dst == 1
    assert src == 0
    assert off == 2
    assert imm == 5


def test_jeq_reg():
    h = _encode_to_hex("r1 = 1\nr2 = 1\nif r1 == r2 goto +1\nexit\nexit")
    code, dst, src, off, _ = unpack(insns(h)[2])
    assert code == 0x1d
    assert dst == 1
    assert src == 2
    assert off == 1


def test_jgt_reg():
    h = _encode_to_hex("r3 = 1\nr4 = 2\nif r3 > r4 goto +1\nexit\nexit")
    code, dst, src, off, _ = unpack(insns(h)[2])
    assert code == 0x2d
    assert dst == 3
    assert src == 4
    assert off == 1


def test_jne_imm():
    h = _encode_to_hex("r0 = 0\nif r0 != 1 goto +1\nexit\nexit")
    code, dst, src, off, imm = unpack(insns(h)[1])
    assert code == 0x55
    assert dst == 0
    assert src == 0
    assert imm == 1


def test_jlt_imm():
    h = _encode_to_hex("r0 = 3\nif r0 < 5 goto +1\nexit\nexit")
    code, _, _, _, imm = unpack(insns(h)[1])
    assert code == 0xa5
    assert imm == 5


def test_jsgt_imm():
    h = _encode_to_hex("r0 = 10\nif r0 s> 0 goto +1\nexit\nexit")
    code, _, _, _, imm = unpack(insns(h)[1])
    assert code == 0x65
    assert imm == 0


# ---------------------------------------------------------------------------
# Memory load / store
# ---------------------------------------------------------------------------

def test_ldx_dw():
    # store then load u64
    h = _encode_to_hex("r0 = 1\n*(u64 *)(r10 -8) = r0\nr1 = *(u64 *)(r10 -8)\nexit")
    code_st, dst_st, src_st, off_st, _ = unpack(insns(h)[1])
    assert code_st == 0x7b   # STX u64
    assert dst_st == 10
    assert src_st == 0
    assert off_st == -8
    code_ld, dst_ld, src_ld, off_ld, _ = unpack(insns(h)[2])
    assert code_ld == 0x79   # LDX u64
    assert dst_ld == 1
    assert src_ld == 10
    assert off_ld == -8


def test_stx_w():
    h = _encode_to_hex("r0 = 7\n*(u32 *)(r10 -4) = r0\nexit")
    code, dst, src, off, _ = unpack(insns(h)[1])
    assert code == 0x63   # STX u32
    assert dst == 10
    assert src == 0
    assert off == -4


def test_ldx_u32():
    h = _encode_to_hex("r1 = *(u32 *)(r2 + 4)\nexit")
    code, dst, src, off, _ = unpack(insns(h)[0])
    assert code == 0x61   # LDX u32
    assert dst == 1
    assert src == 2
    assert off == 4


def test_st_imm():
    h = _encode_to_hex("*(u32 *)(r1 + 0) = 42\nexit")
    code, dst, src, off, imm = unpack(insns(h)[0])
    assert code == 0x62   # ST u32 (immediate)
    assert dst == 1
    assert src == 0
    assert off == 0
    assert imm == 42


# ---------------------------------------------------------------------------
# lddw — hex-address wide immediates silently skipped
# ---------------------------------------------------------------------------

def test_lddw_silently_skipped():
    # r9 = 0xffff88800a11a800 is a map-pointer lddw; must not raise and not appear in output
    h = _encode_to_hex("r9 = 0xffff88800a11a800\nexit")
    assert h is not None
    parts = insns(h)
    assert len(parts) == 1          # only exit survives
    assert unpack(parts[0])[0] == 0x95


def test_lddw_mixed_rest_encodes():
    # lddw in the middle; surrounding instructions still encode correctly
    h = _encode_to_hex("r0 = 1\nr9 = 0xffff88800a11a800\nr1 = 2\nexit")
    assert h is not None
    parts = insns(h)
    assert len(parts) == 3   # r0=1, r1=2, exit (lddw skipped)
    assert unpack(parts[0])[0] == 0xb7
    assert unpack(parts[1])[0] == 0xb7
    assert unpack(parts[2])[0] == 0x95


# ---------------------------------------------------------------------------
# Noise filtering — verifier annotations must NOT appear in output
# ---------------------------------------------------------------------------

def test_verifier_state_lines_skipped():
    asm = """\
0: R1=ctx() R10=fp0
r0 = 0
exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 2


def test_mark_precise_lines_skipped():
    asm = """\
r0 = 1
mark_precise: frame0: last_idx 0 first_idx 0
exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 2


def test_state_transition_lines_skipped():
    asm = """\
r0 = 0
from 0 to 2: R0_w=0 R1=ctx() R10=fp0
exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 2


# ---------------------------------------------------------------------------
# Multi-instruction programs
# ---------------------------------------------------------------------------

def test_instruction_count():
    asm = """\
r0 = 1
r1 = 2
r0 += r1
r0 <<= 32
r0 s>>= 32
exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 6


def test_output_is_multiple_of_8_bytes():
    asm = """\
r0 = 42
r1 = 7
r0 *= r1
exit"""
    h = _encode_to_hex(asm)
    assert len(h) % 16 == 0


# ---------------------------------------------------------------------------
# call instruction
# ---------------------------------------------------------------------------

def test_call_numeric():
    h = _encode_to_hex("call 1\nexit")
    code, _, _, _, imm = unpack(insns(h)[0])
    assert code == 0x85
    assert imm == 1


def test_call_helper_name():
    # Named helper falls back to fn_id=1
    h = _encode_to_hex("call bpf_map_lookup_elem\nexit")
    code, _, _, _, imm = unpack(insns(h)[0])
    assert code == 0x85
    assert imm == 1
