"""
Tests for the pure-Python BPF bytecode encoder in reward.py.

All tests are offline (no VM, no SSH).  Each test verifies one behaviour of
_encode_to_hex or _encode_insn through the public _encode_to_hex interface.

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


def test_no_opcode_annotation_skipped():
    # Lines without (XX) opcode are skipped; bare "exit" is the exception.
    # "r0 = 5" has no opcode → skipped; "exit" → appended.
    h = _encode_to_hex("r0 = 5\nexit")
    assert h is not None
    assert unpack(insns(h)[-1])[0] == 0x95   # only the exit survives


def test_no_instructions_at_all_returns_none():
    # No opcode-annotated lines AND no bare "exit" → nothing to encode
    assert _encode_to_hex("r0 = 5\nr1 = 99") is None


# ---------------------------------------------------------------------------
# exit instruction
# ---------------------------------------------------------------------------

def test_exit_appended_when_missing():
    h = _encode_to_hex("0: (b7) r0 = 0")
    parts = insns(h)
    assert len(parts) == 2
    code, _, _, _, _ = unpack(parts[-1])
    assert code == 0x95


def test_exit_not_duplicated_when_present():
    h = _encode_to_hex("0: (b7) r0 = 0\n1: (95) exit")
    parts = insns(h)
    assert len(parts) == 2
    assert unpack(parts[-1])[0] == 0x95


def test_bare_exit_without_opcode_annotation():
    # A bare "exit" line (no N:(XX) prefix) should still be appended
    h = _encode_to_hex("0: (b7) r0 = 1\nexit")
    parts = insns(h)
    assert unpack(parts[-1])[0] == 0x95


# ---------------------------------------------------------------------------
# ground truth: r0=0; exit  (byte-identical to what clang produces)
# ---------------------------------------------------------------------------

def test_minimal_program_byte_identical_to_ground_truth():
    h = _encode_to_hex("0: (b7) r0 = 0\n1: (95) exit")
    assert h == "b7000000000000009500000000000000"


# ---------------------------------------------------------------------------
# MOV64 immediate  (b7)
# ---------------------------------------------------------------------------

def test_mov64_imm_positive():
    h = _encode_to_hex("0: (b7) r3 = 42\n1: (95) exit")
    code, dst, src, off, imm = unpack(insns(h)[0])
    assert code == 0xb7
    assert dst == 3
    assert src == 0
    assert off == 0
    assert imm == 42


def test_mov64_imm_negative():
    h = _encode_to_hex("0: (b7) r0 = -772319294\n1: (95) exit")
    _, _, _, _, imm = unpack(insns(h)[0])
    assert imm == -772319294


def test_mov64_imm_zero():
    h = _encode_to_hex("0: (b7) r5 = 0\n1: (95) exit")
    code, dst, _, _, imm = unpack(insns(h)[0])
    assert code == 0xb7 and dst == 5 and imm == 0


# ---------------------------------------------------------------------------
# MOV64 register  (bf)
# ---------------------------------------------------------------------------

def test_mov64_reg():
    h = _encode_to_hex("0: (bf) r2 = r5\n1: (95) exit")
    code, dst, src, off, imm = unpack(insns(h)[0])
    assert code == 0xbf
    assert dst == 2
    assert src == 5
    assert off == 0
    assert imm == 0


# ---------------------------------------------------------------------------
# ALU64 compound assignment — register and immediate
# ---------------------------------------------------------------------------

def test_add64_reg():
    h = _encode_to_hex("0: (b7) r0 = 1\n1: (b7) r1 = 2\n2: (0f) r0 += r1\n3: (95) exit")
    code, dst, src, _, _ = unpack(insns(h)[2])
    assert code == 0x0f
    assert dst == 0
    assert src == 1


def test_add64_imm():
    h = _encode_to_hex("0: (b7) r0 = 0\n1: (07) r0 += 100\n2: (95) exit")
    code, dst, src, _, imm = unpack(insns(h)[1])
    assert code == 0x07
    assert dst == 0
    assert src == 0
    assert imm == 100


def test_lsh64_imm():
    h = _encode_to_hex("0: (b7) r0 = 1\n1: (67) r0 <<= 32\n2: (95) exit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0x67
    assert dst == 0
    assert imm == 32


def test_arsh64_imm():
    h = _encode_to_hex("0: (b7) r0 = 1\n1: (c7) r0 s>>= 32\n2: (95) exit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0xc7
    assert dst == 0
    assert imm == 32


# ---------------------------------------------------------------------------
# ALU32 (w registers)
# ---------------------------------------------------------------------------

def test_mov32_imm():
    h = _encode_to_hex("0: (b4) w3 = 255\n1: (95) exit")
    code, dst, src, _, imm = unpack(insns(h)[0])
    assert code == 0xb4
    assert dst == 3
    assert src == 0
    assert imm == 255


def test_xor32_imm():
    h = _encode_to_hex("0: (b7) r0 = 0\n1: (a4) w0 ^= 128\n2: (95) exit")
    code, dst, _, _, imm = unpack(insns(h)[1])
    assert code == 0xa4
    assert dst == 0
    assert imm == 128


# ---------------------------------------------------------------------------
# JMP — unconditional goto
# ---------------------------------------------------------------------------

def test_goto_positive():
    h = _encode_to_hex("0: (05) goto +1\n1: (b7) r0 = 99\n2: (95) exit")
    code, _, _, off, _ = unpack(insns(h)[0])
    assert code == 0x05
    assert off == 1


def test_goto_negative():
    # goto -1 from instruction 1 loops back to itself (instruction 0 next = 1+1-1=1? no,
    # next = 1+1+(-1) = 1, which would be self — verifier would reject, but encoding must be correct)
    h = _encode_to_hex("0: (b7) r0 = 0\n1: (05) goto -1\n2: (95) exit")
    _, _, _, off, _ = unpack(insns(h)[1])
    assert off == -1


# ---------------------------------------------------------------------------
# JMP — conditional
# ---------------------------------------------------------------------------

def test_jeq_imm():
    # if r1 == 5 goto +2
    h = _encode_to_hex("0: (b7) r1 = 5\n1: (15) if r1 == 5 goto +2\n2: (b7) r0 = 0\n3: (95) exit\n4: (95) exit")
    code, dst, src, off, imm = unpack(insns(h)[1])
    assert code == 0x15
    assert dst == 1
    assert src == 0
    assert off == 2
    assert imm == 5


def test_jgt_reg():
    # if r3 > r4 goto +1
    h = _encode_to_hex("0: (b7) r3 = 1\n1: (b7) r4 = 2\n2: (2d) if r3 > r4 goto +1\n3: (95) exit\n4: (95) exit")
    code, dst, src, off, _ = unpack(insns(h)[2])
    assert code == 0x2d
    assert dst == 3
    assert src == 4
    assert off == 1


# ---------------------------------------------------------------------------
# Memory load / store
# ---------------------------------------------------------------------------

def test_ldx_dw():
    # r1 = *(u64 *)(r10 -8)
    h = _encode_to_hex("0: (b7) r0 = 1\n1: (7b) *(u64 *)(r10 -8) = r0\n2: (79) r1 = *(u64 *)(r10 -8)\n3: (95) exit")
    # store: insn 1
    code_st, dst_st, src_st, off_st, _ = unpack(insns(h)[1])
    assert code_st == 0x7b
    assert dst_st == 10   # r10
    assert src_st == 0    # r0
    assert off_st == -8
    # load: insn 2
    code_ld, dst_ld, src_ld, off_ld, _ = unpack(insns(h)[2])
    assert code_ld == 0x79
    assert dst_ld == 1    # r1
    assert src_ld == 10   # r10
    assert off_ld == -8


def test_stx_w():
    # *(u32 *)(r10 -4) = r0
    h = _encode_to_hex("0: (b7) r0 = 7\n1: (63) *(u32 *)(r10 -4) = r0\n2: (95) exit")
    code, dst, src, off, _ = unpack(insns(h)[1])
    assert code == 0x63
    assert dst == 10
    assert src == 0
    assert off == -4


# ---------------------------------------------------------------------------
# Noise filtering — verifier annotations must NOT appear in output
# ---------------------------------------------------------------------------

def test_verifier_state_lines_skipped():
    asm = """\
func#0 @0
0: R1=ctx() R10=fp0
0: (b7) r0 = 0 ; R0_w=0
1: (95) exit"""
    h = _encode_to_hex(asm)
    # Only 2 real instructions
    assert len(insns(h)) == 2


def test_mark_precise_lines_skipped():
    asm = """\
0: (b7) r0 = 1 ; R0_w=1
mark_precise: frame0: last_idx 0 first_idx 0
1: (95) exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 2


def test_state_transition_lines_skipped():
    # Lines like "from 3 to 5: R0=..." appear in longer verifier logs
    asm = """\
0: (b7) r0 = 0
from 0 to 2: R0_w=0 R1=ctx() R10=fp0
1: (95) exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 2


def test_inline_comment_stripped():
    # "; R0_w=0xdeadbeef" must not affect the encoded imm
    h1 = _encode_to_hex("0: (b7) r0 = 5\n1: (95) exit")
    h2 = _encode_to_hex("0: (b7) r0 = 5 ; R0_w=0x5\n1: (95) exit")
    assert h1 == h2


# ---------------------------------------------------------------------------
# Multi-instruction programs
# ---------------------------------------------------------------------------

def test_instruction_count():
    asm = """\
0: (b7) r0 = 1
1: (b7) r1 = 2
2: (0f) r0 += r1
3: (67) r0 <<= 32
4: (c7) r0 s>>= 32
5: (95) exit"""
    h = _encode_to_hex(asm)
    assert len(insns(h)) == 6


def test_output_is_multiple_of_8_bytes():
    asm = """\
0: (b7) r0 = 42
1: (b7) r1 = 7
2: (2f) r0 *= r1
3: (95) exit"""
    h = _encode_to_hex(asm)
    assert len(h) % 16 == 0


# ---------------------------------------------------------------------------
# call instruction
# ---------------------------------------------------------------------------

def test_call_numeric():
    h = _encode_to_hex("0: (85) call 1\n1: (95) exit")
    code, _, _, _, imm = unpack(insns(h)[0])
    assert code == 0x85
    assert imm == 1
