"""Unit tests for tools/figures_lib.py — the pure data-prep behind the thesis figures.

These exercise external behaviour only (counts, curve shape), no I/O, no VM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from figures_lib import cumulative_unique_curve, insn_count_from_hex  # noqa: E402


# --- insn_count_from_hex -----------------------------------------------------

def test_insn_count_one_instruction_is_16_hex_chars():
    assert insn_count_from_hex("b7" + "0" * 14) == 1


def test_insn_count_counts_whole_instructions():
    assert insn_count_from_hex("a" * 48) == 3  # 48 hex / 16 = 3


def test_insn_count_empty_and_none_are_zero():
    assert insn_count_from_hex("") == 0
    assert insn_count_from_hex(None) == 0


def test_insn_count_tolerates_0x_prefix_and_whitespace():
    assert insn_count_from_hex("  0x" + "f" * 32 + "\n") == 2


def test_insn_count_floors_a_trailing_half_instruction():
    assert insn_count_from_hex("f" * 16 + "ff") == 1  # 18 hex → still 1 whole insn


# --- cumulative_unique_curve -------------------------------------------------

def test_cumulative_curve_is_monotonic_and_ends_at_total_distinct():
    pc_sets = [{1, 2}, {2, 3}, {3, 4}]
    curve = cumulative_unique_curve(pc_sets)
    ys = [u for _, u in curve]
    assert curve[0] == (1, 2)
    assert ys == sorted(ys)            # non-decreasing
    assert curve[-1] == (3, 4)         # {1,2,3,4}


def test_cumulative_curve_saturates_when_no_new_pcs_arrive():
    pc_sets = [{1, 2, 3}, {1, 2}, {3}]  # nothing new after program 1
    curve = cumulative_unique_curve(pc_sets)
    assert [u for _, u in curve] == [3, 3, 3]


def test_cumulative_curve_honours_sample_points_clamped_and_deduped():
    pc_sets = [{i} for i in range(10)]
    curve = cumulative_unique_curve(pc_sets, sample_points=[0, 1, 5, 5, 99])
    # 0 clamps to 1, 99 clamps to 10, duplicate 5 collapses
    assert [n for n, _ in curve] == [1, 5, 10]
    assert curve[-1] == (10, 10)


def test_cumulative_curve_empty_input():
    assert cumulative_unique_curve([]) == []
