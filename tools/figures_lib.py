#!/usr/bin/env python3
"""
figures_lib.py — pure data-prep functions for the thesis figures.

No I/O, no matplotlib: just the deterministic transforms the figure scripts need,
factored out so they can be unit-tested without candidate files, a GPU, or the VM.

Used by:
  - tools/plot_depth_collapse.py   (F1: instruction-count histograms)
  - tools/plot_saturation.py       (F2: cumulative unique-valid-PC curve)
"""

from __future__ import annotations

from typing import Iterable, Sequence


def insn_count_from_hex(hex_str: str | None) -> int:
    """Number of BPF instructions encoded in a bytecode hex string.

    The codebase convention (benchmark_lib.ComplexityAnalyzer, ml/reward.py) is one
    BPF instruction per 8 bytes = 16 hex characters. We floor-divide, so a trailing
    half-instruction (malformed tail) is ignored rather than counted. Whitespace and
    a leading ``0x`` are tolerated. Returns 0 for empty/None input.
    """
    if not hex_str:
        return 0
    s = hex_str.strip()
    if s[:2].lower() == "0x":
        s = s[2:]
    return len(s) // 16


def cumulative_unique_curve(
    pc_sets: Sequence[Iterable[int]],
    sample_points: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Cumulative count of distinct PCs as programs are accumulated in order.

    Given per-program PC collections in generation order, returns ``(n, unique)``
    pairs: after the first ``n`` programs, how many distinct PCs have been seen in
    total. The curve is monotonically non-decreasing and bounded by the global
    distinct-PC count — the shape that makes saturation visible (F2).

    ``sample_points`` selects which prefix lengths to emit (e.g. log-spaced); when
    omitted, every prefix length ``1..len(pc_sets)`` is emitted. Points are clamped
    to ``[1, len(pc_sets)]`` and de-duplicated.
    """
    n_total = len(pc_sets)
    if n_total == 0:
        return []

    if sample_points is None:
        points = list(range(1, n_total + 1))
    else:
        points = sorted({min(max(p, 1), n_total) for p in sample_points})

    curve: list[tuple[int, int]] = []
    seen: set[int] = set()
    cursor = 0  # number of programs folded into `seen` so far
    for p in points:
        while cursor < p:
            seen.update(pc_sets[cursor])
            cursor += 1
        curve.append((p, len(seen)))
    return curve
