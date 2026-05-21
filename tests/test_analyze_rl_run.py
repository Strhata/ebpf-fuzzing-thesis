"""Tests for tools/analyze_rl_run.py."""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from analyze_rl_run import aggregate, parse_log, write_pcs_csv, write_tiers_csv

_SINGLE_SEGMENT_LOG = """\
2026-01-01 00:00:00,000 batch=0 i=0 ACCETTATO new_pcs=3
2026-01-01 00:00:01,000 batch=0 i=1 RIFIUTATO insns=5
2026-01-01 00:00:02,000 batch=0 i=2 ENCODE_FAIL insns=0
2026-01-01 00:00:03,000 batch=0 i=3 COMPILE_FAIL insns=0
2026-01-01 00:01:00,000 batch=1 i=0 ACCETTATO new_pcs=0
2026-01-01 00:01:01,000 batch=1 i=1 ACCETTATO new_pcs=2
2026-01-01 00:01:02,000 batch=1 i=2 SSH_TIMEOUT
2026-01-01 00:01:03,000 batch=1 i=3 RIFIUTATO insns=3
"""

_TWO_SEGMENT_LOG = """\
2026-01-01 00:00:00,000 batch=0 i=0 RIFIUTATO insns=5
2026-01-01 00:00:01,000 batch=1 i=0 ACCETTATO new_pcs=1
2026-01-01 00:00:02,000 batch=1 i=1 RIFIUTATO insns=3
2026-01-01 00:01:00,000 batch=0 i=0 ACCETTATO new_pcs=5
2026-01-01 00:01:01,000 batch=0 i=1 ENCODE_FAIL insns=0
2026-01-01 00:01:02,000 batch=1 i=0 ACCETTATO new_pcs=0
2026-01-01 00:01:03,000 batch=1 i=1 ACCETTATO new_pcs=2
"""


def _write_log(content: str, directory: Path) -> Path:
    p = directory / "grpo_completions.log"
    p.write_text(content)
    return p


def test_single_segment_detected():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_SINGLE_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    assert len(segs) == 1


def test_two_segments_detected():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_TWO_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    assert len(segs) == 2


def test_segment_restart_splits_on_batch0_after_nonzero():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_TWO_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    # first segment: batch 0 and 1 from first invocation
    assert all(r["batch"] in (0, 1) for r in segs[0])
    # second segment starts fresh at batch 0
    assert segs[1][0]["batch"] == 0


def test_tier_counts_batch0():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_SINGLE_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    rows = aggregate(segs[0])
    b0 = next(r for r in rows if r["batch"] == 0)
    assert b0["new_pcs"] == 1    # ACCETTATO new_pcs=3
    assert b0["rejected"] == 1   # RIFIUTATO
    assert b0["encode_fail"] == 2  # ENCODE_FAIL + COMPILE_FAIL
    assert b0["valid"] == 0
    assert b0["crash"] == 0


def test_valid_vs_new_pcs_split():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_SINGLE_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    rows = aggregate(segs[0])
    b1 = next(r for r in rows if r["batch"] == 1)
    # ACCETTATO new_pcs=0 → valid, ACCETTATO new_pcs=2 → new_pcs
    assert b1["valid"] == 1
    assert b1["new_pcs"] == 1
    assert b1["crash"] == 1   # SSH_TIMEOUT
    assert b1["rejected"] == 1


def test_cumulative_pcs_accumulates():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_SINGLE_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    rows = aggregate(segs[0])
    # batch 0: 3 new_pcs, batch 1: 2 new_pcs → cumulative at batch 1 = 5
    assert rows[0]["cumulative_pcs"] == 3
    assert rows[1]["cumulative_pcs"] == 5


def test_tiers_csv_columns():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_SINGLE_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
        rows = [aggregate(s) for s in segs]
        out = Path(d) / "tiers.csv"
        write_tiers_csv(rows, out)
        with out.open() as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            data = list(reader)
    assert "segment" in cols
    assert "batch" in cols
    assert "new_pcs" in cols
    assert "valid" in cols
    assert "rejected" in cols
    assert "encode_fail" in cols
    assert "reward_mean" in cols
    assert "reward_std" in cols
    assert len(data) == 2  # 2 batches in single segment


def test_pcs_csv_columns():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_SINGLE_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
        rows = [aggregate(s) for s in segs]
        out = Path(d) / "pcs.csv"
        write_pcs_csv(rows, out)
        with out.open() as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
    assert "segment" in cols
    assert "batch" in cols
    assert "cumulative_pcs" in cols


def test_two_segment_cumulative_pcs_independent():
    with tempfile.TemporaryDirectory() as d:
        log = _write_log(_TWO_SEGMENT_LOG, Path(d))
        segs = parse_log(log)
    rows0 = aggregate(segs[0])
    rows1 = aggregate(segs[1])
    # segment 0: batch0=RIFIUTATO (0 new_pcs), batch1=ACCETTATO new_pcs=1 → cumulative=[0,1]
    assert rows0[0]["cumulative_pcs"] == 0
    assert rows0[1]["cumulative_pcs"] == 1
    # segment 1: batch0=5 new_pcs, batch1=2 new_pcs → cumulative=[5,7]
    assert rows1[0]["cumulative_pcs"] == 5
    assert rows1[1]["cumulative_pcs"] == 7
