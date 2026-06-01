"""
Tests for ml/enrich_dataset.py.

All tests are offline — SSH is mocked; no VM required.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

from enrich_dataset import (
    assign_bin,
    assign_novelty_bin,
    compute_novelty_thresholds,
    compute_tertile_thresholds,
    enrich,
)


# ---------------------------------------------------------------------------
# Mock SSH client
# ---------------------------------------------------------------------------

class _MockSSH:
    """Returns preset (pc_count, verdict[, pcs]) keyed by hex_str.

    If the entry has only 2 elements, pcs defaults to list(range(pc_count)).
    If the entry has 3 elements, the third is used as the explicit PC list —
    useful for novelty tests that need precise control over which PCs overlap.
    """

    def __init__(self, results: dict[str, tuple]):
        self._results = results

    def run(self, cmd: str, timeout: int = 30):
        hex_str = cmd.split()[-1]
        entry = self._results.get(hex_str, (0, "ERROR", []))
        if len(entry) == 2:
            pc_count, verdict = entry
            pcs = list(range(pc_count))
        else:
            pc_count, verdict, pcs = entry
        body = json.dumps({"verdict": verdict, "pcs": pcs})
        return 0, body, ""


def _ssh_factory(results: dict):
    return lambda: _MockSSH(results)


def _sample(hex_str: str, is_valid: bool, **extra) -> dict:
    return {"bytecode_hex": hex_str, "is_valid": is_valid, **extra}


# ---------------------------------------------------------------------------
# compute_tertile_thresholds
# ---------------------------------------------------------------------------

def test_linear_scale_on_uniform_distribution():
    counts = list(range(1, 101))
    _, _, scale = compute_tertile_thresholds(counts)
    assert scale == "linear"


def test_log_scale_triggered_on_heavy_right_skew():
    counts = [1] * 97 + [10000, 20000, 30000]
    _, _, scale = compute_tertile_thresholds(counts)
    assert scale == "log"


def test_skew_ratio_boundary():
    t33, _, _ = compute_tertile_thresholds([1] * 99 + [100_000])
    p0, p100 = 1, 100_000
    ratio = (t33 - p0) / (p100 - p0)
    assert ratio < 0.05


def test_thresholds_ordered():
    counts = list(range(0, 300))
    t33, t67, _ = compute_tertile_thresholds(counts)
    assert t33 < t67


# ---------------------------------------------------------------------------
# assign_bin
# ---------------------------------------------------------------------------

def test_assign_bin_low():
    assert assign_bin(0, t33=10.0, t67=20.0, scale="linear") == "low"


def test_assign_bin_medium():
    assert assign_bin(15, t33=10.0, t67=20.0, scale="linear") == "medium"


def test_assign_bin_high():
    assert assign_bin(25, t33=10.0, t67=20.0, scale="linear") == "high"


def test_assign_bin_log_scale_produces_valid_bin():
    t33, t67, scale = compute_tertile_thresholds([1] * 97 + [10000, 20000, 30000])
    assert scale == "log"
    for pc in (0, 1, 500, 10000):
        b = assign_bin(pc, t33, t67, scale)
        assert b in ("low", "medium", "high"), f"pc={pc} → {b!r}"


# ---------------------------------------------------------------------------
# compute_novelty_thresholds / assign_novelty_bin
# ---------------------------------------------------------------------------

def test_novelty_thresholds_ordered():
    scores = [i / 100 for i in range(100)]
    t33, t67 = compute_novelty_thresholds(scores)
    assert t33 < t67


def test_assign_novelty_bin_low():
    assert assign_novelty_bin(0.1, t33=0.3, t67=0.6) == "low"


def test_assign_novelty_bin_medium():
    assert assign_novelty_bin(0.5, t33=0.3, t67=0.6) == "medium"


def test_assign_novelty_bin_high():
    assert assign_novelty_bin(0.8, t33=0.3, t67=0.6) == "high"


# ---------------------------------------------------------------------------
# enrich() — coverage acceptance tests
# ---------------------------------------------------------------------------

def test_enriched_output_has_required_fields():
    samples = [_sample("aa", True), _sample("bb", False)]
    ssh_r = {"aa": (50, "ACCEPTED"), "bb": (20, "REJECTED")}
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)

    assert len(enriched) == 2
    for row in enriched:
        assert "pc_count" in row
        assert "coverage_bin" in row
        assert isinstance(row["pc_count"], int)
        assert row["coverage_bin"] in ("low", "medium", "high")


def test_thresholds_computed_from_valid_only():
    samples = [
        _sample("v1", True),
        _sample("v2", True),
        _sample("v3", True),
        _sample("i1", False),
        _sample("i2", False),
    ]
    ssh_r = {
        "v1": (100, "ACCEPTED"),
        "v2": (200, "ACCEPTED"),
        "v3": (300, "ACCEPTED"),
        "i1": (5,   "REJECTED"),
        "i2": (5,   "REJECTED"),
    }
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)

    for row in enriched:
        if row["bytecode_hex"] in ("i1", "i2"):
            assert row["coverage_bin"] == "low"


def test_invalid_samples_receive_bin_not_null():
    samples = [_sample("vv", True), _sample("ii", False)]
    ssh_r = {"vv": (100, "ACCEPTED"), "ii": (30, "REJECTED")}
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    ii = next(r for r in enriched if r["bytecode_hex"] == "ii")
    assert ii.get("coverage_bin") in ("low", "medium", "high")


def test_original_fields_preserved():
    samples = [_sample("aa", True, verifier_log="exit", error_reason="")]
    ssh_r = {"aa": (50, "ACCEPTED")}
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    assert enriched[0]["verifier_log"] == "exit"
    assert enriched[0]["error_reason"] == ""


def test_input_file_not_modified():
    samples = [_sample("aa", True)]
    original_keys = set(samples[0].keys())
    ssh_r = {"aa": (50, "ACCEPTED")}
    enrich(samples, _ssh_factory(ssh_r), workers=1)
    assert set(samples[0].keys()) == original_keys


def test_skew_detection_log_scale_in_result():
    samples = [_sample(f"{i:02x}", True) for i in range(100)]
    ssh_r = {f"{i:02x}": (1, "ACCEPTED") for i in range(97)}
    ssh_r.update({f"{i:02x}": (10_000 + i, "ACCEPTED") for i in range(97, 100)})
    _, scale, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    assert scale == "log"


def test_uniform_distribution_gives_linear_scale():
    samples = [_sample(f"{i:02x}", True) for i in range(100)]
    ssh_r = {f"{i:02x}": (i, "ACCEPTED") for i in range(100)}
    _, scale, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    assert scale == "linear"


def test_bin_distribution_covers_all_three_tiers():
    samples = [_sample(f"{i:03x}", True) for i in range(99)]
    ssh_r = {f"{i:03x}": (i, "ACCEPTED") for i in range(99)}
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    bins = {r["coverage_bin"] for r in enriched}
    assert bins == {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# enrich() — novelty tests
# ---------------------------------------------------------------------------

def test_novelty_fields_present():
    """enriched rows must have novelty_score (float) and novelty_bin (str)."""
    samples = [_sample("aa", True), _sample("bb", True)]
    ssh_r = {
        "aa": (3, "ACCEPTED", [0, 1, 2]),
        "bb": (3, "ACCEPTED", [3, 4, 5]),
    }
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    for row in enriched:
        assert "novelty_score" in row
        assert "novelty_bin" in row
        assert isinstance(row["novelty_score"], float)
        assert row["novelty_bin"] in ("low", "medium", "high")


def test_novelty_score_in_unit_interval():
    """novelty_score must be in [0, 1] for all programs."""
    samples = [_sample(f"{i:02x}", True) for i in range(10)]
    ssh_r = {f"{i:02x}": (5, "ACCEPTED", list(range(i, i + 5))) for i in range(10)}
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    for row in enriched:
        assert 0.0 <= row["novelty_score"] <= 1.0, row["novelty_score"]


def test_fully_shared_pcs_give_low_novelty():
    """Two programs with identical PC sets — each PC appears in all programs → score ≈ 0."""
    samples = [_sample("aa", True), _sample("bb", True)]
    shared_pcs = [100, 200, 300]
    ssh_r = {
        "aa": (3, "ACCEPTED", shared_pcs),
        "bb": (3, "ACCEPTED", shared_pcs),
    }
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    for row in enriched:
        # pc_freq[pc] = 2 = N → 1 - 2/2 = 0 for every PC
        assert row["novelty_score"] == pytest.approx(0.0)


def test_unique_pcs_give_high_novelty():
    """Two programs with disjoint PC sets — each PC appears in only 1 of 2 programs → score = 0.5."""
    samples = [_sample("aa", True), _sample("bb", True)]
    ssh_r = {
        "aa": (3, "ACCEPTED", [0, 1, 2]),
        "bb": (3, "ACCEPTED", [3, 4, 5]),
    }
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    for row in enriched:
        # pc_freq[pc] = 1, N = 2 → 1 - 1/2 = 0.5 for every PC
        assert row["novelty_score"] == pytest.approx(0.5)


def test_unique_program_scores_higher_than_common():
    """Program C (unique PCs) should score higher than A and B (shared PCs)."""
    samples = [
        _sample("aa", True),  # PCs shared with bb
        _sample("bb", True),  # PCs shared with aa
        _sample("cc", True),  # PCs shared with nobody
    ]
    ssh_r = {
        "aa": (2, "ACCEPTED", [0, 1]),
        "bb": (2, "ACCEPTED", [0, 1]),
        "cc": (2, "ACCEPTED", [99, 100]),
    }
    enriched, _, _ = enrich(samples, _ssh_factory(ssh_r), workers=1)
    by_hex = {r["bytecode_hex"]: r for r in enriched}
    assert by_hex["cc"]["novelty_score"] > by_hex["aa"]["novelty_score"]
    assert by_hex["cc"]["novelty_score"] > by_hex["bb"]["novelty_score"]


def test_novelty_scale_is_always_linear():
    """enrich() returns 'linear' as the novelty_scale regardless of distribution."""
    samples = [_sample("aa", True), _sample("bb", True)]
    ssh_r = {"aa": (2, "ACCEPTED", [0, 1]), "bb": (2, "ACCEPTED", [2, 3])}
    _, _, novelty_scale = enrich(samples, _ssh_factory(ssh_r), workers=1)
    assert novelty_scale == "linear"
