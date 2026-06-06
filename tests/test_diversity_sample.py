"""Offline tests for tools/diversity_sample.py — the batched validation mapping.

The generate→validate end-to-end run needs a GPU + KCOV VM; here we test the pure
result-mapping (encode-fail → ERROR, batched results mapped back by index, chunking,
whole-chunk failure) by patching the batch-validate wrapper.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

import diversity_sample as ds  # noqa: E402
from benchmark_lib import ValidationResult  # noqa: E402


def test_validations_map_back_by_index_with_encode_fails():
    hexes = ["aa", None, "bb", "cc"]  # index 1 failed to encode

    def fake_batch(hex_list, ssh):
        assert hex_list == ["aa", "bb", "cc"]  # only encoded ones are sent
        return [
            {"verdict": "ACCEPTED", "pcs": [1, 2]},
            {"verdict": "REJECTED", "pcs": [3]},
            {"verdict": "ACCEPTED", "pcs": []},
        ]

    with patch.object(ds, "_batch_validate_fn", side_effect=fake_batch):
        res = ds.validations_from_batch(hexes, ssh=None)

    assert res[0] == ValidationResult("ACCEPTED", [1, 2])
    assert res[1] == ValidationResult("ERROR", [])       # encode fail, never sent
    assert res[2] == ValidationResult("REJECTED", [3])
    assert res[3] == ValidationResult("ACCEPTED", [])


def test_whole_chunk_failure_leaves_error():
    hexes = ["aa", "bb"]
    with patch.object(ds, "_batch_validate_fn", return_value=None):
        res = ds.validations_from_batch(hexes, ssh=None)
    assert all(r.verdict == "ERROR" and r.pcs == [] for r in res)


def test_chunking_one_call_per_chunk():
    hexes = ["a", "b", "c", "d", "e"]  # 5 programs, chunk_size 2 → 3 calls
    calls = []

    def fake_batch(hex_list, ssh):
        calls.append(list(hex_list))
        return [{"verdict": "ACCEPTED", "pcs": []} for _ in hex_list]

    with patch.object(ds, "_batch_validate_fn", side_effect=fake_batch):
        res = ds.validations_from_batch(hexes, ssh=None, chunk_size=2)

    assert [len(c) for c in calls] == [2, 2, 1]
    assert len(res) == 5
    assert all(r.verdict == "ACCEPTED" for r in res)
