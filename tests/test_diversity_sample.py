"""Offline tests for tools/diversity_sample.py — the batched validation mapping.

The generate→validate end-to-end run needs a GPU + KCOV VM; here we test the pure
result-mapping (encode-fail → ERROR, batched results mapped back by index, chunking,
whole-chunk failure) by patching the batch-validate wrapper.
"""

import json
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


def test_candidates_roundtrip_preserves_meta_and_records(tmp_path):
    meta = {"model": "m", "n": 3, "seed": 42, "tokens_per_sec": 123.4, "peak_run_gpu_mb": 999.0}
    assemblies = ["asm0", "asm1 with\nnewline", "asm2"]
    hexes = ["aa", None, "cc"]  # index 1 failed to encode → null in the artifact

    path = tmp_path / "candidates.jsonl"
    ds.write_candidates(path, meta, assemblies, hexes)
    got_meta, got_asm, got_hex = ds.read_candidates(path)

    assert got_meta == meta
    assert got_asm == assemblies
    assert got_hex == hexes


def test_candidates_first_line_is_meta_header(tmp_path):
    path = tmp_path / "candidates.jsonl"
    ds.write_candidates(path, {"model": "m"}, ["asm0"], ["aa"])
    first = path.read_text().splitlines()[0]
    assert json.loads(first) == {"_meta": {"model": "m"}}


def test_aggregate_streaming_known_values():
    from benchmark_lib import ComplexityResult, GeneratorResult

    assemblies = ["", "", ""]          # complexity patched out below
    hexes = ["a", "b", "c"]            # all encoded → all sent to the VM
    batch = [
        {"verdict": "ACCEPTED", "pcs": [1, 2, 2, 3]},   # distinct {1,2,3}, len 4
        {"verdict": "REJECTED", "pcs": [2, 4]},          # distinct {2,4}
        {"verdict": "ACCEPTED", "pcs": []},              # distinct {}
    ]
    comp = ComplexityResult(encoded=False, insn_count=None, opcode_diversity=None, jump_count=None)

    with patch.object(ds, "analyze", return_value=comp):
        kpis, valid_only = ds.aggregate_streaming(
            assemblies, hexes,
            batch_validate=lambda hx: batch,
            timing=GeneratorResult(assemblies=assemblies, tokens_per_sec=42.0),
            peak_run_gpu_mb=7.0, chunk_size=500,
        )

    # union over ALL programs = {1,2,3,4}; freq = {1:1, 2:2, 3:1, 4:1}; n=3
    assert kpis.total_unique_pcs == 4
    assert kpis.max_pcs == 4
    assert abs(kpis.pass_rate - 2 / 3) < 1e-9
    # novelty = mean(1 - freq/3) over {1,2,3,4} = (2/3 + 1/3 + 2/3 + 2/3)/4
    assert abs(kpis.novelty_score - (2/3 + 1/3 + 2/3 + 2/3) / 4) < 1e-9
    assert abs(kpis.avg_pcs - 2.0) < 1e-9          # (4 + 0) / 2 accepted
    # valid-only: accepted programs 0 and 2 → union {1,2,3}
    assert valid_only["n_accepted"] == 2
    assert valid_only["valid_unique_pcs"] == 3
    assert abs(valid_only["avg_distinct_pcs_per_valid"] - 1.5) < 1e-9  # (3 + 0) / 2


def test_aggregate_streaming_whole_chunk_failure_no_pcs():
    from benchmark_lib import ComplexityResult, GeneratorResult

    comp = ComplexityResult(encoded=False, insn_count=None, opcode_diversity=None, jump_count=None)
    with patch.object(ds, "analyze", return_value=comp):
        kpis, valid_only = ds.aggregate_streaming(
            ["", ""], ["a", "b"],
            batch_validate=lambda hx: None,   # whole-chunk VM failure
            timing=GeneratorResult(assemblies=["", ""], tokens_per_sec=0.0),
            peak_run_gpu_mb=0.0,
        )
    assert kpis.total_unique_pcs == 0
    assert kpis.pass_rate == 0.0
    assert valid_only == {"n_accepted": 0, "valid_unique_pcs": 0, "avg_distinct_pcs_per_valid": 0.0}
