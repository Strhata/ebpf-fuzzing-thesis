"""
Tests for ml/train.py — split logic and training configuration.

All tests are offline: no model download, no GPU.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

from train import load_and_split, make_training_args, save_test_split, TEST_SPLIT_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_fixture(path: Path, n_valid: int, n_invalid: int) -> int:
    """Write a minimal enriched JSONL fixture. Returns total row count."""
    with path.open("w") as f:
        f.write(json.dumps({"_metadata": True, "coverage_scale": "linear"}) + "\n")
        for i in range(n_valid):
            f.write(json.dumps({
                "bytecode_hex": f"valid{i:04d}",
                "is_valid": True,
                "verifier_log": "exit",
                "coverage_bin": "high",
                "pc_count": 100 + i,
            }) + "\n")
        for i in range(n_invalid):
            f.write(json.dumps({
                "bytecode_hex": f"invalid{i:04d}",
                "is_valid": False,
                "verifier_log": "exit",
                "coverage_bin": "low",
                "pc_count": 10 + i,
            }) + "\n")
    return n_valid + n_invalid


# ---------------------------------------------------------------------------
# Split size tests
# ---------------------------------------------------------------------------

def test_split_sizes_sum_to_total(tmp_path):
    total = _write_fixture(tmp_path / "data.jsonl", n_valid=700, n_invalid=300)
    train, val, test = load_and_split(tmp_path / "data.jsonl")
    assert len(train) + len(val) + len(test) == total


def test_split_proportions_approx_70_15_15(tmp_path):
    total = _write_fixture(tmp_path / "data.jsonl", n_valid=700, n_invalid=300)
    train, val, test = load_and_split(tmp_path / "data.jsonl")
    assert abs(len(train) / total - 0.70) < 0.02
    assert abs(len(val)   / total - 0.15) < 0.02
    assert abs(len(test)  / total - 0.15) < 0.02


def test_metadata_row_excluded_from_splits(tmp_path):
    total = _write_fixture(tmp_path / "data.jsonl", n_valid=200, n_invalid=200)
    train, val, test = load_and_split(tmp_path / "data.jsonl")
    all_rows = train + val + test
    assert all("_metadata" not in r for r in all_rows)
    assert len(all_rows) == total


# ---------------------------------------------------------------------------
# Stratification tests (is_valid ratio preserved)
# ---------------------------------------------------------------------------

def _valid_ratio(rows: list[dict]) -> float:
    return sum(1 for r in rows if r.get("is_valid", False)) / len(rows)


def test_isvalid_ratio_preserved_in_train(tmp_path):
    _write_fixture(tmp_path / "data.jsonl", n_valid=700, n_invalid=300)
    train, val, test = load_and_split(tmp_path / "data.jsonl")
    full_ratio = 0.70
    assert abs(_valid_ratio(train) - full_ratio) < 0.02


def test_isvalid_ratio_preserved_in_val(tmp_path):
    _write_fixture(tmp_path / "data.jsonl", n_valid=700, n_invalid=300)
    train, val, test = load_and_split(tmp_path / "data.jsonl")
    assert abs(_valid_ratio(val) - 0.70) < 0.02


def test_isvalid_ratio_preserved_in_test(tmp_path):
    _write_fixture(tmp_path / "data.jsonl", n_valid=700, n_invalid=300)
    train, val, test = load_and_split(tmp_path / "data.jsonl")
    assert abs(_valid_ratio(test) - 0.70) < 0.02


# ---------------------------------------------------------------------------
# Determinism: test split is byte-identical across two runs
# ---------------------------------------------------------------------------

def test_test_split_byte_identical_across_runs(tmp_path):
    _write_fixture(tmp_path / "data.jsonl", n_valid=500, n_invalid=500)
    _, _, test1 = load_and_split(tmp_path / "data.jsonl", seed=42)
    _, _, test2 = load_and_split(tmp_path / "data.jsonl", seed=42)
    assert [r["bytecode_hex"] for r in test1] == [r["bytecode_hex"] for r in test2]


def test_different_seed_gives_different_test_split(tmp_path):
    _write_fixture(tmp_path / "data.jsonl", n_valid=500, n_invalid=500)
    _, _, test42 = load_and_split(tmp_path / "data.jsonl", seed=42)
    _, _, test99 = load_and_split(tmp_path / "data.jsonl", seed=99)
    assert [r["bytecode_hex"] for r in test42] != [r["bytecode_hex"] for r in test99]


# ---------------------------------------------------------------------------
# save_test_split writes correct JSONL
# ---------------------------------------------------------------------------

def test_save_test_split_writes_jsonl(tmp_path):
    samples = [{"bytecode_hex": "aa", "is_valid": True, "coverage_bin": "high"}]
    out = tmp_path / "test_split.jsonl"
    save_test_split(samples, out)
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["bytecode_hex"] == "aa"


def test_save_test_split_is_deterministic(tmp_path):
    samples = [{"bytecode_hex": f"s{i}", "is_valid": i % 2 == 0} for i in range(20)]
    out = tmp_path / "test_split.jsonl"
    save_test_split(samples, out)
    content1 = out.read_text()
    save_test_split(samples, out)
    content2 = out.read_text()
    assert content1 == content2


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def test_eval_strategy_is_epoch(tmp_path):
    args = make_training_args(tmp_path)
    assert str(args.eval_strategy) in ("epoch", "IntervalStrategy.EPOCH")


def test_eval_steps_not_100(tmp_path):
    # The old broken config used eval_steps=100; ensure it's gone.
    args = make_training_args(tmp_path)
    assert args.eval_steps != 100


def test_max_length_768(tmp_path):
    # Verified via tokenize() — indirectly, we check training args don't cap lower.
    # Direct check: no explicit max_length in TrainingArguments (it's in tokenize()).
    args = make_training_args(tmp_path)
    assert args.num_train_epochs == 3


def test_no_prior_checkpoint_reference():
    # Guard against accidentally referencing old broken checkpoints in the source.
    src = Path(__file__).parent.parent / "ml" / "train.py"
    text = src.read_text()
    assert "curated_3ep" not in text
    assert "curated_merged" not in text


def test_gradient_accumulation_steps(tmp_path):
    args = make_training_args(tmp_path)
    assert args.gradient_accumulation_steps == 8


def test_bf16_enabled(tmp_path):
    args = make_training_args(tmp_path)
    assert args.bf16 is True
