"""
Tests for strip_verifier_log (ml/reward.py) and build_prompt (ml/train.py).

All tests are offline — no tokenizer download, no VM.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

from reward import _encode_to_hex, strip_verifier_log


class _FakeTok:
    eos_token = "<|endoftext|>"


_tok = _FakeTok()


# ---------------------------------------------------------------------------
# strip_verifier_log — structural checks
# ---------------------------------------------------------------------------

def test_strip_removes_line_number_prefixes():
    result = strip_verifier_log("0: (b7) r0 = 1\n1: (95) exit")
    assert not re.search(r'^\d+:', result, re.MULTILINE)


def test_strip_removes_opcode_byte_annotations():
    result = strip_verifier_log("0: (b7) r0 = 1\n1: (95) exit")
    assert '(b7)' not in result
    assert '(95)' not in result
    assert not re.search(r'\([0-9a-fA-F]{2}\)', result)


def test_strip_removes_semicolon_comments():
    result = strip_verifier_log("0: (b7) r0 = 1 ; R0_w=1\n1: (95) exit ; R0_w=0")
    assert ';' not in result


def test_strip_removes_register_state_lines():
    asm = "0: R1=ctx() R10=fp0\n0: (b7) r0 = 0\n1: (95) exit"
    result = strip_verifier_log(asm)
    assert not re.search(r'R\d+=', result)


def test_strip_removes_from_to_transition_lines():
    asm = "0: (b7) r0 = 0\nfrom 0 to 2: R0_w=0 R1=ctx() R10=fp0\n1: (95) exit"
    result = strip_verifier_log(asm)
    assert 'from 0 to 2' not in result


def test_strip_removes_mark_precise_lines():
    asm = "0: (b7) r0 = 1\nmark_precise: frame0: last_idx 0 first_idx 0\n1: (95) exit"
    result = strip_verifier_log(asm)
    assert 'mark_precise' not in result


def test_strip_normalises_goto_pc():
    result = strip_verifier_log("0: (05) goto pc+3")
    assert 'goto +3' in result
    assert 'pc' not in result


def test_strip_normalises_goto_pc_negative():
    result = strip_verifier_log("1: (05) goto pc-2")
    assert 'goto -2' in result
    assert 'pc' not in result


def test_strip_keeps_bare_instructions():
    result = strip_verifier_log("0: (b7) r0 = 42\n1: (95) exit")
    lines = result.splitlines()
    assert lines[0] == 'r0 = 42'
    assert lines[1] == 'exit'


def test_strip_empty_input():
    assert strip_verifier_log("") == ""


def test_strip_all_noise_returns_empty():
    asm = "0: R1=ctx() R10=fp0\nfrom 0 to 1: R0_w=0\nmark_precise: frame0"
    assert strip_verifier_log(asm) == ""


# ---------------------------------------------------------------------------
# strip_verifier_log → _encode_to_hex round-trip
# ---------------------------------------------------------------------------

def test_stripped_output_encodes_without_raising():
    raw = (
        "0: R1=ctx() R10=fp0\n"
        "0: (b7) r0 = 0 ; R0_w=0\n"
        "1: (95) exit\n"
    )
    stripped = strip_verifier_log(raw)
    result = _encode_to_hex(stripped)
    assert result is not None


def test_stripped_multi_instruction_encodes_correctly():
    raw = (
        "0: (b7) r0 = 1 ; R0_w=1\n"
        "1: (b7) r1 = 2 ; R1_w=2\n"
        "2: (0f) r0 += r1 ; R0_w=3\n"
        "3: (95) exit\n"
    )
    stripped = strip_verifier_log(raw)
    result = _encode_to_hex(stripped)
    assert result is not None
    # 4 instructions: r0=1, r1=2, r0+=r1, exit
    assert len(result) == 4 * 16


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def _build_prompt(sample):
    """Import and call build_prompt from train.py with a fake tokenizer."""
    import importlib
    import train as tr
    return tr.build_prompt(sample, _tok)


def _sample(coverage_bin="medium", novelty_bin="medium", is_valid=True, verifier_log="0: (95) exit"):
    return {
        "verifier_log": verifier_log,
        "coverage_bin": coverage_bin,
        "novelty_bin":  novelty_bin,
        "is_valid":     is_valid,
    }


def test_build_prompt_starts_with_coverage_token():
    p = _build_prompt(_sample(coverage_bin="medium"))
    assert p["formatted_prompt"].startswith("[coverage=medium]")


def test_build_prompt_contains_novelty_token():
    p = _build_prompt(_sample(novelty_bin="high"))
    assert "[novelty=high]" in p["formatted_prompt"]


def test_build_prompt_coverage_before_novelty():
    p = _build_prompt(_sample(coverage_bin="low", novelty_bin="high"))
    text = p["formatted_prompt"]
    assert text.index("[coverage=") < text.index("[novelty=")


def test_build_prompt_all_three_coverage_bins():
    for bin_val in ("low", "medium", "high"):
        p = _build_prompt(_sample(coverage_bin=bin_val))
        assert f"[coverage={bin_val}]" in p["formatted_prompt"], bin_val


def test_build_prompt_all_three_novelty_bins():
    for bin_val in ("low", "medium", "high"):
        p = _build_prompt(_sample(novelty_bin=bin_val))
        assert f"[novelty={bin_val}]" in p["formatted_prompt"], bin_val


def test_build_prompt_invalid_sample_uses_bin_not_status_label():
    p = _build_prompt(_sample(coverage_bin="low", novelty_bin="low", is_valid=False))
    assert "INVALID" not in p["formatted_prompt"]
    assert "VALID" not in p["formatted_prompt"]
    assert "[coverage=low]" in p["formatted_prompt"]


def test_build_prompt_valid_sample_uses_bin_not_status_label():
    p = _build_prompt(_sample(coverage_bin="high", novelty_bin="high", is_valid=True))
    assert "VALID" not in p["formatted_prompt"]
    assert "[coverage=high]" in p["formatted_prompt"]


def test_build_prompt_contains_assembly_header():
    p = _build_prompt(_sample(verifier_log="0: (b7) r0 = 0\n1: (95) exit"))
    assert "### ASSEMBLY:" in p["formatted_prompt"]


def test_build_prompt_ends_with_eos():
    p = _build_prompt(_sample())
    assert p["formatted_prompt"].endswith(_tok.eos_token)


def test_build_prompt_missing_novelty_bin_defaults_to_low():
    sample = {"verifier_log": "0: (95) exit", "coverage_bin": "high", "is_valid": True}
    p = _build_prompt(sample)
    assert "[novelty=low]" in p["formatted_prompt"]


def test_build_prompt_stripped_log_encodes():
    sample = {
        "verifier_log": (
            "0: R1=ctx() R10=fp0\n"
            "0: (b7) r0 = 0 ; R0_w=0\n"
            "1: (95) exit\n"
        ),
        "coverage_bin": "medium",
        "novelty_bin":  "medium",
        "is_valid": True,
    }
    p = _build_prompt(sample)
    prompt = p["formatted_prompt"]
    asm_section = prompt.split("### ASSEMBLY:\n", 1)[1].replace(_tok.eos_token, "").strip()
    assert _encode_to_hex(asm_section) is not None
