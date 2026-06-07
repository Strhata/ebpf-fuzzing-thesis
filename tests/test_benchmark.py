"""
Tests for benchmark_lib modules 2, 5, 6, and 7.

All tests are offline (no VM, no GPU) unless marked @pytest.mark.integration.
Import paths mirror the pattern in test_reward_encoder.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make tools/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

from benchmark_lib import (
    ComplexityResult,
    GeneratorResult,
    ProgramRecord,
    SlotKPIs,
    ValidationResult,
    VMLifecycle,
    VMShutdownError,
    aggregate,
    analyze,
    generate_batch,
    validate,
)


# ---------------------------------------------------------------------------
# Module 4: generate_batch (issue #23) — offline orchestration test with fakes
# ---------------------------------------------------------------------------

class _FakeTok:
    """Minimal tokenizer: 1 token per character; left/right pad to batch max."""
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, chunk, return_tensors=None, padding=None):
        import torch
        seqs = [[1] * len(s) for s in chunk]  # toy: one id per char
        L = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = L - len(s)
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + s)
                mask.append([0] * pad + [1] * len(s))
            else:
                ids.append(s + [self.pad_token_id] * pad)
                mask.append([1] * len(s) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def decode(self, tensor, skip_special_tokens=True):
        return f"GEN{int(tensor.sum().item())}"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=4, **kw):
        import torch
        self.calls += 1
        B = input_ids.shape[0]
        # append deterministic new tokens [1..max_new_tokens] to every row
        new = torch.arange(1, max_new_tokens + 1).repeat(B, 1)
        return torch.cat([input_ids, new], dim=1)


def test_generate_batch_orchestration():
    tok = _FakeTok()
    model = _FakeModel()
    prompts = ["a", "bb", "ccc", "dddd", "e"]  # varying lengths → padding exercised

    res = generate_batch(model, tok, prompts, max_new_tokens=4, batch_size=2)

    # one output per prompt, in order
    assert len(res.texts) == len(prompts)
    # 5 prompts at batch_size 2 → 3 generate calls
    assert model.calls == 3
    # new tokens are [1,2,3,4] for every row → decode sees only the new slice (sum=10),
    # i.e. the (left-)padded prompt was correctly stripped regardless of prompt length
    assert all(t == "GEN10" for t in res.texts)
    # total new tokens = 5 prompts * 4 tokens
    assert res.total_new_tokens == 20
    # padding_side restored after the call
    assert tok.padding_side == "right"


def test_generate_batch_empty():
    tok = _FakeTok()
    model = _FakeModel()
    res = generate_batch(model, tok, [], max_new_tokens=4, batch_size=2)
    assert res.texts == []
    assert model.calls == 0


# ---------------------------------------------------------------------------
# Module 2: VMLifecycle
# ---------------------------------------------------------------------------

class TestVMLifecycle:
    """Context manager boots the VM script on enter and kills the PID on exit."""

    def _make_popen(self, pid: int = 9999):
        proc = MagicMock()
        proc.pid = pid
        proc.wait.return_value = 0
        return proc

    def test_enter_spawns_boot_script(self, tmp_path):
        proc = self._make_popen()
        with patch("subprocess.Popen", return_value=proc) as mock_popen, \
             patch("benchmark_lib._wait_for_pid"), \
             patch("os.kill"):
            vm = VMLifecycle()
            vm._PID_FILE = tmp_path / "vm.pid"
            vm._PID_FILE.write_text("1234")
            vm.__enter__()
            args, _ = mock_popen.call_args
            assert "run_eval_vm.sh" in args[0][-1]

    def test_exit_kills_pid_from_file(self, tmp_path):
        proc = self._make_popen(pid=8888)
        with patch("subprocess.Popen", return_value=proc), \
             patch("benchmark_lib._wait_for_pid") as mock_wait, \
             patch("os.kill") as mock_kill:
            vm = VMLifecycle()
            vm._PID_FILE = tmp_path / "vm.pid"
            vm._PID_FILE.write_text("1234")
            vm.__enter__()
            vm.__exit__(None, None, None)
            mock_kill.assert_called_once_with(1234, __import__("signal").SIGTERM)

    def test_exception_inside_with_still_triggers_shutdown(self, tmp_path):
        proc = self._make_popen(pid=7777)
        killed = []

        with patch("subprocess.Popen", return_value=proc), \
             patch("benchmark_lib._wait_for_pid"), \
             patch("os.kill", side_effect=lambda pid, sig: killed.append(pid)):
            vm = VMLifecycle()
            vm._PID_FILE = tmp_path / "vm.pid"
            vm._PID_FILE.write_text("5678")
            try:
                with vm:
                    raise RuntimeError("synthetic failure")
            except RuntimeError:
                pass
            assert 5678 in killed

    def test_missing_pid_file_falls_back_to_popen_pid(self, tmp_path):
        proc = self._make_popen(pid=4242)
        killed = []

        with patch("subprocess.Popen", return_value=proc), \
             patch("benchmark_lib._wait_for_pid"), \
             patch("os.kill", side_effect=lambda pid, sig: killed.append(pid)):
            vm = VMLifecycle()
            vm._PID_FILE = tmp_path / "missing_vm.pid"   # does not exist
            vm.__enter__()
            vm.__exit__(None, None, None)
            assert 4242 in killed

    def test_both_missing_raises_vm_shutdown_error(self, tmp_path):
        """No vm.pid file AND no Popen object → VMShutdownError."""
        vm = VMLifecycle()
        vm._PID_FILE = tmp_path / "missing_vm.pid"
        vm._proc = None
        with pytest.raises(VMShutdownError):
            vm.shutdown()

    def test_invalid_pid_file_falls_back_to_popen_pid(self, tmp_path):
        """Non-integer vm.pid → fall back to Popen PID."""
        proc = self._make_popen(pid=3333)
        killed = []

        with patch("subprocess.Popen", return_value=proc), \
             patch("benchmark_lib._wait_for_pid"), \
             patch("os.kill", side_effect=lambda pid, sig: killed.append(pid)):
            vm = VMLifecycle()
            vm._PID_FILE = tmp_path / "vm.pid"
            vm._PID_FILE.write_text("not_a_number")
            vm.__enter__()
            vm.__exit__(None, None, None)
            assert 3333 in killed


# ---------------------------------------------------------------------------
# Module 5: ComplexityAnalyzer
# ---------------------------------------------------------------------------

class TestComplexityAnalyzer:
    """Uses real _encode_to_hex — no mocking."""

    def test_single_alu_insn(self):
        result = analyze("r0 = 1\nexit")
        assert result.encoded is True
        # r0=1 + exit = 2 instructions
        assert result.insn_count == 2
        # exit opcode 0x95 has class 0x95 & 0x07 = 0x05 = BPF_JMP, so it counts
        assert result.jump_count == 1

    def test_single_alu_insn_no_explicit_exit(self):
        # "r0 = 1" alone → encoder appends exit → 2 instructions
        result = analyze("r0 = 1")
        assert result.encoded is True
        assert result.insn_count == 2
        # auto-appended exit counts as JMP class
        assert result.jump_count == 1

    def test_jump_count_one(self):
        # conditional jump + exit = 2 JMP-class instructions
        asm = "r0 = 0\nif r0 == 0 goto +1\nr0 = 1\nexit"
        result = analyze(asm)
        assert result.encoded is True
        assert result.jump_count == 2   # jeq (0x15, class=0x05) + exit (0x95, class=0x05)

    def test_goto_counts_as_jump(self):
        asm = "r0 = 0\ngoto +1\nr0 = 99\nexit"
        result = analyze(asm)
        assert result.encoded is True
        # goto (0x05) and exit (0x95) both have BPF_JMP class
        assert result.jump_count == 2

    def test_opcode_diversity_two_same_opcodes(self):
        # Two MOV64-imm instructions share opcode 0xb7; plus exit (opcode 0x95) = 3 insns, 2 unique
        asm = "r0 = 1\nr1 = 2\nexit"
        result = analyze(asm)
        assert result.encoded is True
        assert result.insn_count == 3
        # opcodes: 0xb7, 0xb7, 0x95 → 2 unique / 3 total
        assert abs(result.opcode_diversity - 2 / 3) < 1e-9

    def test_opcode_diversity_all_same(self):
        # Two identical MOV64-imm + exit: diversity = 2 unique (0xb7, 0x95) / 3 total
        # To get truly all-same we'd need a program with only exit instruction — 1/1 = 1.0
        result = analyze("exit")
        assert result.encoded is True
        assert result.insn_count == 1
        # Only one instruction (exit), one unique opcode → 1/1 = 1.0
        assert result.opcode_diversity == 1.0

    def test_opcode_diversity_mixed_improves(self):
        # Add, sub, and load → more diverse than pure mov
        asm = "r0 = 1\nr1 = 2\nr0 += r1\nr0 -= r1\nexit"
        result = analyze(asm)
        assert result.encoded is True
        assert result.opcode_diversity > 0.0

    def test_malformed_input_returns_not_encoded(self):
        result = analyze("this is not bpf assembly at all")
        assert result.encoded is False
        assert result.insn_count is None
        assert result.opcode_diversity is None
        assert result.jump_count is None

    def test_empty_string_not_encoded(self):
        result = analyze("")
        assert result.encoded is False

    def test_lddw_skipped_but_rest_encodes(self):
        asm = "r0 = 1\nr9 = 0xffff88800a11a800\nexit"
        result = analyze(asm)
        # lddw is skipped; r0=1 + exit encode
        assert result.encoded is True
        assert result.insn_count == 2


# ---------------------------------------------------------------------------
# Module 6: QEMUValidator
# ---------------------------------------------------------------------------

class TestQEMUValidator:
    """Unit tests mock _validate_on_vm; no VM required."""

    def _fake_ssh(self):
        return MagicMock()

    def test_none_return_maps_to_error(self):
        ssh = self._fake_ssh()
        with patch("benchmark_lib._vm_validate_fn", return_value=None):
            result = validate("deadbeef", ssh)
        assert result.verdict == "ERROR"
        assert result.pcs == []

    def test_accepted_result_mapped_correctly(self):
        ssh = self._fake_ssh()
        with patch("benchmark_lib._vm_validate_fn",
                   return_value={"verdict": "ACCEPTED", "pcs": [1, 2, 3]}):
            result = validate("deadbeef", ssh)
        assert result.verdict == "ACCEPTED"
        assert result.pcs == [1, 2, 3]

    def test_rejected_result_mapped_correctly(self):
        ssh = self._fake_ssh()
        with patch("benchmark_lib._vm_validate_fn",
                   return_value={"verdict": "REJECTED", "pcs": []}):
            result = validate("deadbeef", ssh)
        assert result.verdict == "REJECTED"
        assert result.pcs == []

    def test_missing_pcs_key_defaults_to_empty(self):
        ssh = self._fake_ssh()
        with patch("benchmark_lib._vm_validate_fn",
                   return_value={"verdict": "ACCEPTED"}):
            result = validate("deadbeef", ssh)
        assert result.verdict == "ACCEPTED"
        assert result.pcs == []

    @pytest.mark.integration
    def test_real_vm_accepts_minimal_program(self):
        """Requires a running QEMU VM on port 10022."""
        from reward import SSHClient, _encode_to_hex
        hex_str = _encode_to_hex("r0 = 0\nexit")
        assert hex_str is not None
        ssh = SSHClient()
        result = validate(hex_str, ssh)
        assert result.verdict in ("ACCEPTED", "REJECTED")
        assert isinstance(result.pcs, list)


# ---------------------------------------------------------------------------
# Module 7: KPIAggregator
# ---------------------------------------------------------------------------

def _make_program(verdict: str, pcs: list[int], encoded: bool = True,
                  insn_count: int = 5, opcode_diversity: float = 0.5,
                  jump_count: int = 1) -> ProgramRecord:
    cx = ComplexityResult(
        encoded=encoded,
        insn_count=insn_count if encoded else None,
        opcode_diversity=opcode_diversity if encoded else None,
        jump_count=jump_count if encoded else None,
    )
    vr = ValidationResult(verdict=verdict, pcs=pcs)
    return ProgramRecord(assembly="r0 = 0\nexit", hex_str=None, complexity=cx, validation=vr)


def _timing(tps: float = 100.0) -> GeneratorResult:
    return GeneratorResult(assemblies=[], tokens_per_sec=tps)


class TestKPIAggregator:

    def test_all_accepted_uniform_pcs(self):
        programs = [
            _make_program("ACCEPTED", [10, 20, 30]),
            _make_program("ACCEPTED", [10, 20, 30]),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=512.0, peak_run_gpu_mb=600.0)
        assert kpis.pass_rate == 1.0
        assert kpis.encode_rate == 1.0
        assert kpis.avg_pcs == 3.0
        assert kpis.max_pcs == 3
        assert kpis.total_unique_pcs == 3

    def test_all_rejected_pass_rate_zero(self):
        programs = [
            _make_program("REJECTED", []),
            _make_program("REJECTED", []),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert kpis.pass_rate == 0.0
        assert kpis.avg_pcs == 0.0

    def test_mixed_encode_failures(self):
        programs = [
            _make_program("ACCEPTED", [1, 2], encoded=True),
            _make_program("ERROR", [], encoded=False),
            _make_program("ERROR", [], encoded=False),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert abs(kpis.encode_rate - 1 / 3) < 1e-9
        # insn metrics only over encoded subset (1 program)
        assert kpis.avg_insn_count == 5.0

    def test_novelty_score_identical_pc_sets(self):
        # Both programs hit exactly the same PCs → each PC has freq=2, N=2 → novelty=0
        programs = [
            _make_program("ACCEPTED", [1, 2, 3]),
            _make_program("ACCEPTED", [1, 2, 3]),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert kpis.novelty_score == 0.0

    def test_novelty_score_disjoint_pc_sets(self):
        # Each program hits unique PCs → freq=1 for every PC, N=2 → novelty = mean(1 - 1/2) = 0.5
        # Wait — disjoint means each PC appears in exactly 1 of 2 programs → 1 - 1/2 = 0.5 per PC
        # PRD says: "2 programs, disjoint PCs → novelty_score=1.0"
        # Re-read: novelty_score = mean(1 - freq[pc]/N). If disjoint and N=2,
        # each pc has freq=1 → 1 - 1/2 = 0.5. But PRD says 1.0.
        # PRD note says disjoint → 1.0, implying N=1 per set effectively.
        # Interpretation: "2 programs with disjoint PCs" where each program has unique PCs →
        # novelty_score = 0.5 by formula. PRD may have intended single-program sets.
        # We test the actual formula result: 0.5 for N=2 disjoint.
        programs = [
            _make_program("ACCEPTED", [1, 2]),
            _make_program("ACCEPTED", [3, 4]),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        # Each PC has freq=1, N=2 → 1 - 1/2 = 0.5 per PC → mean = 0.5
        assert abs(kpis.novelty_score - 0.5) < 1e-9

    def test_novelty_fully_unique_single_program(self):
        # Only one program: freq[pc]=1, N=1 → 1 - 1/1 = 0.0 per PC → novelty=0.0
        programs = [_make_program("ACCEPTED", [10, 20])]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert kpis.novelty_score == 0.0

    def test_empty_programs_list(self):
        kpis = aggregate([], _timing(tps=50.0), peak_load_gpu_mb=128.0, peak_run_gpu_mb=256.0)
        assert kpis.pass_rate == 0.0
        assert kpis.encode_rate == 0.0
        assert kpis.avg_pcs == 0.0
        assert kpis.max_pcs == 0
        assert kpis.total_unique_pcs == 0
        assert kpis.novelty_score == 0.0
        assert kpis.avg_insn_count == 0.0
        assert kpis.avg_opcode_diversity == 0.0
        assert kpis.avg_jump_count == 0.0
        assert kpis.tokens_per_sec == 50.0
        assert kpis.peak_load_gpu_mb == 128.0
        assert kpis.peak_run_gpu_mb == 256.0

    def test_peak_gpu_mb_passed_through(self):
        programs = [_make_program("ACCEPTED", [1])]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=1024.5, peak_run_gpu_mb=2048.0)
        assert kpis.peak_load_gpu_mb == 1024.5
        assert kpis.peak_run_gpu_mb == 2048.0

    def test_tokens_per_sec_passed_through(self):
        programs = [_make_program("ACCEPTED", [])]
        kpis = aggregate(programs, _timing(tps=333.3), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert abs(kpis.tokens_per_sec - 333.3) < 1e-6

    def test_avg_pcs_only_over_accepted(self):
        programs = [
            _make_program("ACCEPTED", [1, 2, 3, 4, 5]),
            _make_program("REJECTED", [9, 9, 9]),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert kpis.avg_pcs == 5.0

    def test_max_pcs_across_all_programs(self):
        programs = [
            _make_program("ACCEPTED", [1, 2, 3]),
            _make_program("REJECTED", [4, 5, 6, 7]),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert kpis.max_pcs == 4

    def test_total_unique_pcs_union(self):
        programs = [
            _make_program("ACCEPTED", [1, 2, 3]),
            _make_program("ACCEPTED", [3, 4, 5]),
        ]
        kpis = aggregate(programs, _timing(), peak_load_gpu_mb=0.0, peak_run_gpu_mb=0.0)
        assert kpis.total_unique_pcs == 5
