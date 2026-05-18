"""
Tests for SanityCheckCallback in rl_grpo.py.

All offline — no VM, no SSH.  The SSH client is stubbed.
"""

import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

import trl.import_utils as _trl_utils
_trl_utils.is_vllm_available = lambda: False

from transformers import TrainerControl, TrainerState
from rl_grpo import SanityCheckCallback, _SANITY_LOG, _GRPO_DEBUG_LOG
import reward as rw


def _state(step: int = 0) -> TrainerState:
    s = TrainerState()
    s.global_step = step
    return s


def _ssh_up() -> MagicMock:
    ssh = MagicMock()
    ssh.run.return_value = (0, "pong", "")
    return ssh


def _ssh_down() -> MagicMock:
    ssh = MagicMock()
    ssh.run.side_effect = Exception("Connection refused")
    return ssh


# ---------------------------------------------------------------------------
# Timing gate
# ---------------------------------------------------------------------------

def test_does_not_fire_before_interval():
    cb = SanityCheckCallback(_ssh_up(), interval_secs=3600)
    with patch.object(cb, '_run_check') as mock_check:
        cb.on_step_end(None, _state(1), TrainerControl())
        mock_check.assert_not_called()


def test_fires_when_interval_elapsed():
    cb = SanityCheckCallback(_ssh_up(), interval_secs=3600)
    cb._next_check = time.time() - 1  # already past due
    with patch.object(cb, '_run_check') as mock_check:
        cb.on_step_end(None, _state(10), TrainerControl())
        mock_check.assert_called_once_with(10)


def test_reschedules_after_firing():
    cb = SanityCheckCallback(_ssh_up(), interval_secs=3600)
    cb._next_check = time.time() - 1
    with patch.object(cb, '_run_check'):
        cb.on_step_end(None, _state(5), TrainerControl())
    assert cb._next_check > time.time()


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def test_recent_verdicts_empty_when_no_log(tmp_path):
    cb = SanityCheckCallback(_ssh_up())
    with patch('rl_grpo._GRPO_DEBUG_LOG', tmp_path / "missing.log"):
        counts = cb._recent_verdicts(200)
    assert counts == {}


def test_recent_verdicts_counts_correctly(tmp_path):
    log = tmp_path / "grpo.log"
    log.write_text(
        "batch=0 i=0 ACCETTATO new_pcs=1\n"
        "batch=0 i=1 RIFIUTATO insns=5\n"
        "batch=0 i=2 RIFIUTATO insns=3\n"
        "batch=0 i=3 ENCODE_FAIL insns=0\n"
    )
    cb = SanityCheckCallback(_ssh_up())
    with patch('rl_grpo._GRPO_DEBUG_LOG', log):
        counts = cb._recent_verdicts(200)
    assert counts == {"ACCETTATO": 1, "RIFIUTATO": 2, "ENCODE_FAIL": 1}


def test_recent_verdicts_respects_n_limit(tmp_path):
    log = tmp_path / "grpo.log"
    # 10 ACCETTATO followed by 5 RIFIUTATO
    lines = ["ACCETTATO\n"] * 10 + ["RIFIUTATO\n"] * 5
    log.write_text("".join(lines))
    cb = SanityCheckCallback(_ssh_up())
    with patch('rl_grpo._GRPO_DEBUG_LOG', log):
        counts = cb._recent_verdicts(5)  # only last 5
    assert counts.get("ACCETTATO", 0) == 0
    assert counts.get("RIFIUTATO", 0) == 5


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

def test_report_written_to_sanity_log(tmp_path):
    cb = SanityCheckCallback(_ssh_up())
    sanity = tmp_path / "sanity_checks.log"
    debug = tmp_path / "grpo.log"
    debug.write_text("ACCETTATO\nRIFIUTATO\n")
    with patch('rl_grpo._SANITY_LOG', sanity), \
         patch('rl_grpo._GRPO_DEBUG_LOG', debug):
        cb._run_check(step=99)
    content = sanity.read_text()
    assert "step=99" in content
    assert "cumulative_pcs" in content
    assert "vm_status" in content


def test_report_shows_vm_up(tmp_path, capsys):
    cb = SanityCheckCallback(_ssh_up())
    sanity = tmp_path / "s.log"
    with patch('rl_grpo._SANITY_LOG', sanity), \
         patch('rl_grpo._GRPO_DEBUG_LOG', tmp_path / "g.log"):
        cb._run_check(step=1)
    out = capsys.readouterr().out
    assert "UP" in out


def test_report_shows_vm_down(tmp_path, capsys):
    cb = SanityCheckCallback(_ssh_down())
    sanity = tmp_path / "s.log"
    with patch('rl_grpo._SANITY_LOG', sanity), \
         patch('rl_grpo._GRPO_DEBUG_LOG', tmp_path / "g.log"):
        cb._run_check(step=1)
    out = capsys.readouterr().out
    assert "DOWN" in out
    assert "WARNING" in out


def test_warning_on_high_encode_fail_rate(tmp_path, capsys):
    log = tmp_path / "grpo.log"
    log.write_text("ENCODE_FAIL\n" * 9 + "ACCETTATO\n")  # 90% fail
    cb = SanityCheckCallback(_ssh_up())
    sanity = tmp_path / "s.log"
    with patch('rl_grpo._SANITY_LOG', sanity), \
         patch('rl_grpo._GRPO_DEBUG_LOG', log):
        cb._run_check(step=50)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "ENCODE_FAIL" in out


def test_no_warning_on_healthy_run(tmp_path, capsys):
    log = tmp_path / "grpo.log"
    log.write_text("ACCETTATO\n" * 5 + "RIFIUTATO\n" * 15)
    cb = SanityCheckCallback(_ssh_up())
    sanity = tmp_path / "s.log"
    with patch('rl_grpo._SANITY_LOG', sanity), \
         patch('rl_grpo._GRPO_DEBUG_LOG', log):
        cb._run_check(step=20)
    out = capsys.readouterr().out
    assert "WARNING" not in out
