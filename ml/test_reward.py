"""Unit tests for ml/reward.py (all 5 reward tiers + PC set persistence)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Run from repo root:   python ml/test_reward.py
# or from ml/ dir:     python test_reward.py

import importlib
import importlib.util

_ML_DIR = Path(__file__).parent
_REWARD_PATH = _ML_DIR / "reward.py"


def _fresh_reward(pc_set_path: str = ""):
    """Return a freshly-imported reward module with a custom PC set path."""
    spec = importlib.util.spec_from_file_location("reward", _REWARD_PATH)
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    if pc_set_path:
        rw._PC_SET_PATH = Path(pc_set_path)
        rw._load_pc_set()  # re-run with the test path
    return rw


# Minimal valid BPF assembly (passes verifier)
_VALID_ASM = "0: (b7) r0 = 0\n1: (95) exit"
# Assembly that will fail to parse/compile (no valid instructions)
_BAD_ASM = "this is not assembly"


def _mock_ssh(verdict: str, pcs: list[int] | None = None) -> MagicMock:
    """Return an SSHClient mock that returns the given verdict."""
    client = MagicMock()
    payload = json.dumps({"verdict": verdict, "pcs": pcs or []})
    # rc=0 for ACCETTATO, rc=1 for RIFIUTATO
    rc = 0 if verdict == "ACCETTATO" else 1
    client.run.return_value = (rc, payload, "")
    return client


class TestRewardTiers(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pc_path = str(Path(self._tmpdir) / "rl_pc_set.json")

    def _rw(self):
        return _fresh_reward(self._pc_path)

    def test_compile_fail_returns_zero(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCETTATO")
        rewards = rw.compute_rewards([_BAD_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        ssh.run.assert_not_called()

    def test_rejected_returns_point_one(self):
        rw = self._rw()
        ssh = _mock_ssh("RIFIUTATO")
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.1])

    def test_valid_new_pcs_returns_one(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCETTATO", pcs=[0x1000, 0x1001])
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [1.0])
        self.assertEqual(rw._pc_set, {0x1000, 0x1001})

    def test_valid_no_new_pcs_returns_point_four(self):
        rw = self._rw()
        rw._pc_set = {0x1000, 0x1001}
        ssh = _mock_ssh("ACCETTATO", pcs=[0x1000, 0x1001])
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.4])

    def test_ssh_timeout_returns_crash_reward(self):
        rw = self._rw()
        ssh = MagicMock()
        ssh.run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"), \
             patch.object(rw, "_watchdog") as mock_watchdog:
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [2.0])
        mock_watchdog.assert_called_once_with(ssh)

    def test_accettato_not_treated_as_errore(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCETTATO", pcs=[0x5555])
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertIn(rewards[0], (0.4, 1.0))
        self.assertNotEqual(rewards[0], 0.0)

    def test_pc_set_grows_across_calls(self):
        rw = self._rw()
        ssh1 = _mock_ssh("ACCETTATO", pcs=[0xA])
        ssh2 = _mock_ssh("ACCETTATO", pcs=[0xB])
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"):
            rw.compute_rewards([_VALID_ASM], ssh1)
            rw.compute_rewards([_VALID_ASM], ssh2)
        self.assertIn(0xA, rw._pc_set)
        self.assertIn(0xB, rw._pc_set)

    def test_pc_set_written_every_100_calls(self):
        rw = self._rw()
        rw._pc_set = {0x42}
        rw._call_count = 99
        ssh = _mock_ssh("ACCETTATO", pcs=[0x43])
        with patch.object(rw, "_compile_to_hex", return_value="deadbeef"):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertTrue(Path(self._pc_path).exists(), "PC set file not written at call 100")
        with open(self._pc_path) as f:
            saved = set(json.load(f))
        self.assertIn(0x42, saved)
        self.assertIn(0x43, saved)

    def test_pc_set_reloads_from_disk(self):
        pc_data = [0xDEAD, 0xBEEF]
        with open(self._pc_path, "w") as f:
            json.dump(pc_data, f)
        rw = _fresh_reward(self._pc_path)
        self.assertEqual(rw._pc_set, {0xDEAD, 0xBEEF})


if __name__ == "__main__":
    unittest.main(verbosity=2)
