"""Unit tests for ml/reward.py — depth-based verdict-blind reward formula."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib
import importlib.util

_ML_DIR = Path(__file__).parent.parent / "ml"
_REWARD_PATH = _ML_DIR / "reward.py"


def _fresh_reward(pc_set_path: str = "", max_pcs_seen_path: str = ""):
    """Return a freshly-imported reward module with isolated state."""
    spec = importlib.util.spec_from_file_location("reward", _REWARD_PATH)
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    rw._pc_set = set()       # isolate from results/rl_pc_set.json on disk
    rw._max_pcs_seen = 1     # isolate from results/rl_max_pcs_seen.json on disk
    if pc_set_path:
        rw._PC_SET_PATH = Path(pc_set_path)
        rw._load_pc_set()
    if max_pcs_seen_path:
        rw._MAX_PCS_SEEN_PATH = Path(max_pcs_seen_path)
        rw._load_max_pcs_seen()
    return rw


# Minimal valid BPF assembly (passes verifier)
_VALID_ASM = "0: (b7) r0 = 0\n1: (95) exit"
# Assembly that will fail to parse/compile (no valid instructions)
_BAD_ASM = "this is not assembly"
# Hex string representing 20 8-byte instructions — passes the instruction floor (≥15)
_VALID_HEX = "00" * (20 * 8)


def _mock_ssh(verdict: str, pcs: list[int] | None = None) -> MagicMock:
    """Return an SSHClient mock that speaks the `kcov_validator --batch` protocol:
    one JSON array sized to the stdin payload, every entry carrying the given
    verdict/pcs. The real batch binary exits 0 on success regardless of verdicts."""
    client = MagicMock()

    def _run(cmd, timeout=30, input=None):
        lines = [ln for ln in (input or "").splitlines() if ln.strip()]
        arr = [{"index": j, "verdict": verdict, "pcs": pcs or []} for j in range(len(lines))]
        return (0, json.dumps(arr), "")

    client.run.side_effect = _run
    return client


class TestRewardFormula(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pc_path = str(Path(self._tmpdir) / "rl_pc_set.json")
        self._max_pcs_path = str(Path(self._tmpdir) / "rl_max_pcs_seen.json")

    def _rw(self):
        return _fresh_reward(self._pc_path, self._max_pcs_path)

    # --- special cases (unchanged from v1) ---

    def test_compile_fail_returns_zero(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED")
        rewards = rw.compute_rewards([_BAD_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        ssh.run.assert_not_called()

    def test_ssh_timeout_returns_crash_reward(self):
        rw = self._rw()
        ssh = MagicMock()
        ssh.run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX), \
             patch.object(rw, "_watchdog") as mock_watchdog:
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [2.0])
        mock_watchdog.assert_called_once_with(ssh)

    # --- depth component ---

    def test_depth_scales_with_pcs(self):
        cases = [(50, 0.25), (100, 0.5), (200, 0.5)]  # (n_pcs, expected_depth)
        for n_pcs, expected in cases:
            with self.subTest(n_pcs=n_pcs):
                rw = self._rw()
                rw._max_pcs_seen = 100
                pcs = list(range(n_pcs))
                rw._pc_set = set(pcs)  # pre-seed so no discovery bonus
                ssh = _mock_ssh("ACCEPTED", pcs=pcs)
                with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
                    rewards = rw.compute_rewards([_VALID_ASM], ssh)
                self.assertAlmostEqual(rewards[0], expected, places=5)

    def test_max_pcs_seen_initialises_to_one(self):
        rw = self._rw()
        self.assertEqual(rw._max_pcs_seen, 1)

    def test_max_pcs_seen_updates_on_record(self):
        rw = self._rw()
        rw._max_pcs_seen = 10
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(25)))
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rw._max_pcs_seen, 25)

    def test_max_pcs_seen_does_not_decrease(self):
        rw = self._rw()
        rw._max_pcs_seen = 50
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(10)))
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rw._max_pcs_seen, 50)

    # --- discovery bonus ---

    def test_discovery_bonus_fires_on_new_pc(self):
        rw = self._rw()
        rw._max_pcs_seen = 10
        ssh = _mock_ssh("ACCEPTED", pcs=[0x9999])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        depth = min(0.5, 1 / 10 * 0.5)
        self.assertAlmostEqual(rewards[0], depth + 1.0)

    def test_discovery_bonus_zero_no_new_pcs(self):
        rw = self._rw()
        rw._max_pcs_seen = 10
        rw._pc_set = {0x9999}
        ssh = _mock_ssh("ACCEPTED", pcs=[0x9999])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        depth = min(0.5, 1 / 10 * 0.5)
        self.assertAlmostEqual(rewards[0], depth)

    # --- batched validation (issue #24) ---

    def test_single_ssh_call_for_whole_batch(self):
        # N completions must result in exactly ONE ssh.run (one --batch call).
        rw = self._rw()
        rw._max_pcs_seen = 100
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(50)))
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM] * 5, ssh)
        self.assertEqual(len(rewards), 5)
        ssh.run.assert_called_once()

    def test_encode_fail_isolated_in_batch(self):
        # One un-encodable completion scores 0.0 at its index; the rest validate.
        rw = self._rw()
        rw._max_pcs_seen = 100
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(50)))

        def fake_encode(asm):
            return None if asm == _BAD_ASM else _VALID_HEX

        with patch.object(rw, "_encode_to_hex", side_effect=fake_encode):
            rewards = rw.compute_rewards([_BAD_ASM, _VALID_ASM, _BAD_ASM], ssh)
        self.assertEqual(rewards[0], 0.0)
        self.assertGreater(rewards[1], 0.0)
        self.assertEqual(rewards[2], 0.0)
        ssh.run.assert_called_once()  # only the one valid program was sent

    def test_batch_parity_mixed_verdicts(self):
        # Parity: a single batched call with mixed verdicts produces exactly the
        # reward vector the per-program formula yields, mapped back by index.
        rw = self._rw()
        rw._max_pcs_seen = 100
        arr = [
            {"index": 0, "verdict": "ACCEPTED", "pcs": list(range(40))},
            {"index": 1, "verdict": "REJECTED", "pcs": list(range(40, 70))},  # 30 pcs
            {"index": 2, "verdict": "ERROR", "pcs": []},
        ]
        ssh = MagicMock()
        ssh.run.return_value = (0, json.dumps(arr), "")
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM, _VALID_ASM], ssh)
        # max_pcs_seen=100 throughout (40,30 < 100); both have new PCs → +1.0 bonus
        self.assertAlmostEqual(rewards[0], min(0.5, 40 / 100 * 0.5) + 1.0)
        self.assertAlmostEqual(rewards[1], min(0.5, 30 / 100 * 0.5) + 1.0)
        self.assertEqual(rewards[2], 0.0)  # ERROR → 0.0
        ssh.run.assert_called_once()

    def test_batch_timeout_gives_crash_reward_to_all(self):
        # Whole-batch SSH timeout → crash reward for every validated program + watchdog.
        rw = self._rw()
        ssh = MagicMock()
        ssh.run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX), \
             patch.object(rw, "_watchdog") as mock_watchdog:
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        self.assertEqual(rewards, [2.0, 2.0])
        mock_watchdog.assert_called_once_with(ssh)

    def test_within_batch_snapshot(self):
        # Both completions in one call return the same new PC.
        # Both should earn the discovery bonus because the pc_set snapshot
        # is taken before either completion is evaluated.
        rw = self._rw()
        rw._max_pcs_seen = 10
        ssh = _mock_ssh("ACCEPTED", pcs=[0x1000])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        depth = min(0.5, 1 / 10 * 0.5)
        self.assertEqual(len(rewards), 2)
        self.assertAlmostEqual(rewards[0], depth + 1.0)
        self.assertAlmostEqual(rewards[1], depth + 1.0)

    # --- verdict-blind behaviour ---

    def test_rifiutato_gets_depth_reward(self):
        rw = self._rw()
        rw._max_pcs_seen = 10
        ssh = _mock_ssh("REJECTED", pcs=[0x1000, 0x1001])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        depth = min(0.5, 2 / 10 * 0.5)
        self.assertAlmostEqual(rewards[0], depth + 1.0)  # new pcs → discovery bonus
        self.assertNotEqual(rewards[0], 0.1)             # old rejected tier is gone

    def test_accettato_not_treated_as_errore(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED", pcs=[0x5555])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertGreater(rewards[0], 0.0)

    # --- PC set persistence ---

    def test_pc_set_grows_across_calls(self):
        rw = self._rw()
        ssh1 = _mock_ssh("ACCEPTED", pcs=[0xA])
        ssh2 = _mock_ssh("ACCEPTED", pcs=[0xB])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh1)
            rw.compute_rewards([_VALID_ASM], ssh2)
        self.assertIn(0xA, rw._pc_set)
        self.assertIn(0xB, rw._pc_set)

    def test_pc_set_written_every_100_calls(self):
        rw = self._rw()
        rw._pc_set = {0x42}
        rw._call_count = 99
        ssh = _mock_ssh("ACCEPTED", pcs=[0x43])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertTrue(Path(self._pc_path).exists(), "PC set file not written at call 100")
        with open(self._pc_path) as f:
            saved = set(json.load(f))
        self.assertIn(0x42, saved)
        self.assertIn(0x43, saved)

    def test_max_pcs_seen_written_on_cadence(self):
        rw = self._rw()
        rw._max_pcs_seen = 5
        rw._call_count = 99
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(8)))  # 8 pcs > 5
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertTrue(Path(self._max_pcs_path).exists(),
                        "max_pcs_seen file not written at call 100")
        with open(self._max_pcs_path) as f:
            saved = int(json.load(f))
        self.assertEqual(saved, 8)

    def test_pc_set_reloads_from_disk(self):
        pc_data = [0xDEAD, 0xBEEF]
        with open(self._pc_path, "w") as f:
            json.dump(pc_data, f)
        rw = _fresh_reward(self._pc_path, self._max_pcs_path)
        self.assertEqual(rw._pc_set, {0xDEAD, 0xBEEF})

    def test_max_pcs_seen_reloads_from_disk(self):
        with open(self._max_pcs_path, "w") as f:
            json.dump(42, f)
        rw = _fresh_reward(self._pc_path, self._max_pcs_path)
        self.assertEqual(rw._max_pcs_seen, 42)


    # --- instruction floor ---

    def _hex_of_n_insns(self, n: int) -> str:
        """Return a hex string representing exactly n 8-byte BPF instructions."""
        return "00" * (n * 8)

    def test_instruction_floor_fires_below_15(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(100)))
        with patch.object(rw, "_encode_to_hex", return_value=self._hex_of_n_insns(5)):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        ssh.run.assert_not_called()   # VM must not be contacted when floor fires

    def test_instruction_floor_fires_at_exactly_14(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(100)))
        with patch.object(rw, "_encode_to_hex", return_value=self._hex_of_n_insns(14)):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        ssh.run.assert_not_called()

    def test_instruction_floor_does_not_fire_at_15(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(50)))
        with patch.object(rw, "_encode_to_hex", return_value=self._hex_of_n_insns(15)):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        # Floor must NOT fire — VM is called and depth reward returned
        ssh.run.assert_called_once()
        self.assertGreater(rewards[0], 0.0)

    def test_instruction_floor_20_insns_no_new_pcs_gives_depth_only(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        pcs = list(range(40))
        rw._pc_set = set(pcs)   # pre-seed so discovery bonus is 0
        ssh = _mock_ssh("ACCEPTED", pcs=pcs)
        with patch.object(rw, "_encode_to_hex", return_value=self._hex_of_n_insns(20)):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        expected_depth = min(0.5, 40 / 100 * 0.5)
        self.assertAlmostEqual(rewards[0], expected_depth)

    def test_instruction_floor_20_insns_new_pcs_gives_depth_plus_bonus(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(40)))
        with patch.object(rw, "_encode_to_hex", return_value=self._hex_of_n_insns(20)):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        expected_depth = min(0.5, 40 / 100 * 0.5)
        self.assertAlmostEqual(rewards[0], expected_depth + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
