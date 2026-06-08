"""Unit tests for ml/reward.py — RL-v2 validity-gated, per-group-novelty reward.

Ladder under test:
    encode fail / < floor insns   → 0.0
    VM ERROR / whole-batch crash  → 0.0
    REJECTED                      → W_REJECT_MAX * min(1, len(pcs)/max_pcs_seen)
    ACCEPTED                      → W_VALID + W_NOVELTY * group_novelty(p)
Expected values are computed from rw.W_* so the suite is robust to env overrides.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    rw._global_pc_freq = {}  # isolate from results/rl_global_pc_freq.json on disk
    if pc_set_path:
        rw._PC_SET_PATH = Path(pc_set_path)
        rw._load_pc_set()
    if max_pcs_seen_path:
        rw._MAX_PCS_SEEN_PATH = Path(max_pcs_seen_path)
        rw._load_max_pcs_seen()
    return rw


_VALID_ASM = "0: (b7) r0 = 0\n1: (95) exit"
_BAD_ASM = "this is not assembly"
_VALID_HEX = "00" * (20 * 8)  # 20 instructions — passes the floor (≥15)


def _mock_ssh(verdict: str, pcs: list[int] | None = None) -> MagicMock:
    """SSHClient mock: every program in the batch gets the same verdict/pcs."""
    client = MagicMock()

    def _run(cmd, timeout=30, input=None):
        lines = [ln for ln in (input or "").splitlines() if ln.strip()]
        arr = [{"index": j, "verdict": verdict, "pcs": pcs or []} for j in range(len(lines))]
        return (0, json.dumps(arr), "")

    client.run.side_effect = _run
    return client


def _mock_ssh_multi(specs: list[tuple[str, list[int]]]) -> MagicMock:
    """SSHClient mock returning a distinct (verdict, pcs) per program, in stdin order.

    Requires every completion to pass encode+floor (patch _encode_to_hex to _VALID_HEX)
    so that batch index j corresponds to completion j.
    """
    client = MagicMock()

    def _run(cmd, timeout=30, input=None):
        lines = [ln for ln in (input or "").splitlines() if ln.strip()]
        assert len(lines) == len(specs), "mock_ssh_multi: spec/payload size mismatch"
        arr = [{"index": j, "verdict": v, "pcs": p} for j, (v, p) in enumerate(specs)]
        return (0, json.dumps(arr), "")

    client.run.side_effect = _run
    return client


class TestRewardLadder(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pc_path = str(Path(self._tmpdir) / "rl_pc_set.json")
        self._max_pcs_path = str(Path(self._tmpdir) / "rl_max_pcs_seen.json")

    def _rw(self):
        return _fresh_reward(self._pc_path, self._max_pcs_path)

    # --- zero cases ---

    def test_compile_fail_returns_zero(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED")
        rewards = rw.compute_rewards([_BAD_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        ssh.run.assert_not_called()

    def test_ssh_timeout_returns_zero_not_crash_bonus(self):
        # RL-v2: infra timeout is noise, not a reward (RL-v1 paid 2.0 here).
        rw = self._rw()
        ssh = MagicMock()
        ssh.run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX), \
             patch.object(rw, "_watchdog") as mock_watchdog:
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        mock_watchdog.assert_called_once_with(ssh)
        self.assertEqual(rw._last_verdict_counts["crash"], 1)

    def test_error_verdict_returns_zero(self):
        rw = self._rw()
        ssh = _mock_ssh("ERROR", pcs=[])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        self.assertEqual(rw._last_verdict_counts["error"], 1)

    # --- invalid soft floor ---

    def test_rejected_soft_floor_scales_with_depth(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        cases = [(50, 0.5), (100, 1.0)]  # (n_pcs, fraction of W_REJECT_MAX)
        for n_pcs, frac in cases:
            with self.subTest(n_pcs=n_pcs):
                rw = self._rw()
                rw._max_pcs_seen = 100
                ssh = _mock_ssh("REJECTED", pcs=list(range(n_pcs)))
                with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
                    rewards = rw.compute_rewards([_VALID_ASM], ssh)
                self.assertAlmostEqual(rewards[0], rw.W_REJECT_MAX * frac, places=6)

    def test_rejected_floor_capped_at_w_reject_max(self):
        # pcs beyond max_pcs_seen pushes max up, so fraction caps at 1.0.
        rw = self._rw()
        rw._max_pcs_seen = 100
        ssh = _mock_ssh("REJECTED", pcs=list(range(200)))
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertAlmostEqual(rewards[0], rw.W_REJECT_MAX, places=6)

    # --- valid: W_VALID + per-group novelty ---

    def test_lone_valid_gets_w_valid_only(self):
        # One accepted program → nothing to be novel against → novelty 0.
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED", pcs=[1, 2, 3])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertAlmostEqual(rewards[0], rw.W_VALID, places=6)

    def test_per_group_novelty_rewards_unique_paths(self):
        # P0 covers two PCs no sibling touches; P1 covers only shared PCs.
        rw = self._rw()
        rw._max_pcs_seen = 100
        specs = [("ACCEPTED", [1, 2, 3, 4]), ("ACCEPTED", [1, 2])]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        # freq: 1→2, 2→2, 3→1, 4→1 ; n_acc=2
        nov0 = (0 + 0 + 0.5 + 0.5) / 4   # = 0.25
        nov1 = (0 + 0) / 2               # = 0.0
        self.assertAlmostEqual(rewards[0], rw.W_VALID + rw.W_NOVELTY * nov0, places=6)
        self.assertAlmostEqual(rewards[1], rw.W_VALID + rw.W_NOVELTY * nov1, places=6)
        self.assertGreater(rewards[0], rewards[1])

    def test_valid_always_beats_invalid(self):
        # Monotonic ladder: even a maximally-deep rejected program < any accepted.
        rw = self._rw()
        rw._max_pcs_seen = 100
        specs = [("ACCEPTED", [7]), ("REJECTED", list(range(100)))]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        self.assertGreater(rewards[0], rewards[1])
        self.assertAlmostEqual(rewards[1], rw.W_REJECT_MAX, places=6)  # capped invalid
        self.assertLess(rw.W_REJECT_MAX, rw.W_VALID)                   # gate holds by design

    # --- phase B: decayed-global novelty ---

    def test_global_term_off_by_default(self):
        # W_GLOBAL defaults to 0 → reward is phase-A only (W_VALID for a lone valid).
        rw = self._rw()
        self.assertEqual(rw.W_GLOBAL, 0.0)
        ssh = _mock_ssh("ACCEPTED", pcs=[1, 2, 3])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertAlmostEqual(rewards[0], rw.W_VALID, places=6)

    def test_global_novelty_decays_with_frequency(self):
        # A PC seen many times globally must pay less than a brand-new PC.
        rw = self._rw()
        rw.W_GLOBAL = 2.0
        rw.W_NOVELTY = 0.0                  # isolate the global term
        rw._global_pc_freq = {99: 99.0}     # PC 99 hit 99× before; PC 7 unseen
        specs = [("ACCEPTED", [7]), ("ACCEPTED", [99])]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        # fresh PC 7: 1/(1+0)=1 ; stale PC 99: 1/(1+99)=0.01
        self.assertAlmostEqual(rewards[0], rw.W_VALID + 2.0 * 1.0, places=6)
        self.assertAlmostEqual(rewards[1], rw.W_VALID + 2.0 * (1.0 / 100.0), places=6)
        self.assertGreater(rewards[0], rewards[1])

    def test_global_freq_accumulates_accepted_only(self):
        rw = self._rw()
        rw.W_GLOBAL = 1.0
        specs = [("ACCEPTED", [1, 2]), ("REJECTED", [3, 4])]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        self.assertEqual(rw._global_pc_freq.get(1), 1.0)
        self.assertEqual(rw._global_pc_freq.get(2), 1.0)
        self.assertNotIn(3, rw._global_pc_freq)   # rejected PCs do NOT enter the frontier
        self.assertNotIn(4, rw._global_pc_freq)

    def test_global_freq_snapshot_within_batch(self):
        # Two accepted programs hitting the same fresh PC both score against pre-batch freq=0.
        rw = self._rw()
        rw.W_GLOBAL = 2.0
        ssh = _mock_ssh("ACCEPTED", pcs=[5000])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        # both see freq=0 → global novelty 1.0 each; group novelty 0 (both hit it)
        for r in rewards:
            self.assertAlmostEqual(r, rw.W_VALID + 2.0 * 1.0, places=6)
        self.assertEqual(rw._global_pc_freq.get(5000), 2.0)  # then incremented twice

    def test_global_decay_ages_counts(self):
        rw = self._rw()
        rw.W_GLOBAL = 1.0
        rw.GLOBAL_DECAY = 0.5
        rw._global_pc_freq = {1: 10.0}
        ssh = _mock_ssh("ACCEPTED", pcs=[2])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rw._global_pc_freq.get(1), 5.0)   # aged 10 * 0.5
        self.assertEqual(rw._global_pc_freq.get(2), 1.0)   # this batch's PC added after ageing

    def test_global_freq_reloads_from_disk(self):
        path = str(Path(self._tmpdir) / "gf.json")
        with open(path, "w") as f:
            json.dump({"123": 4.0}, f)
        rw = _fresh_reward(self._pc_path, self._max_pcs_path)
        rw._GLOBAL_FREQ_PATH = Path(path)
        rw._load_global_freq()
        self.assertEqual(rw._global_pc_freq.get(123), 4.0)

    # --- the RL-v1 guard: within-group reward variance ---

    def test_all_rejected_group_still_has_variance(self):
        # An all-invalid group must NOT collapse to identical rewards (reward_std=0
        # was the RL-v1 gradient-starvation failure). Different depths → different rewards.
        rw = self._rw()
        rw._max_pcs_seen = 100
        specs = [("REJECTED", list(range(10))),
                 ("REJECTED", list(range(50))),
                 ("REJECTED", list(range(90)))]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM] * 3, ssh)
        self.assertEqual(len(set(rewards)), 3)  # three distinct rewards → std > 0

    # --- batching mechanics (unchanged) ---

    def test_single_ssh_call_for_whole_batch(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(50)))
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM] * 5, ssh)
        self.assertEqual(len(rewards), 5)
        ssh.run.assert_called_once()

    def test_encode_fail_isolated_in_batch(self):
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
        ssh.run.assert_called_once()
        self.assertEqual(rw._last_verdict_counts["encode_fail"], 2)

    def test_batch_mixed_verdicts_mapped_by_index(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        specs = [("ACCEPTED", list(range(40))),
                 ("REJECTED", list(range(40, 70))),  # 30 pcs
                 ("ERROR", [])]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM, _VALID_ASM], ssh)
        # only one accepted → novelty 0 → W_VALID
        self.assertAlmostEqual(rewards[0], rw.W_VALID, places=6)
        self.assertAlmostEqual(rewards[1], rw.W_REJECT_MAX * (30 / 100), places=6)
        self.assertEqual(rewards[2], 0.0)
        ssh.run.assert_called_once()

    def test_batch_timeout_gives_zero_to_all(self):
        rw = self._rw()
        ssh = MagicMock()
        ssh.run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX), \
             patch.object(rw, "_watchdog") as mock_watchdog:
            rewards = rw.compute_rewards([_VALID_ASM, _VALID_ASM], ssh)
        self.assertEqual(rewards, [0.0, 0.0])
        mock_watchdog.assert_called_once_with(ssh)

    def test_verdict_counts_populated(self):
        rw = self._rw()
        rw._max_pcs_seen = 100
        specs = [("ACCEPTED", [1]), ("REJECTED", [2]), ("ERROR", [])]
        ssh = _mock_ssh_multi(specs)
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM] * 3, ssh)
        vc = rw._last_verdict_counts
        self.assertEqual(vc["accepted"], 1)
        self.assertEqual(vc["rejected"], 1)
        self.assertEqual(vc["error"], 1)

    # --- max_pcs_seen accounting ---

    def test_max_pcs_seen_initialises_to_one(self):
        self.assertEqual(self._rw()._max_pcs_seen, 1)

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

    # --- PC set persistence ---

    def test_pc_set_grows_across_calls(self):
        rw = self._rw()
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], _mock_ssh("ACCEPTED", pcs=[0xA]))
            rw.compute_rewards([_VALID_ASM], _mock_ssh("ACCEPTED", pcs=[0xB]))
        self.assertIn(0xA, rw._pc_set)
        self.assertIn(0xB, rw._pc_set)

    def test_pc_set_grows_on_rejected_too(self):
        # Coverage accrues regardless of verdict (telemetry / future global reward).
        rw = self._rw()
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], _mock_ssh("REJECTED", pcs=[0xC]))
        self.assertIn(0xC, rw._pc_set)

    def test_pc_set_written_every_100_calls(self):
        rw = self._rw()
        rw._pc_set = {0x42}
        rw._call_count = 99
        ssh = _mock_ssh("ACCEPTED", pcs=[0x43])
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertTrue(Path(self._pc_path).exists())
        with open(self._pc_path) as f:
            saved = set(json.load(f))
        self.assertIn(0x42, saved)
        self.assertIn(0x43, saved)

    def test_max_pcs_seen_written_on_cadence(self):
        rw = self._rw()
        rw._max_pcs_seen = 5
        rw._call_count = 99
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(8)))
        with patch.object(rw, "_encode_to_hex", return_value=_VALID_HEX):
            rw.compute_rewards([_VALID_ASM], ssh)
        self.assertTrue(Path(self._max_pcs_path).exists())
        with open(self._max_pcs_path) as f:
            self.assertEqual(int(json.load(f)), 8)

    def test_pc_set_reloads_from_disk(self):
        with open(self._pc_path, "w") as f:
            json.dump([0xDEAD, 0xBEEF], f)
        rw = _fresh_reward(self._pc_path, self._max_pcs_path)
        self.assertEqual(rw._pc_set, {0xDEAD, 0xBEEF})

    def test_max_pcs_seen_reloads_from_disk(self):
        with open(self._max_pcs_path, "w") as f:
            json.dump(42, f)
        rw = _fresh_reward(self._pc_path, self._max_pcs_path)
        self.assertEqual(rw._max_pcs_seen, 42)

    # --- instruction floor (VM not contacted below floor) ---

    def _hex_of_n_insns(self, n: int) -> str:
        return "00" * (n * 8)

    def test_instruction_floor_fires_below_15(self):
        rw = self._rw()
        ssh = _mock_ssh("ACCEPTED", pcs=list(range(100)))
        with patch.object(rw, "_encode_to_hex", return_value=self._hex_of_n_insns(5)):
            rewards = rw.compute_rewards([_VALID_ASM], ssh)
        self.assertEqual(rewards, [0.0])
        ssh.run.assert_not_called()

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
        ssh.run.assert_called_once()
        self.assertGreater(rewards[0], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
