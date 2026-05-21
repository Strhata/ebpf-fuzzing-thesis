"""Tests for tools/reward_server.py — auth, response shape, state propagation."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_TOOLS_DIR = Path(__file__).parent.parent / "tools"
_ML_DIR = Path(__file__).parent.parent / "ml"
_SERVER_PATH = _TOOLS_DIR / "reward_server.py"

# Set key before the module-level RuntimeError check fires on import.
os.environ.setdefault("REWARD_API_KEY", "test-key-123")
_VALID_KEY = os.environ["REWARD_API_KEY"]

sys.path.insert(0, str(_TOOLS_DIR))
sys.path.insert(0, str(_ML_DIR))


def _load_server():
    spec = importlib.util.spec_from_file_location("reward_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_srv = _load_server()
_rw = _srv._reward
_client = TestClient(_srv.app)
_HEADERS_OK = {"X-API-Key": _VALID_KEY}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_valid_key_returns_200():
    with patch.object(_rw, "compute_rewards", return_value=[0.5]):
        _rw._last_pcs_per_program = [10]
        resp = _client.post("/rewards", json={"completions": ["asm"]}, headers=_HEADERS_OK)
    assert resp.status_code == 200


def test_missing_key_returns_401():
    resp = _client.post("/rewards", json={"completions": ["asm"]})
    assert resp.status_code == 401


def test_wrong_key_returns_401():
    resp = _client.post("/rewards", json={"completions": ["asm"]},
                        headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

def test_response_shape():
    _rw._pc_set = {1, 2, 3}
    _rw._max_pcs_seen = 20
    _rw._last_pcs_per_program = [10, 15]
    with patch.object(_rw, "compute_rewards", return_value=[0.5, 1.5]):
        resp = _client.post("/rewards", json={"completions": ["a", "b"]},
                            headers=_HEADERS_OK)
    body = resp.json()
    assert body["rewards"] == [0.5, 1.5]
    assert body["cumulative_pcs"] == 3
    assert body["total_pcs_per_program"] == [10, 15]
    assert body["max_pcs_seen"] == 20


def test_cumulative_pcs_reflects_pc_set_size():
    _rw._pc_set = {10, 20, 30, 40, 50}
    _rw._last_pcs_per_program = [5]
    with patch.object(_rw, "compute_rewards", return_value=[0.3]):
        resp = _client.post("/rewards", json={"completions": ["asm"]},
                            headers=_HEADERS_OK)
    assert resp.json()["cumulative_pcs"] == 5


# ---------------------------------------------------------------------------
# Startup guard
# ---------------------------------------------------------------------------

def test_server_refuses_start_without_api_key():
    saved = os.environ.pop("REWARD_API_KEY")
    try:
        spec = importlib.util.spec_from_file_location("reward_server_nokey", _SERVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        with pytest.raises(RuntimeError, match="REWARD_API_KEY"):
            spec.loader.exec_module(mod)
    finally:
        os.environ["REWARD_API_KEY"] = saved
