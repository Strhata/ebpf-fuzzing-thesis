"""
reward_server.py — FastAPI reward server for remote GRPO training.

Exposes compute_rewards over HTTP so a Colab training process can reach
the local KCOV VM reward function without direct VM access.

Usage:
    REWARD_API_KEY=<key> pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000

State is kept in results/rl_pc_set_colab.json and results/rl_max_pcs_seen_colab.json,
separate from the local rl_grpo_v2 state files.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ml"))

import reward as _reward  # noqa: E402

# Use Colab-run-specific state files, separate from the local rl_grpo_v2 run.
_reward._PC_SET_PATH = _REPO_ROOT / "results" / "rl_pc_set_colab.json"
_reward._MAX_PCS_SEEN_PATH = _REPO_ROOT / "results" / "rl_max_pcs_seen_colab.json"
_reward._pc_set = set()
_reward._max_pcs_seen = 1
_reward._load_pc_set()
_reward._load_max_pcs_seen()

_API_KEY: str | None = os.environ.get("REWARD_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "REWARD_API_KEY environment variable is not set; refusing to start"
    )

_ssh = _reward.SSHClient()

app = FastAPI(title="KCOV Reward Server")


class RewardRequest(BaseModel):
    completions: list[str]


class RewardResponse(BaseModel):
    rewards: list[float]
    cumulative_pcs: int
    total_pcs_per_program: list[int]
    max_pcs_seen: int


@app.post("/rewards", response_model=RewardResponse)
def post_rewards(
    body: RewardRequest,
    x_api_key: str | None = Header(default=None),
) -> RewardResponse:
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    rewards = _reward.compute_rewards(body.completions, _ssh)

    return RewardResponse(
        rewards=rewards,
        cumulative_pcs=len(_reward._pc_set),
        total_pcs_per_program=_reward._last_pcs_per_program,
        max_pcs_seen=_reward._max_pcs_seen,
    )
