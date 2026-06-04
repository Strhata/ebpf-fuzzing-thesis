# Colab Pro — Restart Guide (grpo-depth-reward-v1, beta=0.01)

---

## Context

The GRPO training run on Colab Pro has a ~24h session limit. When the session dies,
the model checkpoint is safe on Google Drive. This guide gets training back in ~5 min.

---

## Before you start (one-time setup, done once)

- [ ] Colab Pro subscription active at colab.research.google.com
- [ ] Notebook saved to Drive: `My Drive/grpo-depth-reward-v1/train_grpo_colab.ipynb`
- [ ] Google Drive folder exists: `My Drive/grpo-depth-reward-v1/`
- [ ] Colab secrets set: `GITHUB_TOKEN`, `WANDB_API_KEY`, `REWARD_API_KEY`, `REWARD_SERVER_URL`
- [ ] Local reward server running (`tmux` session `reward-server` on local machine)
- [ ] ngrok tunnel running (`tmux` session `reward-tunnel` on local machine)
  — on a static domain the URL never changes; only re-run after a host boot
  — see `docs/ngrok_tunnel_setup.md` for setup and reboot recovery

---

## How to tell if the session died

Open the notebook. If you see:

- **"Connect" button** (top right, greyed out RAM/Disk bars) → runtime is dead, needs restart
- **Running RAM/Disk bars** → runtime is alive, nothing to do
- **"Reconnecting…" spinner** → wait 30s, it may recover on its own

---

## Restart steps

### 1 — Open the notebook
Go to `colab.research.google.com` → open `train_grpo_colab.ipynb` from Drive.

### 2 — Connect to runtime
Click **Connect** (top right). Wait for RAM/Disk bars to appear (~30s).

If you get CPU only: Runtime > Change runtime type > T4 GPU > Save, then reconnect.
Verify on first run whether Colab Pro assigns T4 automatically.

### 3 — Run all cells
Runtime > **Run all** (or Ctrl+F9).

Cell 3 installs deps fresh from PyPI every time (~10 min). There is no Drive caching —
this is expected and correct. The run resumes safely once training starts in cell 5.

### 4 — Verify resume picked up correctly

Watch the trainer output cell. You should see:

```
[*] Resuming from <OUTPUT_DIR>/checkpoint-XXXX
...
{'loss': ..., 'epoch': ...}   ← step counter > 0
```

If step counter starts at 0 → checkpoint was not found. Check that the Drive folder
`grpo-depth-reward-v1/` exists and `OUTPUT_DIR` in cell 4 matches it.

### 5 — Verify WandB continuity

Open wandb.ai → your project → confirm the run shows new points being added to the
existing curve, not a new run started.

The run ID is persisted at `{OUTPUT_DIR}/wandb_run_id.txt`. On resume, `rl_grpo.py`
reads this file and sets `WANDB_RUN_ID`/`WANDB_RESUME=must` before trainer init so
WandB appends to the existing run. If a new run appears, confirm the file exists in
the Drive folder.

### 6 — Verify reward server reachable

The trainer output should show reward values flowing within the first batch (~30s
after training starts). If you see retry warnings:

```
[REWARD] server unreachable, retrying in Xs...
```

Check on local machine:
```bash
tmux ls                          # confirm reward-server and reward-tunnel sessions exist
curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $REWARD_API_KEY" \
  -H "ngrok-skip-browser-warning: true" \
  "$REWARD_SERVER_URL/rewards" -d '{"completions":["test"]}'
```

On a static ngrok domain `REWARD_SERVER_URL` never changes — a Colab restart needs no update,
and a host reboot only needs the `reward-server` + `reward-tunnel` tmux sessions restarted
(same URL). See `docs/ngrok_tunnel_setup.md`.

### 7 — Close the browser

Background execution keeps the runtime alive after you close the tab.
You do not need to keep Colab open.

---

## Schedule

Set a recurring phone alarm for every **23 hours** (1h buffer before the 24h limit).
When it fires: check if the run is still alive (open notebook, look at RAM bars).
If dead: run steps 1–7 above.

---

## Checklist per restart

- [ ] Step counter > 0 in trainer output
- [ ] WandB shows new points on existing run (not a new run)
- [ ] Reward values appear within first batch
- [ ] Browser tab closed after confirming
