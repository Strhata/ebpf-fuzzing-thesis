# Colab Pro — Daily Restart Guide (beta=0.1 RL run)

> **Status:** DRAFT — written before first restart. Items marked ⚠️ ASSUMED are unverified.
> Tune this file after the first real restart.

---

## Context

The GRPO training run on Colab Pro has a ~24h session limit. When the session dies,
the model checkpoint is safe on Google Drive. This guide gets training back in ~5 min.

---

## Before you start (one-time setup, done once)

- [ ] Colab Pro subscription active at colab.research.google.com
- [ ] Notebook saved to Drive: `My Drive/ebpf-thesis/train_grpo_colab.ipynb`
- [ ] Google Drive folder exists: `My Drive/ebpf-thesis/rl_grpo_colab/`
- [ ] Local reward server running (`tmux` session `reward-server` on local machine)
- [ ] Cloudflare Tunnel running (`tmux` session `cf-tunnel` on local machine)
- [ ] Tunnel URL saved as Colab secret `REWARD_SERVER_URL` ⚠️ ASSUMED: tunnel URL is stable per tmux session; verify after first cloudflared restart

---

## How to tell if the session died

Open the notebook. If you see:

- **"Connect" button** (top right, greyed out RAM/Disk bars) → runtime is dead, needs restart
- **Running RAM/Disk bars** → runtime is alive, nothing to do
- **"Reconnecting…" spinner** → wait 30s, it may recover on its own

---

## Restart steps (~5 min)

### 1 — Open the notebook
Go to `colab.research.google.com` → open `train_grpo_colab.ipynb` from Drive.

### 2 — Connect to runtime
Click **Connect** (top right). Wait for RAM/Disk bars to appear (~30s).

⚠️ ASSUMED: Colab Pro assigns a T4 automatically. If you get CPU only, go to
Runtime > Change runtime type > T4 GPU > Save, then reconnect.

### 3 — Run all cells
Runtime > **Run all** (or Ctrl+F9).

The first cell mounts Drive and installs deps (~3–5 min). ⚠️ ASSUMED: deps are
cached on Drive to skip re-download; verify this is actually faster on first restart.

### 4 — Verify resume picked up correctly

Watch the trainer output cell. You should see:

```
[*] Resuming from checkpoints/rl_grpo_colab/checkpoint-XXXX
...
{'loss': ..., 'epoch': ...}   ← step counter > 0
```

If step counter starts at 0 → checkpoint was not found. Check Drive folder exists
and output_dir in the notebook matches `My Drive/ebpf-thesis/rl_grpo_colab/`.

### 5 — Verify WandB continuity

Open wandb.ai → your project → confirm the run shows new points being added to the
existing curve, not a new run started. ⚠️ ASSUMED: `wandb_run_id.txt` on Drive is
read automatically; if a new run appears instead, check the file exists in the
checkpoint folder.

### 6 — Verify reward server reachable

The trainer output should show reward values flowing within the first batch (~30s
after training starts). If you see retry warnings:

```
[REWARD] server unreachable, retrying in Xs...
```

Check on local machine:
```bash
tmux ls                          # confirm reward-server and cf-tunnel sessions exist
tmux attach -t cf-tunnel         # check tunnel URL hasn't changed
```

If tunnel URL changed, update the `REWARD_SERVER_URL` Colab secret and re-run
the setup cell only (not Run all). ⚠️ ASSUMED: only the setup cell reads the secret.

### 7 — Close the browser

Background execution keeps the runtime alive after you close the tab.
You do not need to keep Colab open.

---

## Schedule

Set a recurring phone alarm for every **23 hours** (1h buffer before the 24h limit).
When it fires: check if the run is still alive (open notebook, look at RAM bars).
If dead: run steps 1–7 above. Total time: ~5 min.

---

## Checklist per restart

- [ ] Step counter > 0 in trainer output
- [ ] WandB shows new points on existing run (not a new run)
- [ ] Reward values appear within first batch
- [ ] Browser tab closed after confirming

---

## What to update in this guide after first restart

- ⚠️ Confirm dep caching actually speeds up cell 1
- ⚠️ Confirm tunnel URL stability (does cloudflared keep the same URL across sessions?)
- ⚠️ Confirm `wandb_run_id.txt` is found automatically — note actual filename/path
- ⚠️ Confirm T4 is assigned automatically without manual runtime type change
- ⚠️ Note actual time for steps 1–7 (replace the ~5 min estimate)
