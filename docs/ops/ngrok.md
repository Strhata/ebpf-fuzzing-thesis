# ngrok Tunnel Setup

Exposes the local reward server (`tools/reward_server.py` on `:8000`) to Colab over HTTPS.
ngrok free tier gives **one static domain** per account — claim it once and `REWARD_SERVER_URL`
**never changes again**, even across host reboots (the main win over a quick tunnel).

> The reward client (`ml/rl_grpo.py`) already sends `ngrok-skip-browser-warning: true`, so
> ngrok's free interstitial does not corrupt the JSON response. No extra config needed there.

---

## 1 — Install ngrok (once)

```bash
# WSL / linux-amd64 — user-local install, no sudo
mkdir -p ~/.local/bin
curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C ~/.local/bin
~/.local/bin/ngrok version
# add to PATH (append to ~/.bashrc to persist):
export PATH="$HOME/.local/bin:$PATH"
```

> The agent **authtoken** (step 2) is a ~49-char string from the *"Your Authtoken"* dashboard page.
> It is **not** the same as an API key (those start `rd_…`) — using an API key fails with ERR_NGROK_105.

## 2 — Authenticate (once, persists in ~/.config/ngrok/ngrok.yml)

Sign up free at https://dashboard.ngrok.com → copy your authtoken, then:

```bash
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
```

## 3 — Claim your static domain (once)

Dashboard → **Domains** → create → you get something like `your-name.ngrok-free.app`.
Use it everywhere below as `<STATIC_DOMAIN>`. (Skip this and you get a random URL each start —
see "Ephemeral fallback" at the bottom.)

## 4 — Generate the reward API key (once)

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
Save it — goes in both the reward-server command and the Colab `REWARD_API_KEY` secret.

---

## 5 — Start everything (after each host boot)

```bash
# 0. eval VM must be up — the reward server SSHes to it on :10022
./fuzzing/run_eval_vm.sh

# 1. reward server (paste your KEY)
tmux new-session -d -s reward-server \
  'cd ~/tesi/ebpf-fuzzing-thesis && REWARD_API_KEY=<YOUR-KEY> pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000'

# 2. ngrok on the static domain (URL is constant → set Colab secret once, never touch again)
tmux new-session -d -s reward-tunnel \
  'ngrok http --url=https://<STATIC_DOMAIN> 8000'
```

## 6 — Verify

```bash
URL=https://<STATIC_DOMAIN>
# 200 — server reachable through the tunnel
curl -s -o /dev/null -w "%{http_code}\n" -H "ngrok-skip-browser-warning: true" $URL/docs
# 401 — tunnel live, API key enforced
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/rewards \
  -H "Content-Type: application/json" -H "ngrok-skip-browser-warning: true" \
  -d '{"completions":["test"]}'
```

ngrok's own request inspector is at http://localhost:4040 (handy for debugging Colab calls).

## 7 — Colab secrets

Colab → key icon (Secrets) → set, toggle "Notebook access" on:
- `REWARD_SERVER_URL` = `https://<STATIC_DOMAIN>`
- `REWARD_API_KEY`    = the key from step 4

With a static domain these are set **once**. A host reboot only requires re-running step 5 —
the URL is unchanged.

---

## Ephemeral fallback (no static domain)

```bash
tmux new-session -d -s reward-tunnel 'ngrok http 8000'
# fetch the random URL from ngrok's local API:
sleep 5 && curl -s localhost:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```
This URL changes every `ngrok` restart → you must update `REWARD_SERVER_URL` in Colab each time.
Prefer the static domain.
