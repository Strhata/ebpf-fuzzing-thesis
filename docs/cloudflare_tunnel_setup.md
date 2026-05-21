# Cloudflare Quick Tunnel Setup

Uses Cloudflare's free Quick Tunnel (`*.trycloudflare.com`). No domain or account needed.
The URL is stable while `cloudflared` is running; it changes only on machine reboot.

---

## 1 — Install cloudflared

```bash
sudo curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared
sudo chmod +x /usr/local/bin/cloudflared
cloudflared --version
```

---

## 2 — Generate an API key (one-time)

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Save this value — it goes in both the tmux start command and the Colab `REWARD_API_KEY` secret.

---

## 3 — Start both servers in tmux

```bash
# Reward server
tmux new-session -d -s reward-server \
  'cd ~/tesi/ebpf-fuzzing-thesis && REWARD_API_KEY=<YOUR-KEY> pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000'

# Cloudflare tunnel
tmux new-session -d -s reward-tunnel \
  'cloudflared tunnel --url localhost:8000 2>&1 | tee /tmp/cf-tunnel.log'
```

---

## 4 — Get the tunnel URL

```bash
sleep 8 && grep -o 'https://[^ ]*trycloudflare\.com' /tmp/cf-tunnel.log | head -1
```

---

## 5 — Verify

```bash
export TUNNEL_URL=$(grep -o 'https://[^ ]*trycloudflare\.com' /tmp/cf-tunnel.log | head -1)

# Should return 200
curl -s -o /dev/null -w "%{http_code}" $TUNNEL_URL/docs

# Should return 401 (tunnel live, key required)
curl -s -o /dev/null -w "%{http_code}" \
  -X POST $TUNNEL_URL/rewards \
  -H "Content-Type: application/json" \
  -d '{"completions":["test"]}'
```

---

## 6 — Save secrets in Colab

1. Open colab.research.google.com
2. Click the key icon (Secrets) in the left sidebar
3. Add/update:
   - `REWARD_SERVER_URL` = the `trycloudflare.com` URL from step 4
   - `REWARD_API_KEY` = the key from step 2
4. Toggle "Notebook access" on for both

---

## After a machine reboot

The URL changes on reboot. Restart the sessions and update the Colab secret:

```bash
tmux new-session -d -s reward-server \
  'cd ~/tesi/ebpf-fuzzing-thesis && REWARD_API_KEY=<YOUR-KEY> pixi run uvicorn tools.reward_server:app --host 0.0.0.0 --port 8000'
tmux new-session -d -s reward-tunnel \
  'cloudflared tunnel --url localhost:8000 2>&1 | tee /tmp/cf-tunnel.log'
sleep 8 && grep -o 'https://[^ ]*trycloudflare\.com' /tmp/cf-tunnel.log | head -1
```

Then update `REWARD_SERVER_URL` in Colab secrets with the new URL.
A Colab restart alone (no machine reboot) does **not** require updating the secret.
