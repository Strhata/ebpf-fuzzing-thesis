#!/bin/bash
# vm_watchdog.sh — Ensure the eval VM is reachable, restarting it if needed.
#
# Usage:
#   tools/vm_watchdog.sh [host] [port] [ssh_key]
#
# Defaults: localhost 10022 ~/fuzzing_lab/trixie.id_rsa
#
# Exit 0: VM is reachable (was already up or successfully restarted).
# Exit 1: VM did not respond within TIMEOUT seconds.
#
# Called by ml/reward.py when an SSH timeout is detected.

set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-10022}"
KEY="${3:-$HOME/fuzzing_lab/trixie.id_rsa}"
TIMEOUT=300
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PID_FILE="$REPO_ROOT/data/vm.pid"

ssh_ok() {
    ssh -q -p "$PORT" -i "$KEY" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=3 \
        -o BatchMode=yes \
        root@"$HOST" "echo ok" >/dev/null 2>&1
}

# 1. Quick check — transient failures don't need a restart
if ssh_ok; then
    exit 0
fi

echo "[watchdog] VM unreachable at $HOST:$PORT — attempting restart" >&2

# 2. Kill existing QEMU (best-effort; ignore errors if already dead)
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    sleep 1
fi

# 3. Launch VM — run_eval_vm.sh blocks until SSH is ready, so we wrap with timeout
if ! timeout "$TIMEOUT" "$REPO_ROOT/fuzzing/run_eval_vm.sh" >&2; then
    echo "[watchdog] VM did not start within ${TIMEOUT}s" >&2
    exit 1
fi

# 4. Make corpus tools executable (kcov_validator added after initial vm.sh was written)
ssh -q -p "$PORT" -i "$KEY" \
    -o StrictHostKeyChecking=no \
    -o BatchMode=yes \
    root@"$HOST" \
    "chmod +x /mnt/corpus/kcov_validator 2>/dev/null || true" >/dev/null 2>&1 || true

echo "[watchdog] VM ready at $HOST:$PORT" >&2
exit 0
