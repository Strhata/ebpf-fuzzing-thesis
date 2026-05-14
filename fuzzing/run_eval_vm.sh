#!/bin/bash
# run_eval_vm.sh — Start a persistent evaluation VM.
#
# Boots trixie.img with bzImage_kasan_kcov (KASAN + KCOV).
# Mounts eval_workspace/ via virtio-9p at /mnt/corpus inside the VM.
# ebpf_validator must be compiled and placed in eval_workspace/ before running.
#
# Usage (from repo root):
#   ./fuzzing/run_eval_vm.sh
#
# Prerequisites:
#   make build-validator          # compiles ebpf_validator into eval_workspace/
#   FUZZING_LAB=~/fuzzing_lab     # override if assets are elsewhere

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
FUZZING_LAB="${FUZZING_LAB:-$HOME/fuzzing_lab}"

KERNEL="$FUZZING_LAB/linux/arch/x86/boot/bzImage_kasan_kcov"
DISK="$FUZZING_LAB/trixie.img"
SSH_KEY="$FUZZING_LAB/trixie.id_rsa"
WORKSPACE="$REPO_ROOT/data/corpus"
OVERLAY="$REPO_ROOT/data/vm_eval.qcow2"
SSH_PORT=10022

# Sanity checks
for f in "$KERNEL" "$DISK" "$SSH_KEY"; do
    if [[ ! -f "$f" ]]; then
        echo "[!] Missing: $f"
        echo "[!] Set FUZZING_LAB= to the directory containing trixie.img and bzImage_kasan_kcov."
        exit 1
    fi
done

mkdir -p "$WORKSPACE"

if [[ ! -f "$WORKSPACE/ebpf_validator" ]]; then
    echo "[!] $WORKSPACE/ebpf_validator not found."
    echo "[!] Run: make build-validator"
    exit 1
fi

# Fresh QCOW2 overlay — leaves trixie.img untouched
echo "[*] Creating fresh QCOW2 overlay..."
qemu-img create -f qcow2 -b "$DISK" -F raw "$OVERLAY" 2G >/dev/null

echo "[*] Booting VM (bzImage_kasan_kcov, KASAN + KCOV)..."
qemu-system-x86_64 \
    -m 4G -smp 2 \
    -kernel "$KERNEL" \
    -append "console=ttyS0 root=/dev/sda rw earlyprintik=serial net.ifnames=0 panic=1" \
    -drive file="$OVERLAY",format=qcow2 \
    -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:${SSH_PORT}-:22 \
    -net nic,model=e1000 \
    -fsdev local,security_model=none,id=fsdev_eval,path="$WORKSPACE" \
    -device virtio-9p-pci,id=fs_eval,fsdev=fsdev_eval,mount_tag=corpus_share \
    -enable-kvm -display none -daemonize \
    -serial file:"$REPO_ROOT/data/vm.log" \
    -pidfile "$REPO_ROOT/data/vm.pid"

echo "[*] Waiting for SSH..."
until ssh -q -p "$SSH_PORT" -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
    root@127.0.0.1 "echo ok" >/dev/null 2>&1; do
    sleep 2
done

echo "[*] Mounting corpus share inside VM..."
ssh -q -p "$SSH_PORT" -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    root@127.0.0.1 \
    "mkdir -p /mnt/corpus && mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 corpus_share /mnt/corpus && chmod +x /mnt/corpus/ebpf_validator"

echo "[+] VM ready. SSH: ssh -p $SSH_PORT -i $SSH_KEY root@127.0.0.1"
echo "[+] Workspace mounted at /mnt/corpus inside VM."
echo "[+] To stop: kill \$(cat $REPO_ROOT/data/vm.pid)"
