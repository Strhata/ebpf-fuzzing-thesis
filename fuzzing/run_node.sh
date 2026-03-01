#!/bin/bash

MODE=""

# 1. Standard UNIX getopts parsing
while getopts "bs" opt; do
    case ${opt} in
        b ) MODE="blind" ;;
        s ) MODE="smart" ;;
        \? ) 
            echo "Usage: $0 [-b | -s] <Node_ID>"
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))
ID=$1

if [ -z "$MODE" ] || [ -z "$ID" ]; then
    echo "====================================================="
    echo "Usage: $0 [-b | -s] <Node_ID>"
    echo ""
    echo "Options:"
    echo "  -b : Blind Mode (Uses bzImage_blind, NO KCOV)"
    echo "  -s : Smart Mode (Uses bzImage_smart, uploads vmlinux)"
    echo "====================================================="
    exit 1
fi

PORT=$((10020 + ID))
MAC="52:54:00:12:34:5${ID}"
LOG="vm${ID}_log.txt"

# Dynamic Kernel Selection
if [ "$MODE" == "smart" ]; then
    KERNEL_IMG="linux/arch/x86/boot/bzImage_smart"
    MEMORY="6G"
else
    KERNEL_IMG="linux/arch/x86/boot/bzImage_blind"
    MEMORY="2G"
fi

# --- NODE SPECIFIC CLEANUP ---
cleanup() {
    echo -e "\n[*] Node $ID shutting down cleanly..."
    kill -9 $(cat vm${ID}.pid 2>/dev/null) 2>/dev/null
    rm -f vm${ID}.pid
    exit 0
}
trap cleanup SIGINT SIGTERM

mkdir -p crash_logs
rm -f vm${ID}.qcow2
qemu-img create -f qcow2 -b bullseye.img -F raw vm${ID}.qcow2 2G >/dev/null

while true; do
    echo "[Node $ID] Booting QEMU on port $PORT using $KERNEL_IMG (2 Cores)..."
    
    qemu-system-x86_64 \
        -m $MEMORY -smp 2 \
        -kernel $KERNEL_IMG \
        -append "console=ttyS0 root=/dev/sda rw earlyprintk=serial net.ifnames=0 panic=1" \
        -drive file=vm${ID}.qcow2,format=qcow2 \
        -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:${PORT}-:22 \
        -net nic,model=e1000,macaddr=$MAC \
        -enable-kvm -display none -daemonize \
        -serial file:$LOG -pidfile vm${ID}.pid

    echo "[Node $ID] Waiting for VM to boot..."
    until ssh -q -p $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" -o "ConnectTimeout=2" root@localhost "echo 1" >/dev/null 2>&1; do
        sleep 2
    done

    # ==========================================
    # DYNAMIC MODE EXECUTION
    # ==========================================
    if [ "$MODE" == "smart" ]; then
        echo "[Node $ID] MODE: SMART (-s) -> Uploading fuzzer and vmlinux (~500MB)..."
        scp -q -P $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" buzzer_bin linux/vmlinux root@localhost:/root/

        echo "[Node $ID] Launching Attack! (Coverage-Guided / High-Depth)"
        ssh -q -p $PORT -i bullseye.id_rsa \
            -o "StrictHostKeyChecking no" \
            -o "ServerAliveInterval=3" \
            -o "ServerAliveCountMax=2" \
            root@localhost "chmod +x buzzer_bin && while true; do ./buzzer_bin --strategy=pointer_arithmetic --vmlinux_path=/root/vmlinux; done"

    else
        echo "[Node $ID] MODE: BLIND (-b) -> Uploading fuzzer only (High-Speed)..."
        scp -q -P $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" buzzer_bin root@localhost:/root/

        echo "[Node $ID] Launching Attack! (Brute Force / High-Throughput)"
        
        ssh -q -p $PORT -i bullseye.id_rsa \
            -o "StrictHostKeyChecking no" \
            -o "ServerAliveInterval=3" \
            -o "ServerAliveCountMax=2" \
            root@localhost "chmod +x buzzer_bin && while true; do ./buzzer_bin --strategy=pointer_arithmetic; done"
    fi
    # ==========================================

    echo "[Node $ID] ⚠️ CONNECTION LOST! Checking for kernel panic..."
    if grep -qiE "panic|KASAN|Call Trace|Oops" $LOG; then
        CRASH_FILE="crash_logs/vm${ID}_crash_$(date +%s).txt"
        echo "[Node $ID] 🚨 CRASH DETECTED! Saved to $CRASH_FILE"
        cp $LOG $CRASH_FILE
    else
        echo "[Node $ID] No panic. VM locked up naturally."
    fi
    
    echo "[Node $ID] Restarting QEMU in 2 seconds..."
    kill -9 $(cat vm${ID}.pid 2>/dev/null) 2>/dev/null
    rm -f vm${ID}.pid
    sleep 2 
    > $LOG
done