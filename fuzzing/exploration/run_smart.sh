#!/bin/bash

ID=$1

if [ -z "$ID" ]; then
    echo "====================================================="
    echo "Usage: $0 <Node_ID>"
    echo "Example: $0 1"
    echo "====================================================="
    exit 1
fi

PORT=$((10020 + ID))
METRICS_PORT=$((8080 + ID))
MAC="52:54:00:12:34:5${ID}"
LOG="vm${ID}_log.txt"

# Smart Mode specific variables
KERNEL_IMG="linux/arch/x86/boot/bzImage_kasan_kcov"
MEMORY="6G" # Needed to fit the 500MB vmlinux into the RAM disk safely

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
    echo "[Node $ID] Booting QEMU on port $PORT (Metrics UI on $METRICS_PORT)..."
    
    # Added hostfwd for the Metrics Server (8080 -> 8080+ID)
    qemu-system-x86_64 \
        -m $MEMORY -smp 2 \
        -kernel $KERNEL_IMG \
        -append "console=ttyS0 root=/dev/sda rw earlyprintk=serial net.ifnames=0 panic=1" \
        -drive file=vm${ID}.qcow2,format=qcow2 \
        -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:${PORT}-:22,hostfwd=tcp:0.0.0.0:${METRICS_PORT}-:8080 \
        -net nic,model=e1000,macaddr=$MAC \
        -enable-kvm -display none -daemonize \
        -serial file:$LOG -pidfile vm${ID}.pid

    echo "[Node $ID] Waiting for VM to boot..."
    until ssh -q -p $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" -o "ConnectTimeout=2" root@localhost "echo 1" >/dev/null 2>&1; do
        sleep 2
    done

    # ==========================================
    # SMART MODE EXECUTION
    # ==========================================
    echo "[Node $ID] Uploading fuzzer to /root/ ..."
    scp -q -P $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" buzzer_bin root@localhost:/root/
    
    echo "[Node $ID] Uploading vmlinux (500MB) to RAM DISK (/dev/shm) ... This might take a moment."
    scp -q -P $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" linux/vmlinux root@localhost:/dev/shm/

    echo "[Node $ID] 🚀 Launching Attack! (Silenced output, high threshold)"
    echo "[Node $ID] 📊 View Live Coverage at: http://0.0.0.0:$METRICS_PORT"
    
    # Note on strategy: Change '--strategy=pointer_arithmetic' to '--strategy=coverage' 
    # if you want it to focus purely on reaching new lines of code.
    ssh -q -p $PORT -i bullseye.id_rsa \
        -o "StrictHostKeyChecking no" \
        -o "ServerAliveInterval=3" \
        -o "ServerAliveCountMax=2" \
        root@localhost "chmod +x /root/buzzer_bin && while true; do /root/buzzer_bin --strategy=pointer_arithmetic --vmlinux_path=/dev/shm/vmlinux --metrics_threshold=5000 > /dev/null 2>&1; done"

    # ==========================================

    echo "[Node $ID] ⚠️ CONNECTION LOST! Checking for kernel panic..."
    if grep -qiE "panic|KASAN|Call Trace|Oops" $LOG; then
        CRASH_FILE="crash_logs/vm${ID}_crash_$(date +%s).txt"
        echo "[Node $ID] 🚨 CRASH DETECTED! Saved to $CRASH_FILE"
        cp $LOG $CRASH_FILE
    else
        echo "[Node $ID] No panic. VM locked up naturally or OOM killed fuzzer."
    fi
    
    echo "[Node $ID] Restarting QEMU in 2 seconds..."
    kill -9 $(cat vm${ID}.pid 2>/dev/null) 2>/dev/null
    rm -f vm${ID}.pid
    sleep 2 
    > $LOG
done