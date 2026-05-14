#!/bin/bash
echo "[*] Initializing Auto-Restarting Fuzzing Swarm..."

# --- BULLETPROOF CLEANUP TRAP ---
# We will store the exact Process IDs of the 3 background workers here
WORKER_PIDS=""

cleanup() {
    echo -e "\n[*] Ctrl+C detected! Shutting down the swarm..."
    
    # 1. Kill the exact background worker loops so they don't restart QEMU
    if [ -n "$WORKER_PIDS" ]; then
        kill -9 $WORKER_PIDS 2>/dev/null
    fi
    
    # 2. Kill the QEMU instances using their exact lock files
    for i in 1 2 3; do
        if [ -f "vm${i}.pid" ]; then
            kill -9 $(cat vm${i}.pid) 2>/dev/null
            rm -f vm${i}.pid
        fi
    done
    
    # 3. Final sweep just to be safe
    pkill -9 -f qemu-system-x86 2>/dev/null
    
    echo "[*] Swarm cleanly shut down. Zero zombies remain."
    exit 0
}
trap cleanup SIGINT SIGTERM
# --------------------------------

mkdir -p crash_logs

# 1. Create fresh QCOW2 overlays
for i in 1 2 3; do
  rm -f vm${i}.qcow2
  qemu-img create -f qcow2 -b bullseye.img -F raw vm${i}.qcow2 2G >/dev/null
done

# 2. Define the Worker Function
run_worker() {
    local ID=$1
    local PORT=$((10020 + ID))
    local MAC="52:54:00:12:34:5${ID}"
    local LOG="vm${ID}_log.txt"

    while true; do
        echo "[Worker $ID] Booting QEMU on port $PORT..."
        
        qemu-system-x86_64 \
            -m 2G -smp 2 \
            -kernel linux/arch/x86/boot/bzImage \
            -append "console=ttyS0 root=/dev/sda rw earlyprintk=serial net.ifnames=0 panic=1" \
            -drive file=vm${ID}.qcow2,format=qcow2 \
            -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:${PORT}-:22 \
            -net nic,model=e1000,macaddr=$MAC \
            -enable-kvm -display none -daemonize \
            -serial file:$LOG -pidfile vm${ID}.pid

        echo "[Worker $ID] Waiting for VM to boot and SSH to open..."
        until ssh -q -p $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" -o "ConnectTimeout=2" root@localhost "echo 1" >/dev/null 2>&1; do
            sleep 2
        done

        echo "[Worker $ID] VM is UP! Uploading fuzzer..."
        scp -q -P $PORT -i bullseye.id_rsa -o "StrictHostKeyChecking no" buzzer_bin root@localhost:/root/

        echo "[Worker $ID] Launching Attack! (Monitoring for crashes...)"
        
        # Fuzzer is trapped in an infinite loop INSIDE the VM.
        ssh -q -p $PORT -i bullseye.id_rsa \
            -o "StrictHostKeyChecking no" \
            -o "ServerAliveInterval=3" \
            -o "ServerAliveCountMax=2" \
            root@localhost "chmod +x buzzer_bin && while true; do ./buzzer_bin --strategy=pointer_arithmetic; done"

        # If we reach here, the SSH connection broke (Kernel Panic!)
        echo "[Worker $ID] ⚠️ CONNECTION LOST! Checking for kernel panic..."
        
        if grep -qiE "panic|KASAN|Call Trace|Oops" $LOG; then
            CRASH_FILE="crash_logs/vm${ID}_crash_$(date +%s).txt"
            echo "[Worker $ID] 🚨 CRASH DETECTED! Saving log to $CRASH_FILE"
            cp $LOG $CRASH_FILE
        else
            echo "[Worker $ID] No panic found. VM locked up naturally."
        fi
        
        echo "[Worker $ID] Cleaning up dead QEMU process before restart..."
        kill -9 $(cat vm${ID}.pid 2>/dev/null) 2>/dev/null
        rm -f vm${ID}.pid
        sleep 2 
        > $LOG
    done
}

# 3. Launch the workers and save their exact PIDs to our trap variable!
run_worker 1 &
WORKER_PIDS="$WORKER_PIDS $!"

run_worker 2 &
WORKER_PIDS="$WORKER_PIDS $!"

run_worker 3 &
WORKER_PIDS="$WORKER_PIDS $!"

echo "========================================="
echo "[SUCCESS] The Resilient Swarm is ALIVE!"
echo "Leave this terminal open. Press Ctrl+C to cleanly kill the swarm."
echo "========================================="

wait
