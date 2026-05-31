#!/bin/bash
# entrypoint.sh

echo "[*] Container started. Creating ephemeral QCOW2 disk..."
qemu-img create -f qcow2 -b trixie.img -F raw vm_disk.qcow2 2G >/dev/null

echo "[*] Starting QEMU with KVM acceleration..."
qemu-system-x86_64 \
    -m 3G -smp 2 \
    -kernel bzImage \
    -append "console=ttyS0 root=/dev/sda rw earlyprintk=serial net.ifnames=0 panic=1" \
    -drive file=vm_disk.qcow2,format=qcow2 \
    -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:10022-:22 \
    -net nic,model=e1000 \
    -fsdev local,security_model=none,id=fsdev_corpus,path=/shared_corpus \
    -device virtio-9p-pci,id=fs_corpus,fsdev=fsdev_corpus,mount_tag=corpus_share \
    -enable-kvm -display none -daemonize \
    -pidfile qemu.pid

echo "[*] Waiting for VM boot..."
until ssh -q -p 10022 -i trixie.id_rsa -o "StrictHostKeyChecking no" -o "ConnectTimeout=2" root@127.0.0.1 "echo 1" >/dev/null 2>&1; do
    sleep 2
done

echo "[*] Connected. Starting JSONL data extraction..."
ssh -q -p 10022 -i trixie.id_rsa \
    -o "StrictHostKeyChecking no" \
    root@127.0.0.1 "mkdir -p /mnt/corpus && mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 corpus_share /mnt/corpus && /mnt/corpus/buzzer -strategy=pointer_arithmetic"

echo "[!] Fuzzer exited or VM crashed. Container shutting down."