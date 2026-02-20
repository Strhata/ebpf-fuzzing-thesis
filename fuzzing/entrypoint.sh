#!/bin/bash
# entrypoint.sh

echo "[*] Container Partito. Genero disco QCOW2 temporaneo..."
# Creiamo un disco usa-e-getta nella RAM/disco del container
qemu-img create -f qcow2 -b trixie.img -F raw vm_disk.qcow2 2G >/dev/null

echo "[*] Avvio QEMU con accelerazione KVM..."
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

echo "[*] Attesa boot VM..."
until ssh -q -p 10022 -i trixie.id_rsa -o "StrictHostKeyChecking no" -o "ConnectTimeout=2" root@127.0.0.1 "echo 1" >/dev/null 2>&1; do
    sleep 2
done

echo "[*] Connesso! Avvio estrazione dati JSONL..."
# Monta la cartella condivisa mappata su Docker e lancia il fuzzer
ssh -q -p 10022 -i trixie.id_rsa \
    -o "StrictHostKeyChecking no" \
    root@127.0.0.1 "mkdir -p /mnt/corpus && mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 corpus_share /mnt/corpus && /mnt/corpus/buzzer -strategy=pointer_arithmetic"

echo "[!] Fuzzer terminato o VM crashata. Il container si spegnerà."