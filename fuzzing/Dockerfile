# Usiamo un'immagine base leggerissima
FROM debian:bookworm-slim

# Installiamo QEMU e SSH (client)
RUN apt-get update && apt-get install -y \
    qemu-system-x86 \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Creiamo la cartella di lavoro
WORKDIR /fuzzing_lab

# Copiamo gli asset fondamentali dall'host dentro l'immagine Docker
COPY linux/arch/x86/boot/bzImage_blind /fuzzing_lab/bzImage
COPY trixie.img /fuzzing_lab/trixie.img
COPY trixie.id_rsa /fuzzing_lab/trixie.id_rsa

# Sistemiamo i permessi della chiave SSH
RUN chmod 600 /fuzzing_lab/trixie.id_rsa

# Creiamo il punto di mount per la cartella condivisa (dove metteremo il fuzzer e i log)
RUN mkdir -p /shared_corpus

# Copiamo lo script di avvio
COPY entrypoint.sh /fuzzing_lab/entrypoint.sh
RUN chmod +x /fuzzing_lab/entrypoint.sh

# Quando il container parte, lancia lo script
ENTRYPOINT ["/fuzzing_lab/entrypoint.sh"]