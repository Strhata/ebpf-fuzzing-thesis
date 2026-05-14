# Recap tesi — Fuzzing del verifier eBPF

> Documento ricostruito dai file in `/home/stefano-u/fuzzing_lab/` e `/home/stefano-u/fuzzing_ml_env/`
> e dalle annotazioni in `note.txt`. Serve come recap personale e come base per la stesura della tesi.
> Due sezioni richiedono un intervento manuale: sono marcate con **[DA COMPLETARE]**.

---

## 1. Obiettivo iniziale

Il relatore ha chiesto di fare fuzzing del **verifier eBPF del kernel Linux usando AFL**.
Questo è il mio primo lavoro di ricerca: parte della narrazione qui sotto è proprio la
curva di apprendimento — scelte che col senno di poi cambierei, e cambi di rotta
motivati dall'aver capito strada facendo cosa stavo usando davvero.

---

## 2. Il malinteso su buzzer

- **Assunzione iniziale:** pensavo che [buzzer](https://github.com/google/buzzer) fosse un
  wrapper AFL specializzato sul verifier eBPF, coerente con la richiesta del relatore.
- **Realtà:** buzzer è un fuzzer eBPF **standalone**, scritto in Go da Google, con
  strategie proprie (es. `--strategy=pointer_arithmetic`, `--strategy=coverage_based`).
  Non usa AFL sotto.
- **Come me ne sono accorto:** dal *mismatch di comportamento* — non c'era la UI tipica
  di `afl-fuzz`, non c'era la shared-memory coverage map (la `__AFL_SHM_ID`), le
  metriche e i flag non corrispondevano affatto a quelli di AFL/AFL++.
- **Conseguenza:** nel lavoro svolto fin qui **AFL non è mai stato integrato**. La
  richiesta del relatore resta quindi aperta — vedi sezione 10.

---

## 3. Fase 1 — Infrastruttura VM/kernel (1–3 marzo)

- [`create-image.sh`](fuzzing_lab/create-image.sh) adattato da syzkaller → immagini
  Debian `bullseye.img` e `trixie.img` (~2 GB ciascuna) con chiavi SSH per root
  passwordless.
- Sorgenti di **Linux 6.8.0** in `fuzzing_lab/linux/`, compilati in tre varianti:
  - `bzImage` — kernel standard
  - `bzImage_kasan` — **KCOV disattivato**, baseline di throughput del fuzzer
  - `bzImage_kasan_kcov` — **KCOV + KASAN + UBSAN** attivi, coverage-guided
- [`note.txt`](fuzzing_lab/note.txt) contiene le linee di comando QEMU provate,
  incluso il montaggio via `virtfs` 9p per condividere kernel e corpus col guest.

---

## 4. Fase 2 — Run con buzzer e swarm (5–10 marzo)

Script runner in `fuzzing_lab/`:

| Script | Ruolo |
|--------|-------|
| [`start_swarm.sh`](fuzzing_lab/start_swarm.sh) | Lancia 3 VM QEMU in parallelo, con auto-restart on crash |
| [`run_node.sh`](fuzzing_lab/run_node.sh) | Runner con flag `-b` (blind) / `-s` (smart); carica `buzzer` e `vmlinux` nel guest |
| [`run_smart.sh`](fuzzing_lab/run_smart.sh) | Modalità smart con UI metriche su porta `8080 + ID` |
| [`run_fake.sh`](fuzzing_lab/run_fake.sh) | Sanity check con `vmlinux` fasullo, per isolare problemi di coverage |

Tentativi di profiling (per capire dove si spendevano i cicli):

- [`diagnose1.sh`](fuzzing_lab/diagnose1.sh), [`diagnose2.sh`](fuzzing_lab/diagnose2.sh),
  [`diagnose3.sh`](fuzzing_lab/diagnose3.sh), [`diagnose_bottleneck.sh`](fuzzing_lab/diagnose_bottleneck.sh)
  → `strace`, `perf`, `GODEBUG=gctrace,schedtrace`.
- Output raccolti in `fuzzing_lab/diagnostic_data/`.

Risultati raccolti:

- **`crash_logs/`** — 30 dump di kernel panic / KASAN (4–5 marzo), distribuiti
  sulle 4 VM attive (`vm1…vm4_log.txt`).

---

## 5. Fase 3 — Containerizzazione (26–29 marzo)

Per rendere riproducibile il setup e poter scalare i nodi:

- [`Dockerfile`](fuzzing_lab/Dockerfile) + [`entrypoint.sh`](fuzzing_lab/entrypoint.sh) →
  container che lancia QEMU **annidato** (`--device /dev/kvm`), aspetta SSH, poi esegue
  `/mnt/corpus/buzzer --strategy=pointer_arithmetic`.
- Corpus condiviso host ↔ guest tramite **virtio-9p**.
- Il container è `fuzzer_node_1` (nome referenziato in `rotate_log.txt` e in `note.txt`).

---

## 6. Fase 4 — Pivot: generazione di programmi eBPF con un LLM (27 marzo → 16 aprile)

Tutto il lavoro ML vive in `fuzzing_ml_env/`.

### Motivazione del pivot

**[DA COMPLETARE]** — scrivi qui perché hai deciso di passare dal fuzzing "classico"
alla generazione di bytecode via LLM: era perché buzzer non era AFL e volevi
comunque produrre input interessanti? perché il corpus di syzkaller era ricco e
sfruttabile come dataset di training? perché coverage-guided su eBPF non stava dando
bug nuovi? solo tu conosci il vero perché.

### Setup di training

- **Modello base:** Qwen2.5-Coder-1.5B.
- **Tecnica:** **QLoRA** (4-bit NF4), rank-16 su `Q/K/V/O`; ~0.14% parametri
  addestrati (≈ 2.1 M su 1.5 B).
- **Iperparametri:** batch effettivo 8 (gradient accumulation), `max_seq_len` 768,
  learning rate 1–2e-4, AdamW 8-bit.
- **Hardware:** RTX 4070 Laptop, 8.59 GB VRAM, compute 8.9, BF16 ok
  (verificato in [`gpu_check.ipynb`](fuzzing_ml_env/gpu_check.ipynb)).
- **Ambiente:** [`pixi.toml`](fuzzing_ml_env/pixi.toml) — Python 3.11, PyTorch 2.5.1 + CUDA 12.1,
  Transformers ≥ 5.4, PEFT 0.18.1, BitsAndBytes, SentencePiece.

### Notebooks

| Notebook | Data | Contenuto |
|----------|------|-----------|
| [`Untitled.ipynb`](fuzzing_ml_env/Untitled.ipynb) | 29 mar | Banco di prova. Prompt universale: `Status: VALID \| Complexity: N insns` per i validi, `Status: INVALID \| Error: … \| Instr: …` per gli invalidi. 2000 step, transfer learning a partire da `modello_ebpf_produzione/checkpoint-1000`. |
| [`data_analisys.ipynb`](fuzzing_ml_env/data_analisys.ipynb) | 10 apr | Analisi dei 73 dump syzkaller (27 mar → 9 apr): 13 153 validi + 14 361 invalidi. Normalizzazione errori del verifier (mask di numeri e indirizzi). Cap 2 000 esempi per classe di errore → `dataset_final_qwen.jsonl` (27 514 esempi). |
| [`qwen_fine_tuning.ipynb`](fuzzing_ml_env/qwen_fine_tuning.ipynb) | 16 apr | Versione intermedia del fine-tuning (fp16). |
| [`SFT_tesi.ipynb`](fuzzing_ml_env/SFT_tesi.ipynb) | 16 apr | Versione "tesi": BF16 + FlashAttention-2, 1500 step, split 90/10, salvataggio best-on-val. |
| [`gpu_check.ipynb`](fuzzing_ml_env/gpu_check.ipynb) | 16 apr | Diagnostica hardware. |

### Directory dei modelli (in ordine cronologico)

| Directory | Ruolo |
|-----------|-------|
| [`modello_ebpf_lora/`](fuzzing_ml_env/modello_ebpf_lora/) | Primi tentativi LoRA (checkpoint 100–500) |
| [`modello_ebpf_finetuned/`](fuzzing_ml_env/modello_ebpf_finetuned/) | Baseline SFT iniziale |
| [`modello_ebpf_fase2/`](fuzzing_ml_env/modello_ebpf_fase2/) | Run "fase 2" |
| [`modello_ebpf_produzione/`](fuzzing_ml_env/modello_ebpf_produzione/) | `checkpoint-1000` usato come base per transfer learning |
| [`modello_ebpf_3000/`](fuzzing_ml_env/modello_ebpf_3000/) | Run completo 2000 step, checkpoint 500–2000 |
| [`modello_ebpf_definitivo/`](fuzzing_ml_env/modello_ebpf_definitivo/) | Adapter LoRA finale (~8.7 MB) |
| [`adattatore_ebpf_finale/`](fuzzing_ml_env/adattatore_ebpf_finale/) | Copia / finalizzazione dell'adapter |

### Output generati

I file [`risultati_checkpoint-*.txt`](fuzzing_ml_env/) contengono il bytecode eBPF
generato dai vari checkpoint, in formato esadecimale, con terminazione
`9500000000000000` (istruzione `BPF_EXIT`).

---

## 7. Fase 5 — Valutazione closed-loop (documentata in `note.txt`)

Pipeline costruita a mano per misurare la qualità delle generazioni:

1. Genero bytecode con il checkpoint fine-tuned (dai notebook sopra).
2. Dentro la VM QEMU, monto il corpus via 9p in `/mnt/corpus`.
3. Passo il file `risultati_checkpoint-*.txt` riga per riga al mio tool
   **`ebpf_validator`** (scritto da me), che chiama il verifier e stampa
   `VERDETTO: ACCETTATO` / `VERDETTO: RIFIUTATO`.
4. Conto gli accettati e calcolo il pass-rate per checkpoint.
5. Per i primi 25 programmi, dump completo del log del verifier per classificare
   le cause di rifiuto.

Il loop completo è in `note.txt` (linee 56–87), inclusa la versione `docker exec`
che invoca `ebpf_validator` sul container `fuzzer_node_1`.

### Pass-rate misurati

**[DA COMPLETARE]** — riempi con i numeri che hai raccolto:

| Checkpoint | File | Accettati / Totali | Pass-rate |
|------------|------|--------------------|-----------|
| 500 | `risultati_checkpoint-500.txt` | … / 21 | … % |
| 1000 (run 1) | `risultati_checkpoint-1000.txt` | … / 21 | … % |
| 1000 (run 2) | `risultati_checkpoint-1000-1.txt` | … / 15 | … % |
| 1000 (run 3) | `risultati_checkpoint-1000-2.txt` | … / 15 | … % |
| 1000 (run 4) | `risultati_checkpoint-1000-3.txt` | … / 15 | … % |
| 2000 | `risultati_checkpoint-2000.txt` | … / 50 | … % |
| 3000 | `risultati_checkpoint_3000.txt` | … / 50 | … % |

---

## 8. Fase 6 — Rotazione corpus (8–21 aprile)

- [`rotate_dataset.sh`](fuzzing_lab/rotate_dataset.sh) — pausa `fuzzer_node_1`,
  comprime il JSONL con gzip + timestamp, riavvia il container.
- Il container *può* crashare (è il punto del fuzzing del kernel): per questo c'è
  il restart automatico. Le righe `[!] fuzzer_node_1 is not running` in
  [`rotate_log.txt`](fuzzing_lab/rotate_log.txt) sono la rotazione che incontra il
  nodo tra un crash e il restart — comportamento atteso, non un bug.
- [`shared_corpus/`](fuzzing_lab/shared_corpus/) — ~13 GB compressi,
  `dataset_syzkaller_347…405.jsonl.gz` (8–9 aprile), ~600k righe ciascuno, campo
  `bytecode_hex`.

---

## 9. Cosa NON è stato fatto

- **AFL non integrato.** La richiesta originale del relatore resta aperta.
- **Nessun confronto quantitativo** tra le run di buzzer e i bytecode generati da
  Qwen fine-tuned.

---

## 10. Prossimi passi (proposte, non decisioni)

1. Integrare davvero **AFL / AFL++** con un harness dedicato al verifier eBPF, per
   chiudere la richiesta iniziale del relatore.
2. Confronto sistematico, a parità di tempo-macchina:
   - buzzer con strategia coverage-guided,
   - bytecode generati dal modello fine-tuned,
   - (ideale) AFL++ una volta integrato.
   Metriche: pass-rate sul verifier, coverage del kernel (KCOV), bug unici.
3. Pulire il dataset dei bytecode generati e pubblicarlo come artefatto di tesi,
   insieme all'adapter LoRA finale.

---

## Appendice A — Timeline dei file chiave

| Data | File / directory | Descrizione |
|------|------------------|-------------|
| 1 mar | `bullseye/`, `bullseye.id_rsa` | Immagine Debian bullseye + ssh key |
| 1 mar | `trixie/`, `trixie.id_rsa` | Immagine Debian trixie + ssh key |
| 1 mar | `create-image.sh` | Builder immagine (adattato da syzkaller) |
| 3 mar | `buzzer_bin` | Binario buzzer compilato |
| 4–5 mar | `crash_logs/` | 30 dump di kernel panic / KASAN |
| 5 mar | `start_swarm.sh`, `run_node.sh` | Orchestrazione 3 VM + runner nodo |
| 6 mar | `run_smart.sh`, `run_fake.sh` | Modalità smart / sanity check |
| 7–10 mar | `diagnose*.sh`, `diagnostic_data/` | Profiling del fuzzer |
| 10 mar | `linux/` (ultimo build) | Linux 6.8.0 con 3 `bzImage` (standard/blind/smart) |
| 26 mar | `bullseye.img` | Immagine finale bullseye |
| 27 mar | `Dockerfile`, `entrypoint.sh` | Containerizzazione |
| 28 mar | `pixi.toml`, prime run LoRA | Setup ambiente ML + `modello_ebpf_lora/` |
| 28 mar | `modello_ebpf_fase2/`, `modello_ebpf_produzione/` | Fase 2 + ckpt produzione |
| 28–29 mar | `risultati_checkpoint-500/1000/2000/3000.txt` | Generazioni dei checkpoint |
| 29 mar | `Untitled.ipynb`, `trixie.img` | Banco di prova training + immagine trixie finale |
| 8 apr | `rotate_dataset.sh`, `note.txt` (aggiornato) | Rotazione corpus + pipeline validazione |
| 8–9 apr | `shared_corpus/dataset_syzkaller_347…405.jsonl.gz` | ~13 GB di corpus ruotato |
| 10 apr | `data_analisys.ipynb` | Dataset finale 27 514 esempi |
| 16 apr | `SFT_tesi.ipynb`, `qwen_fine_tuning.ipynb`, `gpu_check.ipynb` | Versione "tesi" del training |
| 21 apr | `rotate_log.txt` (ultima riga) | Ultima esecuzione registrata della rotazione |

---

## Appendice B — Comandi chiave da `note.txt`

**Avvio QEMU con sharing del sorgente kernel via 9p**
```bash
qemu-system-x86_64 \
    -m 4G -smp 4 \
    -kernel $HOME/tesi/linux/arch/x86/boot/bzImage \
    -append "console=ttyS0 root=/dev/sda earlyprintk=serial net.ifnames=0" \
    -drive file=$HOME/vm_image/trixie.img,format=raw \
    -net user,hostfwd=tcp::10022-:22,hostfwd=tcp::10250-:10250 \
    -net nic,model=e1000 \
    -display none -pidfile vm.pid \
    -virtfs local,path=$HOME/tesi/linux,mount_tag=host_linux,security_model=none,id=linux_src \
    -daemonize
```

**Upload di `vmlinux` e `buzzer` nel guest**
```bash
scp -i trixie.id_rsa -P 10022 -o "StrictHostKeyChecking no" \
    $HOME/tesi/linux/vmlinux root@localhost:/root/vmlinux
scp -i trixie.id_rsa -P 10022 -o "StrictHostKeyChecking no" \
    $HOME/tesi/buzzer/bazel-bin/buzzer_/buzzer root@localhost:/root/buzzer
```

**QEMU con `bzImage_kasan` e corpus condiviso**
```bash
qemu-system-x86_64 \
    -m 2G -smp 2 \
    -kernel linux/arch/x86/boot/bzImage_kasan \
    -append "console=ttyS0 root=/dev/sda rw earlyprintk=serial net.ifnames=0" \
    -drive file=trixie.img,format=raw \
    -nographic \
    -fsdev local,security_model=none,id=fsdev_corpus,path=./shared_corpus \
    -device virtio-9p-pci,id=fs_corpus,fsdev=fsdev_corpus,mount_tag=corpus_share

# dentro il guest:
mkdir -p /mnt/corpus
mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 corpus_share /mnt/corpus
```

**Loop di validazione dei bytecode generati**
```bash
cd /mnt/corpus
contatore=1
accettati=0
while read -r line; do
    res=$(./ebpf_validator "$line" | grep "VERDETTO")
    if [[ $res == *"ACCETTATO"* ]]; then
        accettati=$((accettati+1))
        echo "[+] Prog $contatore: OK"
    else
        echo "[-] Prog $contatore: FAIL"
    fi
    contatore=$((contatore+1))
done < risultati_checkpoint-1000-1.txt

echo " RISULTATO FINALE: $accettati su $((contatore-1))"
echo " PASS RATE: $((accettati * 100 / (contatore-1)))%"
```

**Validazione di un singolo bytecode via `docker exec`**
```bash
docker exec -it fuzzer_node_1 \
    ssh -p 10022 -i trixie.id_rsa -o "StrictHostKeyChecking no" root@127.0.0.1 \
    "/mnt/corpus/ebpf_validator <hex_bytecode>"
```

**Run del container fuzzer**
```bash
docker run -d \
    --name fuzzer_node_1 \
    --device /dev/kvm \
    -v /home/stefano-u/fuzzing_lab/shared_corpus:/shared_corpus \
    ebpffuzzer:v1
```
