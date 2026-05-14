# ============================================================
# ebpf-fuzzing-thesis — top-level Makefile
#
# Quick start (new machine):
#   1. Install missing system deps printed by `make check-deps`
#   2. make setup
#
# Individual build steps can be run independently after setup.
# ============================================================

REPO_ROOT   := $(shell git rev-parse --show-toplevel)
PIXI        := pixi run

# Load .env if present (keeps secrets out of shell history)
-include $(REPO_ROOT)/.env
export

# Linux 6.8.0 source
KERNEL_VER     := 6.8
KERNEL_TARBALL := linux-$(KERNEL_VER).tar.xz
KERNEL_URL     := https://cdn.kernel.org/pub/linux/kernel/v6.x/$(KERNEL_TARBALL)
KERNEL_SHA256  := c969dea4e8bb6be991bbf7c010ba0e0a5643a3a8d8fb0a2aaa053406f1e965f3
KERNEL_CACHE   := $(REPO_ROOT)/build/linux/$(KERNEL_TARBALL)
KERNEL_SRC     := $(REPO_ROOT)/build/linux/linux-$(KERNEL_VER)

.PHONY: all setup check-deps \
        build-kernel build-buzzer build-validator build-image \
        build-baseline train-curated train-baseline eval help

# ── DEFAULT ─────────────────────────────────────────────────
all: help

help:
	@echo ""
	@echo "  make setup            Full environment setup (run once on a new machine)"
	@echo "  make check-deps       Check for missing system dependencies"
	@echo ""
	@echo "  make build-kernel     Compile bzImage_kasan and bzImage_kasan_kcov (Linux 6.8)"
	@echo "  make build-buzzer     Build custom buzzer binary with Bazel"
	@echo "  make build-validator  Build ebpf_validator Go binary"
	@echo "  make build-image      Build trixie QEMU disk image + SSH keys"
	@echo ""
	@echo "  make build-baseline   Build baseline dataset from raw corpus"
	@echo "  make train-curated    Run Phase 2 curated SFT training (overnight)"
	@echo "  make train-baseline   Run Phase 2 baseline SFT training (overnight)"
	@echo "  make eval ADAPTER=<path>  Run Phase 3 pass-rate evaluation"
	@echo ""

# ── SETUP ───────────────────────────────────────────────────
# Full setup: check deps, install Python env, download ML data,
# then compile all fuzzing components from source.
setup: check-deps
	@echo "[*] Installing Python environment..."
	pixi install
	@echo ""
	@echo "[*] Login to HuggingFace (browser will open)..."
	$(PIXI) huggingface-cli login
	@echo ""
	@echo "[*] Downloading ML datasets from Strhata/ebpf-corpus..."
	$(PIXI) huggingface-cli download Strhata/ebpf-corpus --repo-type dataset --local-dir $(REPO_ROOT)/data
	@echo ""
	@echo "[*] Downloading Phase 1 adapter from Strhata/ebpf-checkpoints..."
	$(PIXI) huggingface-cli download Strhata/ebpf-checkpoints --repo-type model --local-dir $(REPO_ROOT)/checkpoints/sft_fase1
	@echo ""
	$(MAKE) build-kernel
	$(MAKE) build-buzzer
	$(MAKE) build-validator
	$(MAKE) build-image
	@echo ""
	@echo "[+] Setup complete. Run 'make train-curated' to start Phase 2 training."
	@echo "    (Set WANDB_API_KEY or run 'wandb login' before training.)"

# ── DEPENDENCY CHECK ────────────────────────────────────────
# Checks for required system tools. Prints missing ones and exits if any.
REQUIRED_BINS := clang llvm-objcopy bazel go debootstrap qemu-system-x86_64 \
                 qemu-img wget xz ssh scp curl gcc make flex bison libelf-dev

check-deps:
	@echo "[*] Checking system dependencies..."
	@MISSING=""; \
	for cmd in $(REQUIRED_BINS); do \
	    if ! command -v $$cmd >/dev/null 2>&1; then \
	        MISSING="$$MISSING $$cmd"; \
	    fi; \
	done; \
	if [ -n "$$MISSING" ]; then \
	    echo ""; \
	    echo "[!] Missing dependencies:$$MISSING"; \
	    echo ""; \
	    echo "    Install with:"; \
	    echo "    sudo apt install clang llvm golang-go bazel debootstrap \\"; \
	    echo "        qemu-system-x86 qemu-utils wget xz-utils openssh-client \\"; \
	    echo "        gcc make flex bison libelf-dev libssl-dev bc"; \
	    echo ""; \
	    exit 1; \
	fi
	@echo "[+] All dependencies found."

# ── KERNEL BUILD ────────────────────────────────────────────
# Downloads Linux 6.8 tarball (cached in build/linux/), verifies SHA256,
# then compiles bzImage_kasan and bzImage_kasan_kcov using committed configs.
build-kernel: $(REPO_ROOT)/fuzzing/bzImage_kasan $(REPO_ROOT)/fuzzing/bzImage_kasan_kcov

$(KERNEL_CACHE):
	@mkdir -p $(REPO_ROOT)/build/linux
	@echo "[*] Downloading Linux $(KERNEL_VER) tarball..."
	wget -q --show-progress -O $(KERNEL_CACHE) $(KERNEL_URL)
	@echo "[*] Verifying SHA256..."
	@echo "$(KERNEL_SHA256)  $(KERNEL_CACHE)" | sha256sum -c -
	@echo "[+] Tarball verified."

$(KERNEL_SRC): $(KERNEL_CACHE)
	@echo "[*] Extracting kernel source..."
	tar -xf $(KERNEL_CACHE) -C $(REPO_ROOT)/build/linux
	@echo "[+] Extracted to $(KERNEL_SRC)"

$(REPO_ROOT)/fuzzing/bzImage_kasan: $(KERNEL_SRC)
	@echo "[*] Building bzImage_kasan (KASAN only)..."
	cp $(REPO_ROOT)/fuzzing/kernel_kasan.config $(KERNEL_SRC)/.config
	$(MAKE) -C $(KERNEL_SRC) olddefconfig
	$(MAKE) -C $(KERNEL_SRC) bzImage -j$(shell nproc)
	cp $(KERNEL_SRC)/arch/x86/boot/bzImage $(REPO_ROOT)/fuzzing/bzImage_kasan
	@echo "[+] bzImage_kasan built."

$(REPO_ROOT)/fuzzing/bzImage_kasan_kcov: $(KERNEL_SRC)
	@echo "[*] Building bzImage_kasan_kcov (KASAN + KCOV)..."
	cp $(REPO_ROOT)/fuzzing/kernel_kasan_kcov.config $(KERNEL_SRC)/.config
	$(MAKE) -C $(KERNEL_SRC) olddefconfig
	$(MAKE) -C $(KERNEL_SRC) bzImage -j$(shell nproc)
	cp $(KERNEL_SRC)/arch/x86/boot/bzImage $(REPO_ROOT)/fuzzing/bzImage_kasan_kcov
	@echo "[+] bzImage_kasan_kcov built."

# ── BUZZER BUILD ────────────────────────────────────────────
# Builds the custom buzzer binary (google/buzzer fork with ML data collection)
# using Bazel. Output goes to data/corpus/ so the VM can access it via 9p mount.
build-buzzer:
	@echo "[*] Building buzzer with Bazel..."
	cd $(REPO_ROOT)/fuzzing/buzzer && bazel build //:buzzer
	@mkdir -p $(REPO_ROOT)/data/corpus
	cp $(REPO_ROOT)/fuzzing/buzzer/bazel-bin/buzzer $(REPO_ROOT)/data/corpus/buzzer
	chmod +x $(REPO_ROOT)/data/corpus/buzzer
	@echo "[+] buzzer → data/corpus/buzzer"

# ── VALIDATOR BUILD ─────────────────────────────────────────
# Builds the ebpf_validator Go binary for use inside the QEMU VM.
# Output goes to data/corpus/ so the VM can access it via 9p mount.
build-validator:
	@echo "[*] Building ebpf_validator..."
	cd $(REPO_ROOT)/tools/ebpf_validator && \
	    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o ebpf_validator .
	@mkdir -p $(REPO_ROOT)/data/corpus
	cp $(REPO_ROOT)/tools/ebpf_validator/ebpf_validator $(REPO_ROOT)/data/corpus/ebpf_validator
	chmod +x $(REPO_ROOT)/data/corpus/ebpf_validator
	@echo "[+] ebpf_validator → data/corpus/ebpf_validator"

# ── VM IMAGE BUILD ──────────────────────────────────────────
# Runs create-image.sh to build the Debian trixie VM disk image and
# generate a fresh SSH keypair. Requires sudo for debootstrap/mount.
build-image:
	@echo "[*] Building trixie VM image (requires sudo for debootstrap)..."
	@echo "[*] This may take 10-15 minutes..."
	cd $(REPO_ROOT)/fuzzing && sudo bash create-image.sh -d trixie -o trixie
	mv $(REPO_ROOT)/fuzzing/trixie.id_rsa $(REPO_ROOT)/fuzzing/trixie.id_rsa 2>/dev/null || true
	chmod 600 $(REPO_ROOT)/fuzzing/trixie.id_rsa
	@echo "[+] trixie.img and trixie.id_rsa ready in fuzzing/"

# ── ML: BASELINE DATASET ────────────────────────────────────
# Regenerates the baseline dataset from raw corpus files.
# Requires data/corpus/*.jsonl.gz (downloaded by make setup).
build-baseline:
	@echo "[*] Building baseline dataset..."
	$(PIXI) python ml/build_baseline_dataset.py
	@echo "[+] data/dataset_baseline_qwen.jsonl ready."

# ── ML: TRAINING ────────────────────────────────────────────
# Run Phase 2 curated training overnight.
# Resumes from checkpoints/sft_fase1/checkpoint-1500.
# Set WANDB_API_KEY env var or run `wandb login` before this target.
train-curated:
	@echo "[*] Starting curated training run..."
	$(PIXI) python ml/train.py --run curated

# Run Phase 2 baseline training overnight (trains from scratch).
# Set WANDB_API_KEY env var or run `wandb login` before this target.
train-baseline: build-baseline
	@echo "[*] Starting baseline training run..."
	$(PIXI) python ml/train.py --run baseline

# ── ML: EVALUATION ──────────────────────────────────────────
# Run Phase 3 pass-rate evaluation.
# ADAPTER is required: path to a trained adattatore_ebpf_v1 directory.
# VM_KEY defaults to fuzzing/trixie.id_rsa (built by make build-image).
#
# Example:
#   make eval ADAPTER=checkpoints/curated_3ep/adattatore_ebpf_v1
#   make eval ADAPTER=checkpoints/curated_3ep/adattatore_ebpf_v1 VM_KEY=/path/to/key
eval:
ifndef ADAPTER
	@echo "[!] ADAPTER is required. Example:"
	@echo "    make eval ADAPTER=checkpoints/curated_3ep/adattatore_ebpf_v1"
	@exit 1
endif
	$(PIXI) python tools/evaluate_passrate.py \
	    --adapter $(ADAPTER) \
	    --label $(notdir $(ADAPTER)) \
	    $(if $(VM_KEY),--vm-key $(VM_KEY),)
