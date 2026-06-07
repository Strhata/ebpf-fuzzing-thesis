package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"unsafe"

	"golang.org/x/sys/unix"
)

// KCOV ioctl numbers (x86_64, kernel 6.8+)
const (
	kcovInitTrace uintptr = 0x80086301
	kcovEnable    uintptr = 0x6364
	kcovDisable   uintptr = 0x6365
	kcovTracePC   uintptr = 0
	coverSize             = 256 << 10 // 256k entries — large enough for complex programs

	bpfProgLoad      uintptr = 5
	bpfSocketFilter  uint32  = 1
	bpfLogLevelStats uint32  = 1
)

// bpfProgLoadAttr matches the first fields of union bpf_attr for BPF_PROG_LOAD.
// Fields beyond ExpectedAttachType are zero-initialised by Go, which is correct.
type bpfProgLoadAttr struct {
	ProgType           uint32
	InsnCnt            uint32
	Insns              uint64
	License            uint64
	LogLevel           uint32
	LogSize            uint32
	LogBuf             uint64
	KernVersion        uint32
	ProgFlags          uint32
	ProgName           [16]byte
	ProgIfindex        uint32
	ExpectedAttachType uint32
}

// result is the single-program JSON contract: {"verdict": ..., "pcs": [...]}.
type result struct {
	Verdict string   `json:"verdict"`
	PCs     []uint64 `json:"pcs"`
}

// indexedResult is one element of the --batch JSON array, carrying the program's
// position in the input so callers can map results back by index.
type indexedResult struct {
	Index   int      `json:"index"`
	Verdict string   `json:"verdict"`
	PCs     []uint64 `json:"pcs"`
}

func main() {
	// KCOV is per-thread; lock goroutine to OS thread for the whole run.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Batch mode: read newline-delimited hex programs from stdin, emit a JSON array.
	// KCOV is initialised once and reused across every program in the batch.
	if len(os.Args) == 2 && os.Args[1] == "--batch" {
		k, err := kcovSetup()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer k.close()
		if err := runBatch(k, os.Stdin, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	// Single-program mode — unchanged JSON contract (one {verdict, pcs} object).
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: ./kcov_validator <bytecode_hex>  |  ./kcov_validator --batch < programs.txt")
		os.Exit(1)
	}
	k, err := kcovSetup()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		out, _ := json.Marshal(result{Verdict: "ERROR", PCs: []uint64{}})
		fmt.Println(string(out))
		os.Exit(1)
	}
	defer k.close()

	res := validateOne(os.Args[1], k)
	out, _ := json.Marshal(res)
	fmt.Println(string(out))
	if res.Verdict == "ERROR" {
		os.Exit(1)
	}
}

// runBatch reads one hex program per line from r and writes a JSON array of
// indexed results to w. Blank lines are skipped. A program that fails to
// validate becomes an ERROR entry rather than aborting the batch; only an I/O
// error on the input stream is returned.
func runBatch(k *kcov, r io.Reader, w io.Writer) error {
	results := []indexedResult{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 64<<20) // allow very long hex lines (large programs)

	idx := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		res := validateOne(line, k)
		results = append(results, indexedResult{Index: idx, Verdict: res.Verdict, PCs: res.PCs})
		idx++
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("read stdin: %w", err)
	}

	enc, err := json.Marshal(results)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(w, string(enc))
	return err
}

// kcov holds a one-time KCOV setup (fd + mmap'd cover buffer) reused across many
// program validations so the batch path pays the init cost once.
type kcov struct {
	fd    *os.File
	area  []byte
	cover []uint64
}

func kcovSetup() (*kcov, error) {
	fd, err := os.OpenFile("/sys/kernel/debug/kcov", os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("open kcov: %w", err)
	}
	kfd := fd.Fd()

	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, kfd, kcovInitTrace, coverSize); errno != 0 {
		fd.Close()
		return nil, fmt.Errorf("KCOV_INIT_TRACE: %v", errno)
	}

	area, err := unix.Mmap(int(kfd), 0, coverSize*8, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		fd.Close()
		return nil, fmt.Errorf("mmap kcov: %w", err)
	}

	cover := unsafe.Slice((*uint64)(unsafe.Pointer(&area[0])), coverSize)
	return &kcov{fd: fd, area: area, cover: cover}, nil
}

func (k *kcov) close() {
	unix.Munmap(k.area)
	k.fd.Close()
}

// decodeProgram parses a hex string into BPF bytecode. Kernel-free; returns
// (bytecode, true) on success, or (nil, false) if the hex is malformed or not a
// whole number of 8-byte instructions.
func decodeProgram(hexStr string) ([]byte, bool) {
	bytecode, err := hex.DecodeString(hexStr)
	if err != nil || len(bytecode) == 0 || len(bytecode)%8 != 0 {
		return nil, false
	}
	return bytecode, true
}

// validateOne loads one program under an already-initialised KCOV, resetting the
// coverage counter first so PCs do not leak between programs in a batch.
// Declared as a var so tests can stub out the kernel interaction.
var validateOne = func(hexStr string, k *kcov) result {
	bytecode, ok := decodeProgram(hexStr)
	if !ok {
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}

	kfd := k.fd.Fd()
	k.cover[0] = 0 // reset coverage counter for this program (isolation)

	license := []byte("GPL\x00")
	logBuf := make([]byte, 1<<20) // 1 MiB verifier log

	attr := bpfProgLoadAttr{
		ProgType: bpfSocketFilter,
		InsnCnt:  uint32(len(bytecode) / 8),
		Insns:    uint64(uintptr(unsafe.Pointer(&bytecode[0]))),
		License:  uint64(uintptr(unsafe.Pointer(&license[0]))),
		LogLevel: bpfLogLevelStats,
		LogSize:  uint32(len(logBuf)),
		LogBuf:   uint64(uintptr(unsafe.Pointer(&logBuf[0]))),
	}

	// KCOV_ENABLE: trace only this thread's syscalls (the BPF_PROG_LOAD below).
	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, kfd, kcovEnable, kcovTracePC); errno != 0 {
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}

	progFd, _, sysErrno := unix.Syscall(
		unix.SYS_BPF,
		bpfProgLoad,
		uintptr(unsafe.Pointer(&attr)),
		unsafe.Sizeof(attr),
	)

	unix.Syscall(unix.SYS_IOCTL, kfd, kcovDisable, 0) // always disable before reading

	pcs := readPCs(k.cover)

	if sysErrno == 0 {
		unix.Close(int(progFd))
		return result{Verdict: "ACCEPTED", PCs: pcs}
	}

	// Verifier ran if it wrote anything to the log buffer.
	for _, b := range logBuf {
		if b != 0 {
			return result{Verdict: "REJECTED", PCs: pcs}
		}
	}

	// BPF syscall failed before the verifier (e.g. bad insn count, missing map FD)
	return result{Verdict: "ERROR", PCs: []uint64{}}
}

func readPCs(cover []uint64) []uint64 {
	n := cover[0]
	if n > coverSize-1 {
		n = coverSize - 1
	}
	pcs := make([]uint64, n)
	copy(pcs, cover[1:n+1])
	return pcs
}
