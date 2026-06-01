package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"
)

// KCOV ioctl numbers (x86_64, kernel 6.8+)
const (
	kcovInitTrace uintptr = 0x80086301
	kcovEnable    uintptr = 0x6364
	kcovDisable   uintptr = 0x6365
	kcovTracePC   uintptr = 0
	coverSize              = 256 << 10 // 256k entries — large enough for complex programs

	bpfProgLoad       uintptr = 5
	bpfSocketFilter   uint32  = 1
	bpfLogLevelStats  uint32  = 1
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

type result struct {
	Verdict string   `json:"verdict"`
	PCs     []uint64 `json:"pcs"`
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: ./kcov_validator <bytecode_hex>")
		os.Exit(1)
	}

	// KCOV is per-thread; lock goroutine to OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	res := validate(os.Args[1])
	out, _ := json.Marshal(res)
	fmt.Println(string(out))
	if res.Verdict == "ERROR" {
		os.Exit(1)
	}
}

func validate(hexStr string) result {
	bytecode, err := hex.DecodeString(hexStr)
	if err != nil || len(bytecode) == 0 || len(bytecode)%8 != 0 {
		fmt.Fprintln(os.Stderr, "invalid bytecode")
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}

	// --- KCOV setup ---
	kcovFd, err := os.OpenFile("/sys/kernel/debug/kcov", os.O_RDWR, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open kcov: %v\n", err)
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}
	defer kcovFd.Close()

	kfd := kcovFd.Fd()

	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, kfd, kcovInitTrace, coverSize); errno != 0 {
		fmt.Fprintf(os.Stderr, "KCOV_INIT_TRACE: %v\n", errno)
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}

	area, err := unix.Mmap(int(kfd), 0, coverSize*8, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mmap kcov: %v\n", err)
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}
	defer unix.Munmap(area)

	cover := unsafe.Slice((*uint64)(unsafe.Pointer(&area[0])), coverSize)
	cover[0] = 0

	// --- BPF_PROG_LOAD attr ---
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

	// --- KCOV_ENABLE: trace only the BPF_PROG_LOAD syscall ---
	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, kfd, kcovEnable, kcovTracePC); errno != 0 {
		fmt.Fprintf(os.Stderr, "KCOV_ENABLE: %v\n", errno)
		return result{Verdict: "ERROR", PCs: []uint64{}}
	}

	progFd, _, sysErrno := unix.Syscall(
		unix.SYS_BPF,
		bpfProgLoad,
		uintptr(unsafe.Pointer(&attr)),
		unsafe.Sizeof(attr),
	)

	unix.Syscall(unix.SYS_IOCTL, kfd, kcovDisable, 0) // always disable before reading

	pcs := readPCs(cover)

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
