package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
	"golang.org/x/sys/unix"
)

// KCOV ioctl numbers for x86_64, kernel 6.8+ (from /usr/include/linux/kcov.h):
//   KCOV_INIT_TRACE = _IOR('c', 1, unsigned long)
//                  = (2<<30)|(8<<16)|('c'<<8)|1 = 0x80086301
//   KCOV_ENABLE    = _IO('c', 100) = 0x6364
//   KCOV_DISABLE   = _IO('c', 101) = 0x6365
const (
	kcovInitTrace uintptr = 0x80086301
	kcovEnable    uintptr = 0x6364
	kcovDisable   uintptr = 0x6365
	kcovTracePC   uintptr = 0
	coverSize              = 64 << 10 // 64k uint64 entries
)

type result struct {
	Verdict string   `json:"verdict"`
	PCs     []uint64 `json:"pcs"`
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Uso: ./kcov_validator <bytecode_hex>")
		os.Exit(1)
	}

	// KCOV is per-thread; lock goroutine to OS thread so the BPF load
	// and the KCOV enable/disable happen on the same thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	res := validate(os.Args[1])
	out, _ := json.Marshal(res)
	fmt.Println(string(out))
	if res.Verdict == "ERRORE" {
		os.Exit(1)
	}
}

func validate(hexStr string) result {
	bytecode, err := hex.DecodeString(hexStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "decodifica hex: %v\n", err)
		return result{Verdict: "ERRORE", PCs: []uint64{}}
	}

	var insns asm.Instructions
	reader := bytes.NewReader(bytecode)
	for reader.Len() > 0 {
		var ins asm.Instruction
		if err := ins.Unmarshal(reader, binary.LittleEndian, ""); err != nil {
			fmt.Fprintf(os.Stderr, "parsing istruzione %d: %v\n", len(insns), err)
			return result{Verdict: "ERRORE", PCs: []uint64{}}
		}
		insns = append(insns, ins)
	}

	// Open KCOV device (requires O_RDWR to enable coverage)
	kcovFd, err := os.OpenFile("/sys/kernel/debug/kcov", os.O_RDWR, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "apertura kcov: %v\n", err)
		return result{Verdict: "ERRORE", PCs: []uint64{}}
	}
	defer kcovFd.Close()

	fd := kcovFd.Fd()

	// KCOV_INIT_TRACE: tell kernel to allocate coverSize-entry buffer
	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, kcovInitTrace, coverSize); errno != 0 {
		fmt.Fprintf(os.Stderr, "KCOV_INIT_TRACE: %v\n", errno)
		return result{Verdict: "ERRORE", PCs: []uint64{}}
	}

	// mmap the shared buffer: coverSize uint64 entries = coverSize*8 bytes
	area, err := unix.Mmap(int(fd), 0, coverSize*8, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mmap kcov: %v\n", err)
		return result{Verdict: "ERRORE", PCs: []uint64{}}
	}
	defer unix.Munmap(area)

	// View the mmap area as a []uint64: cover[0]=count, cover[1..n]=PCs
	cover := unsafe.Slice((*uint64)(unsafe.Pointer(&area[0])), coverSize)
	cover[0] = 0

	// KCOV_ENABLE with KCOV_TRACE_PC (flat PC array, no comparison tracing)
	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, kcovEnable, kcovTracePC); errno != 0 {
		fmt.Fprintf(os.Stderr, "KCOV_ENABLE: %v\n", errno)
		return result{Verdict: "ERRORE", PCs: []uint64{}}
	}

	// Load BPF program — this is the syscall KCOV traces
	spec := &ebpf.ProgramSpec{
		Type:         ebpf.SocketFilter,
		License:      "GPL",
		Instructions: insns,
	}
	prog, loadErr := ebpf.NewProgramWithOptions(spec, ebpf.ProgramOptions{
		LogLevel: ebpf.LogLevel(2),
	})

	// KCOV_DISABLE — always call before reading cover[0]
	unix.Syscall(unix.SYS_IOCTL, fd, kcovDisable, 0)

	if loadErr != nil {
		var ve *ebpf.VerifierError
		if errors.As(loadErr, &ve) {
			return result{Verdict: "RIFIUTATO", PCs: []uint64{}}
		}
		return result{Verdict: "ERRORE", PCs: []uint64{}}
	}
	prog.Close()

	n := cover[0]
	if n > coverSize-1 {
		n = coverSize - 1
	}
	pcs := make([]uint64, n)
	copy(pcs, cover[1:n+1])
	return result{Verdict: "ACCETTATO", PCs: pcs}
}
