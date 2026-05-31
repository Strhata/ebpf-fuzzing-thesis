package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: ./validator <hex_bytecode>")
		os.Exit(1)
	}

	hexStr := os.Args[1]

	bytecode, err := hex.DecodeString(hexStr)
	if err != nil {
		fmt.Printf("[-] hex decode error: %v\n", err)
		os.Exit(1)
	}

	var insns asm.Instructions
	reader := bytes.NewReader(bytecode)

	for reader.Len() > 0 {
		var ins asm.Instruction
		if err := ins.Unmarshal(reader, binary.LittleEndian, ""); err != nil {
			fmt.Printf("[-] eBPF instruction parse error at insn %d: %v\n", len(insns), err)
			os.Exit(1)
		}
		insns = append(insns, ins)
	}

	fmt.Printf("[*] Bytecode parsed: %d instructions\n", len(insns))

	spec := &ebpf.ProgramSpec{
		Type:         ebpf.SocketFilter,
		License:      "GPL",
		Instructions: insns,
	}

	fmt.Println("[*] Submitting to kernel verifier...")
	prog, err := ebpf.NewProgramWithOptions(spec, ebpf.ProgramOptions{
		LogLevel: ebpf.LogLevel(2),
	})

	var ve *ebpf.VerifierError
	if err != nil {
		if errors.As(err, &ve) {
			fmt.Println("\n--- VERIFIER LOG (RIFIUTATO) ---")
			fmt.Printf("%+v\n", ve)
			fmt.Println("--------------------------------")
		} else {
			fmt.Printf("\n[!] Non-verifier error: %v\n", err)
		}
		fmt.Printf("\n[!] VERDETTO: RIFIUTATO!\n")
		os.Exit(1)
	}

	fmt.Printf("\n[+] VERDETTO: ACCETTATO!\n")
	prog.Close()
}
