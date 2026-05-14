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
		fmt.Println("Uso: ./validator <bytecode_esadecimale>")
		os.Exit(1)
	}

	hexStr := os.Args[1]

	// 1. Decodifica la stringa hex in byte crudi
	bytecode, err := hex.DecodeString(hexStr)
	if err != nil {
		fmt.Printf("[-] Errore decodifica hex: %v\n", err)
		os.Exit(1)
	}

	// 2. Parsiamo l'assembly ciclando sulle singole istruzioni (8 byte l'una)
	var insns asm.Instructions
	reader := bytes.NewReader(bytecode)

	for reader.Len() > 0 {
		var ins asm.Instruction
		// Il terzo parametro "platform" è usato per log o estensioni specifiche, una stringa vuota va benissimo.
		if err := ins.Unmarshal(reader, binary.LittleEndian, ""); err != nil {
			fmt.Printf("[-] Errore parsing assembly eBPF all'istruzione %d: %v\n", len(insns), err)
			os.Exit(1)
		}
		insns = append(insns, ins)
	}

	fmt.Printf("[*] Bytecode parsato con successo: %d istruzioni\n", len(insns))

	// 3. Prepara le specifiche del programma per il Kernel
	spec := &ebpf.ProgramSpec{
		Type:         ebpf.SocketFilter, // Tipo di programma sicuro
		License:      "GPL",
		Instructions: insns,
	}

	// 4. Spara il programma nel Verifier
	fmt.Println("[*] Invio al Verifier del Kernel...")
	prog, err := ebpf.NewProgramWithOptions(spec, ebpf.ProgramOptions{
		LogLevel: ebpf.LogLevel(2), // Livello 2: Log iper-dettagliato
	})

	// 5. Analizza il verdetto
	var ve *ebpf.VerifierError
	if err != nil {
		if errors.As(err, &ve) {
			fmt.Println("\n--- LOG DEL VERIFIER (RIFIUTATO) ---")
			fmt.Printf("%+v\n", ve) // Il format %+v stampa tutto il dump formattato
			fmt.Println("------------------------------------")
		} else {
			fmt.Printf("\n[!] Altro Errore (non dal verifier): %v\n", err)
		}
		fmt.Printf("\n[!] VERDETTO: RIFIUTATO!\n")
		os.Exit(1)
	}

	fmt.Printf("\n[+] VERDETTO: ACCETTATO!\n")
	prog.Close()
}
