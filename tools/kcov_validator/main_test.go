package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// minimalProg is "BPF_MOV64_IMM(R0, 0); BPF_EXIT" — a valid socket-filter program
// (returns 0). Used by the integration tests as a known-good, deterministic input.
const minimalProg = "b7000000000000009500000000000000"

// --- kernel-free unit tests (run anywhere) ---------------------------------

func TestDecodeProgram(t *testing.T) {
	cases := []struct {
		name string
		hex  string
		ok   bool
	}{
		{"valid one-insn", "b700000000000000", true},
		{"valid two-insn", minimalProg, true},
		{"non-hex", "zzzz", false},
		{"empty", "", false},
		{"not-8-multiple", "b7010000", false}, // 4 bytes
	}
	for _, c := range cases {
		if _, ok := decodeProgram(c.hex); ok != c.ok {
			t.Errorf("%s: decodeProgram ok=%v, want %v", c.name, ok, c.ok)
		}
	}
}

func TestRunBatchFramingAndOrder(t *testing.T) {
	orig := validateOne
	defer func() { validateOne = orig }()
	// Stub: "BAD" -> ERROR (empty PCs), anything else -> ACCEPTED with one PC.
	validateOne = func(hexStr string, k *kcov) result {
		if hexStr == "BAD" {
			return result{Verdict: "ERROR", PCs: []uint64{}}
		}
		return result{Verdict: "ACCEPTED", PCs: []uint64{uint64(len(hexStr))}}
	}

	// blank line in the middle must be skipped, not indexed.
	in := strings.NewReader("aa\n\nBAD\nbbbb\n")
	var out bytes.Buffer
	if err := runBatch(nil, in, &out); err != nil {
		t.Fatalf("runBatch: %v", err)
	}

	var got []indexedResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal %q: %v", out.String(), err)
	}
	if len(got) != 3 {
		t.Fatalf("want 3 results (blank skipped), got %d: %q", len(got), out.String())
	}

	want := []struct {
		idx     int
		verdict string
	}{{0, "ACCEPTED"}, {1, "ERROR"}, {2, "ACCEPTED"}}
	for i, w := range want {
		if got[i].Index != w.idx || got[i].Verdict != w.verdict {
			t.Errorf("result %d = {idx:%d, %s}, want {idx:%d, %s}",
				i, got[i].Index, got[i].Verdict, w.idx, w.verdict)
		}
	}
	// malformed entry stays isolated: ERROR has empty PCs, neighbours do not.
	if len(got[1].PCs) != 0 {
		t.Errorf("ERROR entry should have empty PCs, got %v", got[1].PCs)
	}
	if len(got[0].PCs) == 0 || len(got[2].PCs) == 0 {
		t.Errorf("ACCEPTED entries should carry PCs")
	}
}

func TestRunBatchEmptyStdin(t *testing.T) {
	orig := validateOne
	defer func() { validateOne = orig }()
	validateOne = func(string, *kcov) result {
		t.Fatal("validateOne must not be called on empty input")
		return result{}
	}

	var out bytes.Buffer
	if err := runBatch(nil, strings.NewReader(""), &out); err != nil {
		t.Fatalf("runBatch: %v", err)
	}
	if got := strings.TrimSpace(out.String()); got != "[]" {
		t.Errorf("empty stdin should produce [], got %q", got)
	}
}

// --- integration tests (require a running kernel with the KCOV device) -------

func skipWithoutKCOV(t *testing.T) {
	t.Helper()
	if _, err := os.Stat("/sys/kernel/debug/kcov"); err != nil {
		t.Skip("no KCOV device (/sys/kernel/debug/kcov) — run on the eval VM")
	}
}

func TestSingleModeKnownGoodIntegration(t *testing.T) {
	skipWithoutKCOV(t)
	k, err := kcovSetup()
	if err != nil {
		t.Fatalf("kcovSetup: %v", err)
	}
	defer k.close()

	res := validateOne(minimalProg, k)
	if res.Verdict != "ACCEPTED" {
		t.Fatalf("minimal program verdict = %s, want ACCEPTED", res.Verdict)
	}
	if len(res.PCs) == 0 {
		t.Errorf("ACCEPTED program should report non-empty PCs")
	}
}

// TestBatchResetIsolationIntegration validates the same program twice in one
// batch. With the per-program cover-counter reset, both runs must report the
// same number of PCs; if the reset were broken, the second run would accumulate
// the first run's PCs and report more.
func TestBatchResetIsolationIntegration(t *testing.T) {
	skipWithoutKCOV(t)
	k, err := kcovSetup()
	if err != nil {
		t.Fatalf("kcovSetup: %v", err)
	}
	defer k.close()

	in := strings.NewReader(minimalProg + "\n" + minimalProg + "\n")
	var out bytes.Buffer
	if err := runBatch(k, in, &out); err != nil {
		t.Fatalf("runBatch: %v", err)
	}

	var got []indexedResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 results, got %d", len(got))
	}
	if got[0].Verdict != "ACCEPTED" || got[1].Verdict != "ACCEPTED" {
		t.Fatalf("both runs should be ACCEPTED, got %s/%s", got[0].Verdict, got[1].Verdict)
	}
	if len(got[0].PCs) != len(got[1].PCs) {
		t.Errorf("cover reset broken: run0 has %d PCs, run1 has %d (should be equal)",
			len(got[0].PCs), len(got[1].PCs))
	}
}
