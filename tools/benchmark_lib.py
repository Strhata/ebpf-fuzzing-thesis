"""
benchmark_lib.py — Importable modules for the eBPF model benchmark pipeline.

Modules:
    1.  BenchmarkConfig    — typed config loaded from benchmark.json
    2.  VMLifecycle        — context manager that boots/shuts down QEMU eval VM
    3.  ModelLoader        — loads model + tokenizer under a given strategy
    4.  Generator          — inference loop; returns assemblies + timing
    5.  ComplexityAnalyzer — bytecode-level complexity metrics
    6.  QEMUValidator      — wraps _validate_on_vm; never returns None
    7.  KPIAggregator       — aggregates per-slot KPIs
    8.  ResultsWriter       — writes timestamped JSON run + appends summary.csv
    9.  Reporter            — formats terminal + markdown output
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Module 1: BenchmarkConfig
# ---------------------------------------------------------------------------

@dataclass
class PromptSpec:
    id: str
    text: str


@dataclass
class BenchmarkConfig:
    model_id: str
    model_type: str           # "adapter" | "merged"
    model_path: str
    description: str
    prompts: list[PromptSpec]
    max_new_tokens: int
    temperature: float
    base_model: str | None = None
    skip_strategies: list[str] = field(default_factory=list)

    @property
    def strategies(self) -> list[str]:
        if self.model_type == "adapter":
            candidates = ["adapter_bnb4bit"]
        else:
            candidates = ["merged_fp16", "merged_bnb4bit"]
        return [s for s in candidates if s not in self.skip_strategies]

    @classmethod
    def from_dir(cls, checkpoint_dir: Path) -> BenchmarkConfig:
        cfg_path = checkpoint_dir / "benchmark.json"
        with cfg_path.open() as f:
            raw = json.load(f)
        prompts = [PromptSpec(**p) for p in raw["prompts"]]
        return cls(
            model_id=raw["model_id"],
            model_type=raw["model_type"],
            model_path=raw["model_path"],
            description=raw.get("description", ""),
            prompts=prompts,
            max_new_tokens=raw["max_new_tokens"],
            temperature=raw["temperature"],
            base_model=raw.get("base_model"),
            skip_strategies=raw.get("skip_strategies", []),
        )


# ---------------------------------------------------------------------------
# Module 2: VMLifecycle
# ---------------------------------------------------------------------------

class VMShutdownError(RuntimeError):
    pass


class VMLifecycle:
    """Context manager that boots run_eval_vm.sh and shuts down QEMU on exit."""

    _BOOT_SCRIPT = REPO_ROOT / "fuzzing" / "run_eval_vm.sh"
    _PID_FILE = REPO_ROOT / "data" / "vm.pid"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> VMLifecycle:
        self._proc = subprocess.Popen(
            ["bash", str(self._BOOT_SCRIPT)],
            cwd=str(REPO_ROOT),
        )
        # Script exits only after SSH is confirmed ready
        self._proc.wait()
        return self

    def shutdown(self) -> None:
        pid = self._read_vm_pid()
        if pid is None and self._proc is not None:
            pid = self._proc.pid
        if pid is None:
            raise VMShutdownError(
                "Cannot shut down VM: data/vm.pid is absent/invalid and no Popen PID available."
            )
        os.kill(pid, signal.SIGTERM)
        # Give the process a moment to exit before returning
        _wait_for_pid(pid, timeout=10)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def _read_vm_pid(self) -> int | None:
        try:
            text = self._PID_FILE.read_text().strip()
            return int(text)
        except (FileNotFoundError, ValueError):
            return None


def _wait_for_pid(pid: int, timeout: float) -> None:
    """Wait up to `timeout` seconds for process `pid` to exit; send SIGKILL if it lingers."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    # Process did not exit in time
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ---------------------------------------------------------------------------
# Module 3: ModelLoader
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    peak_load_gpu_mb: float


def load_model(config: BenchmarkConfig, strategy: str) -> LoadedModel:
    """Load model + tokenizer under the given strategy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    torch.cuda.reset_peak_memory_stats()

    bnb4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if strategy == "adapter_bnb4bit":
        base = config.base_model
        if base is None:
            raise ValueError(f"adapter_bnb4bit requires base_model in benchmark.json for {config.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=bnb4bit,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(base)

    elif strategy == "merged_fp16":
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    elif strategy == "merged_bnb4bit":
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb4bit,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    model.eval()
    peak_load_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
    return LoadedModel(model=model, tokenizer=tokenizer, peak_load_gpu_mb=peak_load_gpu_mb)


# ---------------------------------------------------------------------------
# Module 4: Generator
# ---------------------------------------------------------------------------

@dataclass
class GeneratorResult:
    assemblies: list[str]
    tokens_per_sec: float


@dataclass
class BatchGenResult:
    texts: list[str]          # decoded NEW tokens (prompt stripped), one per prompt, in order
    total_new_tokens: int
    seconds: float


def generate_batch(
    model: object,
    tokenizer: object,
    prompts: list[str],
    max_new_tokens: int,
    *,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float | None = None,
    batch_size: int = 16,
) -> BatchGenResult:
    """Generate completions for many prompts using batched, left-padded inference.

    Decoder-only models must be **left-padded** so generation continues from each
    prompt's real last token; an attention mask hides the padding. Returns the
    decoded new tokens per prompt (prompt prefix stripped), in input order.
    """
    import torch

    # Decoder-only generation requires left padding + a pad token.
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    texts: list[str] = []
    total_new = 0
    t0 = time.monotonic()
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start:start + batch_size]
                enc = tokenizer(chunk, return_tensors="pt", padding=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                input_len = enc["input_ids"].shape[1]
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=pad_id,
                )
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
                output = model.generate(**enc, **gen_kwargs)
                for row in output:
                    new = row[input_len:]               # slice off the (padded) prompt
                    total_new += int(new.shape[0])
                    texts.append(tokenizer.decode(new, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side

    return BatchGenResult(texts=texts, total_new_tokens=total_new, seconds=time.monotonic() - t0)


def generate(
    model: object,
    tokenizer: object,
    config: BenchmarkConfig,
    prompt_text: str,
    n: int,
    batch_size: int = 1,
) -> GeneratorResult:
    """Run inference for a single prompt sampled n times; return n assemblies + timing.

    With batch_size > 1 the n samples are generated in batched, left-padded calls
    (sampling makes each row independent); batch_size = 1 preserves the original
    one-at-a-time behaviour.
    """
    res = generate_batch(
        model, tokenizer, [prompt_text] * n, config.max_new_tokens,
        do_sample=True, temperature=config.temperature, batch_size=batch_size,
    )
    tokens_per_sec = res.total_new_tokens / res.seconds if res.seconds > 0 else 0.0
    return GeneratorResult(assemblies=res.texts, tokens_per_sec=tokens_per_sec)


# ---------------------------------------------------------------------------
# Module 5: ComplexityAnalyzer
# ---------------------------------------------------------------------------

_BPF_JMP_CLASS = 0x05
_BPF_JMP32_CLASS = 0x06


@dataclass
class ComplexityResult:
    encoded: bool
    insn_count: int | None
    opcode_diversity: float | None
    jump_count: int | None


def analyze(assembly: str) -> ComplexityResult:
    """Compute bytecode-level complexity metrics from a bare assembly string."""
    # Import inside function — reward.py has module-level side effects
    sys_path_insert = __import__("sys").path
    _ensure_ml_path()
    from reward import _encode_to_hex, strip_verifier_log  # noqa: PLC0415

    hex_str = _encode_to_hex(strip_verifier_log(assembly))
    if hex_str is None:
        return ComplexityResult(encoded=False, insn_count=None, opcode_diversity=None, jump_count=None)

    raw = bytes.fromhex(hex_str)
    insn_count = len(raw) // 8

    if insn_count == 0:
        return ComplexityResult(encoded=True, insn_count=0, opcode_diversity=0.0, jump_count=0)

    opcodes: list[int] = [raw[i * 8] for i in range(insn_count)]
    unique_opcodes = set(opcodes)
    opcode_diversity = len(unique_opcodes) / insn_count
    jump_count = sum(
        1 for op in opcodes
        if (op & 0x07) == _BPF_JMP_CLASS or (op & 0x07) == _BPF_JMP32_CLASS
    )
    return ComplexityResult(
        encoded=True,
        insn_count=insn_count,
        opcode_diversity=opcode_diversity,
        jump_count=jump_count,
    )


def _ensure_ml_path() -> None:
    import sys
    ml_path = str(REPO_ROOT / "ml")
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)


# ---------------------------------------------------------------------------
# Module 6: QEMUValidator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    verdict: str
    pcs: list[int]


def _vm_validate_fn(hex_str: str, ssh: object) -> dict | None:
    """Thin wrapper around reward._validate_on_vm; defined as a module-level name so
    tests can patch benchmark_lib._vm_validate_fn without triggering reward.py's
    module-level side effects at import time."""
    _ensure_ml_path()
    import reward as _reward_mod  # noqa: PLC0415
    return _reward_mod._validate_on_vm(hex_str, ssh)


def validate(hex_str: str, ssh: object) -> ValidationResult:
    """Validate a hex-encoded BPF program on the QEMU VM."""
    result = _vm_validate_fn(hex_str, ssh)
    if result is None:
        return ValidationResult(verdict="ERROR", pcs=[])
    return ValidationResult(
        verdict=result.get("verdict", "ERROR"),
        pcs=result.get("pcs", []),
    )


# ---------------------------------------------------------------------------
# Module 7: KPIAggregator
# ---------------------------------------------------------------------------

@dataclass
class ProgramRecord:
    assembly: str
    hex_str: str | None
    complexity: ComplexityResult
    validation: ValidationResult


@dataclass
class SlotKPIs:
    pass_rate: float
    encode_rate: float
    avg_pcs: float
    max_pcs: int
    total_unique_pcs: int
    novelty_score: float
    avg_insn_count: float
    avg_opcode_diversity: float
    avg_jump_count: float
    tokens_per_sec: float
    peak_load_gpu_mb: float
    peak_run_gpu_mb: float


def aggregate(
    programs: list[ProgramRecord],
    timing: GeneratorResult,
    peak_load_gpu_mb: float,
    peak_run_gpu_mb: float,
) -> SlotKPIs:
    """Compute slot-level KPIs from a list of ProgramRecords and timing info."""
    n = len(programs)
    if n == 0:
        return SlotKPIs(
            pass_rate=0.0,
            encode_rate=0.0,
            avg_pcs=0.0,
            max_pcs=0,
            total_unique_pcs=0,
            novelty_score=0.0,
            avg_insn_count=0.0,
            avg_opcode_diversity=0.0,
            avg_jump_count=0.0,
            tokens_per_sec=timing.tokens_per_sec,
            peak_load_gpu_mb=peak_load_gpu_mb,
            peak_run_gpu_mb=peak_run_gpu_mb,
        )

    accepted = [p for p in programs if p.validation.verdict == "ACCEPTED"]
    encoded = [p for p in programs if p.complexity.encoded]

    pass_rate = len(accepted) / n
    encode_rate = len(encoded) / n

    # avg_pcs over ACCEPTED programs only
    if accepted:
        avg_pcs = sum(len(p.validation.pcs) for p in accepted) / len(accepted)
    else:
        avg_pcs = 0.0

    all_pcs_union: set[int] = set()
    for p in programs:
        all_pcs_union.update(p.validation.pcs)

    max_pcs = max((len(p.validation.pcs) for p in programs), default=0)
    total_unique_pcs = len(all_pcs_union)

    # novelty_score: mean(1 - freq[pc]/N) over all pcs in union
    # freq[pc] = number of *programs* containing pc (not total occurrences)
    if all_pcs_union:
        freq: dict[int, int] = {}
        for p in programs:
            for pc in set(p.validation.pcs):
                freq[pc] = freq.get(pc, 0) + 1
        novelty_score = sum(1.0 - freq[pc] / n for pc in all_pcs_union) / len(all_pcs_union)
    else:
        novelty_score = 0.0

    if encoded:
        avg_insn_count = sum(p.complexity.insn_count for p in encoded) / len(encoded)
        avg_opcode_diversity = sum(p.complexity.opcode_diversity for p in encoded) / len(encoded)
        avg_jump_count = sum(p.complexity.jump_count for p in encoded) / len(encoded)
    else:
        avg_insn_count = 0.0
        avg_opcode_diversity = 0.0
        avg_jump_count = 0.0

    return SlotKPIs(
        pass_rate=pass_rate,
        encode_rate=encode_rate,
        avg_pcs=avg_pcs,
        max_pcs=max_pcs,
        total_unique_pcs=total_unique_pcs,
        novelty_score=novelty_score,
        avg_insn_count=avg_insn_count,
        avg_opcode_diversity=avg_opcode_diversity,
        avg_jump_count=avg_jump_count,
        tokens_per_sec=timing.tokens_per_sec,
        peak_load_gpu_mb=peak_load_gpu_mb,
        peak_run_gpu_mb=peak_run_gpu_mb,
    )


# ---------------------------------------------------------------------------
# Module 8: ResultsWriter
# ---------------------------------------------------------------------------

@dataclass
class SlotRecord:
    model_id: str
    strategy: str
    prompt_id: str
    programs: list[ProgramRecord]
    kpis: SlotKPIs


@dataclass
class RunRecord:
    timestamp: str
    scale: str
    git_commit: str
    slots: list[SlotRecord]


def _slot_to_dict(slot: SlotRecord) -> dict:
    programs_out = []
    for p in slot.programs:
        programs_out.append({
            "assembly": p.assembly,
            "hex_str": p.hex_str,
            "complexity": {
                "encoded": p.complexity.encoded,
                "insn_count": p.complexity.insn_count,
                "opcode_diversity": p.complexity.opcode_diversity,
                "jump_count": p.complexity.jump_count,
            },
            "validation": {
                "verdict": p.validation.verdict,
                "pcs": p.validation.pcs,
            },
        })
    return {
        "model_id": slot.model_id,
        "strategy": slot.strategy,
        "prompt_id": slot.prompt_id,
        "programs": programs_out,
        "kpis": slot.kpis.__dict__,
    }


def write(run: RunRecord, out_dir: Path) -> Path:
    """Write a timestamped run JSON to out_dir; return the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{run.timestamp}_run.json"
    out_path = out_dir / filename
    payload = {
        "timestamp": run.timestamp,
        "scale": run.scale,
        "git_commit": run.git_commit,
        "slots": [_slot_to_dict(s) for s in run.slots],
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return out_path


_SUMMARY_COLUMNS = [
    "timestamp", "model_id", "strategy", "prompt_id",
    "pass_rate", "encode_rate", "avg_pcs", "max_pcs", "total_unique_pcs",
    "novelty_score", "avg_insn_count", "avg_opcode_diversity", "avg_jump_count",
    "tokens_per_sec", "peak_load_gpu_mb", "peak_run_gpu_mb",
]


def append_summary(run: RunRecord, summary_path: Path) -> None:
    """Append one row per slot to summary_path (CSV)."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        for slot in run.slots:
            row = {"timestamp": run.timestamp, "model_id": slot.model_id,
                   "strategy": slot.strategy, "prompt_id": slot.prompt_id}
            row.update(slot.kpis.__dict__)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Module 9: Reporter
# ---------------------------------------------------------------------------

_KPI_COLS = [
    "pass_rate", "encode_rate", "avg_pcs", "max_pcs", "total_unique_pcs",
    "novelty_score", "avg_insn_count", "avg_opcode_diversity", "avg_jump_count",
    "tokens_per_sec", "peak_load_gpu_mb", "peak_run_gpu_mb",
]


def _flag(slot: SlotRecord, baselines: dict[str, float]) -> str:
    baseline = baselines.get(slot.model_id)
    if baseline is None:
        return ""
    if slot.strategy != "adapter_bnb4bit" and slot.kpis.pass_rate < baseline * 0.8:
        return "!"
    return ""


def _build_baselines(run: RunRecord) -> dict[str, float]:
    """Map model_id → adapter_bnb4bit pass_rate for this run."""
    result: dict[str, float] = {}
    for slot in run.slots:
        if slot.strategy == "adapter_bnb4bit":
            result[slot.model_id] = slot.kpis.pass_rate
    return result


def report(run: RunRecord) -> str:
    """Return a terminal-formatted string for the run."""
    baselines = _build_baselines(run)
    try:
        from rich.console import Console
        from rich.table import Table
        import io

        table = Table(title=f"Benchmark run {run.timestamp}  scale={run.scale}  commit={run.git_commit[:8]}")
        table.add_column("flag", style="bold red", width=4)
        table.add_column("model_id")
        table.add_column("strategy")
        table.add_column("prompt_id")
        for col in _KPI_COLS:
            table.add_column(col, justify="right")

        for slot in run.slots:
            flag = _flag(slot, baselines)
            kpi_vals = [f"{getattr(slot.kpis, c):.3f}" if isinstance(getattr(slot.kpis, c), float)
                        else str(getattr(slot.kpis, c)) for c in _KPI_COLS]
            table.add_row(flag, slot.model_id, slot.strategy, slot.prompt_id, *kpi_vals)

        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        console.print(table)
        return buf.getvalue()

    except ImportError:
        # Fallback: plain text
        header = "flag\tmodel_id\tstrategy\tprompt_id\t" + "\t".join(_KPI_COLS)
        rows = [header]
        for slot in run.slots:
            flag = _flag(slot, baselines)
            kpi_vals = "\t".join(
                f"{getattr(slot.kpis, c):.3f}" if isinstance(getattr(slot.kpis, c), float)
                else str(getattr(slot.kpis, c))
                for c in _KPI_COLS
            )
            rows.append(f"{flag}\t{slot.model_id}\t{slot.strategy}\t{slot.prompt_id}\t{kpi_vals}")
        return "\n".join(rows)


def report_md(run: RunRecord) -> str:
    """Return a GitHub-flavored markdown table for the run."""
    baselines = _build_baselines(run)
    cols = ["flag", "model_id", "strategy", "prompt_id"] + _KPI_COLS
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [
        f"## Benchmark run `{run.timestamp}`",
        f"scale={run.scale}  commit={run.git_commit[:8]}",
        "",
        header,
        sep,
    ]
    for slot in run.slots:
        flag = _flag(slot, baselines)
        kpi_vals = [
            f"{getattr(slot.kpis, c):.3f}" if isinstance(getattr(slot.kpis, c), float)
            else str(getattr(slot.kpis, c))
            for c in _KPI_COLS
        ]
        row = "| " + " | ".join([flag, slot.model_id, slot.strategy, slot.prompt_id] + kpi_vals) + " |"
        lines.append(row)
    return "\n".join(lines)
