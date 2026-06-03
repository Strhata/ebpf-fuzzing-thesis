#!/usr/bin/env python3
"""
benchmark.py — eBPF model benchmark pipeline.

Subcommands:
    prepare <checkpoint_dir>              Merge LoRA adapter into a full-precision model.
    run [--models ...] [--scale ...] ...  Run inference + QEMU validation for all models.
    report [run_json]                     Print + write report for a run bundle.

Usage:
    pixi run python tools/benchmark.py prepare checkpoints/sft_retrain/checkpoint-1500
    pixi run python tools/benchmark.py run --scale small
    pixi run python tools/benchmark.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml"))

from benchmark_lib import (
    BenchmarkConfig,
    GeneratorResult,
    LoadedModel,
    ProgramRecord,
    RunRecord,
    SlotKPIs,
    SlotRecord,
    VMLifecycle,
    aggregate,
    analyze,
    append_summary,
    generate,
    load_model,
    report,
    report_md,
    validate,
    write,
)

_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
_BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
_RUNS_DIR = _BENCHMARKS_DIR / "runs"
_SUMMARY_CSV = _BENCHMARKS_DIR / "summary.csv"

_SCALE_N: dict[str, int] = {
    "small": 50,
    "full": 200,
}


# ---------------------------------------------------------------------------
# prepare subcommand
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir

    config = BenchmarkConfig.from_dir(checkpoint_dir)
    if config.model_type != "adapter":
        print(f"[!] Expected model_type=adapter, got {config.model_type!r}. Aborting.", file=sys.stderr)
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = config.base_model
    print(f"[*] Loading base model {base} in bfloat16 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base)

    print(f"[*] Attaching LoRA adapter from {config.model_path} ...")
    peft_model = PeftModel.from_pretrained(base_model, config.model_path)

    print("[*] Merging weights ...")
    merged = peft_model.merge_and_unload()

    merged_dir = Path(str(checkpoint_dir) + "-merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving merged model to {merged_dir} ...")
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    merged_cfg = {
        "model_id": config.model_id + "_merged",
        "model_type": "merged",
        "model_path": str(merged_dir),
        "description": config.description + " [merged]",
        "prompts": [{"id": p.id, "text": p.text} for p in config.prompts],
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "skip_strategies": config.skip_strategies,
    }
    cfg_out = merged_dir / "benchmark.json"
    with cfg_out.open("w") as f:
        json.dump(merged_cfg, f, indent=2)

    print(f"[+] Done. Merged model written to {merged_dir}")
    print(f"[+] Auto-generated {cfg_out}")


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

def _discover_configs(model_ids: list[str] | None) -> list[tuple[Path, BenchmarkConfig]]:
    results: list[tuple[Path, BenchmarkConfig]] = []
    seen_dirs: set[Path] = set()
    # Walk up to two levels deep: checkpoints/<name>/ and checkpoints/<name>/<checkpoint>/
    candidates: list[Path] = []
    for d in sorted(_CHECKPOINTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (d / "benchmark.json").exists():
            candidates.append(d)
        else:
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and (sub / "benchmark.json").exists():
                    candidates.append(sub)
            if not any((d / sub.name / "benchmark.json").exists() for sub in d.iterdir() if sub.is_dir()):
                print(f"[!] No benchmark.json in {d} — skipping.", file=sys.stderr)
    for checkpoint_dir in candidates:
        if checkpoint_dir in seen_dirs:
            continue
        seen_dirs.add(checkpoint_dir)
        config = BenchmarkConfig.from_dir(checkpoint_dir)
        if model_ids and config.model_id not in model_ids:
            continue
        results.append((checkpoint_dir, config))
    return results


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_slot(
    config: BenchmarkConfig,
    strategy: str,
    prompt_spec,
    n: int,
    ssh,
    no_vm: bool,
    loaded: LoadedModel,
) -> SlotRecord:
    import torch

    print(f"  [model={config.model_id} strategy={strategy} prompt={prompt_spec.id} n={n}]")

    torch.cuda.reset_peak_memory_stats()
    result = generate(loaded.model, loaded.tokenizer, config, prompt_spec.text, n)
    peak_run_gpu_mb = torch.cuda.max_memory_allocated() / 1e6

    programs: list[ProgramRecord] = []
    for asm in result.assemblies:
        cx = analyze(asm)
        if no_vm or cx.insn_count is None or cx.insn_count == 0:
            vr = ValidationResult_stub(cx, no_vm)
        else:
            vr = validate(cx_hex(asm), ssh)
        programs.append(ProgramRecord(
            assembly=asm,
            hex_str=cx_hex(asm),
            complexity=cx,
            validation=vr,
        ))

    kpis = aggregate(programs, result, loaded.peak_load_gpu_mb, peak_run_gpu_mb)
    return SlotRecord(
        model_id=config.model_id,
        strategy=strategy,
        prompt_id=prompt_spec.id,
        programs=programs,
        kpis=kpis,
    )


def cx_hex(assembly: str) -> str | None:
    """Re-encode assembly to hex; used to populate ProgramRecord.hex_str."""
    _ensure_ml_path()
    from reward import _encode_to_hex
    return _encode_to_hex(assembly)


def _ensure_ml_path() -> None:
    ml = str(REPO_ROOT / "ml")
    if ml not in sys.path:
        sys.path.insert(0, ml)


def ValidationResult_stub(cx, no_vm: bool):
    """Return a no-validation ValidationResult when VM is skipped or encoding failed."""
    from benchmark_lib import ValidationResult
    if not cx.encoded:
        return ValidationResult(verdict="ERROR", pcs=[])
    if no_vm:
        return ValidationResult(verdict="SKIPPED", pcs=[])
    return ValidationResult(verdict="ERROR", pcs=[])


def cmd_run(args: argparse.Namespace) -> None:
    from reward import SSHClient

    model_ids: list[str] | None = args.models or None
    scale: str = args.scale
    n: int = args.n if args.n is not None else _SCALE_N[scale]
    no_vm: bool = args.no_vm
    out_dir = Path(args.out) if args.out else _RUNS_DIR

    configs = _discover_configs(model_ids)
    if not configs:
        print("[!] No benchmark.json configs found.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Running benchmark: scale={scale} n={n} no_vm={no_vm}")
    print(f"[*] Models: {[c.model_id for _, c in configs]}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_commit = _git_commit()
    slots: list[SlotRecord] = []

    def _run_all(ssh) -> None:
        import torch
        for _dir, config in configs:
            for strategy in config.strategies:
                print(f"  [loading model={config.model_id} strategy={strategy}]")
                torch.cuda.reset_peak_memory_stats()
                loaded = load_model(config, strategy)
                for prompt_spec in config.prompts:
                    slot = _run_slot(config, strategy, prompt_spec, n, ssh, no_vm, loaded)
                    slots.append(slot)
                del loaded
                torch.cuda.empty_cache()

    if no_vm:
        _run_all(ssh=None)
    else:
        with VMLifecycle():
            ssh = _build_ssh()
            _run_all(ssh)

    run = RunRecord(timestamp=timestamp, scale=scale, git_commit=git_commit, slots=slots)
    run_path = write(run, out_dir)
    append_summary(run, _SUMMARY_CSV)

    md = report_md(run)
    md_path = out_dir / f"{timestamp}_report.md"
    with md_path.open("w") as f:
        f.write(md)

    print(report(run))
    print(f"[+] Run written to {run_path}")
    print(f"[+] Report written to {md_path}")
    print(f"[+] Summary appended to {_SUMMARY_CSV}")


def _build_ssh():
    _ensure_ml_path()
    from reward import SSHClient
    return SSHClient()


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------

def _load_run(run_json: str | None) -> RunRecord:
    if run_json:
        path = Path(run_json)
    else:
        candidates = sorted(_RUNS_DIR.glob("*_run.json"))
        if not candidates:
            print("[!] No run files found in benchmarks/runs/.", file=sys.stderr)
            sys.exit(1)
        path = candidates[-1]

    with path.open() as f:
        raw = json.load(f)

    from benchmark_lib import (
        ComplexityResult,
        ProgramRecord,
        SlotKPIs,
        SlotRecord,
        ValidationResult,
    )

    slots: list[SlotRecord] = []
    for s in raw["slots"]:
        programs: list[ProgramRecord] = []
        for p in s["programs"]:
            cx = ComplexityResult(**p["complexity"])
            vr = ValidationResult(**p["validation"])
            programs.append(ProgramRecord(
                assembly=p["assembly"],
                hex_str=p.get("hex_str"),
                complexity=cx,
                validation=vr,
            ))
        kpis = SlotKPIs(**s["kpis"])
        slots.append(SlotRecord(
            model_id=s["model_id"],
            strategy=s["strategy"],
            prompt_id=s["prompt_id"],
            programs=programs,
            kpis=kpis,
        ))

    return RunRecord(
        timestamp=raw["timestamp"],
        scale=raw["scale"],
        git_commit=raw["git_commit"],
        slots=slots,
    )


def cmd_report(args: argparse.Namespace) -> None:
    run = _load_run(getattr(args, "run_json", None))

    out_dir = _RUNS_DIR
    md = report_md(run)
    md_path = out_dir / f"{run.timestamp}_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as f:
        f.write(md)

    print(report(run))
    print(f"[+] Report written to {md_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Merge LoRA adapter into a full-precision model.")
    p_prepare.add_argument("checkpoint_dir", help="Path to adapter checkpoint directory.")

    p_run = sub.add_parser("run", help="Run inference + validation for all discovered models.")
    p_run.add_argument("--models", nargs="+", metavar="id", help="Limit to specific model IDs.")
    p_run.add_argument("--scale", choices=["small", "full"], default="small",
                       help="small=50 programs/slot, full=200 programs/slot.")
    p_run.add_argument("--n", type=int, metavar="N", help="Override programs per slot (overrides --scale).")
    p_run.add_argument("--no-vm", action="store_true", help="Skip QEMU validation.")
    p_run.add_argument("--out", metavar="dir", help="Override output directory.")

    p_report = sub.add_parser("report", help="Print report for a run bundle.")
    p_report.add_argument("run_json", nargs="?", help="Path to run JSON (default: latest in benchmarks/runs/).")

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
