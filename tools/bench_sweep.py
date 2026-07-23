#!/usr/bin/env python3
"""
CPI throughput / latency / memory sweep.

Runs cpi across a configurable matrix of:
  - context lengths  (e.g. 128, 512, 2048, 4096)
  - quantization modes  (fp16, int8, int4)
  - compute paths  (CUDA, CPU)

For each configuration the script records:
  - decode_tok_per_s, total_tok_per_s
  - prefill_ms, decode_ms, and per-phase timings
  - peak RAM (RSS via psutil, polled in a background thread)
  - peak VRAM (nvidia-smi, polled in a background thread)

Outputs:
  docs/results/sweep_<model>_<timestamp>.json  – machine-readable
  docs/results/sweep_<model>_<timestamp>.md    – Markdown table

Usage:
  # Minimal: fp16 CUDA across four context sizes
  python tools/bench_sweep.py --model artifacts/mymodel.ll2c --tokenizer artifacts/mymodel/hf/tokenizer.json

  # Full matrix (all quant modes, CPU + CUDA)
  python tools/bench_sweep.py \\
      --model artifacts/mymodel.ll2c \\
      --tokenizer artifacts/mymodel/hf/tokenizer.json \\
      --context-lengths 128 512 2048 4096 \\
      --quant-modes fp16 int8 int4 \\
      --include-cpu

  # CPU-only machine
  python tools/bench_sweep.py \\
      --model artifacts/mymodel.ll2c \\
      --tokenizer artifacts/mymodel/hf/tokenizer.json \\
      --force-cpu-only
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"


def _rel(p) -> str:
    """Repo-root-relative path (forward slashes) for portable, non-leaky result
    metadata — avoids baking an absolute C:\\Users\\<name>\\... path into committed
    JSON. Falls back to the absolute string if the path is outside the repo."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)

BENCH_RE = re.compile(r"^\[bench(?:-avg)?\]\s+(.*)$")
BENCH_PHASE_RE = re.compile(r"^\[bench-phase(?:-avg)?\]\s+(.*)$")
PERF_RE = re.compile(r"^\[perf(?:-avg)?\]\s+(.*)$")


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query_vram_mb() -> Optional[float]:
    """Return total VRAM used (MB) across all GPUs, or None on failure."""
    if not _has_nvidia_smi():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5
        )
        total = sum(float(x.strip()) for x in out.strip().splitlines() if x.strip())
        return total
    except Exception:
        return None


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "none"


def _cpu_info() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name", "/value"], text=True, timeout=5
            )
            for line in out.splitlines():
                if line.lower().startswith("name="):
                    return line.split("=", 1)[1].strip()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_total_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_kv(blob: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for token in blob.strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _parse_output(stdout: str) -> dict[str, Any]:
    bench: dict[str, float] = {}
    phases: dict[str, float] = {}
    perf: dict[str, float] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        m = BENCH_RE.match(stripped)
        if m:
            bench.update(_parse_kv(m.group(1)))
        m2 = BENCH_PHASE_RE.match(stripped)
        if m2:
            phases.update(_parse_kv(m2.group(1)))
        m3 = PERF_RE.match(stripped)
        if m3:
            perf.update(_parse_kv(m3.group(1)))
    return {"bench": bench, "phases": phases, "perf": perf}


# ---------------------------------------------------------------------------
# Memory tracking (background threads)
# ---------------------------------------------------------------------------

class _MemoryTracker:
    """
    Polls peak RSS and VRAM while a subprocess runs.
    Call .start(pid) before the process launches, .stop() after it finishes.
    """

    def __init__(self, poll_ms: int = 200):
        self._poll_s = max(0.05, poll_ms / 1000.0)
        self._pid: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.peak_rss_mb: float = 0.0
        self.peak_vram_mb: float = 0.0
        self._baseline_vram: float = _query_vram_mb() or 0.0

    def start(self, pid: int) -> None:
        self._pid = pid
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._poll_s)

    def _sample(self) -> None:
        # RSS
        try:
            import psutil
            proc = psutil.Process(self._pid)
            # Include child processes (the engine may spawn workers)
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            rss_mb = rss / 1024 / 1024
            if rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = rss_mb
        except Exception:
            pass

        # VRAM
        vram = _query_vram_mb()
        if vram is not None:
            delta = max(0.0, vram - self._baseline_vram)
            if delta > self.peak_vram_mb:
                self.peak_vram_mb = delta


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def _run_one(
    infer_bin: Path,
    model: Path,
    tokenizer: Path,
    context_length: int,
    quant_mode: str,
    force_cpu: bool,
    prompt: str,
    max_new: int,
    benchmark_reps: int,
    benchmark_warmup: int,
    chat_template: str,
    timeout_s: int,
    no_resource_limits: bool = False,
) -> dict[str, Any]:
    cmd = [
        str(infer_bin),
        str(model),
        "--tokenizer", str(tokenizer),
        "--prompt", prompt,
        "--max-new", str(max_new),
        "--max-context", str(context_length),
        "--temp", "0",
        "--top-k", "1",
        "--benchmark",
        "--benchmark-reps", str(benchmark_reps),
        "--benchmark-warmup", str(benchmark_warmup),
        "--benchmark-phases",
        "--runtime-metrics",
    ]
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    if quant_mode != "fp16":
        cmd.extend(["--weight-quant", quant_mode])
    if force_cpu:
        cmd.extend(["--cpu"])
    if no_resource_limits:
        cmd.extend(["--no-resource-limits"])

    tracker = _MemoryTracker(poll_ms=150)
    t0 = time.perf_counter()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO_ROOT),
        )
        tracker.start(proc.pid)
        stdout, _ = proc.communicate(timeout=timeout_s)
        elapsed_s = time.perf_counter() - t0
        tracker.stop()
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        tracker.stop()
        return {
            "status": "timeout",
            "context_length": context_length,
            "quant_mode": quant_mode,
            "force_cpu": force_cpu,
        }
    except Exception as exc:
        tracker.stop()
        return {
            "status": "error",
            "error": str(exc),
            "context_length": context_length,
            "quant_mode": quant_mode,
            "force_cpu": force_cpu,
        }

    if proc.returncode != 0:
        return {
            "status": "fail",
            "returncode": proc.returncode,
            "context_length": context_length,
            "quant_mode": quant_mode,
            "force_cpu": force_cpu,
            "stdout_tail": stdout[-800:] if stdout else "",
        }

    parsed = _parse_output(stdout)
    bench = parsed["bench"]
    phases = parsed["phases"]

    result: dict[str, Any] = {
        "status": "ok",
        "context_length": context_length,
        "quant_mode": quant_mode,
        "force_cpu": force_cpu,
        "elapsed_s": round(elapsed_s, 3),
        # Core throughput
        "decode_tok_per_s": bench.get("decode_tok_per_s", math.nan),
        "total_tok_per_s": bench.get("total_tok_per_s", math.nan),
        # Latency
        "prefill_ms": bench.get("prefill_ms", math.nan),
        "decode_ms": bench.get("decode_ms", math.nan),
        # Phase breakdown
        "phases": {k: v for k, v in phases.items()},
        # Memory
        "peak_rss_mb": round(tracker.peak_rss_mb, 1) if tracker.peak_rss_mb > 0 else None,
        "peak_vram_mb": round(tracker.peak_vram_mb, 1) if tracker.peak_vram_mb > 0 else None,
    }
    return result


# ---------------------------------------------------------------------------
# Markdown table renderer
# ---------------------------------------------------------------------------

def _render_table(runs: list[dict], model_name: str) -> str:
    lines: list[str] = [
        f"## Throughput & Memory — {model_name}",
        "",
        "| Path | Context | Quant | Decode tok/s | Prefill ms | Decode ms | Peak RAM MB | Peak VRAM MB |",
        "|------|--------:|------:|-------------:|-----------:|----------:|------------:|-------------:|",
    ]
    for r in runs:
        if r.get("status") != "ok":
            path_label = "CPU" if r.get("force_cpu") else "CUDA"
            lines.append(
                f"| {path_label} | {r.get('context_length', '?'):,} | {r.get('quant_mode', '?')} "
                f"| — | — | — | — | — | _{r.get('status', 'error')}_ |"
            )
            continue
        path_label = "CPU" if r["force_cpu"] else "CUDA"
        dtps = r["decode_tok_per_s"]
        dtps_s = f"{dtps:.2f}" if math.isfinite(dtps) else "—"
        pfms = r["prefill_ms"]
        pfms_s = f"{pfms:.1f}" if math.isfinite(pfms) else "—"
        dcms = r["decode_ms"]
        dcms_s = f"{dcms:.1f}" if math.isfinite(dcms) else "—"
        ram = f"{r['peak_rss_mb']:.0f}" if r.get("peak_rss_mb") is not None else "—"
        vram = f"{r['peak_vram_mb']:.0f}" if r.get("peak_vram_mb") is not None else "—"
        lines.append(
            f"| {path_label} | {r['context_length']:,} | {r['quant_mode']} "
            f"| {dtps_s} | {pfms_s} | {dcms_s} | {ram} | {vram} |"
        )
    return "\n".join(lines)


def _render_speedup_table(runs: list[dict], model_name: str) -> str:
    """Compute int8 and int4 speedup over fp16 baseline for each (path, context) pair."""
    grouped: dict[tuple, dict] = {}
    for r in runs:
        if r.get("status") != "ok":
            continue
        key = (r["force_cpu"], r["context_length"])
        qm = r["quant_mode"]
        if key not in grouped:
            grouped[key] = {}
        grouped[key][qm] = r["decode_tok_per_s"]

    if not grouped:
        return ""

    lines = [
        f"## Quantization Speedup — {model_name}",
        "",
        "Speedup = quant decode tok/s ÷ fp16 decode tok/s for same path + context.",
        "",
        "| Path | Context | fp16 tok/s | int8 tok/s | int8 ×speedup | int4 tok/s | int4 ×speedup |",
        "|------|--------:|-----------:|-----------:|--------------:|-----------:|--------------:|",
    ]
    for (force_cpu, ctx), qmap in sorted(grouped.items()):
        path_label = "CPU" if force_cpu else "CUDA"
        fp16 = qmap.get("fp16", math.nan)
        int8 = qmap.get("int8", math.nan)
        int4 = qmap.get("int4", math.nan)

        def fmt_tok(v: float) -> str:
            return f"{v:.2f}" if math.isfinite(v) else "—"

        def fmt_speedup(v: float, base: float) -> str:
            if math.isfinite(v) and math.isfinite(base) and base > 0:
                return f"**×{v/base:.2f}**"
            return "—"

        lines.append(
            f"| {path_label} | {ctx:,} | {fmt_tok(fp16)} | {fmt_tok(int8)} | {fmt_speedup(int8, fp16)} "
            f"| {fmt_tok(int4)} | {fmt_speedup(int4, fp16)} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CPI throughput/latency/memory sweep.")
    ap.add_argument("--model", required=True, help=".ll2c model path")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer path (tokenizer.json or .model)")
    ap.add_argument("--infer-bin", default=None,
                    help="Path to cpi binary (auto-detected if omitted)")
    ap.add_argument("--context-lengths", nargs="+", type=int,
                    default=[128, 512, 2048, 4096],
                    help="Context lengths to sweep (default: 128 512 2048 4096)")
    ap.add_argument("--quant-modes", nargs="+",
                    choices=["fp16", "int8", "int4"],
                    default=["fp16"],
                    help="Quantization modes (default: fp16)")
    ap.add_argument("--include-cpu", action="store_true",
                    help="Also run CPU inference for each configuration")
    ap.add_argument("--force-cpu-only", action="store_true",
                    help="Run only CPU inference (skip CUDA)")
    ap.add_argument("--prompt", default="Explain what a neural network is in one sentence.",
                    help="Prompt text for benchmark runs")
    ap.add_argument("--max-new", type=int, default=64,
                    help="Max new tokens per run (default: 64)")
    ap.add_argument("--benchmark-reps", type=int, default=3,
                    help="Benchmark repetitions (default: 3)")
    ap.add_argument("--benchmark-warmup", type=int, default=1,
                    help="Benchmark warmup reps (default: 1)")
    ap.add_argument("--chat-template", default="",
                    help="Chat template name (llama2, llama4, mistral, qwen3_5, …)")
    ap.add_argument("--no-resource-limits", action="store_true",
                    help="Pass --no-resource-limits to cpi (bypass CPU/RAM throttle guards)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-run timeout in seconds (default: 600)")
    ap.add_argument("--model-name", default=None,
                    help="Human-readable model name (default: file stem)")
    ap.add_argument("--out-json", default=None,
                    help="Custom JSON output path")
    ap.add_argument("--out-md", default=None,
                    help="Custom Markdown output path")
    args = ap.parse_args()

    model = Path(args.model).resolve()
    tokenizer = Path(args.tokenizer).resolve()
    if not model.exists():
        print(f"[sweep] model not found: {model}", file=sys.stderr)
        return 2
    if not tokenizer.exists():
        print(f"[sweep] tokenizer not found: {tokenizer}", file=sys.stderr)
        return 2

    # Auto-detect infer binary
    if args.infer_bin:
        infer_bin = Path(args.infer_bin).resolve()
    else:
        candidates = [
            REPO_ROOT / "build" / "Release" / "cpi.exe",
            REPO_ROOT / "build" / "cpi",
            REPO_ROOT / "build" / "Release" / "cpi",
            REPO_ROOT / "build" / "cpu-release" / "cpi",
        ]
        infer_bin = next((c for c in candidates if c.exists()), None)
        if infer_bin is None:
            print("[sweep] could not find cpi binary. "
                  "Build first or pass --infer-bin.", file=sys.stderr)
            return 2

    model_name = args.model_name or model.stem

    # Determine compute paths
    cuda_paths = not args.force_cpu_only
    cpu_paths = args.include_cpu or args.force_cpu_only
    if not cuda_paths and not cpu_paths:
        cpu_paths = True

    configs: list[tuple[int, str, bool]] = []
    for ctx in sorted(set(args.context_lengths)):
        for qm in args.quant_modes:
            if cuda_paths:
                configs.append((ctx, qm, False))
            if cpu_paths:
                configs.append((ctx, qm, True))

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    print(f"[sweep] model: {model_name}")
    print(f"[sweep] binary: {infer_bin}")
    print(f"[sweep] configurations: {len(configs)}")
    print(f"[sweep] GPU: {_gpu_name()}")
    print(f"[sweep] CPU: {_cpu_info()}")
    print()

    runs: list[dict] = []
    for i, (ctx, qm, force_cpu) in enumerate(configs, 1):
        path_label = "CPU" if force_cpu else "CUDA"
        print(f"[sweep] [{i}/{len(configs)}] ctx={ctx} quant={qm} path={path_label} ...",
              end=" ", flush=True)
        r = _run_one(
            infer_bin=infer_bin,
            model=model,
            tokenizer=tokenizer,
            context_length=ctx,
            quant_mode=qm,
            force_cpu=force_cpu,
            prompt=args.prompt,
            max_new=args.max_new,
            benchmark_reps=args.benchmark_reps,
            benchmark_warmup=args.benchmark_warmup,
            chat_template=args.chat_template,
            timeout_s=args.timeout,
            no_resource_limits=args.no_resource_limits,
        )
        runs.append(r)
        if r["status"] == "ok":
            dtps = r["decode_tok_per_s"]
            dtps_s = f"{dtps:.2f} tok/s" if math.isfinite(dtps) else "? tok/s"
            ram_s = f"  ram={r['peak_rss_mb']:.0f}MB" if r.get("peak_rss_mb") else ""
            vram_s = f"  vram={r['peak_vram_mb']:.0f}MB" if r.get("peak_vram_mb") else ""
            print(f"decode={dtps_s}{ram_s}{vram_s}")
        else:
            print(f"FAILED ({r['status']})")

    # Build output document
    report = {
        "schema": "cpi-sweep-v1",
        "timestamp": timestamp,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": _cpu_info(),
        "gpu": _gpu_name(),
        "ram_total_mb": round(_ram_total_mb(), 0),
        "model_name": model_name,
        "model_path": _rel(model),
        "tokenizer_path": _rel(tokenizer),
        "infer_bin": _rel(infer_bin),
        "prompt": args.prompt,
        "max_new": args.max_new,
        "benchmark_reps": args.benchmark_reps,
        "benchmark_warmup": args.benchmark_warmup,
        "runs": runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.out_json) if args.out_json else RESULTS_DIR / f"sweep_{model_name}_{timestamp}.json"
    md_path = Path(args.out_md) if args.out_md else RESULTS_DIR / f"sweep_{model_name}_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"\n[sweep] wrote {json_path}")

    # Markdown report
    md_parts = [
        f"# CPI Benchmark Sweep — {model_name}",
        "",
        f"**Date:** {timestamp}  ",
        f"**Host:** {socket.gethostname()}  ",
        f"**GPU:** {_gpu_name()}  ",
        f"**CPU:** {_cpu_info()}  ",
        f"**Prompt:** _{args.prompt}_  ",
        f"**max-new:** {args.max_new}  reps={args.benchmark_reps}  warmup={args.benchmark_warmup}",
        "",
        _render_table(runs, model_name),
        "",
        _render_speedup_table(runs, model_name),
    ]
    md_path.write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    print(f"[sweep] wrote {md_path}")

    ok = sum(1 for r in runs if r.get("status") == "ok")
    print(f"\n[sweep] {ok}/{len(runs)} runs succeeded.")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
