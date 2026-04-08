#!/usr/bin/env python3
"""
Benchmark preset for Llama2 7B TurboQuant cached decode.

Records:
- decode_tok_per_s
- phase timings
- fallback reason (if any)
- tq3_cached_active flag
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
BENCH_RE = re.compile(r"^\[(bench|bench-avg)\]\s+(.*)$")
PHASE_RE = re.compile(r"^\[(bench-phase|bench-phase-avg)\]\s*(.*)$")
PERF_RE = re.compile(r"^\[(perf|perf-avg)\]\s+(.*)$")
FALLBACK_RE = re.compile(r"\[engine\].*(fallback|policy=|tq_cached|tq3_cached_active)", re.IGNORECASE)


def parse_kv(payload: str) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    for m in KV_RE.finditer(payload):
        k = m.group(1)
        v = m.group(2)
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def parse_stdout_metrics(stdout: str) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    fallback_lines: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        bm = BENCH_RE.match(line)
        if bm:
            metrics.update(parse_kv(bm.group(2)))
        pm = PHASE_RE.match(line)
        if pm:
            metrics.update(parse_kv(pm.group(2)))
        pf = PERF_RE.match(line)
        if pf:
            metrics.update(parse_kv(pf.group(2)))
        if FALLBACK_RE.search(line):
            fallback_lines.append(line)
    metrics["fallback_reason"] = " | ".join(fallback_lines[-3:]) if fallback_lines else ""
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--infer-bin", type=Path, default=Path("build/Release/llama_infer.exe"))
    ap.add_argument("--model", type=Path, default=Path(r"d:\models\llama2_tq3_fixed.ll2c"))
    ap.add_argument("--tokenizer", type=Path, default=Path(r"d:\models\Llama-2-7b-chat-hf\tokenizer.model"))
    ap.add_argument("--prompt", type=str, default="Write one short sentence about quantization.")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--benchmark-reps", type=int, default=3)
    ap.add_argument("--benchmark-warmup", type=int, default=1)
    ap.add_argument("--vram-safety-margin-mb", type=int, default=2048)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--json-out", type=Path, default=Path("artifacts/tq3_llama2_benchmark_latest.json"))
    ap.add_argument("--csv-out", type=Path, default=Path("artifacts/tq3_llama2_benchmark_latest.csv"))
    args = ap.parse_args()

    repo = args.repo.resolve()
    infer_bin = (repo / args.infer_bin).resolve() if not args.infer_bin.is_absolute() else args.infer_bin
    model = args.model.resolve()
    tokenizer = args.tokenizer.resolve()
    json_out = (repo / args.json_out).resolve() if not args.json_out.is_absolute() else args.json_out.resolve()
    csv_out = (repo / args.csv_out).resolve() if not args.csv_out.is_absolute() else args.csv_out.resolve()

    if not infer_bin.exists():
        print(f"[preset] missing infer binary: {infer_bin}")
        return 2
    if not model.exists():
        print(f"[preset] missing model: {model}")
        return 2
    if not tokenizer.exists():
        print(f"[preset] missing tokenizer: {tokenizer}")
        return 2

    cmd = [
        str(infer_bin),
        str(model),
        "--tokenizer", str(tokenizer),
        "--prompt", args.prompt,
        "--max-new", str(args.max_new),
        "--temp", "0",
        "--top-k", "1",
        "--top-p", "1",
        "--benchmark",
        "--benchmark-reps", str(args.benchmark_reps),
        "--benchmark-warmup", str(args.benchmark_warmup),
        "--benchmark-phases",
        "--gpu-cache-all",
        "--enable-tq-cached",
        "--tq-mode", "mse",
        "--vram-safety-margin-mb", str(args.vram_safety_margin_mb),
    ]

    print(f"[preset] $ {' '.join(cmd)}")
    started = time.time()
    cp = subprocess.run(
        cmd,
        cwd=str(repo),
        timeout=args.timeout_s,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started

    if cp.stdout:
        print(cp.stdout.rstrip())
    if cp.stderr:
        print(cp.stderr.rstrip())

    metrics = parse_stdout_metrics(cp.stdout)
    row = {
        "status": "pass" if cp.returncode == 0 else "fail",
        "returncode": cp.returncode,
        "elapsed_s": round(elapsed, 3),
        "model": str(model),
        "decode_tok_per_s": float(metrics.get("decode_tok_per_s", 0.0)),
        "tok_per_s": float(metrics.get("tok_per_s", 0.0)),
        "prefill_ms": float(metrics.get("prefill_ms", 0.0)),
        "decode_ms": float(metrics.get("decode_ms", 0.0)),
        "transfer_ms": float(metrics.get("transfer_ms", 0.0)),
        "rmsnorm_ms": float(metrics.get("rmsnorm_ms", 0.0)),
        "qkv_ms": float(metrics.get("qkv_ms", 0.0)),
        "kv_store_ms": float(metrics.get("kv_store_ms", 0.0)),
        "attention_ms": float(metrics.get("attention_ms", 0.0)),
        "wo_ms": float(metrics.get("wo_ms", 0.0)),
        "mlp_ms": float(metrics.get("mlp_ms", 0.0)),
        "lm_head_ms": float(metrics.get("lm_head_ms", 0.0)),
        "tq3_cached_active": int(metrics.get("tq3_cached_active", 0)),
        "fallback_reason": str(metrics.get("fallback_reason", "")),
        "target_decode_tok_per_s": 42.0,
        "target_pass": bool(float(metrics.get("decode_tok_per_s", 0.0)) > 42.0),
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(row, indent=2), encoding="utf-8")

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[preset] wrote {json_out}")
    print(f"[preset] wrote {csv_out}")
    return 0 if cp.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
