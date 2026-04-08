#!/usr/bin/env python3
"""
TurboQuant smoke/perf matrix runner.

Modes:
  - smoke: quick convert + validate + short inference checks
  - perf-sweep: benchmark matrix and emit JSON/CSV summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


BENCH_RE = re.compile(r"^\[(bench|bench-avg)\]\s+(.*)$")
PERF_RE = re.compile(r"^\[(perf|perf-avg)\]\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
FALLBACK_RE = re.compile(r"\[engine\].*(fallback|guardrail|policy=|tq_cache)", re.IGNORECASE)


def run_cmd(cmd: list[str], timeout_s: int, cwd: Path) -> subprocess.CompletedProcess[str]:
    print(f"[runner] $ {' '.join(cmd)}")
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout_s,
        text=True,
        capture_output=True,
    )
    if cp.stdout:
        print("[runner] stdout:")
        print(cp.stdout.rstrip())
    if cp.stderr:
        print("[runner] stderr:")
        print(cp.stderr.rstrip())
    return cp


def parse_keyvals(payload: str) -> dict[str, float | int | str]:
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


def parse_infer_metrics(stdout: str) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    fallback_lines: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        bm = BENCH_RE.match(line)
        if bm:
            metrics.update(parse_keyvals(bm.group(2)))
        pm = PERF_RE.match(line)
        if pm:
            metrics.update(parse_keyvals(pm.group(2)))
        if FALLBACK_RE.search(line):
            fallback_lines.append(line)
    metrics["fallback_reason"] = " | ".join(fallback_lines[-2:]) if fallback_lines else ""
    return metrics


def normalize_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_infer_cmd(
    infer_bin: Path,
    model: Path,
    tokenizer: Path,
    max_new: int,
    benchmark_reps: int,
    benchmark_warmup: int,
    cache_mode: str,
    enable_tq_cached: bool,
    int8_streaming: bool,
) -> list[str]:
    cmd = [
        str(infer_bin),
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--prompt",
        "Write one short sentence about quantization.",
        "--max-new",
        str(max_new),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--benchmark",
        "--benchmark-reps",
        str(benchmark_reps),
        "--benchmark-warmup",
        str(benchmark_warmup),
        "--vram-safety-margin-mb",
        "2048",
    ]
    if int8_streaming:
        cmd += ["--int8-streaming"]
    if cache_mode == "uncached":
        cmd += ["--gpu-cache-layers", "0"]
    elif cache_mode == "cached":
        cmd += ["--gpu-cache-all"]
        if enable_tq_cached:
            cmd += ["--enable-tq-cached"]
    else:
        raise ValueError(f"unknown cache mode: {cache_mode}")
    return cmd


def write_reports(results: list[dict], json_out: Path | None, csv_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[runner] wrote JSON: {json_out}")

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "model_tag",
            "objective",
            "cache_mode",
            "status",
            "tok_per_s",
            "decode_tok_per_s",
            "prefill_ms",
            "decode_ms",
            "transfer_ms",
            "fallback_reason",
            "returncode",
        ]
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, "") for k in keys})
        print(f"[runner] wrote CSV: {csv_out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--infer-bin", type=Path, default=Path("build/Release/llama_infer.exe"))
    ap.add_argument("--input-model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--mode", choices=["smoke", "perf-sweep"], default="smoke")
    ap.add_argument("--objectives", type=str, default="mse,prod")
    ap.add_argument("--cache-modes", type=str, default="uncached,cached")
    ap.add_argument("--include-fp16", action="store_true")
    ap.add_argument("--qjl-dim", type=int, default=256)
    ap.add_argument("--qjl-seed", type=int, default=17)
    ap.add_argument("--chunk-rows", type=int, default=64)
    ap.add_argument("--max-ram-gb", type=float, default=0.0)
    ap.add_argument("--convert-timeout-s", type=int, default=7200)
    ap.add_argument("--infer-timeout-s", type=int, default=300)
    ap.add_argument("--max-new-smoke", type=int, default=8)
    ap.add_argument("--max-new-perf", type=int, default=128)
    ap.add_argument("--benchmark-reps", type=int, default=3)
    ap.add_argument("--benchmark-warmup", type=int, default=1)
    ap.add_argument("--enable-tq-cached", action="store_true")
    ap.add_argument("--int8-streaming", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    infer_bin = (repo / args.infer_bin).resolve() if not args.infer_bin.is_absolute() else args.infer_bin
    input_model = args.input_model.resolve()
    tokenizer = args.tokenizer.resolve()

    if not infer_bin.exists():
        print(f"[runner] FAIL: infer binary not found: {infer_bin}")
        return 2
    if not input_model.exists():
        print(f"[runner] FAIL: input model not found: {input_model}")
        return 2
    if not tokenizer.exists():
        print(f"[runner] FAIL: tokenizer not found: {tokenizer}")
        return 2

    objectives = normalize_csv_list(args.objectives)
    cache_modes = normalize_csv_list(args.cache_modes)
    if args.mode == "smoke":
        objectives = objectives[:1] if objectives else ["mse"]
        cache_modes = cache_modes[:1] if cache_modes else ["uncached"]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.json_out is None:
        args.json_out = repo / "artifacts" / f"tq_matrix_{stamp}.json"
    if args.csv_out is None:
        args.csv_out = repo / "artifacts" / f"tq_matrix_{stamp}.csv"

    out_dir = Path(tempfile.mkdtemp(prefix="tq_matrix_", dir=str(repo / "artifacts")))
    converted: dict[str, Path] = {}
    results: list[dict] = []

    try:
        # Convert and validate each requested objective once.
        for obj in objectives:
            out_model = out_dir / f"model_tq_{obj}.ll2c"
            convert_cmd = [
                sys.executable,
                "tools/turbo_quant_convert.py",
                str(input_model),
                str(out_model),
                "--chunk-rows",
                str(args.chunk_rows),
                "--objective",
                obj,
            ]
            if obj == "prod":
                convert_cmd += ["--qjl-dim", str(args.qjl_dim), "--qjl-seed", str(args.qjl_seed)]
            if args.max_ram_gb > 0:
                convert_cmd += ["--max-ram-gb", str(args.max_ram_gb)]

            cp = run_cmd(convert_cmd, timeout_s=args.convert_timeout_s, cwd=repo)
            if cp.returncode != 0:
                print(f"[runner] FAIL: conversion failed for objective={obj} (exit {cp.returncode})")
                return 1

            cp = run_cmd([sys.executable, "tools/validate_ll2c.py", str(out_model)], timeout_s=180, cwd=repo)
            if cp.returncode != 0:
                print(f"[runner] FAIL: validate_ll2c failed for objective={obj} (exit {cp.returncode})")
                return 1
            converted[obj] = out_model

        matrix: list[tuple[str, str, Path, str]] = []
        if args.include_fp16:
            for cache_mode in cache_modes:
                matrix.append(("fp16", "fp16", input_model, cache_mode))
        for obj in objectives:
            for cache_mode in cache_modes:
                matrix.append((f"tq_{obj}", obj, converted[obj], cache_mode))

        max_new = args.max_new_smoke if args.mode == "smoke" else args.max_new_perf
        reps = 1 if args.mode == "smoke" else args.benchmark_reps
        warmup = 0 if args.mode == "smoke" else args.benchmark_warmup

        for model_tag, objective, model_path, cache_mode in matrix:
            infer_cmd = build_infer_cmd(
                infer_bin=infer_bin,
                model=model_path,
                tokenizer=tokenizer,
                max_new=max_new,
                benchmark_reps=reps,
                benchmark_warmup=warmup,
                cache_mode=cache_mode,
                enable_tq_cached=args.enable_tq_cached,
                int8_streaming=args.int8_streaming,
            )
            try:
                cp = run_cmd(infer_cmd, timeout_s=args.infer_timeout_s, cwd=repo)
                metrics = parse_infer_metrics(cp.stdout)
                row = {
                    "model_tag": model_tag,
                    "objective": objective,
                    "cache_mode": cache_mode,
                    "status": "pass" if cp.returncode == 0 else "fail",
                    "returncode": cp.returncode,
                    "tok_per_s": float(metrics.get("tok_per_s", 0.0)),
                    "decode_tok_per_s": float(metrics.get("decode_tok_per_s", 0.0)),
                    "prefill_ms": float(metrics.get("prefill_ms", 0.0)),
                    "decode_ms": float(metrics.get("decode_ms", 0.0)),
                    "transfer_ms": float(metrics.get("transfer_ms", 0.0)),
                    "fallback_reason": str(metrics.get("fallback_reason", "")),
                }
            except subprocess.TimeoutExpired:
                row = {
                    "model_tag": model_tag,
                    "objective": objective,
                    "cache_mode": cache_mode,
                    "status": "timeout",
                    "returncode": -1,
                    "tok_per_s": 0.0,
                    "decode_tok_per_s": 0.0,
                    "prefill_ms": 0.0,
                    "decode_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fallback_reason": f"inference timeout after {args.infer_timeout_s}s",
                }
            results.append(row)

        write_reports(results, args.json_out, args.csv_out)

        failures = [r for r in results if r["status"] != "pass"]
        if failures:
            print(f"[runner] FAIL: {len(failures)} matrix entries failed")
            return 1
        print(f"[runner] PASS: {len(results)} matrix entries completed")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
