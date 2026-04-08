#!/usr/bin/env python3
"""
CI gate for MoE CUDA regression checks.

This wrapper runs tools/moe_cuda_bench.py, then enforces additional
performance regression checks:
1) relative decode-throughput ratios between fp16/int8/int4
2) optional absolute decode-throughput floor vs baseline values
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_decode_tok_s(summary: dict[str, Any], mode: str) -> float | None:
    entry = summary.get("models", {}).get(mode)
    if not isinstance(entry, dict):
        return None
    bench = entry.get("bench", {})
    if not isinstance(bench, dict):
        return None
    value = bench.get("decode_tok_per_s")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f <= 0.0:
        return None
    return f


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Run MoE CUDA benchmark/parity gates for CI.")
    ap.add_argument("--llama-infer", type=Path, default=repo_root / "build" / "Release" / "llama_infer.exe")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--fp16-model", type=Path, required=True)
    ap.add_argument("--int8-model", type=Path, required=True)
    ap.add_argument("--int4-model", type=Path, required=True)
    ap.add_argument("--chat-template", default="mistral")
    ap.add_argument("--prompt", default="Describe MoE routing briefly.")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--benchmark-reps", type=int, default=3)
    ap.add_argument("--benchmark-warmup", type=int, default=1)
    ap.add_argument("--drift-topk", type=int, default=256)
    ap.add_argument("--overlap-k", type=int, default=50)
    ap.add_argument("--thresholds", type=Path, default=repo_root / "tools" / "ci" / "moe_gate_thresholds.json")
    ap.add_argument("--result-json", type=Path, default=repo_root / "artifacts" / "ci_moe_gate_result.json")
    args = ap.parse_args()

    thresholds = load_json(args.thresholds.resolve())
    parity_cfg = thresholds.get("parity", {}) if isinstance(thresholds, dict) else {}
    perf_cfg = thresholds.get("perf", {}) if isinstance(thresholds, dict) else {}

    int8_max_mean_abs = float(parity_cfg.get("int8_max_mean_abs", 1.0))
    int4_max_mean_abs = float(parity_cfg.get("int4_max_mean_abs", 2.5))
    int8_min_overlap = float(parity_cfg.get("int8_min_overlap", 0.5))
    int4_min_overlap = float(parity_cfg.get("int4_min_overlap", 0.35))

    bench_script = repo_root / "tools" / "moe_cuda_bench.py"
    cmd = [
        sys.executable,
        str(bench_script),
        "--llama-infer",
        str(args.llama_infer.resolve()),
        "--tokenizer",
        str(args.tokenizer.resolve()),
        "--fp16-model",
        str(args.fp16_model.resolve()),
        "--int8-model",
        str(args.int8_model.resolve()),
        "--int4-model",
        str(args.int4_model.resolve()),
        "--chat-template",
        args.chat_template,
        "--prompt",
        args.prompt,
        "--max-new",
        str(args.max_new),
        "--max-context",
        str(args.max_context),
        "--benchmark-reps",
        str(args.benchmark_reps),
        "--benchmark-warmup",
        str(args.benchmark_warmup),
        "--drift-topk",
        str(args.drift_topk),
        "--overlap-k",
        str(args.overlap_k),
        "--int8-max-mean-abs",
        str(int8_max_mean_abs),
        "--int4-max-mean-abs",
        str(int4_max_mean_abs),
        "--int8-min-overlap",
        str(int8_min_overlap),
        "--int4-min-overlap",
        str(int4_min_overlap),
        "--json-out",
        str(args.result_json.resolve()),
    ]

    print("[moe-gate] running moe_cuda_bench.py...")
    proc = subprocess.run(cmd, cwd=str(repo_root))
    if proc.returncode != 0:
        print("[moe-gate] parity gate failed.")
        return proc.returncode

    summary = load_json(args.result_json.resolve())
    fp16 = get_decode_tok_s(summary, "fp16")
    int8 = get_decode_tok_s(summary, "int8")
    int4 = get_decode_tok_s(summary, "int4")

    failed = False
    min_ratios = perf_cfg.get("min_decode_ratio", {}) if isinstance(perf_cfg, dict) else {}
    ratio_int8_fp16 = float(min_ratios.get("int8_vs_fp16", 1.0))
    ratio_int4_fp16 = float(min_ratios.get("int4_vs_fp16", 1.0))
    ratio_int4_int8 = float(min_ratios.get("int4_vs_int8", 1.0))

    if fp16 and int8:
        actual = int8 / fp16
        print(f"[moe-gate] int8/fp16 decode ratio = {actual:.3f} (min {ratio_int8_fp16:.3f})")
        if actual < ratio_int8_fp16:
            print("[moe-gate] FAILED: int8 decode ratio below threshold.")
            failed = True
    if fp16 and int4:
        actual = int4 / fp16
        print(f"[moe-gate] int4/fp16 decode ratio = {actual:.3f} (min {ratio_int4_fp16:.3f})")
        if actual < ratio_int4_fp16:
            print("[moe-gate] FAILED: int4 decode ratio below threshold.")
            failed = True
    if int8 and int4:
        actual = int4 / int8
        print(f"[moe-gate] int4/int8 decode ratio = {actual:.3f} (min {ratio_int4_int8:.3f})")
        if actual < ratio_int4_int8:
            print("[moe-gate] FAILED: int4 decode ratio vs int8 below threshold.")
            failed = True

    baseline = perf_cfg.get("baseline_decode_tok_per_s", {}) if isinstance(perf_cfg, dict) else {}
    max_reg_pct = float(perf_cfg.get("max_decode_regression_pct_vs_baseline", 25.0))
    keep = max(0.0, 1.0 - max_reg_pct / 100.0)
    for mode, actual in [("fp16", fp16), ("int8", int8), ("int4", int4)]:
        if actual is None:
            continue
        if mode not in baseline:
            continue
        try:
            base = float(baseline[mode])
        except (TypeError, ValueError):
            continue
        if base <= 0.0:
            continue
        floor = base * keep
        print(f"[moe-gate] {mode} decode tok/s = {actual:.2f}, baseline = {base:.2f}, floor = {floor:.2f}")
        if actual < floor:
            print(f"[moe-gate] FAILED: {mode} decode throughput regressed beyond {max_reg_pct:.1f}%")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

