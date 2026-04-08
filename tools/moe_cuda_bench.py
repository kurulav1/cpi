#!/usr/bin/env python3
"""
MoE CUDA benchmark + parity checker for CPI models.

Runs:
1) Perf benchmarks for fp16/int8/int4 model variants.
2) Top-k logits drift checks (fp16 baseline vs int8/int4).

Example:
  py tools/moe_cuda_bench.py ^
    --tokenizer artifacts/hub/hf-internal-testing__tiny-random-MixtralForCausalLM/hf/tokenizer.json ^
    --fp16-model artifacts/mixtral_fp16.ll2c ^
    --int8-model artifacts/mixtral_int8.ll2c ^
    --int4-model artifacts/mixtral_int4.ll2c
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


BENCH_RE = re.compile(r"^\[bench(?:-avg)?\]\s+(.*)$")
BENCH_PHASE_RE = re.compile(r"^\[bench-phase(?:-avg)?\]\s+(.*)$")
NEXT_TOP_RE = re.compile(r"^Next-token top-(\d+):\s*$")
TOP_LINE_RE = re.compile(
    r"^\s*id=(\d+)\s+logit=([+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf))",
    re.IGNORECASE,
)


@dataclass
class BenchResult:
    model: str
    bench: Dict[str, float]
    phases: Dict[str, float]


def parse_kv_pairs(blob: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in blob.strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def run_cmd(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def run_bench(
    infer_bin: Path,
    model_path: Path,
    tokenizer: Path,
    prompt: str,
    chat_template: str,
    quant_mode: str | None,
    max_new: int,
    max_context: int,
    reps: int,
    warmup: int,
    cwd: Path,
) -> BenchResult:
    cmd = [
        str(infer_bin),
        str(model_path),
        "--prompt",
        prompt,
        "--tokenizer",
        str(tokenizer),
        "--chat-template",
        chat_template,
        "--max-new",
        str(max_new),
        "--temp",
        "0",
        "--max-context",
        str(max_context),
        "--benchmark",
        "--benchmark-reps",
        str(reps),
        "--benchmark-warmup",
        str(warmup),
        "--benchmark-phases",
    ]
    if quant_mode:
        cmd.extend(["--weight-quant", quant_mode])
    output = run_cmd(cmd, cwd)
    bench: Dict[str, float] = {}
    phases: Dict[str, float] = {}
    for line in output.splitlines():
        m = BENCH_RE.match(line.strip())
        if m:
            bench = parse_kv_pairs(m.group(1))
        m2 = BENCH_PHASE_RE.match(line.strip())
        if m2:
            phases = parse_kv_pairs(m2.group(1))
    if not bench:
        raise RuntimeError(f"Failed to parse benchmark output for {model_path}\n{output}")
    return BenchResult(model=str(model_path), bench=bench, phases=phases)


def run_topk_logits(
    infer_bin: Path,
    model_path: Path,
    tokenizer: Path,
    prompt: str,
    chat_template: str,
    quant_mode: str | None,
    max_context: int,
    topk: int,
    cwd: Path,
) -> Dict[int, float]:
    cmd = [
        str(infer_bin),
        str(model_path),
        "--prompt",
        prompt,
        "--tokenizer",
        str(tokenizer),
        "--chat-template",
        chat_template,
        "--max-new",
        "1",
        "--temp",
        "0",
        "--max-context",
        str(max_context),
        "--inspect-next-topk",
        str(topk),
    ]
    if quant_mode:
        cmd.extend(["--weight-quant", quant_mode])
    output = run_cmd(cmd, cwd)
    found = False
    logits: Dict[int, float] = {}
    for line in output.splitlines():
        if NEXT_TOP_RE.match(line.strip()):
            found = True
            continue
        if not found:
            continue
        m = TOP_LINE_RE.match(line)
        if not m:
            continue
        tid = int(m.group(1))
        lv = float(m.group(2))
        logits[tid] = lv
    if not logits:
        raise RuntimeError(f"Failed to parse inspect-next-topk output for {model_path}\n{output}")
    return logits


def compare_logits(ref: Dict[int, float], other: Dict[int, float], overlap_k: int) -> Dict[str, float]:
    ref_sorted = sorted(ref.items(), key=lambda kv: kv[1], reverse=True)
    other_sorted = sorted(other.items(), key=lambda kv: kv[1], reverse=True)
    ref_top1 = ref_sorted[0][0]
    other_top1 = other_sorted[0][0]

    ref_topk = {tid for tid, _ in ref_sorted[:overlap_k]}
    other_topk = {tid for tid, _ in other_sorted[:overlap_k]}
    inter = ref_topk.intersection(other_topk)
    overlap = (len(inter) / float(max(1, overlap_k)))

    common_ids = sorted(set(ref.keys()).intersection(other.keys()))
    if not common_ids:
        return {
            "top1_match": 1.0 if ref_top1 == other_top1 else 0.0,
            "overlap_at_k": overlap,
            "mean_abs": math.inf,
            "max_abs": math.inf,
        }

    diffs = []
    for i in common_ids:
        a = ref[i]
        b = other[i]
        if not math.isfinite(a) or not math.isfinite(b):
            diffs.append(float("inf"))
        else:
            diffs.append(abs(a - b))
    mean_abs = sum(diffs) / float(len(diffs))
    max_abs = max(diffs)
    return {
        "top1_match": 1.0 if ref_top1 == other_top1 else 0.0,
        "overlap_at_k": overlap,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
    }


def print_bench(label: str, result: BenchResult) -> None:
    bench = result.bench
    phases = result.phases
    decode_tok_s = bench.get("decode_tok_per_s", float("nan"))
    total_tok_s = bench.get("total_tok_per_s", float("nan"))
    prefill_ms = bench.get("prefill_ms", float("nan"))
    decode_ms = bench.get("decode_ms", float("nan"))
    mlp_ms = phases.get("mlp_ms", float("nan"))
    moe_router_ms = phases.get("moe_router_ms", float("nan"))
    moe_expert_ms = phases.get("moe_expert_ms", float("nan"))
    moe_merge_ms = phases.get("moe_merge_ms", float("nan"))
    print(
        f"[moe-bench] {label} decode_tok_per_s={decode_tok_s:.2f} total_tok_per_s={total_tok_s:.2f} "
        f"prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} mlp_ms={mlp_ms:.2f} "
        f"moe_router_ms={moe_router_ms:.2f} moe_expert_ms={moe_expert_ms:.2f} moe_merge_ms={moe_merge_ms:.2f}"
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Run MoE CUDA perf + parity checks for fp16/int8/int4 model variants.")
    ap.add_argument("--llama-infer", type=Path, default=repo_root / "build" / "Release" / "llama_infer.exe")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--fp16-model", type=Path, required=True)
    ap.add_argument("--int8-model", type=Path)
    ap.add_argument("--int4-model", type=Path)
    ap.add_argument("--prompt", default="Write a short sentence about CUDA kernels.")
    ap.add_argument("--chat-template", default="mistral")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--benchmark-reps", type=int, default=3)
    ap.add_argument("--benchmark-warmup", type=int, default=1)
    ap.add_argument("--drift-topk", type=int, default=256)
    ap.add_argument("--overlap-k", type=int, default=50)
    ap.add_argument("--int8-max-mean-abs", type=float, default=1.0)
    ap.add_argument("--int4-max-mean-abs", type=float, default=2.5)
    ap.add_argument("--int8-min-overlap", type=float, default=0.50)
    ap.add_argument("--int4-min-overlap", type=float, default=0.35)
    ap.add_argument("--json-out", type=Path, help="Write machine-readable benchmark/parity summary JSON.")
    args = ap.parse_args()

    infer_bin = args.llama_infer.resolve()
    if not infer_bin.exists():
        raise FileNotFoundError(f"llama_infer not found: {infer_bin}")

    tokenizer = args.tokenizer.resolve()
    fp16_model = args.fp16_model.resolve()
    int8_model = args.int8_model.resolve() if args.int8_model else None
    int4_model = args.int4_model.resolve() if args.int4_model else None

    for p in [tokenizer, fp16_model]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")
    for p in [int8_model, int4_model]:
        if p and not p.exists():
            raise FileNotFoundError(f"Missing model path: {p}")

    bench_fp16 = run_bench(
        infer_bin,
        fp16_model,
        tokenizer,
        args.prompt,
        args.chat_template,
        None,
        args.max_new,
        args.max_context,
        args.benchmark_reps,
        args.benchmark_warmup,
        repo_root,
    )
    print_bench("fp16", bench_fp16)
    summary = {
        "models": {
            "fp16": {"path": str(fp16_model), "bench": bench_fp16.bench, "phases": bench_fp16.phases},
            "int8": None,
            "int4": None,
        },
        "parity": {"int8": None, "int4": None},
        "thresholds": {
            "int8_max_mean_abs": args.int8_max_mean_abs,
            "int4_max_mean_abs": args.int4_max_mean_abs,
            "int8_min_overlap": args.int8_min_overlap,
            "int4_min_overlap": args.int4_min_overlap,
            "overlap_k": args.overlap_k,
            "drift_topk": args.drift_topk,
        },
    }

    bench_int8 = None
    if int8_model:
        bench_int8 = run_bench(
            infer_bin,
            int8_model,
            tokenizer,
            args.prompt,
            args.chat_template,
            "int8",
            args.max_new,
            args.max_context,
            args.benchmark_reps,
            args.benchmark_warmup,
            repo_root,
        )
        print_bench("int8", bench_int8)
        summary["models"]["int8"] = {
            "path": str(int8_model),
            "bench": bench_int8.bench,
            "phases": bench_int8.phases,
        }

    bench_int4 = None
    if int4_model:
        bench_int4 = run_bench(
            infer_bin,
            int4_model,
            tokenizer,
            args.prompt,
            args.chat_template,
            "int4",
            args.max_new,
            args.max_context,
            args.benchmark_reps,
            args.benchmark_warmup,
            repo_root,
        )
        print_bench("int4", bench_int4)
        summary["models"]["int4"] = {
            "path": str(int4_model),
            "bench": bench_int4.bench,
            "phases": bench_int4.phases,
        }

    ref_logits = run_topk_logits(
        infer_bin,
        fp16_model,
        tokenizer,
        args.prompt,
        args.chat_template,
        None,
        args.max_context,
        args.drift_topk,
        repo_root,
    )

    failed = False
    if int8_model:
        int8_logits = run_topk_logits(
            infer_bin,
            int8_model,
            tokenizer,
            args.prompt,
            args.chat_template,
            "int8",
            args.max_context,
            args.drift_topk,
            repo_root,
        )
        cmp = compare_logits(ref_logits, int8_logits, args.overlap_k)
        print(
            f"[moe-parity] int8 top1_match={int(cmp['top1_match'])} "
            f"overlap@{args.overlap_k}={cmp['overlap_at_k']:.3f} "
            f"mean_abs={cmp['mean_abs']:.4f} max_abs={cmp['max_abs']:.4f}"
        )
        summary["parity"]["int8"] = cmp
        if (not math.isfinite(cmp["mean_abs"]) or not math.isfinite(cmp["max_abs"]) or
                cmp["mean_abs"] > args.int8_max_mean_abs or cmp["overlap_at_k"] < args.int8_min_overlap):
            print("[moe-parity] int8 bounds FAILED")
            failed = True

    if int4_model:
        int4_logits = run_topk_logits(
            infer_bin,
            int4_model,
            tokenizer,
            args.prompt,
            args.chat_template,
            "int4",
            args.max_context,
            args.drift_topk,
            repo_root,
        )
        cmp = compare_logits(ref_logits, int4_logits, args.overlap_k)
        print(
            f"[moe-parity] int4 top1_match={int(cmp['top1_match'])} "
            f"overlap@{args.overlap_k}={cmp['overlap_at_k']:.3f} "
            f"mean_abs={cmp['mean_abs']:.4f} max_abs={cmp['max_abs']:.4f}"
        )
        summary["parity"]["int4"] = cmp
        if (not math.isfinite(cmp["mean_abs"]) or not math.isfinite(cmp["max_abs"]) or
                cmp["mean_abs"] > args.int4_max_mean_abs or cmp["overlap_at_k"] < args.int4_min_overlap):
            print("[moe-parity] int4 bounds FAILED")
            failed = True

    summary["failed"] = bool(failed)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
