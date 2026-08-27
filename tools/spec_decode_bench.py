#!/usr/bin/env python3
"""Speculative-decoding bench (Track 2 harness-first baseline).

Runs the target model alone, then target+draft speculative decoding across a
sweep of --spec-tokens K, on the same prompt, and reports the speedup and
acceptance. Establishes the baseline before any spec-decode optimization, the
same harness-first discipline used for the decode-attention kernel work.

Usage:
  python tools/spec_decode_bench.py \
    --bin build/Release/cpi.exe \
    --target artifacts/hub/Qwen__Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct.ll2c \
    --draft  artifacts/hub/Qwen__Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.ll2c \
    --tokenizer artifacts/hub/Qwen__Qwen2.5-7B-Instruct/hf/tokenizer.json \
    [--k 3 4 5] [--max-new 200] [--prompt "..."]

Draft and target must share a tokenizer. Reads [perf] tok_per_s and [spec]
accept_rate/tokens_per_round from the CLI's stderr.

Measured baseline 2026-07-05 (Qwen2.5 7B target + 0.5B draft, 5090, code prompt):
  target alone      92.3 tok/s
  spec K=4  101 tok/s (1.10x)  accept 0.72  3.28 tok/round
  spec K=5  103 tok/s (1.11x)  accept 0.67  3.77 tok/round
Round cost attributed (K=5): ~19 ms draft + ~18 ms verify. Acceptance is already
good (3.6 tok/round would be ~3.6x if free); the wall is overhead, split two
ways with different causes:
  - Draft (5 x 3.75 ms): the 0.5B draft IS graphed (greedy_decode_graph enabled),
    so launches are amortized. 3.75 ms is genuine execution of ~900 tiny kernels
    (each a few us even under graph replay); ~5.6x off the ~0.67 ms weight
    roofline. Lever: reduce kernel count (forward-path fusion). Large effort.
  - Verify (~18 ms): run_batched_chunk (~37 kernel launches) is not graphed, so
    it pays full per-round launch overhead. Lever: capture it in a fixed-K CUDA
    graph (CPI already graphs single-token decode). Tractable, ~1.2x.
No single change reaches 2x: it needs verify-graphing (~1.2x, tractable) + draft
forward-fusion (large) + possibly tree speculation. Each modest, all real work.
"""
import argparse
import os
import re
import subprocess
import sys

PERF = re.compile(r"tok_per_s=([0-9.]+)")
SPEC = re.compile(r"accept_rate=([0-9.]+) tokens_per_round=([0-9.]+)")


def run(bin_path, args, mutex):
    env = dict(os.environ, CPI_INSTANCE_MUTEX=mutex)
    # Resolve the binary to an absolute path: a relative path is not reliably
    # found by subprocess on Windows.
    bin_path = os.path.abspath(bin_path)
    p = subprocess.run([bin_path, *args], env=env, capture_output=True, text=True, timeout=400)
    out = p.stdout + p.stderr
    tps = PERF.findall(out)
    spec = SPEC.findall(out)
    return (float(tps[-1]) if tps else None, spec[-1] if spec else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/Release/cpi.exe")
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--prompt", default="Write a Python function that computes the nth Fibonacci "
                                        "number using memoization, with a docstring.")
    a = ap.parse_args()
    mutex = f"Local\\cpi_specbench_{os.getpid()}"
    common = ["--tokenizer", a.tokenizer, "--prompt", a.prompt, "--max-new", str(a.max_new),
              "--temp", "0", "--gpu-cache-all", "--no-resource-limits"]

    base_tps, _ = run(a.bin, [a.target, *common], mutex)
    if not base_tps:
        print("baseline produced no tok/s: check the binary/model paths", file=sys.stderr)
        sys.exit(1)
    print(f"{'config':16} {'tok/s':>8} {'speedup':>8} {'accept':>7} {'tok/round':>9}")
    print("-" * 54)
    print(f"{'target alone':16} {base_tps:8.1f} {'1.00x':>8} {'-':>7} {'-':>9}")
    for k in a.k:
        tps, spec = run(a.bin, [a.target, "--draft-model", a.draft, "--spec-tokens", str(k),
                                *common], mutex)
        acc, tpr = (spec if spec else ("-", "-"))
        sp = f"{tps / base_tps:.2f}x" if tps else "n/a"
        acc_s = f"{float(acc):.2f}" if acc != "-" else "-"
        tpr_s = f"{float(tpr):.2f}" if tpr != "-" else "-"
        print(f"{'spec K=' + str(k):16} {(tps or 0):8.1f} {sp:>8} {acc_s:>7} {tpr_s:>9}")


if __name__ == "__main__":
    main()
