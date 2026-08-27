#!/usr/bin/env python3
"""Forward-parity gate: GPU decode forward vs an independent CPU reference.

Runs cpi --parity-check, which computes the full decode forward on the
GPU and again on the CPU (embedding -> N x [RMSNorm, QKV(+bias), RoPE, attention,
o-proj, residual, RMSNorm, SwiGLU, residual] -> final norm -> lm-head), then
compares: the argmax must agree and the max logit diff must be within tolerance.

This is the safety harness for forward-path refactors (kernel fusion): run it
before a change (record max_abs_diff and pass), make the change, run it after;
correctness is preserved iff it still PASSes with an unchanged max_abs_diff. A
byte-identical fusion leaves max_abs_diff untouched; a bug diverges sharply (a
missing QKV bias, for instance, took Qwen2.5's max_abs from 0.07 to 21).

Usage:
  python tools/forward_parity_check.py \
    --bin build/Release/cpi.exe \
    --model artifacts/hub/Qwen__Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.ll2c \
    --tokenizer artifacts/hub/Qwen__Qwen2.5-0.5B-Instruct/hf/tokenizer.json

Exit code 0 = pass, 1 = FAIL (mirrors the CLI), so it drops into a gate.

Verified baseline 2026-07-05 (5090, --gpu-cache-all): Qwen2.5-0.5B max_abs 0.066,
Llama-3.1-8B max_abs 0.012; both pass. The CPU reference handles the fused
attention.bqkv bias, so it is valid for Qwen2-family and Llama-family models.
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/Release/cpi.exe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    a = ap.parse_args()

    env = dict(os.environ, CPI_INSTANCE_MUTEX=f"Local\\cpi_parity_{os.getpid()}")
    cmd = [os.path.abspath(a.bin), a.model, "--tokenizer", a.tokenizer, "--prompt", a.prompt,
           "--parity-check", "--gpu-cache-all", "--no-resource-limits"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=400)
    for line in (p.stdout + p.stderr).splitlines():
        if "[parity]" in line:
            print(line)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
