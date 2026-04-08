#!/usr/bin/env python3
"""
Production-size Mixtral validation runner.

Runs an end-to-end flow:
  download -> convert -> quantize -> warmup -> long-context benchmark -> sustained streaming
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import shutil
from pathlib import Path


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"[mixtral-prod] $ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    print(proc.stdout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def query_gpu_mem_mb() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Validate production-size Mixtral end-to-end.")
    ap.add_argument("--repo-id", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--output-dir", type=Path, default=repo_root / "artifacts" / "e2e_mixtral_prod")
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--llama-infer", type=Path, default=repo_root / "build" / "Release" / "llama_infer.exe")
    ap.add_argument("--chat-template", default="mistral")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--sustained-runs", type=int, default=5)
    ap.add_argument("--min-free-gb", type=float, default=80.0, help="Required free disk space before download.")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(args.output_dir))
    free_gb = usage.free / (1024.0 ** 3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"insufficient free disk space at {args.output_dir}: "
            f"{free_gb:.2f} GB available, {args.min_free_gb:.2f} GB required"
        )
    fp16_model = args.output_dir / f"{args.repo_id.split('/')[-1]}.ll2c"
    int8_model = args.output_dir / f"{args.repo_id.split('/')[-1]}-int8.ll2c"
    int4_model = args.output_dir / f"{args.repo_id.split('/')[-1]}-int4.ll2c"
    tokenizer = args.output_dir / "hf" / "tokenizer.model"

    if not args.skip_download:
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "hf_download.py"),
            "download",
            args.repo_id,
            "--family",
            "mixtral",
            "--output-dir",
            str(args.output_dir),
        ]
        if args.hf_token:
            cmd.extend(["--token", args.hf_token])
        run(cmd, repo_root)

    run(
        [
            sys.executable,
            str(repo_root / "tools" / "quantize_ll2c_streaming.py"),
            "--input",
            str(fp16_model),
            "--output",
            str(int8_model),
            "--mode",
            "int8",
            "--overwrite",
        ],
        repo_root,
    )
    run(
        [
            sys.executable,
            str(repo_root / "tools" / "quantize_ll2c_streaming.py"),
            "--input",
            str(fp16_model),
            "--output",
            str(int4_model),
            "--mode",
            "int4",
            "--overwrite",
        ],
        repo_root,
    )

    if not tokenizer.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer}")
    if not args.llama_infer.exists():
        raise FileNotFoundError(f"llama_infer not found: {args.llama_infer}")

    warmup_prompt = "warmup"
    long_prompt = " ".join(["Explain MoE routing in detail."] * 512)

    for mode, model, quant in [
        ("fp16", fp16_model, None),
        ("int8", int8_model, "int8"),
        ("int4", int4_model, "int4"),
    ]:
        cmd = [
            str(args.llama_infer),
            str(model),
            "--prompt",
            warmup_prompt,
            "--tokenizer",
            str(tokenizer),
            "--chat-template",
            args.chat_template,
            "--max-context",
            str(args.max_context),
            "--max-new",
            "1",
            "--temp",
            "0",
            "--top-k",
            "1",
            "--simple",
        ]
        if quant:
            cmd.extend(["--weight-quant", quant])
        run(cmd, repo_root)

        mem_before = query_gpu_mem_mb()
        bench = [
            str(args.llama_infer),
            str(model),
            "--prompt",
            long_prompt,
            "--tokenizer",
            str(tokenizer),
            "--chat-template",
            args.chat_template,
            "--max-context",
            str(args.max_context),
            "--max-new",
            str(args.max_new),
            "--benchmark",
            "--benchmark-reps",
            "1",
            "--benchmark-warmup",
            "0",
            "--benchmark-phases",
            "--simple",
        ]
        if quant:
            bench.extend(["--weight-quant", quant])
        run(bench, repo_root)
        mem_after = query_gpu_mem_mb()
        if mem_before and mem_after:
            print(
                f"[mixtral-prod] {mode} gpu_mem_before={mem_before[0]}MiB/{mem_before[1]}MiB "
                f"after={mem_after[0]}MiB/{mem_after[1]}MiB"
            )

    sustained = []
    for i in range(max(1, args.sustained_runs)):
        start = time.time()
        out = run(
            [
                str(args.llama_infer),
                str(int8_model),
                "--weight-quant",
                "int8",
                "--prompt",
                f"Run {i}: summarize expert routing briefly.",
                "--tokenizer",
                str(tokenizer),
                "--chat-template",
                args.chat_template,
                "--max-context",
                str(args.max_context),
                "--max-new",
                "32",
                "--simple",
            ],
            repo_root,
        )
        elapsed = (time.time() - start) * 1000.0
        sustained.append({"run": i, "elapsed_ms": elapsed, "ok": out.returncode == 0})

    report = {
        "repo_id": args.repo_id,
        "output_dir": str(args.output_dir),
        "sustained": sustained,
        "llama_infer": str(args.llama_infer),
    }
    report_path = args.output_dir / "production_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[mixtral-prod] report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
