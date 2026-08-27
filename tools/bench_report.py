#!/usr/bin/env python3
"""
CPI benchmark report aggregator.

Reads sweep JSON files and/or perplexity JSON files from docs/results/ (or
explicit paths) and renders a consolidated Markdown report. Optionally updates
the "Latest Results" section of docs/benchmarks.md in place.

Usage:
  # Auto-discover all docs/results/*.json and print report
  python tools/bench_report.py

  # Explicit files
  python tools/bench_report.py docs/results/sweep_llama2_*.json docs/results/ppl_llama2_*.json

  # Write report to a file and patch docs/benchmarks.md
  python tools/bench_report.py --out docs/results/report_latest.md --patch-benchmarks
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import sys
from pathlib import Path

# Ensure stdout handles Unicode on Windows consoles (cp1252 → utf-8)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"
BENCHMARKS_MD = REPO_ROOT / "docs" / "benchmarks.md"

# Sentinel that marks the auto-generated block in benchmarks.md
_BLOCK_START = "<!-- bench-report:start -->"
_BLOCK_END = "<!-- bench-report:end -->"


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[report] warning: could not read {path}: {exc}", file=sys.stderr)
        return None


def _is_sweep(doc: dict) -> bool:
    return doc.get("schema") == "cpi-sweep-v1"


def _is_ppl(doc: dict) -> bool:
    return doc.get("schema") == "cpi-ppl-v1"


# ---------------------------------------------------------------------------
# Sweep report rendering
# ---------------------------------------------------------------------------

def _fmt_float(v: Any, decimals: int = 2, suffix: str = "") -> str:
    try:
        f = float(v)
        if not math.isfinite(f):
            return "; "
        return f"{f:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "; "


def _speedup(a: Any, b: Any) -> str:
    try:
        fa, fb = float(a), float(b)
        if math.isfinite(fa) and math.isfinite(fb) and fb > 0:
            ratio = fa / fb
            return f"**×{ratio:.2f}**"
    except (TypeError, ValueError):
        pass
    return "; "


def _render_sweep(doc: dict) -> str:
    model = doc.get("model_name", "unknown")
    ts = doc.get("timestamp", "")
    gpu = doc.get("gpu", "; ")
    cpu = doc.get("cpu", "; ")
    runs = [r for r in doc.get("runs", []) if r.get("status") == "ok"]

    if not runs:
        return f"### {model} sweep ;  no successful runs\n"

    lines = [
        f"### Sweep: {model}",
        "",
        f"Date: {ts} &nbsp;|&nbsp; GPU: {gpu} &nbsp;|&nbsp; CPU: {cpu}",
        "",
        "#### Throughput & Latency",
        "",
        "| Path | Context | Quant | Decode tok/s | Prefill ms | Decode ms | Peak RAM MB | Peak VRAM MB |",
        "|------|--------:|------:|-------------:|-----------:|----------:|------------:|-------------:|",
    ]
    for r in sorted(runs, key=lambda x: (x["force_cpu"], x["context_length"], x["quant_mode"])):
        path = "CPU" if r["force_cpu"] else "CUDA"
        lines.append(
            f"| {path} | {r['context_length']:,} | {r['quant_mode']} "
            f"| {_fmt_float(r.get('decode_tok_per_s'))} "
            f"| {_fmt_float(r.get('prefill_ms'), 1)} "
            f"| {_fmt_float(r.get('decode_ms'), 1)} "
            f"| {_fmt_float(r.get('peak_rss_mb'), 0)} "
            f"| {_fmt_float(r.get('peak_vram_mb'), 0)} |"
        )

    # Speedup table: group by (force_cpu, context_length)
    from collections import defaultdict
    groups: dict[tuple, dict] = defaultdict(dict)
    for r in runs:
        groups[(r["force_cpu"], r["context_length"])][r["quant_mode"]] = r.get("decode_tok_per_s", math.nan)

    speedup_rows = []
    for (force_cpu, ctx), qmap in sorted(groups.items()):
        fp16 = qmap.get("fp16", math.nan)
        int8 = qmap.get("int8", math.nan)
        int4 = qmap.get("int4", math.nan)
        if math.isnan(fp16):
            continue
        speedup_rows.append((force_cpu, ctx, fp16, int8, int4))

    if speedup_rows:
        lines += [
            "",
            "#### Quantization Speedup (vs fp16 baseline)",
            "",
            "| Path | Context | fp16 tok/s | int8 tok/s | int8 speedup | int4 tok/s | int4 speedup |",
            "|------|--------:|-----------:|-----------:|-------------:|-----------:|-------------:|",
        ]
        for force_cpu, ctx, fp16, int8, int4 in speedup_rows:
            path = "CPU" if force_cpu else "CUDA"
            lines.append(
                f"| {path} | {ctx:,} "
                f"| {_fmt_float(fp16)} "
                f"| {_fmt_float(int8)} | {_speedup(int8, fp16)} "
                f"| {_fmt_float(int4)} | {_speedup(int4, fp16)} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perplexity report rendering
# ---------------------------------------------------------------------------

def _render_ppl(doc: dict) -> str:
    model = doc.get("model_name", "unknown")
    ts = doc.get("timestamp", "")
    corpus = doc.get("corpus", "; ")
    stride = doc.get("stride", "; ")
    max_len = doc.get("max_length", "; ")
    gpu = doc.get("gpu", "; ")
    results = doc.get("results", {})

    if not results:
        return f"### {model} perplexity ;  no results\n"

    lines = [
        f"### Perplexity: {model}",
        "",
        f"Date: {ts} &nbsp;|&nbsp; GPU: {gpu}  ",
        f"Corpus: {corpus} &nbsp;|&nbsp; Stride: {stride} &nbsp;|&nbsp; Window: {max_len}",
        "",
        "| Mode | PPL ↓ | BPB ↓ | NLL ↓ | Tokens | Eval time |",
        "|------|------:|------:|------:|-------:|----------:|",
    ]
    for mode in ["fp16", "int8", "int4"]:
        r = results.get(mode)
        if r is None:
            continue
        if "error" in r:
            lines.append(f"| {mode} | ERROR | ;  | ;  | ;  | ;  |")
            continue
        ppl_delta = ""
        if mode != "fp16" and "fp16" in results and "ppl" in results["fp16"]:
            delta = r["ppl"] - results["fp16"]["ppl"]
            ppl_delta = f" (+{delta:.3f})" if delta > 0 else f" ({delta:.3f})"
        lines.append(
            f"| {mode} | {_fmt_float(r.get('ppl'), 4)}{ppl_delta} "
            f"| {_fmt_float(r.get('bpb'), 4)} "
            f"| {_fmt_float(r.get('loss'), 4)} "
            f"| {r.get('n_tokens', 0):,} "
            f"| {_fmt_float(r.get('eval_s'), 1)}s |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def _build_report(sweep_docs: list[dict], ppl_docs: list[dict]) -> str:
    parts = ["# CPI Research Benchmark Report", ""]

    if not sweep_docs and not ppl_docs:
        parts.append("_No benchmark results found in `docs/results/`. "
                     "Run `tools/bench_sweep.py` and/or `tools/perplexity.py` first._")
        return "\n".join(parts)

    if ppl_docs:
        parts += ["## Perplexity (WikiText-2)", ""]
        for doc in ppl_docs:
            parts += [_render_ppl(doc), ""]

    if sweep_docs:
        parts += ["## Throughput, Latency & Memory", ""]
        for doc in sweep_docs:
            parts += [_render_sweep(doc), ""]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Patch docs/benchmarks.md
# ---------------------------------------------------------------------------

def _patch_benchmarks_md(report_body: str) -> None:
    if not BENCHMARKS_MD.exists():
        print("[report] docs/benchmarks.md not found; skipping patch.", file=sys.stderr)
        return

    original = BENCHMARKS_MD.read_text(encoding="utf-8")
    start_idx = original.find(_BLOCK_START)
    end_idx = original.find(_BLOCK_END)

    block = f"{_BLOCK_START}\n\n{report_body}\n\n{_BLOCK_END}"

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        new_text = original[:start_idx] + block + original[end_idx + len(_BLOCK_END):]
    else:
        # Append if sentinels are missing
        new_text = original.rstrip() + f"\n\n{block}\n"

    BENCHMARKS_MD.write_text(new_text, encoding="utf-8")
    print(f"[report] patched {BENCHMARKS_MD}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate CPI benchmark JSON results into Markdown.")
    ap.add_argument("files", nargs="*",
                    help="JSON result files (default: auto-discover docs/results/*.json)")
    ap.add_argument("--out", default=None,
                    help="Write Markdown report to this file (default: stdout)")
    ap.add_argument("--patch-benchmarks", action="store_true",
                    help="Update the auto-generated block in docs/benchmarks.md")
    ap.add_argument("--latest", action="store_true",
                    help="Only use the most recent sweep and ppl file per model")
    args = ap.parse_args()

    # Collect JSON files
    if args.files:
        paths = [Path(f) for pattern in args.files for f in glob.glob(pattern)]
    else:
        paths = sorted(RESULTS_DIR.glob("*.json"))

    if not paths:
        print("[report] no JSON files found.", file=sys.stderr)

    sweep_docs: list[dict] = []
    ppl_docs: list[dict] = []

    for p in paths:
        doc = _load_json(p)
        if doc is None:
            continue
        if _is_sweep(doc):
            sweep_docs.append(doc)
        elif _is_ppl(doc):
            ppl_docs.append(doc)

    # Sort by timestamp (newest last for --latest)
    sweep_docs.sort(key=lambda d: d.get("timestamp", ""))
    ppl_docs.sort(key=lambda d: d.get("timestamp", ""))

    if args.latest:
        # Keep only newest per model
        def _dedupe(docs: list[dict]) -> list[dict]:
            seen: dict[str, dict] = {}
            for d in docs:
                seen[d.get("model_name", "_")] = d
            return list(seen.values())
        sweep_docs = _dedupe(sweep_docs)
        ppl_docs = _dedupe(ppl_docs)

    report = _build_report(sweep_docs, ppl_docs)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"[report] wrote {out}")
    else:
        print(report)

    if args.patch_benchmarks:
        _patch_benchmarks_md(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
