#!/usr/bin/env python3
"""Per-kernel performance regression gate.

Runs CPI's per-kernel microbenchmarks (int4_gemv_bench, attention_decode_bench),
parses the achieved GB/s for every shape, and compares against a committed
baseline (tools/ci/kernel_perf_baseline.json). Fails if any kernel/shape
regresses by more than --tolerance (default 12%). This is the automatic
per-kernel perf regression check.

  python tools/ci/kernel_perf_gate.py --update      # (re)generate the baseline
  python tools/ci/kernel_perf_gate.py               # gate against the baseline

Notes:
- Run with the GPU otherwise idle (stop the web server) — a co-resident model
  adds contention/clock noise (~8-10%) that inflates false positives.
- The metric is GB/s (higher is better); the gate flags drops below baseline.
- Each shape is measured best-of-N (--repeat, default 3): the max over runs is
  the least clock/thermal/contention-perturbed estimate, so it is reproducible
  run-to-run. Single-shot benching swings 40%+ on the high-batch/long-context
  attention shapes — best-of-N is what makes the tolerance meaningful.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "kernel_perf_baseline.json"

# Per-bench regression tolerance (max fractional GB/s drop before failing).
#
# int4_gemv is rock-stable run-to-run on this box (never false-flagged across four
# independent best-of-3 windows), so it gets a tight, meaningful gate — it is also
# the load-bearing decode signal (weight-streaming GEMVs dominate decode time).
#
# attention_decode is different: its high-batch/long-context shapes have a ~40-45%
# run-to-run NOISE FLOOR here (one shape swung -32/-39/-41/-45% across windows that
# changed nothing). The cause is cross-window thermal/clock drift — best-of-N absorbs
# drift within a window but not the temperature difference between a baseline capture
# and a gate run minutes later. On a consumer WDDM GPU that floor is irreducible for a
# wall-clock-throttled microbench, so the attention gate is deliberately coarse: it
# only catches GROSS regressions (fallback kernel / wrong head_dim path / disabled
# coarsening — all 2-10x cliffs), not subtle ones. For a tight attention signal, run
# the microbench standalone on an idle, thermally-steady GPU and eyeball %-of-roofline.
DEFAULT_TOL = {"int4_gemv": 0.12, "attention_decode": 0.50}

# name  out=..  in=..   <ms> ms   <gbs> GB/s ...
INT4_RE = re.compile(r"^(?P<name>.+?)\s+out=(?P<out>\d+)\s+in=(?P<in>\d+)\s+"
                     r"[\d.]+ ms\s+(?P<gbs>[\d.]+) GB/s")
# model  B  ctx  ms/step  GB/s  %peak
ATTN_RE = re.compile(r"^(?P<model>\S+)\s+(?P<b>\d+)\s+(?P<ctx>\d+)\s+"
                     r"[\d.]+\s+(?P<gbs>[\d.]+)\s+\d+%")


def _bin(name: str) -> Path:
    for p in (REPO / "build" / "Release" / f"{name}.exe", REPO / "build" / name):
        if p.exists():
            return p
    sys.exit(f"[kernel_perf_gate] bench binary not found: {name} (build it first)")


def _run(name: str) -> str:
    return subprocess.run([str(_bin(name))], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def _measure_once() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {"int4_gemv": {}, "attention_decode": {}}
    for line in _run("int4_gemv_bench").splitlines():
        m = INT4_RE.match(line)
        if m:
            key = f"{m['name'].strip()} [{m['out']}x{m['in']}]"
            out["int4_gemv"][key] = float(m["gbs"])
    for line in _run("attention_decode_bench").splitlines():
        m = ATTN_RE.match(line)
        if m:
            key = f"{m['model']} B{m['b']} ctx{m['ctx']}"
            out["attention_decode"][key] = float(m["gbs"])
    return out


def measure(repeat: int) -> dict[str, dict[str, float]]:
    """{bench: {shape_key: gbs}}, best-of-`repeat` per shape (max = least noisy)."""
    best: dict[str, dict[str, float]] = {"int4_gemv": {}, "attention_decode": {}}
    for _ in range(max(1, repeat)):
        run = _measure_once()
        for bench, shapes in run.items():
            for key, gbs in shapes.items():
                cur = best[bench].get(key)
                if cur is None or gbs > cur:
                    best[bench][key] = gbs
    if not best["int4_gemv"] or not best["attention_decode"]:
        sys.exit("[kernel_perf_gate] parsed no rows — bench output format changed?")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="write current numbers as the baseline")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="override the per-bench tolerance for ALL benches (fractional drop)")
    ap.add_argument("--repeat", type=int, default=3, help="best-of-N runs per shape (default 3)")
    args = ap.parse_args()

    current = measure(args.repeat)
    if args.update:
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        n = sum(len(v) for v in current.values())
        print(f"[kernel_perf_gate] wrote baseline ({n} shapes) -> {BASELINE}")
        return 0

    if not BASELINE.exists():
        sys.exit(f"[kernel_perf_gate] no baseline at {BASELINE} — run with --update first")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))

    regressions = []
    checked = 0
    for bench, shapes in base.items():
        tol = args.tolerance if args.tolerance is not None else DEFAULT_TOL.get(bench, 0.12)
        for key, base_gbs in shapes.items():
            cur = current.get(bench, {}).get(key)
            if cur is None:
                print(f"  MISSING  {bench}: {key} (in baseline, not measured)")
                continue
            checked += 1
            drop = (base_gbs - cur) / base_gbs if base_gbs > 0 else 0.0
            if drop > tol:
                regressions.append((bench, key, base_gbs, cur, drop))
                print(f"  REGRESS  {bench}: {key}  {base_gbs:.1f} -> {cur:.1f} GB/s  "
                      f"(-{drop*100:.0f}%, tol {tol*100:.0f}%)")

    tol_desc = (f"{args.tolerance*100:.0f}% (override)" if args.tolerance is not None
                else ", ".join(f"{b} {t*100:.0f}%" for b, t in DEFAULT_TOL.items()))
    print(f"[kernel_perf_gate] {checked} shapes checked, {len(regressions)} regressions "
          f"(tolerance: {tol_desc})")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
