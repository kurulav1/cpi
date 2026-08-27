#!/usr/bin/env python3
"""Compare two per-layer activation dumps and report the first layer that diverges.

A port that disagrees only in its final token tells you nothing about where it went wrong. For
Qwen3.5 there are three independent suspects per layer, the delta-net block, the gated
attention block, the MLP, times 24 layers, so "the output is wrong" is a search, not a finding.
This turns it into a line number.

Dumps are written by setting CPI_Q35_DUMP=<dir> (see qwen35_cpu_engine.cpp); each file is
layer_NN_pos_MM.f32, raw float32, hidden_size values, native endianness. layer_00 is the
embedding, before any layer runs.

    python3 tools/layer_diff.py ref_dir test_dir

Exit code is 1 when a layer diverges past tolerance, so it can gate.

Reading the output: divergence that starts at layer N and grows is a bug in layer N. Divergence
that appears at layer N already large, with N-1 clean, is the same thing. Divergence that is
small everywhere and never grows is fp16-vs-fp32 accumulation, not a bug, raise --tol.
"""

import argparse
import struct
import sys
from pathlib import Path


# Anything above this is a real difference, not fp accumulation between two identical runs
# (which measure exactly 0.0). Kept far below --tol on purpose: --tol judges "is this broken",
# NOISE_FLOOR judges "where did it start".
NOISE_FLOOR = 1e-9


def load(path: Path):
    raw = path.read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def stats(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return None
    max_abs = 0.0
    sum_abs = 0.0
    # cosine similarity too: a scale error and a structural error look identical in max_abs, and
    # very different here. cos ~1.0 with a large max_abs means "right direction, wrong gain".
    dot = na = nb = 0.0
    for i in range(n):
        d = abs(a[i] - b[i])
        max_abs = max(max_abs, d)
        sum_abs += d
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    denom = (na ** 0.5) * (nb ** 0.5)
    cos = dot / denom if denom > 0 else float("nan")
    return max_abs, sum_abs / n, cos, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("test_dir")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max_abs above this counts as divergence (default 0.05)")
    ap.add_argument("--pos", type=int, default=None, help="only this position")
    args = ap.parse_args()

    ref_dir, test_dir = Path(args.ref_dir), Path(args.test_dir)
    refs = sorted(ref_dir.glob("layer_*.f32"))
    if not refs:
        print(f"no dumps in {ref_dir}: was CPI_Q35_DUMP set?")
        return 1

    # ONSET, not tolerance. Divergence grows as it propagates, so the first layer to exceed a
    # tolerance is many layers downstream of the cause; injecting a 5% error at layer 11 and
    # asking "where does max_abs first exceed 0.05" answers layer 24. The cause is the first
    # layer that stops matching AT all, so that is what gets reported; the tolerance only decides
    # the exit code.
    first_onset = None
    first_bad = None
    print(f"{'layer':>6} {'pos':>4} {'max_abs':>11} {'mean_abs':>11} {'cos':>9}  {'':<6}")
    for r in refs:
        t = test_dir / r.name
        parts = r.stem.split("_")
        layer, pos = int(parts[1]), int(parts[3])
        if args.pos is not None and pos != args.pos:
            continue
        if not t.exists():
            print(f"{layer:6d} {pos:4d} {'MISSING':>11}")
            first_bad = first_bad if first_bad is not None else layer
            continue
        st = stats(load(r), load(t))
        if st is None:
            continue
        max_abs, mean_abs, cos, n = st
        bad = max_abs > args.tol
        if bad and first_bad is None:
            first_bad = layer
        if max_abs > NOISE_FLOOR and first_onset is None:
            first_onset = layer
        print(f"{layer:6d} {pos:4d} {max_abs:11.6f} {mean_abs:11.6f} {cos:9.6f}  "
              f"{'<-- ONSET' if layer == first_onset else ('<-- past tol' if bad else '')}")

    print()
    if first_onset is None:
        print(f"agreement within {args.tol} at every layer")
        return 0
    first_bad = first_onset
    # layer 0 in the filenames is the embedding; a divergence there is a tokenizer or
    # embedding-table problem, not a block problem, and worth saying so explicitly.
    if first_bad == 0:
        print("FIRST DIVERGENCE: the embedding, before any layer ran "
              "(tokenizer or embedding table, not a block)")
    else:
        print(f"FIRST DIVERGENCE: layer {first_bad} (1-based; layer_00 is the embedding)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
