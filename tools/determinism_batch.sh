#!/usr/bin/env bash
# Batch-size invariance: decode the same prompt alone and alongside other
# sequences, and compare the text it produced.
#
# This needs its own script because --verify-determinism decodes one sequence,
# so it cannot see the axis at all. The target sequence is identical in every
# run; only the number of unrelated sequences beside it changes. If the hashes
# differ, a request's answer depends on what else the server happened to be
# doing, which for an agent loop means the batch composition is an input.
#
# Note the paths under test are fp16 only: the batched decode refuses
# INT8/INT4 weights, so there is no quantised batch to compare against.
#
# Exits non-zero on divergence. Two ways to watch it do that:
#   CPI_DET_BATCH=0      restores the pre-fix cuBLAS behaviour, where N=1 selects
#                        a different kernel and batch 1 really does disagree.
#   CPI_DET_SELFTEST=<n> corrupts token <n> of the batch-1 run only, so the
#                        comparison must report divergence at index <n>.
#
# Usage: tools/determinism_batch.sh <model> <tokenizer> [cpi-binary]

set -u
MODEL="${1:?usage: determinism_batch.sh <model> <tokenizer> [binary]}"
TOK="${2:?usage: determinism_batch.sh <model> <tokenizer> [binary]}"
BIN="${3:-./build-cuda-ninja/cpi.exe}"
SELFTEST="${CPI_DET_SELFTEST:-}"
N=64
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Pull out the target sequence only, and print its tokens one per line. Sequences
# are interleaved in the output stream, so they have to be reassembled by id.
extract_target() {
  python -c '
import sys, json
acc = {}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    i = d.get("id")
    if d.get("type") == "delta":
        acc[i] = acc.get(i, "") + d.get("delta", "")
    elif d.get("type") == "done" and d.get("text"):
        acc[i] = d["text"]
text = acc.get("TARGET")
if text:
    # One character per line: the transport carries text, not ids, so this is the
    # finest-grained comparison available and it localises a divergence.
    for ch in text:
        print(repr(ch))
'
}

requests() {
  python -c '
import json, sys
n = int(sys.argv[1])
nmax = int(sys.argv[2])
print(json.dumps({"id": "TARGET", "prompt": "The capital of France is", "max_new": nmax}))
for k in range(1, n):
    print(json.dumps({"id": "filler%d" % k, "prompt": "Count from %d to 9." % k, "max_new": nmax}))
' "$1" "$N"
}

run_batch() {
  local b="$1"
  # The self-test corrupts the batch-1 run only, so that the comparison has one
  # side that is known-different and must say so.
  if [ -n "$SELFTEST" ] && [ "$b" -eq 1 ]; then
    requests "$b" | CPI_DET_PERTURB="$SELFTEST" "$BIN" "$MODEL" --tokenizer "$TOK" \
      --interactive-batch --paged-blocks --temp 0 --gpu-cache-all 2>/dev/null | extract_target
  else
    requests "$b" | "$BIN" "$MODEL" --tokenizer "$TOK" \
      --interactive-batch --paged-blocks --temp 0 --gpu-cache-all 2>/dev/null | extract_target
  fi
}

fail=0
ref=""
echo "batch  chars  result"
for b in 1 2 3 5 8; do
  run_batch "$b" > "$TMP/b$b.txt"
  n=$(wc -l < "$TMP/b$b.txt")
  # An empty run is a broken test, not a passing one. Three SHA-256s of the empty
  # string once compared equal here and read as clean invariance.
  if [ "$n" -eq 0 ]; then
    printf '%-6s %-6s BROKEN: produced no output (check stderr)\n' "$b" "0"
    fail=1
    continue
  fi
  if [ -z "$ref" ]; then
    ref="$TMP/b$b.txt"
    printf '%-6s %-6s reference\n' "$b" "$n"
    continue
  fi
  if diff -q "$ref" "$TMP/b$b.txt" >/dev/null; then
    printf '%-6s %-6s identical\n' "$b" "$n"
  else
    first=$(diff --unchanged-line-format='' --old-line-format='%dn
' --new-line-format='' "$ref" "$TMP/b$b.txt" 2>/dev/null | head -1)
    printf '%-6s %-6s DIVERGES from batch 1, first difference at character %s\n' \
      "$b" "$n" "${first:-unknown}"
    fail=1
  fi
done

echo ""
if [ "$fail" -eq 0 ]; then
  echo "PASS: a request's output does not depend on how many others are in flight."
  if [ -n "$SELFTEST" ]; then
    echo "SELF-TEST FAILED: the batch-1 run was corrupted at token $SELFTEST and this"
    echo "comparison still reported no difference, so it is not detecting anything."
    exit 1
  fi
else
  echo "FAIL: batch size changed the answer."
  if [ -n "$SELFTEST" ]; then
    echo "(self-test: this failure is expected, the batch-1 run was corrupted on purpose)"
    exit 0
  fi
fi
exit "$fail"
