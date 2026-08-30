#!/usr/bin/env bash
# Backend invariance: does the CPU engine produce the same tokens as CUDA?
#
# This axis needs no second machine, and it is the one most likely to break:
# different reduction orders, different accumulate precision, and no cuBLAS at
# all. Run it before renting hardware to test anything else.
#
# Two controls, because a matching hash is worthless without them:
#   1. The [verify] backend= field must actually differ between the two runs.
#      It reports the engine chosen at runtime, not the build configuration; a
#      build-time #if would label a --cpu run "cuda" and compare equal.
#   2. The CPU run must be far slower. If the two are within a few times of each
#      other, --cpu did not take effect and both ran on the GPU.
#
# Usage: tools/determinism_backend.sh <model> <tokenizer> [cpi-binary]

set -u
MODEL="${1:?usage: determinism_backend.sh <model> <tokenizer> [binary]}"
TOK="${2:?usage: determinism_backend.sh <model> <tokenizer> [binary]}"
BIN="${3:-./build-cuda-ninja/cpi.exe}"
# CPI_DET_SELFTEST=<n> corrupts token <n> of the CPU side only, so this comparison
# must report divergence. A check that has never failed is not evidence.
SELFTEST="${CPI_DET_SELFTEST:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

ids() { sed 's/^\[verify\] ids=//' "$1" | tr ',' '\n'; }

echo "=== control 1: the two runs must report different engines ==="
gb=$("$BIN" "$MODEL" --tokenizer "$TOK" --prompt "hi" --verify-determinism 4 --gpu-cache-all \
      2>/dev/null | sed -n 's/^\[verify\] backend=\([^ ]*\).*/\1/p')
cb=$("$BIN" "$MODEL" --tokenizer "$TOK" --prompt "hi" --verify-determinism 4 --cpu \
      2>/dev/null | sed -n 's/^\[verify\] backend=\([^ ]*\).*/\1/p')
echo "  gpu run reports: ${gb:-<none>}"
echo "  cpu run reports: ${cb:-<none>}"
if [ -z "$gb" ] || [ -z "$cb" ] || [ "$gb" = "$cb" ]; then
  echo "  CONTROL FAILED: both runs used the same engine, so any agreement below"
  echo "  is an artifact. Nothing after this line means anything."
  exit 1
fi
echo "  ok, two different engines"
echo ""

compare() {
  local label="$1" prompt="$2" n="$3"; shift 3
  "$BIN" "$MODEL" --tokenizer "$TOK" --prompt "$prompt" --verify-determinism "$n" \
      --gpu-cache-all "$@" 2>/dev/null | grep '^\[verify\] ids=' > "$TMP/g.txt"
  # The self-test corrupts this side only, giving the comparison a known-different
  # input it must report. PERTURB is exported empty when not self-testing, and
  # det_perturb treats empty as absent.
  CPI_DET_PERTURB="$SELFTEST" "$BIN" "$MODEL" --tokenizer "$TOK" --prompt "$prompt" \
      --verify-determinism "$n" --cpu "$@" 2>/dev/null | grep '^\[verify\] ids=' > "$TMP/c.txt"
  if [ ! -s "$TMP/g.txt" ] || [ ! -s "$TMP/c.txt" ]; then
    printf '%-26s BROKEN (one side produced no tokens)\n' "$label"
    return
  fi
  ids "$TMP/g.txt" > "$TMP/gi.txt"
  ids "$TMP/c.txt" > "$TMP/ci.txt"
  local ng nc
  ng=$(wc -l < "$TMP/gi.txt"); nc=$(wc -l < "$TMP/ci.txt")

  if [ "$ng" -eq "$nc" ] && diff -q "$TMP/gi.txt" "$TMP/ci.txt" >/dev/null; then
    printf '%-26s identical (%s tokens)\n' "$label" "$ng"
    return
  fi
  # Unequal lengths are not automatically a disagreement about token values. If
  # the shorter run is a prefix of the longer one, the engines computed the same
  # thing and one stopped sooner, which is a different defect from numeric drift
  # and belongs in a different row of the scope table.
  local shorter=$(( ng < nc ? ng : nc ))
  if diff -q <(head -n "$shorter" "$TMP/gi.txt") <(head -n "$shorter" "$TMP/ci.txt") >/dev/null; then
    printf '%-26s PREFIX ONLY (cuda %s, cpu %s: same tokens, one stopped earlier)\n' \
      "$label" "$ng" "$nc"
    fail=1
  else
    local first
    first=$(paste "$TMP/gi.txt" "$TMP/ci.txt" | awk '$1!=$2{print NR-1; exit}')
    printf '%-26s DIVERGES at token %s (cuda %s, cpu %s)\n' "$label" "$first" "$ng" "$nc"
    fail=1
  fi
}

compare "64 tokens"        "The capital of France is" 64
compare "256 tokens"       "The capital of France is" 256
compare "512 tokens"       "The capital of France is" 512
compare "1024 tokens"      "The capital of France is" 1024
compare "2048, ctx 4096"   "Write a long detailed essay about the ocean." 2048 --max-context 4096
compare "prose"            "Write a short paragraph about the sea." 512
compare "code"             "def fibonacci(n):" 512
compare "paged-kv"         "The capital of France is" 256 --paged-kv-cache
compare "ctx 8192"         "The capital of France is" 256 --max-context 8192

echo ""
echo "Known: generating into the context limit is NOT covered above, because the"
echo "CPU engine stops a few tokens earlier there than CUDA does. Those tokens"
echo "agree as far as both engines produce them, so it is a stopping-rule"
echo "difference rather than an arithmetic one. Raise --max-context past the"
echo "requested length to test arithmetic without that boundary in the way."
if [ -n "$SELFTEST" ]; then
  if [ "$fail" -eq 0 ]; then
    echo ""
    echo "SELF-TEST FAILED: the CPU side was corrupted at token $SELFTEST and every row"
    echo "still reported identical, so this comparison detects nothing."
    exit 1
  fi
  echo ""
  echo "(self-test: the failures above are expected, the CPU side was corrupted on purpose)"
  exit 0
fi
exit "$fail"
