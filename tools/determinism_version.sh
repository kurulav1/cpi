#!/usr/bin/env bash
# Version invariance: does today's build still produce the token stream an
# older build produced, from the same weights and settings?
#
# This is the axis that needs no second machine, and it is the one that decides
# whether version stability already exists or has to start being gated now.
#
# Build the old commit in a detached worktree first:
#   git worktree add --detach /tmp/wt <old-commit>
# then configure and build it the way that commit expected, and pass both
# binaries here. Old enough commits name the binary llama_infer rather than cpi.
#
# Use a container both builds can read. GGUF support landed mid-history, so
# comparing across it means using .ll2c, which HEAD separately verifies is
# token-identical to the GGUF of the same checkpoint.
#
# Usage: tools/determinism_version.sh <old-binary> <new-binary> <model> <tokenizer>

set -u
OLD="${1:?usage: determinism_version.sh <old-bin> <new-bin> <model> <tokenizer>}"
NEW="${2:?usage: determinism_version.sh <old-bin> <new-bin> <model> <tokenizer>}"
MODEL="${3:?}"
# CPI_DET_SELFTEST=<n> corrupts token <n> of the new build only, so this comparison
# must report divergence. A check that has never failed is not evidence.
SELFTEST="${CPI_DET_SELFTEST:-}"
TOK="${4:?}"
TMP="${TMPDIR:-/tmp}/cpi-version-axis.$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

fail=0

compare() {
  local label="$1" prompt="$2" n="$3"; shift 3
  "$OLD" "$MODEL" --tokenizer "$TOK" --prompt "$prompt" --max-new "$n" --temp 0 "$@" \
      2>"$TMP/old.err" | grep '^Output tokens:' > "$TMP/old.txt"
  # The self-test corrupts today's build only, so this comparison has a
  # known-different side it is obliged to report.
  CPI_DET_PERTURB="$SELFTEST" "$NEW" "$MODEL" --tokenizer "$TOK" --prompt "$prompt" \
      --max-new "$n" --temp 0 "$@" 2>"$TMP/new.err" | grep '^Output tokens:' > "$TMP/new.txt"

  # A missing token line is a broken test, not a pass. Say which side broke and
  # why, because "both produced nothing" compares equal and looks like success.
  if [ ! -s "$TMP/old.txt" ] || [ ! -s "$TMP/new.txt" ]; then
    local which="both"
    [ -s "$TMP/old.txt" ] && which="new"
    [ -s "$TMP/new.txt" ] && which="old"
    printf '%-34s BROKEN (%s produced no tokens: %s)\n' "$label" "$which" \
      "$(grep -im1 'fatal\|error' "$TMP/old.err" "$TMP/new.err" | head -1 | cut -c1-70)"
    return
  fi

  # One token per line so a difference can be located rather than just detected.
  sed 's/^Output tokens: //' "$TMP/old.txt" | tr ' ' '\n' | grep -v '^$' > "$TMP/oi.txt"
  sed 's/^Output tokens: //' "$TMP/new.txt" | tr ' ' '\n' | grep -v '^$' > "$TMP/ni.txt"
  local no nn
  no=$(wc -l < "$TMP/oi.txt"); nn=$(wc -l < "$TMP/ni.txt")
  if [ "$no" -eq "$nn" ] && diff -q "$TMP/oi.txt" "$TMP/ni.txt" >/dev/null; then
    printf '%-34s identical (%s tokens)\n' "$label" "$no"
  else
    local first
    first=$(paste "$TMP/oi.txt" "$TMP/ni.txt" | awk '$1!=$2{print NR-1; exit}')
    if [ -z "$first" ]; then
      # Same prefix, different length: one build stopped sooner.
      printf '%-34s DIFFER in length only (old %s, new %s tokens)\n' "$label" "$no" "$nn"
    else
      # Position within the printed stream, which includes the prompt. The other
      # scripts read --verify-determinism, whose ids are generated tokens only, so
      # the same corruption appears there at a lower index by the prompt length.
      printf '%-34s DIFFER at output position %s (old %s, new %s tokens)\n' \
        "$label" "$first" "$no" "$nn"
    fi
    fail=1
  fi
}

compare "short, gpu-cache-all"   "The capital of France is" 64  --gpu-cache-all
compare "short, no gpu cache"    "The capital of France is" 64
compare "long, gpu-cache-all"    "The capital of France is" 256 --gpu-cache-all
compare "long, no gpu cache"     "The capital of France is" 256
compare "prose"                  "Write a short paragraph about the sea." 256 --gpu-cache-all
compare "code"                   "def fibonacci(n):" 64 --gpu-cache-all

echo ""
if [ "$fail" -eq 0 ]; then
  echo "No divergence. Note that BROKEN rows are not passes: an old build that"
  echo "cannot run a config tells you nothing about whether it agrees on one."
else
  echo "Divergence found: today's build does not reproduce the older stream."
fi
if [ -n "$SELFTEST" ]; then
  if [ "$fail" -eq 0 ]; then
    echo ""
    echo "SELF-TEST FAILED: the new build was corrupted at token $SELFTEST and every row"
    echo "still reported identical, so this comparison detects nothing."
    exit 1
  fi
  echo ""
  echo "(self-test: the failures above are expected, the new build was corrupted on purpose)"
  exit 0
fi
exit "$fail"
