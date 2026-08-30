#!/usr/bin/env bash
# Prove that the determinism checks can fail.
#
# Every row in docs/determinism-scope.md is some script reporting "identical".
# That is only evidence if the script would have said otherwise. The divergence
# branch is the branch that never runs, and when it finally did run here it
# crashed on a path-format bug it had been carrying unnoticed. Two of the four
# scripts, meanwhile, printed a table and exited 0 whatever they found, so they
# would have passed in CI while reporting failures on screen.
#
# So: run each check against a build corrupted on purpose (CPI_DET_PERTURB
# replaces the token at a chosen index) and require that it notices. A check that
# reports "identical" against a corrupted build is not measuring anything, and
# this script fails when that happens.
#
# Usage: tools/determinism_selftest.sh [model] [tokenizer] [cpi-binary]

set -u
MODEL="${1:-artifacts/llamacpp/Llama-3.2-1B-Instruct-F16.gguf}"
TOK="${2:-/c/models/llama32-1b-hf/tokenizer.json}"
BIN="${3:-./build-cuda-ninja/cpi.exe}"
STEP=10
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

# check <name> <expect-normal-exit> <command...>
# Runs the command and compares its exit status against what that scenario
# requires. Output is kept and shown only on failure, so a green run stays short.
check() {
  local name="$1" want="$2"; shift 2
  local got
  "$@" > "$TMP/out.txt" 2>&1
  got=$?
  if [ "$got" -eq "$want" ]; then
    printf '  ok    %-46s (exit %s)\n' "$name" "$got"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-46s (exit %s, wanted %s)\n' "$name" "$got" "$want"
    sed 's/^/        | /' "$TMP/out.txt" | tail -20
    fail=$((fail + 1))
  fi
}

echo "=== 1. the perturbation switch itself ==="
# Off by default, or every measurement ever taken with this binary is suspect.
h_clean=$("$BIN" "$MODEL" --tokenizer "$TOK" --prompt "The capital of France is" \
            --verify-determinism 32 --gpu-cache-all 2>/dev/null | sed -n 's/.*hash=\([0-9a-f]*\).*/\1/p')
h_dirty=$(CPI_DET_PERTURB="$STEP" "$BIN" "$MODEL" --tokenizer "$TOK" \
            --prompt "The capital of France is" --verify-determinism 32 --gpu-cache-all \
            2>/dev/null | sed -n 's/.*hash=\([0-9a-f]*\).*/\1/p')
if [ -z "$h_clean" ] || [ -z "$h_dirty" ]; then
  echo "  FAIL  could not obtain both hashes (${h_clean:-none} / ${h_dirty:-none})"
  fail=$((fail + 1))
elif [ "$h_clean" = "$h_dirty" ]; then
  echo "  FAIL  CPI_DET_PERTURB=$STEP did not change the token stream ($h_clean)"
  echo "        The engine that ran has no perturbation hook, so every check below"
  echo "        would report a false pass for it."
  fail=$((fail + 1))
else
  printf '  ok    %-46s (%s -> %s)\n' "perturbation changes the stream" "$h_clean" "$h_dirty"
  pass=$((pass + 1))
fi
# And it must announce itself: a corrupted run that looks normal is the failure
# this whole mechanism exists to prevent.
if CPI_DET_PERTURB="$STEP" "$BIN" "$MODEL" --tokenizer "$TOK" --prompt "hi" \
     --verify-determinism 4 --gpu-cache-all 2>&1 >/dev/null | grep -q "det-perturb.*ACTIVE"; then
  printf '  ok    %-46s\n' "corrupted runs announce themselves on stderr"
  pass=$((pass + 1))
else
  echo "  FAIL  a corrupted run produced no banner on stderr"
  fail=$((fail + 1))
fi
echo ""

echo "=== 2. each check passes on an honest build ==="
check "matrix"  0 bash tools/determinism_matrix.sh "$BIN" "$TMP/matrix.tsv"
check "batch"   0 bash tools/determinism_batch.sh "$MODEL" "$TOK" "$BIN"
check "backend" 0 bash tools/determinism_backend.sh "$MODEL" "$TOK" "$BIN"
echo ""

echo "=== 3. each check FAILS on a build corrupted at token $STEP ==="
# Each script exits 0 under CPI_DET_SELFTEST only after confirming it saw the
# corruption, and exits 1 if it reported "identical" anyway. So 0 here means the
# check demonstrated it can detect a difference.
check "matrix detects corruption"  0 env CPI_DET_SELFTEST="$STEP" \
  bash tools/determinism_matrix.sh "$BIN" "$TMP/matrix2.tsv"
check "batch detects corruption"   0 env CPI_DET_SELFTEST="$STEP" \
  bash tools/determinism_batch.sh "$MODEL" "$TOK" "$BIN"
check "backend detects corruption" 0 env CPI_DET_SELFTEST="$STEP" \
  bash tools/determinism_backend.sh "$MODEL" "$TOK" "$BIN"
echo ""

echo "=== 4. the batch fix's own control ==="
# CPI_DET_BATCH=0 restores the cuBLAS N=1 kernel choice, a real divergence rather
# than an injected one, so the batch check must fail outright (exit 1).
check "batch check fails without the fix" 1 env CPI_DET_BATCH=0 \
  bash tools/determinism_batch.sh "$MODEL" "$TOK" "$BIN"
echo ""

echo "$pass passed, $fail failed"
if [ "$fail" -ne 0 ]; then
  echo "A determinism check that cannot fail is not evidence. Fix the check, not this script."
  exit 1
fi
echo "Every determinism check has now been shown to both pass and fail."
echo ""
echo "Not covered here: determinism_version.sh, which needs an old build in a"
echo "worktree. Run it with CPI_DET_SELFTEST set once such a build exists."
exit 0
