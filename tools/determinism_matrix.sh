#!/usr/bin/env bash
# Determinism matrix: run --verify-determinism across models and settings and
# check the hashes against what each row is supposed to do.
#
# What a row means:
#   same config, run twice, same hash        run-to-run determinism
#   same config, another machine, same hash  cross-machine determinism
#   different config, different hash         expected, and not a failure: int4
#                                            weights are different numbers
#
# The interesting rows are the ones that SHOULD not change the answer and do,
# so those are asserted rather than printed: rows in an "invariant group" must
# all agree, and the script exits non-zero when one does not. It used to print
# a table and always exit 0, which reads as a gate in CI without being one.
#
# CPI_DET_SELFTEST=<n> corrupts token <n> of one row in each invariant group, so
# the assertions must fire. A check that has never failed is not evidence.
#
# Usage: tools/determinism_matrix.sh <cpi-binary> [output.tsv]

set -u
BIN="${1:-./build-cuda-ninja/cpi.exe}"
OUT="${2:-determinism_matrix.tsv}"
SELFTEST="${CPI_DET_SELFTEST:-}"
PROMPT="The capital of France is"
N=64

declare -A HASH=()
declare -A CAUGHT=()
declare -A HASSELF=()
fail=0

# run <group> <label> <model> [flags...]
# group is the invariant group: every row sharing one must produce one hash.
# Use "-" for rows that are recorded but not asserted.
run() {
  local group="$1"; shift
  local label="$1"; shift
  local model="$1"; shift
  local perturb=""
  # Corrupt exactly one row per group (the one labelled *-selftest), so the group
  # assertion has something to catch.
  case "$label" in
    *-selftest) perturb="$SELFTEST"; [ -n "$SELFTEST" ] && HASSELF[$group]=1 ;;
  esac
  local line
  line=$(CPI_DET_PERTURB="$perturb" "$BIN" "$model" --prompt "$PROMPT" \
           --verify-determinism "$N" "$@" 2>/dev/null | grep -m1 "hash=")
  if [ -z "$line" ]; then
    printf '%s\t%s\tFAILED-NO-OUTPUT\n' "$label" "$(basename "$model")" | tee -a "$OUT"
    # A row that produced nothing is a broken test, not a passing one: two empty
    # results compare equal and read as agreement.
    fail=1
    return
  fi
  local hash
  hash=$(printf '%s' "$line" | sed -E 's/.*hash=([0-9a-f]+).*/\1/')
  printf '%s\t%s\t%s\n' "$label" "$(basename "$model")" "$hash" | tee -a "$OUT"
  if [ "$group" != "-" ]; then
    if [ -z "${HASH[$group]+x}" ]; then
      HASH[$group]="$hash|$label"
    else
      local want="${HASH[$group]%%|*}"
      local from="${HASH[$group]#*|}"
      if [ "$hash" != "$want" ]; then
        echo "  MISMATCH in group '$group': $label != $from" | tee -a "$OUT"
        fail=1
        CAUGHT[$group]=1
      fi
    fi
  fi
}

: > "$OUT"
echo "label	model	hash" | tee -a "$OUT"

GGUF=artifacts/llamacpp/Llama-3.2-1B-Instruct-F16.gguf
LL2C=/c/models/llama32-1b.ll2c
GEMMA=artifacts/hub/google__gemma-4-E2B-it/hf

# Group "llama": every one of these must produce the same tokens. Repeatability,
# cache policy, paging, context size and container format all belong to it, since
# none of them is allowed to change the answer.
run llama repeat-1            "$GGUF" --gpu-cache-all
run llama repeat-2            "$GGUF" --gpu-cache-all
run llama no-gpu-cache        "$GGUF"
run llama paged-kv            "$GGUF" --gpu-cache-all --paged-kv-cache
run llama paged-blocks        "$GGUF" --gpu-cache-all --paged-blocks
run llama ctx-4096            "$GGUF" --gpu-cache-all --max-context 4096
run llama ctx-8192            "$GGUF" --gpu-cache-all --max-context 8192
# Same weights, different container: the .ll2c and the GGUF hold the same model.
run llama ll2c-baseline       "$LL2C" --gpu-cache-all
run llama llama-selftest      "$GGUF" --gpu-cache-all

# Settings that legitimately change the numbers; recorded, not asserted, to show
# they are distinguishable from the rows above rather than to claim they match.
run - weight-int8             "$GGUF" --gpu-cache-all --weight-quant int8
run - weight-int4             "$GGUF" --gpu-cache-all --weight-quant int4
run - kv-int4                 "$GGUF" --gpu-cache-all --kv-int4

# A second family, and a different engine (op-plan rather than LlamaEngine).
run gemma gemma-baseline      "$GEMMA" --gpu-cache-all
run gemma gemma-repeat        "$GEMMA" --gpu-cache-all
run gemma gemma-no-gpu-cache  "$GEMMA"
run gemma gemma-selftest      "$GEMMA" --gpu-cache-all

echo ""
echo "wrote $OUT"
if [ -n "$SELFTEST" ]; then
  # Every group that had a row corrupted must have caught it. Requiring only one
  # failure overall would let a group whose engine has no perturbation hook pass
  # by riding on another group's mismatch, which is the same false pass this
  # script exists to prevent.
  missed=""
  for g in "${!HASSELF[@]}"; do
    [ -z "${CAUGHT[$g]+x}" ] && missed="$missed $g"
  done
  if [ -n "$missed" ]; then
    echo "SELF-TEST FAILED: corrupted a row in each group at token $SELFTEST, but"
    echo "these groups reported no mismatch:$missed"
    echo "Either the engine behind them has no perturbation hook (see"
    echo "include/engine/det_perturb.hpp) or the group assertion is not running."
    exit 1
  fi
  echo "(self-test: the mismatches above are expected, one row per group was corrupted)"
  exit 0
fi
if [ "$fail" -eq 0 ]; then
  echo "PASS: every row within each invariant group agreed."
else
  echo "FAIL: a setting that must not change the answer changed it."
fi
exit "$fail"
