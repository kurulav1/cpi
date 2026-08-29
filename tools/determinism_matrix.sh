#!/usr/bin/env bash
# Determinism matrix: run --verify-determinism across models and settings and
# print the hash for each, so the scope of any determinism claim is measured
# rather than asserted.
#
# What a row means:
#   same config, run twice, same hash        run-to-run determinism
#   same config, another machine, same hash  cross-machine determinism
#   different config, different hash         expected, and not a failure: int4
#                                            weights are different numbers
#
# The interesting rows are the ones that SHOULD not change the answer and do.
# Usage: tools/determinism_matrix.sh <cpi-binary> [output.tsv]

set -u
BIN="${1:-./build-cuda-ninja/cpi.exe}"
OUT="${2:-determinism_matrix.tsv}"
PROMPT="The capital of France is"
LONG_PROMPT="Summarise the following in one sentence. The heron stands in the shallows at dawn without moving, waiting for the water to forget it is there, and the river carries leaves past its legs while the light climbs the far bank."
N=64

run() {
  # run <label> <model> [extra flags...]
  local label="$1"; shift
  local model="$1"; shift
  local line
  line=$("$BIN" "$model" --prompt "$PROMPT" --verify-determinism "$N" "$@" 2>/dev/null \
         | grep -m1 "hash=")
  if [ -z "$line" ]; then
    printf '%s\t%s\tFAILED\n' "$label" "$(basename "$model")" | tee -a "$OUT"
    return
  fi
  local hash
  hash=$(printf '%s' "$line" | sed -E 's/.*hash=([0-9a-f]+).*/\1/')
  printf '%s\t%s\t%s\n' "$label" "$(basename "$model")" "$hash" | tee -a "$OUT"
}

: > "$OUT"
echo "label	model	hash" | tee -a "$OUT"

GGUF=artifacts/llamacpp/Llama-3.2-1B-Instruct-F16.gguf
LL2C=/c/models/llama32-1b.ll2c
GEMMA=artifacts/hub/google__gemma-4-E2B-it/hf

# Repeatability: the same command twice must agree, or nothing below means anything.
run repeat-1            "$GGUF" --gpu-cache-all
run repeat-2            "$GGUF" --gpu-cache-all

# Settings that should not change the answer.
run baseline            "$GGUF" --gpu-cache-all
run no-gpu-cache        "$GGUF"
run paged-kv            "$GGUF" --gpu-cache-all --paged-kv-cache
run paged-blocks        "$GGUF" --gpu-cache-all --paged-blocks
run ctx-4096            "$GGUF" --gpu-cache-all --max-context 4096
run ctx-8192            "$GGUF" --gpu-cache-all --max-context 8192

# Settings that legitimately change the numbers; recorded to show they are
# distinguishable from the rows above, not to claim they should match.
run weight-int8         "$GGUF" --gpu-cache-all --weight-quant int8
run weight-int4         "$GGUF" --gpu-cache-all --weight-quant int4
run kv-int4             "$GGUF" --gpu-cache-all --kv-int4

# Same weights, different container: the .ll2c and the GGUF hold the same model.
run ll2c-baseline       "$LL2C" --gpu-cache-all

# A second family, and a different engine (op-plan rather than LlamaEngine).
run gemma-baseline      "$GEMMA" --gpu-cache-all
run gemma-repeat        "$GEMMA" --gpu-cache-all
run gemma-no-gpu-cache  "$GEMMA"

echo ""
echo "wrote $OUT"
