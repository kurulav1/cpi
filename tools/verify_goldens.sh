#!/usr/bin/env bash
# Replay the golden token streams against the CURRENT build.
#
# src/tests/golden/*.txt hold token streams the CUDA backend produced once, and
# tools/metal_verify.sh gates Metal by requiring it to reproduce them exactly.
# Nothing checked they still describe what CUDA does. That matters more than it
# sounds: the Metal gate has never run (it is inert until a self-hosted runner
# exists), so if a golden drifted from CUDA in the meantime, the first Metal run
# would compare a new backend against a stale reference and the failure would be
# attributed to Metal.
#
# A golden is a claim about this engine. Claims get checked on the backend that
# made them, not only on the one being judged by them.
#
# Every golden whose checkpoint is missing is skipped LOUDLY and counted, because
# a run that verified nothing and a run that verified everything both print no
# failures. Skips do not fail the run; a mismatch does.
#
# Usage: tools/verify_goldens.sh [cpi-binary] [checkpoint-dir ...]

set -u
BIN="${1:-./build-cuda-ninja/cpi.exe}"
shift 2>/dev/null || true
SEARCH=("$@")
if [ "${#SEARCH[@]}" -eq 0 ]; then
  SEARCH=(/c/models artifacts/ci_mixtral artifacts/llamacpp artifacts)
fi
GOLDEN_DIR="src/tests/golden"

pass=0; skip=0; failn=0

find_ckpt() {
  local d n
  for d in "${SEARCH[@]}"; do
    for n in "$@"; do
      if [ -e "$d/$n" ]; then
        printf '%s' "$d/$n"
        return 0
      fi
    done
  done
  return 1
}

# check <golden-file> <candidate checkpoint names...>
check() {
  local gf="$GOLDEN_DIR/$1"; shift
  local label; label="$(basename "$gf" .txt)"
  if [ ! -f "$gf" ]; then
    printf '  %-34s MISSING GOLDEN FILE\n' "$label"; failn=$((failn + 1)); return
  fi
  local path
  if ! path=$(find_ckpt "$@"); then
    printf '  %-34s skipped (no checkpoint among: %s)\n' "$label" "$*"
    skip=$((skip + 1)); return
  fi
  local prompt expect nprompt nexpect out got
  prompt=$(sed -n 's/^prompt: *//p' "$gf" | head -1)
  expect=$(sed -n 's/^expect: *//p' "$gf" | head -1)
  if [ -z "$prompt" ] || [ -z "$expect" ]; then
    printf '  %-34s UNREADABLE (no prompt/expect line)\n' "$label"; failn=$((failn + 1)); return
  fi
  nprompt=$(printf '%s' "$prompt" | tr ',' '\n' | grep -c .)
  nexpect=$(printf '%s' "$expect" | tr ',' '\n' | grep -c .)

  out=$("$BIN" "$path" --tokens "$prompt" --max-new "$nexpect" --temp 0 --gpu-cache-all \
        2>/dev/null | grep '^Output tokens:' | sed 's/^Output tokens: *//')
  if [ -z "$out" ]; then
    # No output is a broken check, not a passing one.
    printf '  %-34s BROKEN (the model produced no tokens)\n' "$label"
    failn=$((failn + 1)); return
  fi
  # The printed stream carries the prompt as well; drop it to leave what was generated.
  got=$(printf '%s' "$out" | tr ' ' '\n' | tail -n +$((nprompt + 1)) | tr '\n' ',' | sed 's/,$//')
  if [ "$got" = "$expect" ]; then
    printf '  %-34s matches (%s tokens)\n' "$label" "$nexpect"
    pass=$((pass + 1))
  else
    printf '  %-34s DRIFTED\n' "$label"
    printf '      golden: %s\n' "$(printf '%s' "$expect" | cut -c1-70)"
    printf '      now:    %s\n' "$(printf '%s' "$got" | cut -c1-70)"
    failn=$((failn + 1))
  fi
}

echo "Replaying goldens against $BIN"
echo ""
check tiny-mixtral-moe-24.txt        tiny-mixtral.ll2c
check qwen2.5-0.5b-sky-128.txt       qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
check qwen2.5-0.5b-greedy.txt        qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
check qwen2.5-0.5b-longprompt-86x64.txt qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
check qwen3-0.6b-sky-64.txt          Qwen3-0.6B.ll2c
check gemma-2b-capital-32.txt        gemma-2b.ll2c
# A GGUF of the same checkpoint is a fair substitute for the .ll2c: both are read by
# WeightLoader into LlamaEngine, and the two containers are separately verified to
# produce identical tokens. A safetensors DIRECTORY is not a fair substitute, even
# for the same weights, because it dispatches to Llama4CudaEngine instead; a mismatch
# there would mean "a different engine disagrees", not "the golden drifted", and
# reporting it as drift would send the next person after the wrong thing. So the
# HuggingFace copies of the Qwen and Gemma checkpoints on this machine are
# deliberately not listed as candidates.
check llama-3.1-8b-sky-48.txt        llama-3.1-8b.ll2c Llama-3.1-8B-Instruct.ll2c \
                                     llama31-8b-F16.gguf

echo ""
echo "$pass matched, $failn drifted, $skip skipped"
if [ "$skip" -gt 0 ]; then
  echo "Skipped goldens are NOT passes: their checkpoints are not on this machine, so"
  echo "nothing was compared for them. Point this script at a directory that has them."
fi
if [ "$failn" -ne 0 ]; then
  echo ""
  echo "A drifted golden means the reference the Metal gate compares against no longer"
  echo "describes what CUDA does. Decide which one is wrong before touching Metal:"
  echo "regenerating the golden hides a CUDA regression, and keeping it fails Metal for"
  echo "something Metal did not do."
  exit 1
fi
exit 0
