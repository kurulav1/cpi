#!/usr/bin/env bash
# The Metal correctness gate for Apple Silicon.
#
# WHY THIS EXISTS AS A SCRIPT AND NOT JUST CI: GitHub's macOS runners are GPU-less VMs, so
# MTLCreateSystemDefaultDevice() returns nil and every Metal test SKIPs there. The `metal`
# CI job is compile-only BY DESIGN -- a green job means the shaders type-check, the
# Objective-C++ bridge compiles and links. It does NOT mean a single kernel is correct.
# Kernel correctness needs real Apple Silicon, and without this script that check lives
# only in whoever last ran the commands by hand.
#
# Run it on actual Apple Silicon: a dev Mac, or a self-hosted / Tart-based runner (Tart VMs
# DO expose a paravirtual MTLDevice, unlike GitHub's). See .github/workflows/metal-gpu.yml.
#
#   tools/metal_verify.sh [--build DIR] [--models DIR]
#
# Checkpoints are optional: every golden it cannot find is skipped LOUDLY rather than
# silently passing, so a bare checkout still gets the synthetic kernel smoke and the
# summary tells you how much was actually covered.
#
# Exit 0 only when nothing failed.

set -uo pipefail

BUILD=build
MODELS="$HOME"
REQUIRE_GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build)  BUILD="$2";  shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    # Turn "no Metal device" from a skip into a failure. Use this anywhere the whole point
    # of the run is to exercise the GPU (the self-hosted runner): there, a missing device
    # means the host is misconfigured, and a green run that verified nothing is worse than
    # a red one. Without this flag the script stays useful on a GPU-less box.
    --require-gpu) REQUIRE_GPU=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# The engine compiles the shaders at runtime when there is no offline metallib, which is the
# normal case (the `metal` compiler ships with Xcode, not the Command Line Tools).
export CPI_METAL_SOURCE="${CPI_METAL_SOURCE:-$REPO/src/kernels/metal}"

ran=0; failed=0; skipped=0; gpu_missing=0
ok()   { echo "  PASS  $*"; ran=$((ran + 1)); }
bad()  { echo "  FAIL  $*"; ran=$((ran + 1)); failed=$((failed + 1)); }
skip() { echo "  skip  $*"; skipped=$((skipped + 1)); }

find_model() {  # echoes the first candidate that exists
  for c in "$@"; do
    if [ -f "$MODELS/$c" ]; then printf '%s' "$MODELS/$c"; return 0; fi
  done
  return 1
}

echo "== Metal verification =="
echo "  build:  $BUILD"
echo "  models: $MODELS"
echo "  shader: $CPI_METAL_SOURCE"
echo

# ---------------------------------------------------------------------------
# 1. Synthetic kernel smoke. Needs no checkpoint: each kernel is checked against a CPU
#    reference on generated data. This is the only part a GPU-less runner can attempt,
#    and there it SKIPs rather than fails.
# ---------------------------------------------------------------------------
echo "-- kernel smoke (vs CPU reference) --"
if [ ! -x "$BUILD/metal_smoke" ]; then
  bad "metal_smoke not built (is $BUILD configured with -DCPI_ENABLE_METAL=ON?)"
else
  out=$("$BUILD/metal_smoke" 2>&1)
  if echo "$out" | grep -q "ALL PASS"; then
    ok "metal_smoke"
  elif echo "$out" | grep -qiE "skip|no Metal GPU|returned nil"; then
    gpu_missing=1
    skip "metal_smoke -- no Metal GPU on this host (CI VM?), kernels NOT verified"
  else
    bad "metal_smoke"; echo "$out" | tail -6
  fi
fi
echo

# ---------------------------------------------------------------------------
# 2. Golden token streams. fp16 must reproduce the CUDA reference stream EXACTLY; a
#    quantized run is a different model, so it is held to the CPU-oracle logit bound
#    instead (looser, but still a bound).
# ---------------------------------------------------------------------------
run_golden() {  # golden tokens quant candidate...
  golden="$1"; toks="$2"; quant="$3"; shift 3
  if ! path=$(find_model "$@"); then
    skip "$golden${quant:+ (int$quant)} -- no checkpoint among: $*"
    return
  fi
  label="$golden${quant:+ (int$quant)}${LABEL_EXTRA:-}"
  if [ -n "$quant" ]; then
    out=$(CPI_METAL_QUANT="$quant" "$BUILD/metal_decode_test" "$path" "$toks" \
          "$REPO/src/tests/golden/$golden" 2>&1)
  else
    out=$("$BUILD/metal_decode_test" "$path" "$toks" "$REPO/src/tests/golden/$golden" 2>&1)
  fi
  if echo "$out" | grep -q "\[metal_decode\] PASS"; then
    ok "$label"
  elif echo "$out" | grep -q "\[metal_decode\] SKIP"; then
    # A checkpoint is present but the test declined to run it -- almost always a GPU-less
    # host. Not a failure, but not coverage either.
    echo "$out" | grep -q "no Metal GPU" && gpu_missing=1
    skip "$label -- $(echo "$out" | grep '\[metal_decode\] SKIP' | head -1)"
  else
    bad "$label"; echo "$out" | tail -6
  fi
}

echo "-- golden token streams (Metal vs the CUDA reference) --"
if [ ! -x "$BUILD/metal_decode_test" ]; then
  bad "metal_decode_test not built"
else
  # Qwen2.5-0.5B: the dense fp16 baseline, plus both quantizations.
  run_golden qwen2.5-0.5b-sky-128.txt 128 ""  qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
  run_golden qwen2.5-0.5b-sky-128.txt 128 "4" qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
  run_golden qwen2.5-0.5b-sky-128.txt 128 "8" qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
  # The ONLY golden whose prompt (86 tokens) exceeds kGemmMinTokens, so the only one that
  # executes cpi_gemm_f16 at all. Every other prompt here prefills via the GEMV. Verified to
  # catch the real thing: with the GEMM's thread count mismatched again, this FAILS and the
  # 12-token goldens still pass.
  #
  # fp16 ONLY, deliberately. There is no int4 row because the quantized gate compares against
  # the fp32 CPU engine, and across an 86-token prompt int4 legitimately picks a different
  # (equally good) first token -- fp16 opens "Leaves are green because...", int4 opens with a
  # numbered list. Both are correct; an fp32 oracle simply cannot adjudicate an int4 stream at
  # the token level. The quantized GEMM is gated properly instead, by metal_smoke, against a
  # CPU reference over T = 1..200 -- which is a stronger check than this would have been.
  run_golden qwen2.5-0.5b-longprompt-86x64.txt 64 ""  qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c

  # Split-KV decode attention, forced on. It normally waits for 256 keys, which NO golden here
  # ever reaches -- the longest tops out around 150 -- so without this the split kernels ship
  # with the gate green and nothing having run them. CPI_METAL_ATTN_SPLIT_MIN=1 splits at any
  # depth, so these re-runs point the same CUDA-referenced streams at pass 1 and the merge. The
  # split is exact (log-sum-exp merges), so the expected answer is unchanged: same golden.
  LABEL_EXTRA=" [split-kv]"
  export CPI_METAL_ATTN_SPLIT_MIN=1
  run_golden qwen2.5-0.5b-sky-128.txt 128 ""  qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
  run_golden qwen2.5-0.5b-longprompt-86x64.txt 64 ""  qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c
  unset CPI_METAL_ATTN_SPLIT_MIN
  LABEL_EXTRA=""
  # Qwen3: QK-norm, head_dim 128 (the 8B's attention path).
  run_golden qwen3-0.6b-sky-64.txt 64 "" Qwen3-0.6B.ll2c
  # Gemma: GeGLU, sliding window, head_dim 256 -- the scalar attention path.
  run_golden gemma-2b-capital-32.txt 32 "" gemma-2b.ll2c
  # Mixtral MoE: router + top-k + per-expert FFN. The ONLY golden whose plan is not a dense
  # MLP, and the only one that can catch a routing bug.
  run_golden tiny-mixtral-moe-24.txt 24 "" tiny-mixtral.ll2c
  # The 8B at int4: the only model that exercises a real quantized prefill.
  run_golden llama-3.1-8b-sky-48.txt 48 "4" llama8b.ll2c Llama-3.1-8B-Instruct.ll2c
fi
echo

# ---------------------------------------------------------------------------
# 3. Batched paged decode -- continuous batching's core primitive. Checks it against the
#    single-sequence path on the same weights, and that a wrong block table actually
#    changes the answer (otherwise the paged gather could be indexing by raw position).
# ---------------------------------------------------------------------------
echo "-- batched paged decode (vs the single-sequence path) --"
if [ ! -x "$BUILD/metal_batched_test" ]; then
  skip "metal_batched_test not built"
elif ! path=$(find_model qwen.ll2c Qwen2.5-0.5B-Instruct.ll2c); then
  skip "metal_batched_test -- no checkpoint"
else
  out=$("$BUILD/metal_batched_test" "$path" 2>&1)
  if echo "$out" | grep -q "\[metal_batched\] PASS"; then
    ok "metal_batched_test"
  elif echo "$out" | grep -q "\[metal_batched\] SKIP"; then
    echo "$out" | grep -q "no Metal GPU" && gpu_missing=1
    skip "metal_batched_test -- $(echo "$out" | grep '\[metal_batched\] SKIP' | head -1)"
  else
    bad "metal_batched_test"; echo "$out" | tail -8
  fi
fi
echo

echo "== $ran checks ran, $failed failed, $skipped skipped =="
if [ "$skipped" -gt 0 ]; then
  echo "   (skips are NOT passes -- a skipped golden means that path is unverified)"
fi
if [ "$gpu_missing" -eq 1 ] && [ "$REQUIRE_GPU" -eq 1 ]; then
  echo "   FAIL: --require-gpu was passed but this host has no Metal device, so nothing"
  echo "         was actually verified. Green here would be a lie."
  exit 1
fi
[ "$failed" -eq 0 ] || exit 1
