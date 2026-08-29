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
# Usage: tools/determinism_batch.sh <model> <tokenizer> [cpi-binary]

set -u
MODEL="${1:?usage: determinism_batch.sh <model> <tokenizer> [binary]}"
TOK="${2:?usage: determinism_batch.sh <model> <tokenizer> [binary]}"
BIN="${3:-./build-cuda-ninja/cpi.exe}"
N=64

# Pull out the target sequence only, and hash its text. Sequences are
# interleaved in the output stream, so they have to be reassembled by id.
extract_target() {
  python -c '
import sys, json, hashlib
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
text = acc.get("TARGET", "")
if not text:
    print("EMPTY-NO-OUTPUT")
else:
    print(hashlib.sha256(text.encode()).hexdigest()[:12])
'
}

requests() {
  python -c '
import json, sys
n = int(sys.argv[1])
print(json.dumps({"id": "TARGET", "prompt": "The capital of France is", "max_new": '"$N"'}))
for k in range(1, n):
    print(json.dumps({"id": "filler%d" % k, "prompt": "Count from %d to 9." % k, "max_new": '"$N"'}))
' "$1"
}

echo "batch	hash"
for b in 1 2 3 5 8; do
  h=$(requests "$b" | "$BIN" "$MODEL" --tokenizer "$TOK" --interactive-batch \
        --paged-blocks --temp 0 --gpu-cache-all 2>/dev/null | extract_target)
  printf '%s\t%s\n' "$b" "$h"
done

echo ""
echo "All hashes must match. EMPTY-NO-OUTPUT means the run produced nothing,"
echo "which is a broken test rather than a passing one: check stderr."
echo "Set CPI_DET_BATCH=0 to see the pre-fix behaviour, where batch 1 differs."
