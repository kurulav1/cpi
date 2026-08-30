#!/usr/bin/env bash
# Trace invariance: does the agent loop take the same actions under every
# configuration that must not change its answers?
#
# The token-level checks compare ids. This compares what an agent actually did:
# tool name, arguments, order, and number of turns. Token identity implies trace
# identity, so a divergence here that the token checks missed means the token
# checks are not covering the path the server takes.
#
# One server at a time, on purpose: these load a model each, and running two at
# once would both distort the comparison and put two models in memory.
#
# CPI_DET_SELFTEST=<n> corrupts token <n> on the last configuration only, so the
# comparison must report a differing hash. Choose n larger than a tool call is
# long (100 works): the index counts per generation, so a small n corrupts every
# turn including the short tool calls, breaking their JSON so that no trace comes
# back at all. That is a BROKEN row, and it proves nothing.
#
# Usage: tools/determinism_trace.sh <model> <tokenizer> [cpi-binary]

set -u
MODEL="${1:?usage: determinism_trace.sh <model> <tokenizer> [binary]}"
TOK="${2:?usage: determinism_trace.sh <model> <tokenizer> [binary]}"
BIN="${3:-./build-cuda-ninja/cpi.exe}"
SELFTEST="${CPI_DET_SELFTEST:-}"
PORT="${CPI_TRACE_PORT:-8137}"
TMP="$(mktemp -d)"
trap 'stop_server; rm -rf "$TMP"' EXIT

fail=0
# Tracked apart from fail on purpose. A row that could not produce a trace at all
# is BROKEN, and BROKEN is not evidence that this comparison can detect a
# difference; only a differing hash is. The self-test below demands the latter.
diverged=0
ref=""
ref_label=""
SRV_PID=""

stop_server() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
  fi
  # The harness kills by pid, but a model process that outlives its parent would
  # hold the GPU and quietly change the next configuration's numbers.
  ( powershell.exe -NoProfile -Command "Get-Process cpi -ErrorAction SilentlyContinue | Stop-Process -Force" ) >/dev/null 2>&1
  # Killing the process is not the same as getting the VRAM back, and the gap is
  # long enough to matter: a server started too soon finds too little free memory,
  # silently declines to hold the weights resident, and then refuses to run batched
  # decode at all. That surfaced as one configuration BROKEN in one run and fine in
  # the next, which is a harness artifact wearing the costume of a finding. Wait for
  # the GPU to actually release before starting anything else.
  local w
  for w in $(seq 1 40); do
    local apps
    apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null)
    if ! printf '%s' "$apps" | grep -qi 'cpi'; then
      break
    fi
    sleep 2
  done
  sleep 3
}

# start_server <perturb> <flags...>
start_server() {
  local perturb="$1"; shift
  stop_server
  CPI_DET_PERTURB="$perturb" "$BIN" "$MODEL" --tokenizer "$TOK" --serve --port "$PORT" \
    --max-new 200 "$@" > "$TMP/server.log" 2>&1 &
  SRV_PID=$!
  local i
  for i in $(seq 1 100); do
    sleep 3
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "  server exited during startup:"
      grep -i 'fatal\|error' "$TMP/server.log" | head -2 | sed 's/^/    /'
      if grep -qi 'requires --gpu-cache-all' "$TMP/server.log"; then
        echo "    (this usually means the previous server had not released its VRAM;"
        echo "     free memory now: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null))"
      fi
      return 1
    fi
  done
  echo "  server never became ready"
  tail -4 "$TMP/server.log" | sed 's/^/    /'
  return 1
}

# trace <label> <expect-runtime> [extra verify_trace args...]
trace() {
  local label="$1" expect="$2"; shift 2
  local out hash
  out=$(python tools/verify_trace.py --url "http://127.0.0.1:$PORT" --quiet \
          --expect-runtime "$expect" "$@" 2>&1)
  hash=$(printf '%s' "$out" | sed -n 's/.*hash=\([0-9a-f]*\).*/\1/p')
  if [ -z "$hash" ]; then
    # No hash means the trace did not run, which is a broken comparison rather
    # than a passing one. Two runs that both produce nothing compare equal.
    printf '  %-30s BROKEN\n' "$label"
    printf '%s\n' "$out" | sed 's/^/      /' | head -4
    fail=1
    return
  fi
  if [ -z "$ref" ]; then
    ref="$hash"; ref_label="$label"
    printf '  %-30s %s  reference\n' "$label" "$hash"
    return
  fi
  if [ "$hash" = "$ref" ]; then
    printf '  %-30s %s  identical\n' "$label" "$hash"
  else
    printf '  %-30s %s  DIVERGES from %s\n' "$label" "$hash" "$ref_label"
    fail=1
    diverged=1
  fi
}

echo "=== one server, varying only what is in flight ==="
if start_server "" --paged-blocks --gpu-cache-all; then
  trace "baseline"                 "backend=llama-cuda"
  trace "same server, again"       "backend=llama-cuda"
  trace "5 traces concurrently"    "backend=llama-cuda" --concurrent 5
else
  echo "  BROKEN: baseline server did not start"; fail=1
fi

echo ""
echo "=== settings that must not change the answer ==="
if start_server "" --paged-blocks; then
  trace "no gpu cache"             "gpu_cache_all=0"
else
  echo "  BROKEN: no-gpu-cache server did not start"; fail=1
fi

if start_server "" --paged-blocks --gpu-cache-all --max-context 4096; then
  trace "ctx 4096"                 "ctx=4096"
else
  echo "  BROKEN: ctx-4096 server did not start"; fail=1
fi

echo ""
echo "=== a different backend ==="
if start_server "" --cpu --paged-blocks; then
  trace "cpu engine"               "backend=llama-cpu"
else
  echo "  BROKEN: cpu server did not start"; fail=1
fi

if [ -n "$SELFTEST" ]; then
  echo ""
  echo "=== self-test: a corrupted server must be caught ==="
  if start_server "$SELFTEST" --paged-blocks --gpu-cache-all; then
    trace "corrupted at token $SELFTEST" "backend=llama-cuda"
  else
    echo "  BROKEN: corrupted server did not start"; fail=1
  fi
fi

echo ""
if [ -n "$SELFTEST" ]; then
  # Only a differing hash proves this comparison detects anything. Accepting any
  # failure would accept a BROKEN row, and BROKEN means the trace could not be
  # produced, which says nothing either way. That mistake was made here first:
  # corrupting token 10 lands inside turn 0's tool call, breaks its JSON, and
  # yields no trace at all, which this script briefly counted as a pass.
  if [ "$diverged" -eq 0 ]; then
    echo "SELF-TEST INCONCLUSIVE: no row reported a differing hash."
    if [ "$fail" -ne 0 ]; then
      echo "A row is BROKEN, which is not the same thing. Corrupting a token inside a"
      echo "short tool call destroys its JSON, so no trace comes back and the"
      echo "comparison never runs. Choose an index that lands in a long generation"
      echo "(the final answer) so the trace stays valid and merely differs."
    else
      echo "The corrupted server produced the reference trace, so this comparison is"
      echo "not detecting anything."
    fi
    exit 1
  fi
  echo "(self-test: the divergence above is expected)"
  exit 0
fi
if [ "$fail" -eq 0 ]; then
  echo "PASS: the agent loop took the same actions under every configuration."
else
  echo "FAIL: a configuration changed what the agent did."
fi
exit "$fail"
