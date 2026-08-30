#!/usr/bin/env python3
"""Trace-level determinism: does an agent loop take the same actions every time?

--verify-determinism proves the token stream repeats. Nobody running an agent
thinks in tokens; what they need to know is whether the second run calls the same
tools, with the same arguments, in the same order, over the same number of turns.
Token identity implies that, so this is not a new property. It is the same
property expressed in the units of the thing people are afraid of, and it is the
first check that exercises grammar state, prefix reuse across turns, growing
context and the stop rules at once.

Usage:
  python tools/verify_trace.py --url http://127.0.0.1:8080
  python tools/verify_trace.py --url ... --concurrent 5   # traces in flight together
  python tools/verify_trace.py --url ... --expect-runtime backend=llama-cpu

Prints a hash of the canonical trace plus one line per turn. Exit 0 on a usable
trace, 1 if the trace could not be produced or a guard failed.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request

# Canonical form for everything that gets compared or hashed. Sorted keys and
# fixed separators, so a difference in the hash is a difference in content and
# never in formatting.
def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def post(url, payload, timeout=600):
    data = canon(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stub_result(spec, name, args):
    """The tool's answer. A pure function of (name, args): no clock, no random
    source, no filesystem, no iteration over an unordered container."""
    s = spec["stubs"].get(name)
    if s is None:
        return {"error": "no such tool: " + str(name)}
    out = dict(s["constant"])
    for key in s.get("echo", []):
        if isinstance(args, dict) and key in args:
            out[key] = args[key]
    return out


def check_stub_purity(spec, calls):
    """Evaluate every stub twice and require the same answer both times.

    The reference data says the stubs are deterministic. This checks it, because
    a stub that varied would make every trace differ and the harness would be
    reporting itself rather than the engine.
    """
    for c in calls:
        a = canon(stub_result(spec, c["name"], c.get("args_obj")))
        b = canon(stub_result(spec, c["name"], copy.deepcopy(c.get("args_obj"))))
        if a != b:
            return "stub for %s is not deterministic: %s != %s" % (c["name"], a, b)
    return None


def run_trace(base, spec, max_tokens=None, tag="target"):
    """Drive the loop to completion and return the canonical trace records."""
    messages = [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": spec["user"]},
    ]
    records = []
    calls = []
    ended = "max_turns"
    for turn in range(spec["max_turns"]):
        payload = {
            "model": "cpi",
            "messages": messages,
            "tools": spec["tools"],
            "tool_choice": (spec.get("tool_choice_first", "required")
                            if turn < spec.get("forced_turns", 1)
                            else spec.get("tool_choice_rest", "auto")),
            "temperature": 0,
            "max_tokens": max_tokens or spec["max_tokens_per_turn"],
            "seed": 0,
        }
        resp = post(base + "/v1/chat/completions", payload)
        choice = resp["choices"][0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # A turn with no tool call ends the loop. The text is recorded because
            # an agent that answers instead of acting has behaved differently, but
            # it is kept in its own field so a divergence can be attributed to
            # wording rather than to a different action.
            records.append(
                {
                    "turn": turn,
                    "kind": "final",
                    "content": (msg.get("content") or "").strip(),
                    "finish_reason": choice.get("finish_reason"),
                }
            )
            ended = "final"
            break

        messages.append({k: v for k, v in msg.items() if v is not None})
        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            try:
                args_obj = json.loads(raw) if isinstance(raw, str) else raw
                args_canon = canon(args_obj)
                parse_ok = True
            except (ValueError, TypeError):
                # Unparseable arguments are a finding, not a crash: record them
                # verbatim so two runs can still be compared.
                args_obj, args_canon, parse_ok = None, repr(raw), False

            rec = {
                "turn": turn,
                "index": idx,
                "kind": "tool_call",
                "name": name,
                "arguments": args_canon,
                "arguments_parsed": parse_ok,
            }
            records.append(rec)
            calls.append({"name": name, "args_obj": args_obj})

            result = stub_result(spec, name, args_obj)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or ("call_%d_%d" % (turn, idx)),
                    "content": canon(result),
                }
            )
    return {"records": records, "calls": calls, "ended": ended, "tag": tag}


def summarize(trace):
    lines = []
    for r in trace["records"]:
        if r["kind"] == "tool_call":
            flag = "" if r["arguments_parsed"] else "  [arguments not JSON]"
            lines.append(
                "  turn %d.%d  %-18s %s%s" % (r["turn"], r["index"], r["name"], r["arguments"], flag)
            )
        else:
            text = r["content"].replace("\n", " ")
            if len(text) > 60:
                text = text[:57] + "..."
            lines.append('  turn %d    final              "%s"' % (r["turn"], text))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--spec", default=os.path.join(os.path.dirname(__file__), "traces",
                                                   "reference_trace.json"))
    ap.add_argument("--concurrent", type=int, default=1,
                    help="run this many traces at once; the first is the one hashed")
    ap.add_argument("--expect-runtime", default="",
                    help="substring /health must report, e.g. backend=llama-cpu")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--json", default="", help="write the canonical trace here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Guard: the server must be the one the caller thinks it is. Comparing two
    # configurations that were secretly the same configuration is how a check
    # reports agreement it never tested.
    try:
        health = get(base + "/health")
    except (urllib.error.URLError, OSError) as e:
        print("FAIL: cannot reach %s/health: %s" % (base, e))
        return 1
    runtime = health.get("runtime", "")
    if not args.quiet:
        print("server: %s" % (runtime or "<no runtime reported>"))
    if args.expect_runtime:
        if args.expect_runtime not in runtime:
            print("FAIL: server reports runtime %r, which does not contain %r"
                  % (runtime, args.expect_runtime))
            print("      Any comparison against this run would be against the wrong config.")
            return 1

    # Concurrency: identical traces in flight beside the one being measured, so
    # batch composition is varied without varying the request under test.
    results = {}
    def worker(i):
        try:
            results[i] = run_trace(base, spec, args.max_tokens or None, "t%d" % i)
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            results[i] = {"error": "%s: %s" % (type(e).__name__, e)}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(max(1, args.concurrent))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    trace = results.get(0)
    if trace is None or "error" in trace:
        print("FAIL: the trace did not run: %s"
              % (trace.get("error") if trace else "no result"))
        return 1

    # Guard: an empty or trivial trace hashes to something stable and would read
    # as perfect agreement. Three hashes of nothing compared equal here once.
    n_calls = sum(1 for r in trace["records"] if r["kind"] == "tool_call")
    if not trace["records"]:
        print("FAIL: the trace is empty, so its hash means nothing.")
        return 1
    if n_calls < spec.get("min_tool_calls", 1):
        print("FAIL: %d tool call(s), fewer than the %d this trace is supposed to make."
              % (n_calls, spec.get("min_tool_calls", 1)))
        print("      A trace with no actions in it cannot show that actions repeat.")
        return 1

    impure = check_stub_purity(spec, trace["calls"])
    if impure:
        print("FAIL: %s" % impure)
        print("      Every trace would differ and the harness would be measuring itself.")
        return 1

    body = canon(trace["records"])
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]

    print("[trace] hash=%s turns=%d tool_calls=%d ended=%s concurrent=%d"
          % (digest, len(trace["records"]), n_calls, trace["ended"], max(1, args.concurrent)))
    if not args.quiet:
        for line in summarize(trace):
            print(line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
