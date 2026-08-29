#!/usr/bin/env python3
"""Structured-output conformance: does every required tool call come back usable?

Runs N tool calls against a deliberately awkward nested schema and checks each
reply the way a caller would: parse the arguments, then verify them against the
schema that was requested. It counts what is actually wrong rather than reporting
a pass rate over things that were never checked.

It speaks the OpenAI API, so it runs against any server that does, and the
comparison is the point. Run it against CPI, then against llama-server or another
engine, with the same N.

    cpi <model.gguf> --serve --port 8080
    python tools/tool_conformance.py --url http://127.0.0.1:8080 --n 1000

--transcript writes one line per call, so two runs can be diffed byte for byte.
That is how "identical on every backend" gets checked rather than asserted: run it
on CUDA and on Metal and diff the files.

Standard library only.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Nested on purpose. Flat schemas are the easy case; the failures reported in this
# area involve nested objects, arrays, and enums inside them.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "File a ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"enum": ["low", "medium", "high"]},
                    "reporter": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "team": {"enum": ["infra", "ml", "web"]},
                        },
                        "required": ["name", "team"],
                    },
                    # Bounded on purpose. An unbounded array is a loop a greedy
                    # decoder can sit in, and an agent schema that cannot say "at
                    # most four" invites a runaway that reads as truncation.
                    "labels": {"type": "array", "items": {"type": "string"},
                               "minItems": 1, "maxItems": 4},
                },
                "required": ["title", "priority", "reporter", "labels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a measurement",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"enum": ["c", "f", "k"]},
                    "to_unit": {"enum": ["c", "f", "k"]},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

TOOL_NAMES = set(t["function"]["name"] for t in TOOLS)
SCHEMAS = dict((t["function"]["name"], t["function"]["parameters"]) for t in TOOLS)

# Prompts are generated rather than listed. At temperature 0 a fixed list makes N
# calls into a handful of distinct cases repeated: an early run of 1000 produced
# only 8 distinct payloads, which is a much weaker claim than the count suggests.
# Varying the nouns and numbers keeps the schema fixed while the content differs,
# so the grammar is exercised against many different strings and numbers.
_CITIES = ["Lisbon", "Oslo", "Tampere", "Kyoto", "Bogota", "Perth", "Cork", "Ghent",
           "Nantes", "Bergen", "Utrecht", "Aarhus", "Leeds", "Porto", "Vaasa"]
_ISSUES = ["disk filling up on node {n}", "flaky login test on shard {n}",
           "training job stalls at epoch {n}", "cache eviction storm in region {n}",
           "queue backlog on worker {n}", "checkpoint upload times out after {n} minutes",
           "gpu {n} drops off the bus", "index rebuild loops on partition {n}"]
_PEOPLE = ["Ana", "Bo", "Chen", "Dee", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jo"]
_TEAMS = ["infra", "ml", "web"]
_PRIOS = ["low", "medium", "high"]
_UNITS = ["celsius", "fahrenheit", "kelvin"]


def make_prompt(i):
    """Deterministic in i, so two runs are comparable, but varied across i."""
    if i % 2 == 0:
        issue = _ISSUES[(i // 2) % len(_ISSUES)].format(n=(i % 37) + 1)
        who = _PEOPLE[(i // 3) % len(_PEOPLE)]
        team = _TEAMS[(i // 5) % len(_TEAMS)]
        prio = _PRIOS[(i // 7) % len(_PRIOS)]
        return ("Open a %s priority ticket about the %s, reported by %s on the %s team."
                % (prio, issue, who, team))
    value = (i % 97) + 1
    src = _UNITS[(i // 2) % len(_UNITS)]
    dst = _UNITS[((i // 2) + 1 + (i % 2)) % len(_UNITS)]
    city = _CITIES[(i // 4) % len(_CITIES)]
    return "In %s it is %d degrees %s. Convert that to %s." % (city, value, src, dst)


def validate(value, schema, path="args"):
    """Checks value against the subset of JSON Schema used above.

    Returns a list of readable violations; empty means conformant.
    """
    out = []
    if "enum" in schema:
        if value not in schema["enum"]:
            out.append("%s = %r is not one of %r" % (path, value, schema["enum"]))
        return out
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return ["%s is %s, expected object" % (path, type(value).__name__)]
        for key in schema.get("required", []):
            if key not in value:
                out.append("%s is missing required %s" % (path, key))
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                out.extend(validate(value[key], sub, path + "." + key))
        for key in value:
            if key not in props:
                out.append("%s has unexpected key %s" % (path, key))
    elif t == "array":
        if not isinstance(value, list):
            return ["%s is %s, expected array" % (path, type(value).__name__)]
        for i, item in enumerate(value):
            out.extend(validate(item, schema.get("items", {}), "%s[%d]" % (path, i)))
    elif t == "string":
        if not isinstance(value, str):
            out.append("%s is %s, expected string" % (path, type(value).__name__))
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out.append("%s is %s, expected number" % (path, type(value).__name__))
    elif t == "boolean":
        if not isinstance(value, bool):
            out.append("%s is %s, expected boolean" % (path, type(value).__name__))
    return out


def call(url, prompt, timeout, max_tokens, use_tools=True):
    body = {
        "model": "cpi",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    # --no-tools sends the same prompts with no grammar at all. Subtracting that
    # run from the constrained one attributes wall time to generation plus HTTP
    # versus grammar work, without guessing where to put a timer.
    if use_tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "required"
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--timeout", type=int, default=300)
    # Enough room to finish a nested call. The grammar permits whitespace between
    # every token, so a well-formed reply costs more tokens than its text suggests,
    # and a cap that is too low shows up as truncation rather than a defect.
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--transcript", default="")
    ap.add_argument("--no-tools", action="store_true",
                    help="send the prompts unconstrained; for timing decomposition")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    order = ["ok", "truncated", "no_tool_call", "unknown_tool", "arguments_not_json",
             "schema_violation", "http_error"]
    counts = dict((k, 0) for k in order)
    # Wall time is only comparable per token: an unconstrained reply is prose of a
    # different length, so a raw A/B between constrained and free running compares
    # two different amounts of work.
    total_completion_tokens = 0
    examples = {}
    transcript = open(args.transcript, "w", encoding="utf-8") if args.transcript else None
    t0 = time.time()

    for i in range(args.n):
        prompt = make_prompt(i)
        try:
            reply = call(url, prompt, args.timeout, args.max_tokens, not args.no_tools)
        except Exception as e:  # transport, HTTP status, malformed envelope
            counts["http_error"] += 1
            examples.setdefault("http_error", str(e)[:160])
            continue

        total_completion_tokens += int(reply.get("usage", {}).get("completion_tokens", 0) or 0)
        choice = reply.get("choices", [{}])[0]
        msg = choice.get("message", {})
        calls = msg.get("tool_calls") or []
        # Truncation is the caller's max_tokens, not malformed output, so it is
        # counted apart from a server that answered without calling a tool.
        if choice.get("finish_reason") == "length":
            counts["truncated"] += 1
            examples.setdefault("truncated", repr(msg.get("content"))[:120])
            continue
        if args.no_tools:
            counts["ok"] += 1  # timing run: content is expected, not a tool call
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.2f calls/s" % (i + 1, args.n,
                                                 (i + 1) / max(time.time() - t0, 1e-9)), flush=True)
            continue
        if not calls:
            counts["no_tool_call"] += 1
            examples.setdefault("no_tool_call", repr(msg.get("content"))[:160])
            continue

        fn = calls[0].get("function", {})
        name = fn.get("name")
        raw_args = fn.get("arguments", "")
        if transcript:
            transcript.write(
                json.dumps({"i": i, "name": name, "arguments": raw_args}, sort_keys=True) + "\n")
        if name not in TOOL_NAMES:
            counts["unknown_tool"] += 1
            examples.setdefault("unknown_tool", repr(name)[:160])
            continue
        try:
            parsed = json.loads(raw_args)
        except Exception as e:
            counts["arguments_not_json"] += 1
            examples.setdefault("arguments_not_json",
                                "%s | %s" % (str(e)[:60], repr(raw_args)[:100]))
            continue
        problems = validate(parsed, SCHEMAS[name])
        if problems:
            counts["schema_violation"] += 1
            examples.setdefault("schema_violation", "; ".join(problems)[:160])
            continue
        counts["ok"] += 1

        if (i + 1) % 50 == 0:
            done = i + 1
            rate = done / max(time.time() - t0, 1e-9)
            print("  %d/%d  ok=%d  %.2f calls/s" % (done, args.n, counts["ok"], rate), flush=True)

    if transcript:
        transcript.close()

    elapsed = time.time() - t0
    bad = args.n - counts["ok"]
    print("")
    print("conformance against %s" % url)
    print("  calls          %d in %.1fs" % (args.n, elapsed))
    print("  tokens         %d generated, %.2f ms/token, %.1f tok/s"
          % (total_completion_tokens,
             1000.0 * elapsed / max(total_completion_tokens, 1),
             total_completion_tokens / max(elapsed, 1e-9)))
    for k in order:
        line = "  %-18s %d" % (k, counts[k])
        if k in examples:
            line += "   e.g. %s" % examples[k]
        print(line)
    print("")
    if bad == 0:
        print("PASS: %d/%d usable tool calls" % (counts["ok"], args.n))
        return 0
    print("FAIL: %d of %d calls were not usable" % (bad, args.n))
    return 1


if __name__ == "__main__":
    sys.exit(main())
