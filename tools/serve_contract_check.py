#!/usr/bin/env python3
"""End-to-end check that request fields reach the engine.

Every test in this repo checks a layer's implementation. None checks a field's
journey, and that is where the bugs were. On 2026-08-27 a json_schema request to
--serve came back as unconstrained prose, and there were four independent breaks
between the request body and the sampler: the grammar was dropped by a lambda
that declared its constraints parameter unnamed, the schema was never unwrapped
from OpenAI's envelope, --serve never chose a chat template, and min_new was
never parsed at all. Each hid the next. Every one would have failed the checks
below on the first run.

The engines were fine throughout. Request-to-engine plumbing is where things go
missing, because a field that is silently ignored produces a plausible answer.

Needs a running server and a real model, so it is not a ctest. Run it against a
server you started:

    cpi <model> --tokenizer <tok.json> --serve --port 8099
    python tools/serve_contract_check.py --url http://127.0.0.1:8099

Standard library only, matching the rest of the project.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

FAILURES = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    if ok:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAILURES.append(name)


def post(url, body, timeout=600):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def text_of(reply):
    return reply["choices"][0]["message"]["content"]


def tokens_of(reply):
    return reply.get("usage", {}).get("completion_tokens", 0)


def chat(prompt, **extra):
    body = {
        "model": "cpi",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0,
    }
    body.update(extra)
    return body


SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
    "required": ["city", "temp_c"],
}

WEATHER = "Describe the weather in Paris in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
            r.read()
    except (urllib.error.URLError, OSError) as e:
        print("cannot reach %s: %s" % (url, e))
        return 2

    # A control first. Without it, "the schema had no effect" and "the model
    # generated nothing" look identical, which is exactly how an earlier version
    # of this comparison wasted a cycle: both arms returned a single newline.
    print("chat template reaches the model")
    plain = text_of(post(url, chat(WEATHER)))
    check("plain request generates prose", len(plain.strip()) > 20, repr(plain[:60]))

    print("json_schema reaches the grammar")
    nested = text_of(
        post(
            url,
            chat(
                WEATHER,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "w", "strict": True, "schema": SCHEMA},
                },
            ),
        )
    )
    check("nested OpenAI form differs from unconstrained", nested != plain)
    ok_json, parsed = True, None
    try:
        parsed = json.loads(nested)
    except ValueError as e:
        ok_json = False
        parsed = str(e)
    check("nested OpenAI form returns valid JSON", ok_json, repr(nested[:80]))
    check(
        "nested OpenAI form obeys the schema",
        isinstance(parsed, dict) and "city" in parsed and "temp_c" in parsed,
        repr(parsed)[:80],
    )

    bare = text_of(post(url, chat(WEATHER, json_schema=SCHEMA)))
    ok_bare = True
    try:
        pb = json.loads(bare)
    except ValueError:
        ok_bare, pb = False, None
    check(
        "bare CPI form obeys the schema",
        ok_bare and isinstance(pb, dict) and "city" in pb,
        repr(bare[:80]),
    )

    print("min_new reaches the decode loop")
    short = chat("Reply with exactly the word: ok", max_tokens=80)
    n_plain = tokens_of(post(url, short))
    floor = 40
    n_floor = tokens_of(post(url, chat("Reply with exactly the word: ok", max_tokens=80,
                                       min_new=floor)))
    check("without min_new the reply is short", n_plain < 10, "tokens=%d" % n_plain)
    check("min_new holds generation past the floor", n_floor >= floor, "tokens=%d" % n_floor)

    print("stop reaches the sampler")
    # Calibrate the stop string against what this model actually said. An earlier
    # version asked it to stop at "fourteen" and failed against a correct server
    # that happened to count in digits: the check encoded an assumption about the
    # model, not about CPI. A harness other people run against arbitrary models
    # cannot assume anything about the content of a reply.
    #
    # The phrase is taken from the middle and is several words long on purpose.
    # CPI matches a stop that encodes to a SINGLE token by token id rather than as
    # text, because a marker like <|eot_id|> decodes to nothing and can only be
    # caught by id. That is deliberate, and it diverges from OpenAI, where stop is
    # a plain substring: "four" does not stop inside "fourteen", since the token
    # there is a different one. A multi-word phrase is unambiguously text-matched.
    counting = "Count from one to twenty, separated by spaces."
    unstopped = text_of(post(url, chat(counting)))
    words = unstopped.split()
    if len(words) < 8:
        check("stop: model gave too little output to calibrate against", False,
              repr(unstopped[:60]))
    else:
        mid = len(words) // 2
        phrase = " ".join(words[mid:mid + 3])
        stopped = text_of(post(url, chat(counting, stop=[phrase])))
        check("the calibration phrase is present unstopped", phrase in unstopped, repr(phrase))
        check("a stop string truncates the reply", phrase not in stopped,
              "%r still in %r" % (phrase, stopped[-60:]))
        check("the truncated reply is shorter", len(stopped) < len(unstopped),
              "%d vs %d" % (len(stopped), len(unstopped)))

    print("")
    if FAILURES:
        print("%d of %d checks FAILED: %s" % (len(FAILURES), CHECKS, ", ".join(FAILURES)))
        return 1
    print("all %d checks passed" % CHECKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
