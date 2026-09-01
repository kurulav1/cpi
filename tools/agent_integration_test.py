#!/usr/bin/env python3
"""Drive a real agent framework against CPI and see whether the run repeats.

Everything else in tools/ is a harness I wrote testing an engine I wrote, which is
the weakest kind of evidence: the trace check uses my own loop, my own prompt shape,
my own idea of what a tool call looks like. This one hands the wheel to the OpenAI
Agents SDK. It decides the message shapes, the tool schemas, when to stop, and how
to parse what comes back, and none of that is mine.

The task is real rather than a toy: find which log file has the most ERROR lines and
write a summary. Three to five tool turns depending on how the model goes about it.

Determinism of the ENVIRONMENT comes first, because a tool that answers differently
between runs makes the engine look non-deterministic and there is no way to tell the
two apart afterwards. So: a sandbox rebuilt byte-identically before every run, file
listings sorted explicitly rather than taken in directory order, no clock, no
network, no random ids, and no reading of anything outside the sandbox.

Usage:
  python tools/agent_integration_test.py --url http://127.0.0.1:8081/v1 --tag run1
  python tools/agent_integration_test.py --url ... --tag divergence --alt-prompt
"""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import shutil
import sys

SANDBOX = pathlib.Path(os.environ.get("CPI_AGENT_SANDBOX", "")) or None

# The sandbox contents, spelled out here rather than generated, so the fixture is a
# constant of this file and not of whatever the filesystem happened to contain.
FIXTURE = {
    "alpha.log": (
        "INFO service started\n"
        "ERROR disk read failed\n"
        "INFO retry scheduled\n"
        "ERROR disk read failed\n"
        "WARN slow response\n"
    ),
    "beta.log": (
        "INFO cache warm\n"
        "WARN evicting entries\n"
        "INFO cache warm\n"
    ),
    "gamma.log": (
        "ERROR auth rejected\n"
        "ERROR auth rejected\n"
        "ERROR token expired\n"
        "INFO session closed\n"
    ),
}
# gamma.log has 3 ERROR lines, alpha 2, beta 0. The right answer is gamma.log,
# and it is only reachable by reading more than one file.
EXPECTED_ANSWER_SUBSTR = "gamma"


def reset_sandbox(root: pathlib.Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name, body in FIXTURE.items():
        (root / name).write_text(body, encoding="utf-8", newline="\n")


def sandbox_digest(root: pathlib.Path) -> str:
    """Proof the environment really was identical, rather than an assumption."""
    h = hashlib.sha256()
    for name in sorted(p.name for p in root.iterdir() if p.is_file()):
        h.update(name.encode())
        h.update(b"\0")
        h.update((root / name).read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def build_tools():
    from agents import function_tool

    @function_tool
    def list_log_files() -> str:
        """List the log files available in the working directory."""
        # sorted(), not iterdir() order: directory order is not guaranteed stable and
        # would make the engine look non-deterministic when the filesystem moved.
        names = sorted(p.name for p in SANDBOX.iterdir() if p.suffix == ".log")
        return json.dumps(names)

    @function_tool
    def read_log_file(filename: str) -> str:
        """Read the full contents of one log file."""
        p = SANDBOX / pathlib.Path(filename).name
        if not p.exists():
            return json.dumps({"error": "no such file", "filename": filename})
        return json.dumps({"filename": p.name, "contents": p.read_text(encoding="utf-8")})

    @function_tool
    def count_error_lines(filename: str) -> str:
        """Count how many lines in a log file start with ERROR."""
        p = SANDBOX / pathlib.Path(filename).name
        if not p.exists():
            return json.dumps({"error": "no such file", "filename": filename})
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                if line.startswith("ERROR"))
        return json.dumps({"filename": p.name, "error_lines": n})

    @function_tool
    def write_summary(text: str) -> str:
        """Write the final one-line summary to summary.txt."""
        (SANDBOX / "summary.txt").write_text(text.strip() + "\n", encoding="utf-8")
        return json.dumps({"written": True, "bytes": len(text.strip()) + 1})

    return [list_log_files, read_log_file, count_error_lines, write_summary]


TASK = (
    "Work out which of the log files has the most ERROR lines. "
    "Then write a one-line summary naming that file and its ERROR count using write_summary."
)
# Used only by the deliberate-divergence control: a different question that should
# produce a different trace. If the comparison cannot tell these apart it is not
# comparing anything.
TASK_ALT = (
    "Work out which of the log files has the FEWEST ERROR lines. "
    "Then write a one-line summary naming that file and its ERROR count using write_summary."
)


def extract_trace(result) -> dict:
    """What the agent DID: tool names, canonical arguments, order."""
    from agents.items import ToolCallItem

    steps = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = getattr(raw, "name", None)
            args = getattr(raw, "arguments", None)
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
                canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
                ok = True
            except (ValueError, TypeError):
                canon, ok = repr(args), False
            steps.append({"tool": name, "args": canon, "args_parsed": ok})
    return {"steps": steps, "final": (result.final_output or "").strip()}


async def run_once(base_url: str, api_key: str, model: str, use_alt: bool, tool_choice: str = "", temperature: float = 0.0):
    from agents import Agent, Runner, set_tracing_disabled
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    # Tracing off: it phones home to OpenAI, which is both a network dependency and
    # a thing this test has no business doing.
    set_tracing_disabled(True)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    # The SDK talks the Responses API by default, which CPI does not implement.
    # OpenAIChatCompletionsModel is what points it at /v1/chat/completions.
    agent = Agent(
        name="log triage",
        instructions=(
            "You are a log triage assistant. Use the provided tools to inspect log files. "
            "Call one tool at a time. When you know the answer, write the summary and stop."
        ),
        tools=build_tools(),
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
    )
    # temperature is pinned here, and that is a finding rather than housekeeping.
    # The SDK sends no temperature at all unless asked, so the effective value is
    # whatever the server defaults to, and CPI's CLI default is 0.8. A framework
    # pointed at CPI therefore SAMPLES by default: the deterministic path exists but
    # is not the one you get. Three identical runs diverged before this line existed.
    from agents import ModelSettings
    agent.model_settings = ModelSettings(temperature=temperature)
    if tool_choice:
        # A standard framework sends `tools` and leaves tool_choice unset. CPI reads
        # that as "auto", which compiles to no grammar, and a 1B model then narrates a
        # fake tool session instead of emitting a call. Setting it explicitly is what
        # separates "the loop cannot work here" from "the loop needs one field set".
        # The SDK resets it to auto after the first call (Agent.reset_tool_choice), so
        # this forces the opening move rather than every move.
        agent.model_settings = ModelSettings(temperature=temperature, tool_choice=tool_choice)
    result = await Runner.run(agent, TASK_ALT if use_alt else TASK, max_turns=10)
    return result


def main() -> int:
    global SANDBOX
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", default="cpi")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tool-choice", default="",
                    help="force tool_choice (required/auto); unset means the framework default")
    ap.add_argument("--alt-prompt", action="store_true",
                    help="a different question; the deliberate-divergence control")
    args = ap.parse_args()

    SANDBOX = pathlib.Path(args.sandbox or (pathlib.Path(os.environ["TEMP"]) / "cpi-agent-sandbox"))
    reset_sandbox(SANDBOX)
    before = sandbox_digest(SANDBOX)

    try:
        result = asyncio.run(run_once(args.url, args.api_key, args.model, args.alt_prompt, args.tool_choice, args.temperature))
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        print("[%s] FRAMEWORK ERROR: %s: %s" % (args.tag, type(e).__name__, e))
        return 2

    trace = extract_trace(result)
    trace["sandbox_before"] = before
    trace["task"] = "alt" if args.alt_prompt else "main"
    # Whether it COMPLETED is a separate question from whether it repeats, and a
    # repeat of a failed run is worth nothing.
    summary_path = SANDBOX / "summary.txt"
    trace["summary_written"] = summary_path.exists()
    trace["summary"] = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else ""
    correct_target = "beta" if args.alt_prompt else EXPECTED_ANSWER_SUBSTR
    trace["task_completed"] = bool(
        trace["summary_written"] and correct_target in trace["summary"].lower()
    )

    body = json.dumps({"steps": trace["steps"], "final": trace["final"]},
                      sort_keys=True, separators=(",", ":"))
    trace["trace_hash"] = hashlib.sha256(body.encode()).hexdigest()[:16]

    print("[%s] trace_hash=%s steps=%d completed=%s"
          % (args.tag, trace["trace_hash"], len(trace["steps"]), trace["task_completed"]))
    for i, s in enumerate(trace["steps"]):
        print("   %d. %-18s %s" % (i, s["tool"], s["args"][:80]))
    print("   summary: %s" % (trace["summary"] or "<none written>"))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
