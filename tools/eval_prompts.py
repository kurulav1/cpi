#!/usr/bin/env python3
"""
Simple quality/regression harness for llama_infer.

Runs a fixed prompt set and computes lightweight quality heuristics:
- non-empty generated text
- repetition penalty (repeated lines / repeated 3-grams)
- artifact penalty (chat/control-token leakage, replacement chars)
- printability ratio

Outputs per-prompt and aggregate scores in JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ARTIFACT_PATTERNS = [
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|system\|>",
    r"\[INST\]",
    r"<<SYS>>",
    r"�",
]


def load_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompts JSON must be a list")
    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"prompt entry {i} must be an object")
        pid = str(item.get("id", f"prompt_{i}"))
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt entry {i} missing non-empty 'prompt'")
        out.append(item | {"id": pid, "prompt": prompt.strip()})
    return out


def parse_decoded_text(stdout: str) -> str:
    marker = "Decoded text:"
    idx = stdout.rfind(marker)
    if idx < 0:
        return ""
    tail = stdout[idx + len(marker) :]
    lines = []
    for ln in tail.splitlines():
        if ln.startswith("[perf]"):
            break
        lines.append(ln)
    return "\n".join(lines).strip()


def parse_output_tokens(stdout: str) -> list[int]:
    marker = "Output tokens:"
    idx = stdout.rfind(marker)
    if idx < 0:
        return []
    line = stdout[idx + len(marker) :].splitlines()[0].strip()
    toks: list[int] = []
    for p in line.split():
        try:
            toks.append(int(p))
        except ValueError:
            pass
    return toks


def count_repeated_ngrams(text: str, n: int = 3) -> int:
    words = re.findall(r"\S+", text)
    if len(words) < n:
        return 0
    seen: dict[tuple[str, ...], int] = {}
    rep = 0
    for i in range(len(words) - n + 1):
        g = tuple(words[i : i + n])
        c = seen.get(g, 0)
        if c >= 1:
            rep += 1
        seen[g] = c + 1
    return rep


def repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return 0.0
    seen: dict[str, int] = {}
    repeats = 0
    for ln in lines:
        c = seen.get(ln, 0)
        if c >= 1:
            repeats += 1
        seen[ln] = c + 1
    return repeats / max(len(lines), 1)


def printability_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = 0
    for ch in text:
        if ch in "\n\r\t" or ch.isprintable():
            good += 1
    return good / len(text)


def artifact_count(text: str) -> int:
    cnt = 0
    for pat in ARTIFACT_PATTERNS:
        cnt += len(re.findall(pat, text))
    return cnt


def quality_score(decoded: str, checks: dict[str, Any]) -> tuple[float, dict[str, float]]:
    text = decoded.strip()
    if not text:
        return 0.0, {
            "empty_penalty": 1.0,
            "artifact_penalty": 1.0,
            "repetition_penalty": 1.0,
            "printability_penalty": 1.0,
        }

    rep_lines = repeated_line_ratio(text)
    rep_ngrams = count_repeated_ngrams(text, n=3)
    words = max(1, len(re.findall(r"\S+", text)))
    rep_ngram_ratio = min(1.0, rep_ngrams / words)
    rep_penalty = min(1.0, 0.6 * rep_lines + 0.4 * rep_ngram_ratio)

    art = artifact_count(text)
    art_penalty = min(1.0, art / 4.0)

    printable = printability_ratio(text)
    print_penalty = max(0.0, 1.0 - printable)

    words = re.findall(r"\S+", text)
    word_count = len(words)
    sentence_count = len(re.findall(r"[.!?]+", text))
    step_lines = len(re.findall(r"(?m)^\s*\d+\s*[\.\)]", text))

    keyword_penalty = 0.0
    req_keywords = checks.get("require_keywords", [])
    if isinstance(req_keywords, list) and req_keywords:
      missing = 0
      lower = text.lower()
      for kw in req_keywords:
        if not isinstance(kw, str):
          continue
        if kw.lower() not in lower:
          missing += 1
      keyword_penalty = missing / max(len(req_keywords), 1)

    steps_penalty = 0.0
    req_steps = checks.get("require_numbered_steps")
    if isinstance(req_steps, int) and req_steps > 0:
      steps_penalty = min(1.0, abs(step_lines - req_steps) / req_steps)

    length_penalty = 0.0
    min_words = checks.get("min_words")
    if isinstance(min_words, int) and min_words > 0 and word_count < min_words:
      length_penalty = max(length_penalty, min(1.0, (min_words - word_count) / min_words))
    max_words = checks.get("max_words")
    if isinstance(max_words, int) and max_words > 0 and word_count > max_words:
      length_penalty = max(length_penalty, min(1.0, (word_count - max_words) / max_words))
    max_sentences = checks.get("max_sentences")
    sentence_penalty = 0.0
    if isinstance(max_sentences, int) and max_sentences > 0 and sentence_count > max_sentences:
      sentence_penalty = min(1.0, (sentence_count - max_sentences) / max_sentences)

    score = 1.0 - (
        0.30 * rep_penalty
        + 0.20 * art_penalty
        + 0.10 * print_penalty
        + 0.20 * keyword_penalty
        + 0.15 * steps_penalty
        + 0.03 * length_penalty
        + 0.02 * sentence_penalty
    )
    score = max(0.0, min(1.0, score))
    details = {
        "empty_penalty": 0.0,
        "artifact_penalty": art_penalty,
        "repetition_penalty": rep_penalty,
        "printability_penalty": print_penalty,
        "keyword_penalty": keyword_penalty,
        "steps_penalty": steps_penalty,
        "length_penalty": length_penalty,
        "sentence_penalty": sentence_penalty,
    }
    return score, details


def run_one(
    exe: Path,
    model: Path,
    tokenizer: Path,
    prompt_item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = str(prompt_item["prompt"])
    cmd = [
        str(exe),
        str(model),
        "--prompt",
        prompt,
        "--tokenizer",
        str(tokenizer),
        "--max-new",
        str(args.max_new),
        "--temp",
        str(args.temp),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--repeat-penalty",
        str(args.repeat_penalty),
        "--no-repeat-ngram",
        str(args.no_repeat_ngram),
        "--max-context",
        str(args.max_context),
    ]
    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])
    if args.weight_quant and args.weight_quant != "none":
        cmd.extend(["--weight-quant", args.weight_quant])
    for stop in args.stop_text:
        cmd.extend(["--stop-text", stop])

    env = os.environ.copy()
    if args.python_exe:
        env["LLAMA_PYTHON_EXE"] = args.python_exe

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout_sec,
        env=env,
    )
    dt = time.perf_counter() - t0
    out = proc.stdout
    decoded = parse_decoded_text(out)
    toks = parse_output_tokens(out)
    score, detail = quality_score(decoded, prompt_item)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(dt, 4),
        "tokens_count": len(toks),
        "decoded": decoded,
        "score": round(score, 4),
        "detail": detail,
        "raw_output": out if args.keep_raw else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build/Release/llama_infer.exe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", default="tools/eval_prompts.json")
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--chat-template", default="")
    ap.add_argument("--weight-quant", choices=["none", "int8", "int4"], default="none")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--repeat-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--stop-text", action="append", default=["6."])
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--python-exe", default="")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    exe = Path(args.exe)
    model = Path(args.model)
    tokenizer = Path(args.tokenizer)
    prompts_path = Path(args.prompts)
    out_path = Path(args.out)

    if not exe.exists():
        print(f"[eval] missing exe: {exe}", file=sys.stderr)
        return 2
    if not model.exists():
        print(f"[eval] missing model: {model}", file=sys.stderr)
        return 2
    if not tokenizer.exists():
        print(f"[eval] missing tokenizer: {tokenizer}", file=sys.stderr)
        return 2
    if not prompts_path.exists():
        print(f"[eval] missing prompts file: {prompts_path}", file=sys.stderr)
        return 2

    prompts = load_prompts(prompts_path)
    results: list[dict[str, Any]] = []

    print(f"[eval] running {len(prompts)} prompts...")
    for item in prompts:
        pid = item["id"]
        prompt = item["prompt"]
        print(f"[eval] {pid}")
        row = run_one(exe, model, tokenizer, item, args)
        row["id"] = pid
        row["prompt"] = prompt
        results.append(row)
        status = "ok" if row["ok"] else "fail"
        print(f"  -> {status} score={row['score']} sec={row['seconds']} tokens={row['tokens_count']}")

    scores = [r["score"] for r in results if r["ok"]]
    aggregate = {
        "count": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
    }

    report = {
        "config": {
            "exe": str(exe),
            "model": str(model),
            "tokenizer": str(tokenizer),
            "chat_template": args.chat_template,
            "weight_quant": args.weight_quant,
            "max_new": args.max_new,
            "temp": args.temp,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "no_repeat_ngram": args.no_repeat_ngram,
            "max_context": args.max_context,
            "stop_text": args.stop_text,
        },
        "aggregate": aggregate,
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] wrote {out_path}")
    print(f"[eval] mean_score={aggregate['mean_score']} ok={aggregate['ok_count']}/{aggregate['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
