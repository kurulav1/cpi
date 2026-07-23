#!/usr/bin/env python3
"""Interactive Gemma chat REPL.

Runs `cpi --interactive` as a resident subprocess (the model loads once),
applies the Gemma 4 turn template (<|turn>user … <turn|> <|turn>model), and
streams the reply token-by-token. Type a question, press enter; /quit to exit.

Usage:
  python tools/gemma_chat.py \
      --model artifacts/hub/google__gemma-4-12B-it/gemma4-12b.cpi \
      --tokenizer artifacts/hub/google__gemma-4-12B-it/hf/tokenizer.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--infer-bin", default="build/Release/cpi.exe")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    infer = Path(args.infer_bin)
    if not infer.is_absolute():
        infer = (Path.cwd() / infer).resolve()
    if not infer.exists():
        sys.exit(f"[gemma_chat] cpi not found at {infer} — pass --infer-bin or run from the repo root")

    proc = subprocess.Popen(
        [str(infer), args.model, "--tokenizer", args.tokenizer, "--interactive",
         "--max-new", str(args.max_new), "--temp", str(args.temp)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    print("[gemma_chat] loading the model (first reply may take a while)…  /quit to exit\n",
          file=sys.stderr, flush=True)

    rid = 0
    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user in ("/quit", "/exit", "/q"):
                break
            rid += 1
            # Gemma 4 turn format; the tokenizer prepends <bos> (add_bos).
            prompt = f"<|turn>user\n{user}<turn|>\n<|turn>model\n"
            req = {"id": str(rid), "prompt": prompt, "max_new": args.max_new,
                   "temp": args.temp, "add_bos": True, "stop_texts": ["<turn|>"]}
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            sys.stdout.write("gemma> ")
            sys.stdout.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    print("\n[gemma_chat] model process ended.")
                    return
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "delta":
                    sys.stdout.write(ev.get("delta", ""))
                    sys.stdout.flush()
                elif t == "done":
                    print("\n")
                    break
                elif t == "error":
                    print("\n[error]", ev.get("error"))
                    break
    finally:
        try:
            proc.stdin.write('{"shutdown":true}\n')
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.terminate()


if __name__ == "__main__":
    main()
