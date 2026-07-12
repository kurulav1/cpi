# Does the wide GEMV change what the models SAY?
#
# The wide GEMV (int4 loads, one warp per row) accumulates each row in a different order
# than the tiled kernel, so it is deliberately NOT bit-identical -- fp32 sums are
# order-dependent. The question that actually matters is whether that last-bit difference
# ever changes a greedy argmax. Run every model both ways and diff the decoded text.
#
#   LLAMA_INFER_TILED_GEMV=1 -> old tiled kernel (half2 loads)
#   LLAMA_INFER_TILED_GEMV=0 -> new wide kernel (int4 loads)

import os
import subprocess
import sys

HUB = "artifacts/hub"
EXE = os.path.abspath("build/Release/llama_infer.exe")
PROMPT = "Explain in two sentences why the sky is blue."
MAX_NEW = 96

MODELS = [
    ("Qwen2.5-0.5B", f"{HUB}/Qwen__Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.ll2c",
     f"{HUB}/Qwen__Qwen2.5-0.5B-Instruct/hf/tokenizer.json"),
    ("Qwen3-0.6B", f"{HUB}/Qwen__Qwen3-0.6B/Qwen3-0.6B.ll2c",
     f"{HUB}/Qwen__Qwen3-0.6B/hf/tokenizer.json"),
    ("Qwen2.5-Coder-3B", f"{HUB}/Qwen__Qwen2.5-Coder-3B-Instruct/Qwen2.5-Coder-3B-Instruct.ll2c",
     f"{HUB}/Qwen__Qwen2.5-Coder-3B-Instruct/hf/tokenizer.json"),
    ("gemma-2b", f"{HUB}/unsloth__gemma-2b/gemma-2b.ll2c",
     f"{HUB}/unsloth__gemma-2b/hf/tokenizer.json"),
    ("Llama-3.1-8B", f"{HUB}/meta-llama__Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct.ll2c",
     f"{HUB}/meta-llama__Llama-3.1-8B-Instruct/hf/tokenizer.json"),
    ("gemma-4-E2B", f"{HUB}/google__gemma-4-E2B-it/gemma4-e2b.cpi",
     f"{HUB}/google__gemma-4-E2B-it/hf/tokenizer.json"),
]


def run(weights, tokenizer, tiled):
    env = dict(os.environ, LLAMA_INFER_TILED_GEMV="1" if tiled else "0")
    out = subprocess.run(
        [EXE, weights, "--tokenizer", tokenizer, "--prompt", PROMPT,
         "--max-new", str(MAX_NEW), "--gpu-cache-all"],
        capture_output=True, text=True, env=env, timeout=900, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        return None
    # Drop ALL bracketed diagnostic lines ([perf], [startup], ...) -- they carry wall-clock
    # timings that differ between every pair of runs, and comparing them would make the gate
    # scream about a difference it created itself.
    return "\n".join(l for l in out.stdout.splitlines() if not l.startswith("[")).strip()


def main():
    same = diff = skipped = 0
    for name, w, t in MODELS:
        if not (os.path.exists(w) and os.path.exists(t)):
            print(f"  SKIP  {name:20} (not converted)")
            skipped += 1
            continue
        old = run(w, t, tiled=True)
        new = run(w, t, tiled=False)
        if old is None or new is None:
            print(f"  SKIP  {name:20} (run failed)")
            skipped += 1
        elif old == new:
            print(f"  SAME  {name:20} {len(new)} chars, identical text")
            same += 1
        else:
            diff += 1
            print(f"  DIFF  {name:20}")
            for a, b in zip(old.splitlines(), new.splitlines()):
                if a != b:
                    print(f"        tiled: {a[:70]}")
                    print(f"        wide : {b[:70]}")
                    break
    print(f"\n  {same} identical, {diff} differing, {skipped} skipped")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())

