#!/usr/bin/env python3
"""
WikiText-2 perplexity evaluation for CPI models.

Loads a HuggingFace-format model directory (safetensors + tokenizer.json) and
computes perplexity on the WikiText-2 test split using a standard sliding-window
approach. The fp16 / int8 / int4 modes use the same model weights; quantization
is applied at load time via bitsandbytes when requested.

Output JSON is written to docs/results/ by default and can be aggregated with
tools/bench_report.py.

Requirements (on top of requirements.txt):
  pip install transformers datasets torch

Optional (int8/int4 quantization):
  pip install bitsandbytes accelerate

Usage:
  # fp16 baseline
  python tools/perplexity.py --model-dir artifacts/mymodel/hf

  # int8 quantization (bitsandbytes)
  python tools/perplexity.py --model-dir artifacts/mymodel/hf --mode int8

  # int4 quantization (bitsandbytes)
  python tools/perplexity.py --model-dir artifacts/mymodel/hf --mode int4

  # Use a local plain-text corpus instead of downloading WikiText-2
  python tools/perplexity.py --model-dir artifacts/mymodel/hf --local-text path/to/corpus.txt

  # Run all three modes and write a single combined JSON
  python tools/perplexity.py --model-dir artifacts/mymodel/hf --all-modes
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"


def _rel(p) -> str:
    """Repo-root-relative path (forward slashes) for portable, non-leaky result
    metadata -- avoids baking an absolute C:\\Users\\<name>\\... path into committed
    JSON. Falls back to the absolute string if the path is outside the repo."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

def _check_import(pkg: str, install: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _require_import(pkg: str, install: str) -> None:
    if not _check_import(pkg, install):
        print(f"[ppl] missing package '{pkg}'. Install with:  {install}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Hardware info helpers
# ---------------------------------------------------------------------------

def _gpu_info() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "none"


def _cpu_info() -> str:
    try:
        import subprocess
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name", "/value"], text=True, timeout=5
            )
            for line in out.splitlines():
                if line.lower().startswith("name="):
                    return line.split("=", 1)[1].strip()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_wikitext2(split: str = "test") -> str:
    """Download WikiText-2 test split and return raw text."""
    _require_import("datasets", "pip install datasets")
    from datasets import load_dataset  # type: ignore
    print(f"[ppl] downloading WikiText-2 {split} split ...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-v1", split=split)
    return "\n".join(ds["text"])  # type: ignore


def _load_local_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"[ppl] text file not found: {p}", file=sys.stderr)
        sys.exit(2)
    return p.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def _bnb_config(mode: str):
    """Return a BitsAndBytesConfig for the requested quantization mode."""
    _require_import("bitsandbytes", "pip install bitsandbytes accelerate")
    _require_import("accelerate", "pip install accelerate")
    import torch
    from transformers import BitsAndBytesConfig  # type: ignore

    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    raise ValueError(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Core PPL computation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model_dir: str,
    mode: str = "fp16",
    stride: int = 512,
    max_length: int = 1024,
    max_tokens: Optional[int] = None,
    device_map: str = "auto",
    text: Optional[str] = None,
    local_text_file: Optional[str] = None,
    dataset_split: str = "test",
) -> dict:
    """
    Compute perplexity using a sliding-window approach (Dettmers et al. 2022).

    Returns a dict with keys: ppl, bpb, loss, n_tokens, elapsed_s.
    """
    _require_import("transformers", "pip install transformers")
    _require_import("torch", "pip install torch")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"[ppl] model directory not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    # Load corpus
    if text is not None:
        corpus = text
    elif local_text_file:
        corpus = _load_local_text(local_text_file)
    else:
        corpus = _load_wikitext2(dataset_split)

    print(f"[ppl] loading tokenizer from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print(f"[ppl] loading model ({mode}) from {model_path} ...", flush=True)
    load_kwargs: dict = {"device_map": device_map}
    if mode == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif mode in ("int8", "int4"):
        load_kwargs["quantization_config"] = _bnb_config(mode)
    else:
        raise ValueError(f"unsupported mode: {mode!r}. Choose fp16, int8, or int4.")

    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    load_s = time.perf_counter() - t_load
    print(f"[ppl] model loaded in {load_s:.1f}s", flush=True)

    print("[ppl] tokenizing corpus ...", flush=True)
    encodings = tokenizer(corpus, return_tensors="pt")
    input_ids: torch.Tensor = encodings["input_ids"]

    seq_len = input_ids.size(1)
    if max_tokens and seq_len > max_tokens:
        input_ids = input_ids[:, :max_tokens]
        seq_len = max_tokens

    print(f"[ppl] corpus tokens: {seq_len:,}  stride: {stride}  window: {max_length}", flush=True)

    # Sliding-window NLL accumulation (Meister & Cotterell 2021 style)
    nlls: list[torch.Tensor] = []
    prev_end = 0
    t_eval = time.perf_counter()

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens we score this window

        window_ids = input_ids[:, begin:end]
        target_ids = window_ids.clone()
        # Mask the context tokens (already scored in previous windows)
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(window_ids, labels=target_ids)
            # loss is mean NLL over the non-masked target tokens
            nlls.append(outputs.loss * target_len)

        prev_end = end
        pct = 100.0 * end / seq_len
        print(f"\r[ppl] {end:,}/{seq_len:,} tokens ({pct:.1f}%)", end="", flush=True)
        if end >= seq_len:
            break

    print(flush=True)
    eval_s = time.perf_counter() - t_eval

    n_tokens = prev_end
    mean_nll = torch.stack(nlls).sum().item() / n_tokens
    ppl = math.exp(mean_nll)
    # bits per byte: NLL (nats) → bits, divided by avg bytes per token (≈ 4 for EN)
    bpb = mean_nll / math.log(2) / 4.0

    print(f"[ppl] mode={mode}  ppl={ppl:.4f}  bpb={bpb:.4f}  loss={mean_nll:.4f}  "
          f"n_tokens={n_tokens:,}  eval_s={eval_s:.1f}s", flush=True)

    return {
        "mode": mode,
        "ppl": round(ppl, 6),
        "bpb": round(bpb, 6),
        "loss": round(mean_nll, 6),
        "n_tokens": n_tokens,
        "load_s": round(load_s, 2),
        "eval_s": round(eval_s, 2),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="WikiText-2 perplexity for CPI HuggingFace-format models."
    )
    ap.add_argument("--model-dir", required=True,
                    help="HuggingFace model directory (config.json + safetensors)")
    ap.add_argument("--mode", choices=["fp16", "int8", "int4"], default="fp16",
                    help="Quantization mode (default: fp16)")
    ap.add_argument("--all-modes", action="store_true",
                    help="Run fp16, int8, and int4 sequentially and write one combined JSON")
    ap.add_argument("--stride", type=int, default=512,
                    help="Sliding-window stride in tokens (default: 512)")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Maximum context window size (default: 1024)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Truncate corpus to this many tokens (default: full test set)")
    ap.add_argument("--dataset-split", default="test",
                    choices=["test", "validation", "train"],
                    help="WikiText-2 split to use (default: test)")
    ap.add_argument("--local-text", default=None,
                    help="Use a local plain-text file instead of downloading WikiText-2")
    ap.add_argument("--device-map", default="auto",
                    help="HuggingFace device_map (default: auto)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: docs/results/ppl_<model>_<mode>.json)")
    ap.add_argument("--model-name", default=None,
                    help="Human-readable model name for the output (default: directory name)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    model_name = args.model_name or model_dir.name

    modes = ["fp16", "int8", "int4"] if args.all_modes else [args.mode]

    # Pre-load corpus once so it isn't re-downloaded for each mode
    if args.local_text:
        corpus_text = _load_local_text(args.local_text)
        corpus_source = f"local:{args.local_text}"
    else:
        _require_import("datasets", "pip install datasets")
        corpus_text = _load_wikitext2(args.dataset_split)
        corpus_source = f"wikitext-2-v1/{args.dataset_split}"

    results: dict = {}
    for m in modes:
        try:
            r = compute_perplexity(
                model_dir=str(model_dir),
                mode=m,
                stride=args.stride,
                max_length=args.max_length,
                max_tokens=args.max_tokens,
                device_map=args.device_map,
                text=corpus_text,
                dataset_split=args.dataset_split,
            )
            results[m] = r
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[ppl] ERROR running mode {m}: {exc}", file=sys.stderr)
            results[m] = {"mode": m, "error": str(exc)}

    report = {
        "schema": "cpi-ppl-v1",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "host": os.environ.get("CPI_BENCH_HOST", "redacted"),
        "platform": platform.platform(),
        "cpu": _cpu_info(),
        "gpu": _gpu_info(),
        "model_name": model_name,
        "model_dir": _rel(model_dir),
        "corpus": corpus_source,
        "stride": args.stride,
        "max_length": args.max_length,
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "all" if args.all_modes else args.mode
        out_path = RESULTS_DIR / f"ppl_{model_name}_{suffix}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ppl] wrote {out_path}")

    # Quick summary
    print("\n[ppl] === Summary ===")
    header = f"{'Mode':<8}  {'PPL':>10}  {'BPB':>8}  {'Loss':>8}  {'Tokens':>10}  {'EvalTime':>10}"
    print(header)
    print("-" * len(header))
    for m, r in results.items():
        if "error" in r:
            print(f"{m:<8}  ERROR: {r['error']}")
        else:
            print(f"{m:<8}  {r['ppl']:>10.4f}  {r['bpb']:>8.4f}  {r['loss']:>8.4f}"
                  f"  {r['n_tokens']:>10,}  {r['eval_s']:>9.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
