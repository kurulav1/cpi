#!/usr/bin/env python3
"""
HuggingFace model downloader and CPI converter.

Commands:
  search <query>  [--limit N] [--token TOKEN]
  download <repo_id>  [--output-dir DIR] [--family FAMILY] [--token TOKEN]

All output is JSON lines:
  {"type": "log",     "msg": "..."}
  {"type": "results", "models": [...]}    # search only
  {"type": "done",    "path": "..."}      # download only
  {"type": "error",   "msg": "..."}
  {"type": "status",  "status": "..."}    # download only, final state
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT   = SCRIPT_DIR.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

HF_API = os.environ.get("CPI_HF_API", "https://huggingface.co/api").rstrip("/")
HF_USER_AGENT = os.environ.get("CPI_HF_USER_AGENT", "cpi-hf-downloader/1.0")
HF_TIMEOUT_SECONDS = 20


# ── output helpers ────────────────────────────────────────────────────────────

def emit(obj: dict):
    print(json.dumps(obj), flush=True)

def log(msg: str):
    emit({"type": "log", "msg": msg})

def done(p: Path):
    emit({"type": "done", "path": str(p)})
    emit({"type": "status", "status": "done"})

def error(msg: str):
    emit({"type": "error", "msg": msg})
    emit({"type": "status", "status": "error"})


# ── HF API ────────────────────────────────────────────────────────────────────

def hf_get(url: str, token: str | None = None) -> object:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", HF_USER_AGENT)
    with urllib.request.urlopen(req, timeout=HF_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def run_logged_subprocess(cmd: list[str], failure_prefix: str):
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        log(line)
    if result.returncode != 0:
        err_detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"{failure_prefix}: {err_detail}")


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_search(args):
    q = urllib.parse.quote(args.query)
    limit = max(1, min(50, args.limit))
    url = (
        f"{HF_API}/models"
        f"?search={q}&limit={limit}"
        f"&filter=text-generation"
        f"&sort=downloads&direction=-1"
    )
    try:
        models = hf_get(url, args.token)
    except urllib.error.HTTPError as exc:
        error(f"HF API error {exc.code}: {exc.reason}")
        sys.exit(1)
    except Exception as exc:
        error(str(exc))
        sys.exit(1)

    results = [
        {
            "id":           m.get("id", ""),
            "downloads":    m.get("downloads", 0),
            "likes":        m.get("likes", 0),
            "private":      m.get("private", False),
            "tags":         m.get("tags", []),
            "lastModified": m.get("lastModified", ""),
            "gated":        m.get("gated", False),
        }
        for m in (models if isinstance(models, list) else [])
    ]
    emit({"type": "results", "models": results})


def cmd_download(args):
    repo_id = args.repo_id
    safe_id = repo_id.replace("/", "__")

    if args.output_dir:
        root_dir = Path(args.output_dir).resolve()
    else:
        root_dir = ARTIFACTS_DIR / "hub" / safe_id

    root_dir.mkdir(parents=True, exist_ok=True)
    hf_dir   = root_dir / "hf"
    bins_dir = root_dir / "bins"
    bins_dir.mkdir(exist_ok=True)
    ll2c_path = root_dir / f"{repo_id.split('/')[-1]}.ll2c"

    # ── step 1: download ──────────────────────────────────────────────────────
    log(f"Connecting to HuggingFace: {repo_id} …")
    try:
        import huggingface_hub as hf  # type: ignore
    except ImportError:
        error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    SKIP_PATTERNS = (
        ".pt", ".ot", ".msgpack", ".onnx",
    )
    SKIP_PREFIXES = ("flax_", "tf_", "rust_")
    SKIP_EXACT = {".gitattributes"}

    def should_skip(filename: str) -> bool:
        name = filename.lower()
        if any(name.endswith(p) for p in SKIP_PATTERNS):
            return True
        base = Path(filename).name.lower()
        if base in SKIP_EXACT:
            return True
        return any(base.startswith(p) for p in SKIP_PREFIXES)

    try:
        all_files = list(hf.list_repo_files(
            repo_id=repo_id,
            token=args.token or None,
        ))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            if args.token:
                error("Access denied while listing repo files. Your HF token may be invalid, missing permissions, or you may need to accept the model license.")
            else:
                error("Access denied while listing repo files. Add an HF token and ensure you've accepted the model license on Hugging Face.")
        else:
            error(f"Failed to list repo files: HTTP {exc.code} {exc.reason}")
        sys.exit(1)
    except Exception as exc:
        error(f"Failed to list repo files: {exc}")
        sys.exit(1)

    files_to_get = [f for f in all_files if not should_skip(f)]
    log(f"Downloading {len(files_to_get)} files (skipping {len(all_files) - len(files_to_get)} non-model files) …")

    hf_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any stale lock / incomplete files left by interrupted prior runs
    dl_cache = hf_dir / ".cache" / "huggingface" / "download"
    if dl_cache.exists():
        import glob as _glob
        for stale in _glob.glob(str(dl_cache / "*.lock")) + _glob.glob(str(dl_cache / "*.incomplete")):
            try:
                Path(stale).unlink()
                log(f"Removed stale: {Path(stale).name}")
            except OSError:
                pass  # still locked by another process — will fail on download

    total_files = len(files_to_get)

    def stream_file(url: str, dest: Path, filename: str, file_idx: int) -> None:
        """Stream a single file, emitting progress events every ~2 MB."""
        import time
        req = urllib.request.Request(url)
        if args.token:
            req.add_header("Authorization", f"Bearer {args.token}")
        req.add_header("User-Agent", HF_USER_AGENT)

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            chunk = 2 * 1024 * 1024  # 2 MB
            speed_window = []  # [(time, bytes)] for rolling speed

            emit({"type": "progress", "file": filename,
                  "file_idx": file_idx, "file_total": total_files,
                  "downloaded": 0, "total": total, "pct": 0, "speed": 0})

            with open(tmp, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    now = time.monotonic()
                    speed_window.append((now, len(buf)))
                    # Keep only last 5 seconds for rolling average
                    speed_window = [(t, b) for t, b in speed_window if now - t <= 5.0]
                    elapsed_window = now - speed_window[0][0] if len(speed_window) > 1 else 1.0
                    speed = sum(b for _, b in speed_window) / max(elapsed_window, 0.1)
                    pct = round(downloaded / total * 100, 1) if total else -1
                    eta = round((total - downloaded) / speed) if speed > 0 and total > 0 else -1
                    emit({"type": "progress", "file": filename,
                          "file_idx": file_idx, "file_total": total_files,
                          "downloaded": downloaded, "total": total,
                          "pct": pct, "speed": round(speed), "eta": eta})

        tmp.rename(dest)

    for i, filename in enumerate(files_to_get, 1):
        dest = hf_dir / filename
        if dest.exists():
            log(f"[{i}/{total_files}] {filename} (cached)")
            continue

        log(f"[{i}/{total_files}] {filename}")
        try:
            url = hf.hf_hub_url(repo_id=repo_id, filename=filename)
            stream_file(url, dest, filename, i)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                if args.token:
                    error(
                        f"Failed to download {filename}: HTTP {exc.code} {exc.reason}. "
                        "Your HF token may be invalid, missing permissions, or you may need to accept the model license."
                    )
                else:
                    error(
                        f"Failed to download {filename}: HTTP {exc.code} {exc.reason}. "
                        "Add an HF token and ensure you've accepted the model license on Hugging Face."
                    )
            else:
                error(f"Failed to download {filename}: HTTP {exc.code} {exc.reason}")
            sys.exit(1)
        except Exception as exc:
            error(f"Failed to download {filename}: {exc}")
            sys.exit(1)

    log("Download complete.")

    # ── step 2: convert ───────────────────────────────────────────────────────
    log("Converting weights to .bin files …")
    convert_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "convert_hf_to_bins.py"),
        "--hf-dir", str(hf_dir),
        "--out-dir", str(bins_dir),
    ]
    if args.family:
        convert_cmd += ["--family", args.family]

    try:
        run_logged_subprocess(convert_cmd, "Conversion failed")
    except RuntimeError as exc:
        error(str(exc))
        sys.exit(1)

    log("Conversion complete.")

    # ── step 3: pack ──────────────────────────────────────────────────────────
    log(f"Packing into {ll2c_path.name} …")
    pack_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "pack_ll2c.py"),
        "--input-dir", str(bins_dir),
        "--output", str(ll2c_path),
    ]

    try:
        run_logged_subprocess(pack_cmd, "Pack failed")
    except RuntimeError as exc:
        error(str(exc))
        sys.exit(1)

    log(f"Packed: {ll2c_path}")
    done(ll2c_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models and convert to CPI .ll2c format."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p_search = sub.add_parser("search", help="Search HuggingFace models.")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--token", default="")

    # download
    p_dl = sub.add_parser("download", help="Download and convert a model.")
    p_dl.add_argument("repo_id", help="HuggingFace repo id, e.g. meta-llama/Llama-3.2-1B")
    p_dl.add_argument("--output-dir", default="")
    p_dl.add_argument("--family",
                      choices=["llama2", "llama3", "mistral", "mixtral", "phi3", "phimoe", "qwen2"],
                      default="")
    p_dl.add_argument("--token", default="")

    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()
