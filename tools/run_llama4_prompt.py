import argparse
import re
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer


STOP_IDS = {200001, 200007, 200008}


def build_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|header_start|>system<|header_end|>\n\n"
        f"{system_prompt}"
        "<|eot|>"
        "<|header_start|>user<|header_end|>\n\n"
        f"{user_prompt}"
        "<|eot|>"
        "<|header_start|>assistant<|header_end|>\n\n"
    )


def extract_output_tokens(stdout: str) -> list[int]:
    match = re.search(r"Output tokens:(.*)", stdout)
    if not match:
        raise RuntimeError("llama_infer did not print an output token line")
    token_text = match.group(1).strip()
    if not token_text:
        return []
    return [int(piece) for piece in token_text.split()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Llama4 chat prompt through llama_infer using the HF tokenizer.json locally."
    )
    parser.add_argument("model_dir")
    parser.add_argument("prompt")
    parser.add_argument("--binary", default=str(Path("build") / "Release" / "llama_infer.exe"))
    parser.add_argument("--system-prompt", default="You are a concise helpful assistant.")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt_text = build_prompt(args.system_prompt, args.prompt)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids

    cmd = [
        args.binary,
        str(model_dir),
        "--tokens",
        ",".join(str(tok) for tok in prompt_ids),
        "--max-new",
        str(args.max_new),
        "--temp",
        str(args.temp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode

    output_ids = extract_output_tokens(result.stdout)
    generated = output_ids[len(prompt_ids):]
    cut = len(generated)
    for idx, token_id in enumerate(generated):
        if token_id in STOP_IDS:
            cut = idx
            break
    answer_ids = generated[:cut]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    print("\nDecoded answer:\n" + answer_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
