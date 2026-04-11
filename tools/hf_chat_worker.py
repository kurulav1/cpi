import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_event(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load_worker(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    if tokenizer.eos_token_id is not None and model.generation_config.pad_token_id is None:
      model.generation_config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def generate(tokenizer, model, request):
    messages = request.get("messages") or []
    if not messages:
        raise ValueError("messages are required")

    max_new = int(request.get("max_new") or 64)
    temperature = float(request.get("temperature") or 0.0)
    do_sample = temperature > 0.0

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    generation_kwargs = {
        "max_new_tokens": max_new,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    tokenizer, model = load_worker(args.model_dir)
    write_event({"type": "ready"})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        req = json.loads(raw)
        req_id = str(req.get("id") or "")

        if req.get("shutdown"):
            write_event({"type": "done", "id": req_id, "text": "", "elapsed_ms": 0})
            break

        started = time.perf_counter()
        try:
            text = generate(tokenizer, model, req)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            write_event({
                "type": "done",
                "id": req_id,
                "text": text,
                "elapsed_ms": elapsed_ms,
                "generated_tokens": None,
                "tok_per_s": None,
                "metrics": None,
            })
        except Exception as exc:
            write_event({"type": "error", "id": req_id, "error": str(exc)})


if __name__ == "__main__":
    main()
