import argparse
import array
import json
import math
import random
import struct
import sys
import time
from pathlib import Path


def write_event(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_safetensors(path):
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError(f"Invalid safetensors file: {path}")

    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header_start = 8
    header_end = header_start + header_len
    if header_end > len(raw):
        raise ValueError(f"Invalid safetensors header length: {path}")

    header = json.loads(raw[header_start:header_end].decode("utf-8"))
    data_start = header_end
    tensors = {}

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = str(meta.get("dtype", "")).upper()
        if dtype != "F32":
            raise ValueError(f"{name} uses unsupported dtype {dtype}; CPT exports should be F32")
        shape = [int(x) for x in meta.get("shape", [])]
        offsets = [int(x) for x in meta.get("data_offsets", [])]
        if len(offsets) != 2:
            raise ValueError(f"{name} has invalid data_offsets")
        start, end = data_start + offsets[0], data_start + offsets[1]
        values = array.array("f")
        values.frombytes(raw[start:end])
        if sys.byteorder != "little":
            values.byteswap()
        expected = 1
        for dim in shape:
            expected *= dim
        if len(values) != expected:
            raise ValueError(f"{name} has {len(values)} values, expected {expected}")
        tensors[name] = {"shape": shape, "data": values}

    return tensors


def row(tensor, index):
    shape = tensor["shape"]
    data = tensor["data"]
    width = int(shape[-1])
    start = index * width
    return data[start:start + width]


def vector_tensor(tensors, name, size):
    tensor = tensors.get(name)
    if tensor is None:
        raise KeyError(f"Missing tensor {name}")
    if len(tensor["data"]) != size:
        raise ValueError(f"{name} has wrong size")
    return tensor["data"]


def matvec(tensor, x):
    shape = tensor["shape"]
    if len(shape) != 2:
        raise ValueError("matvec tensor must be 2D")
    rows, cols = shape
    if len(x) != cols:
        raise ValueError(f"matvec input has {len(x)} values, expected {cols}")
    data = tensor["data"]
    out = [0.0] * rows
    for r in range(rows):
        base = r * cols
        acc = 0.0
        for c, xv in enumerate(x):
            acc += data[base + c] * xv
        out[r] = acc
    return out


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def layer_norm(x, weight, bias, eps):
    mean = sum(x) / len(x)
    var = sum((v - mean) * (v - mean) for v in x) / len(x)
    inv = 1.0 / math.sqrt(var + eps)
    return [(x[i] - mean) * inv * weight[i] + bias[i] for i in range(len(x))]


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def softmax(scores):
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [v / total for v in exps]


def choose_token(logits, temperature=0.0, top_k=0):
    if temperature <= 0.0:
        return max(range(len(logits)), key=lambda i: logits[i])

    ranked = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    if top_k and top_k > 0:
        ranked = ranked[:max(1, min(int(top_k), len(ranked)))]

    temp = max(float(temperature), 1.0e-6)
    scaled = [logits[i] / temp for i in ranked]
    probs = softmax(scaled)
    pick = random.random()
    cdf = 0.0
    for idx, prob in zip(ranked, probs):
        cdf += prob
        if pick <= cdf:
            return idx
    return ranked[-1]


def messages_to_prompt(messages):
    parts = []
    for message in messages or []:
        content = str(message.get("content") or "")
        if content:
            parts.append(content)
    return "\n".join(parts)


class CptGptModel:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.config = read_json(self.model_dir / "config.json")
        if str(self.config.get("model_type", "")).lower() != "cpt_gpt":
            raise ValueError("config.json is not a cpt_gpt model")

        self.tensors = load_safetensors(self.model_dir / "model.safetensors")
        self.vocab_size = int(self.config["vocab_size"])
        self.hidden_size = int(self.config["hidden_size"])
        self.intermediate_size = int(self.config["intermediate_size"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.max_positions = int(self.config["max_position_embeddings"])
        self.layer_norm_eps = float(self.config.get("layer_norm_eps", 1.0e-5))
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads

        self.hf_tokenizer = None
        hf_tokenizer_path = self.model_dir / "tokenizer.json"
        byte_tokenizer_path = self.model_dir / "byte_tokenizer.json"
        if hf_tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "This cpt_gpt export uses tokenizer.json; install the Python tokenizers package to run it."
                ) from exc
            self.hf_tokenizer = Tokenizer.from_file(str(hf_tokenizer_path))
            self.id_to_byte = {}
        elif byte_tokenizer_path.exists():
            payload = read_json(byte_tokenizer_path)
            self.id_to_byte = {
                int(k): int(v)
                for k, v in dict(payload.get("id_to_byte", {})).items()
            }
        else:
            self.id_to_byte = {i: i for i in range(min(self.vocab_size, 256))}

    def encode(self, text):
        if self.hf_tokenizer is not None:
            return [
                int(token)
                for token in self.hf_tokenizer.encode(str(text), add_special_tokens=False).ids
            ]

        byte_to_id = {byte: idx for idx, byte in self.id_to_byte.items()}
        return [
            byte_to_id.get(b, b if b < self.vocab_size else 0)
            for b in str(text).encode("utf-8", errors="replace")
        ]

    def decode(self, ids):
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.decode([int(idx) for idx in ids], skip_special_tokens=True)

        raw = bytearray()
        for idx in ids:
            byte = self.id_to_byte.get(int(idx))
            if byte is not None and 0 <= byte <= 255:
                raw.append(byte)
        return raw.decode("utf-8", errors="replace")

    def tokenizer_kind(self):
        return "hf_tokenizer" if self.hf_tokenizer is not None else "byte"

    def validate_token_id(self, token_id):
        token = int(token_id)
        if token < 0 or token >= self.vocab_size:
            return 0
        return token

    def prompt_tokens(self, prompt):
        tokens = [self.validate_token_id(token) for token in self.encode(prompt)]
        return tokens or [0]

    def forward_logits(self, tokens):
        if not tokens:
            tokens = [0]
        tokens = tokens[-self.max_positions:]
        states = []
        tok_emb = self.tensors["transformer.wte.weight"]
        pos_emb = self.tensors["transformer.wpe.weight"]

        for pos, token_id in enumerate(tokens):
            safe_id = int(token_id) % self.vocab_size
            states.append(add(row(tok_emb, safe_id), row(pos_emb, pos)))

        for layer in range(self.num_layers):
            prefix = f"transformer.h.{layer}"
            ln1_w = vector_tensor(self.tensors, f"{prefix}.ln_1.weight", self.hidden_size)
            ln1_b = vector_tensor(self.tensors, f"{prefix}.ln_1.bias", self.hidden_size)
            normed = [layer_norm(x, ln1_w, ln1_b, self.layer_norm_eps) for x in states]
            queries = [matvec(self.tensors[f"{prefix}.attn.q_proj.weight"], x) for x in normed]
            keys = [matvec(self.tensors[f"{prefix}.attn.k_proj.weight"], x) for x in normed]
            values = [matvec(self.tensors[f"{prefix}.attn.v_proj.weight"], x) for x in normed]

            attended = []
            scale = 1.0 / math.sqrt(self.head_dim)
            for pos, q in enumerate(queries):
                merged = [0.0] * self.hidden_size
                for head in range(self.num_heads):
                    start = head * self.head_dim
                    end = start + self.head_dim
                    scores = []
                    for prev in range(pos + 1):
                        score = 0.0
                        k = keys[prev]
                        for i in range(start, end):
                            score += q[i] * k[i]
                        scores.append(score * scale)
                    weights = softmax(scores)
                    for prev, weight in enumerate(weights):
                        v = values[prev]
                        for i in range(start, end):
                            merged[i] += weight * v[i]
                attended.append(merged)

            attn_out = [
                matvec(self.tensors[f"{prefix}.attn.out_proj.weight"], x)
                for x in attended
            ]
            states = [add(x, y) for x, y in zip(states, attn_out)]

            ln2_w = vector_tensor(self.tensors, f"{prefix}.ln_2.weight", self.hidden_size)
            ln2_b = vector_tensor(self.tensors, f"{prefix}.ln_2.bias", self.hidden_size)
            next_states = []
            for x in states:
                h = layer_norm(x, ln2_w, ln2_b, self.layer_norm_eps)
                h = matvec(self.tensors[f"{prefix}.mlp.fc_in.weight"], h)
                bias = vector_tensor(self.tensors, f"{prefix}.mlp.fc_in.bias", self.intermediate_size)
                h = [gelu(h[i] + bias[i]) for i in range(self.intermediate_size)]
                h = matvec(self.tensors[f"{prefix}.mlp.fc_out.weight"], h)
                bias = vector_tensor(self.tensors, f"{prefix}.mlp.fc_out.bias", self.hidden_size)
                h = [h[i] + bias[i] for i in range(self.hidden_size)]
                next_states.append(add(x, h))
            states = next_states

        # Pre-norm models apply a final LayerNorm (ln_f) to the last block's output before the
        # head. Older exports predate ln_f, so apply it only when the tensors are present.
        final_state = states[-1]
        if "transformer.ln_f.weight" in self.tensors and "transformer.ln_f.bias" in self.tensors:
            ln_f_w = vector_tensor(self.tensors, "transformer.ln_f.weight", self.hidden_size)
            ln_f_b = vector_tensor(self.tensors, "transformer.ln_f.bias", self.hidden_size)
            final_state = layer_norm(final_state, ln_f_w, ln_f_b, self.layer_norm_eps)
        return matvec(self.tensors["lm_head.weight"], final_state)

    def generate(self, prompt, max_new=64, temperature=0.0, top_k=0):
        prompt_tokens = self.prompt_tokens(prompt)
        tokens = prompt_tokens[:]
        generated = []
        for _ in range(max(0, int(max_new))):
            logits = self.forward_logits(tokens)
            token = choose_token(logits, temperature=temperature, top_k=top_k)
            tokens.append(token)
            generated.append(token)
        return self.decode(generated), len(generated)


def generate(model, request):
    prompt = request.get("prompt")
    if prompt is None:
        prompt = messages_to_prompt(request.get("messages") or [])
    max_new = int(request.get("max_new") or 64)
    temperature = float(request.get("temperature") or 0.0)
    top_k = int(request.get("top_k") or 0)
    return model.generate(prompt, max_new=max_new, temperature=temperature, top_k=top_k)


def run_worker(args):
    model = CptGptModel(args.model_dir)
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
            text, generated_tokens = generate(model, req)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            tok_per_s = None
            if elapsed_ms > 0:
                tok_per_s = generated_tokens / (elapsed_ms / 1000.0)
            write_event({
                "type": "done",
                "id": req_id,
                "text": text,
                "elapsed_ms": elapsed_ms,
                "generated_tokens": generated_tokens,
                "tok_per_s": tok_per_s,
                "metrics": None,
            })
        except Exception as exc:
            write_event({"type": "error", "id": req_id, "error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Run CPT cpt_gpt Hugging Face-style exports.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    if args.prompt is not None:
        model = CptGptModel(args.model_dir)
        text, _ = model.generate(
            args.prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(text)
        return

    run_worker(args)


if __name__ == "__main__":
    main()
