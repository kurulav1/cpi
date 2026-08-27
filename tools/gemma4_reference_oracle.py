#!/usr/bin/env python3
"""Gemma 4 (E2B) reference oracle.

Loads the HF text model, runs a fixed prompt, and dumps per-stage activations
so the CPI Gemma4Engine can be parity-checked layer-by-layer. This is the
ground truth; every scale/norm/rope detail that is ambiguous in the source is
pinned here numerically rather than guessed (cf. the wrong-BOS multi-day hunt).

Outputs:
  <outdir>/manifest.json          shapes + token ids + top tokens + per-stage rms/max
  <outdir>/<name>.f32             raw little-endian float32 for each dumped tensor

Usage:
  python tools/gemma4_reference_oracle.py \
      --model artifacts/hub/google__gemma-4-E2B-it/hf \
      --prompt "The capital of France is" --outdir <scratch>/g4ref
"""
import argparse
import json
import os
import sys
import numpy as np


def rms(x):
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", default=None, help="comma-separated ids, overrides --prompt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--layers", default="0,1,4,14,15,34", help="which layer outputs to dump in full")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    # 10GB multimodal checkpoint: fp32 (~20GB) is too tight for 32GB RAM, so load
    # bf16 (which is how the model actually runs anyway); RMSNorm upcasts to fp32
    # internally. Try text-only CausalLM first, else the full conditional-gen model.
    model = None
    for loader in ("Gemma4ForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers as tf
            cls = getattr(tf, loader, None) or (AutoModelForCausalLM if loader == "AutoModelForCausalLM" else None)
            if cls is None:
                continue
            model = cls.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            print("loaded via", loader)
            break
        except Exception as e:
            print(f"loader {loader} failed: {e}")
    if model is None:
        sys.exit("could not load model")
    model.eval()

    # locate the text model (multimodal wrapper -> .model.language_model, or direct)
    root = model
    lm = None
    for path in ["model.language_model", "language_model", "model"]:
        obj = root
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
                lm = obj
                break
        except AttributeError:
            continue
    if lm is None:
        print("could not locate text model; children:", [n for n, _ in model.named_children()])
        sys.exit(1)
    print("text model located:", type(lm).__name__, "num layers:", len(lm.layers))

    # Dump rope inv_freq per layer type + attention scaling (pins partial-rope +
    # the scaling=1.0 question numerically for the engine).
    rot = getattr(lm, "rotary_emb", None)
    if rot is not None:
        for lt in ("sliding_attention", "full_attention"):
            for suf in ("_inv_freq", "_attention_scaling"):
                buf = getattr(rot, lt + suf, None)
                if buf is None:
                    continue
                if hasattr(buf, "detach"):
                    arr = buf.detach().float().cpu().numpy()
                    arr.astype("<f4").tofile(os.path.join(args.outdir, f"rope_{lt}{suf}.f32"))
                    print(f"rope {lt}{suf}: len={arr.size} first5={arr.reshape(-1)[:5]} last3={arr.reshape(-1)[-3:]}")
                else:
                    print(f"rope {lt}{suf} = {buf}")
        print("attn scaling (layer0):", getattr(lm.layers[0].self_attn, "scaling", "?"),
              "| head_dim:", lm.layers[0].self_attn.head_dim,
              "| layer4 head_dim:", lm.layers[4].self_attn.head_dim)

    if args.tokens:
        ids = [int(x) for x in args.tokens.split(",")]
    else:
        ids = tok(args.prompt, return_tensors="pt")["input_ids"][0].tolist()
    print("token ids:", ids)
    input_ids = torch.tensor([ids], dtype=torch.long)

    dumps = {}  # name -> np.array (float32, squeezed batch)
    want_layers = set(int(x) for x in args.layers.split(",") if x != "")

    # ---- hooks ----
    handles = []

    def save(name, t):
        a = t.detach().to(torch.float32).cpu().numpy()
        if a.ndim >= 1 and a.shape[0] == 1:
            a = a[0]
        dumps[name] = a

    # embeddings + per-layer inputs: hook the text model forward via wrapper
    # capture inputs_embeds (post-scale) and per_layer_inputs by monkeypatching
    orig_project = getattr(lm, "project_per_layer_inputs", None)
    if orig_project is not None:
        def wrapped_project(inputs_embeds, per_layer_inputs=None):
            out = orig_project(inputs_embeds, per_layer_inputs)
            save("per_layer_inputs", out)  # [seq, num_layers, ple_dim]
            return out
        lm.project_per_layer_inputs = wrapped_project

    orig_embed = lm.embed_tokens.forward
    def wrapped_embed(x):
        out = orig_embed(x)
        save("embeds", out)
        return out
    lm.embed_tokens.forward = wrapped_embed

    for i, layer in enumerate(lm.layers):
        def mk(idx):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                dumps[f"__layer_rms_{idx}"] = np.array([rms(o.detach().float().cpu().numpy())])
                if idx in want_layers:
                    save(f"layer_{idx}_out", o)
            return hook
        handles.append(layer.register_forward_hook(mk(i)))

    if hasattr(lm, "norm"):
        def norm_hook(mod, inp, out):
            save("final_norm", out)
        handles.append(lm.norm.register_forward_hook(norm_hook))

    # post-attention hidden (= input to pre_feedforward_layernorm) per layer
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "pre_feedforward_layernorm"):
            def mkpre(idx):
                def preh(mod, inp):
                    save(f"postattn_{idx}", inp[0])
                return preh
            handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(mkpre(i)))
        # raw attention output (post o_proj, pre post_attention_layernorm)
        def mkattn(idx):
            def hook(mod, inp, out):
                save(f"attnout_{idx}", out[0] if isinstance(out, tuple) else out)
            return hook
        handles.append(layer.self_attn.register_forward_hook(mkattn(i)))

    # ---- run ----
    # use_cache=True so KV sharing (num_kv_shared_layers) is ACTIVE; this is the
    # intended inference behavior. With use_cache=False, past_key_values is None
    # and HF skips sharing (each layer computes its own K/V), a different graph.
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[0]  # [seq, vocab]
    save("logits_last", logits[-1:])
    last = logits[-1]
    topv, topi = torch.topk(last, 10)
    top = [(int(i), float(v)) for i, v in zip(topi, topv)]
    print("top-10 next tokens:")
    for tid, v in top:
        print(f"  {tid:7d}  {v:9.4f}  {tok.decode([tid])!r}")

    # ---- write ----
    manifest = {
        "model": args.model,
        "prompt": args.prompt,
        "token_ids": ids,
        "num_layers": len(lm.layers),
        "top10": top,
        "argmax_last": int(last.argmax()),
        "tensors": {},
        "per_layer_rms": {},
    }
    for k, v in dumps.items():
        if k.startswith("__layer_rms_"):
            manifest["per_layer_rms"][k.replace("__layer_rms_", "")] = float(v[0])
            continue
        v = np.ascontiguousarray(v, dtype="<f4")
        v.tofile(os.path.join(args.outdir, f"{k}.f32"))
        manifest["tensors"][k] = {"shape": list(v.shape), "rms": rms(v), "max": float(np.abs(v).max())}

    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("\nper-stage summary:")
    for k in sorted(manifest["tensors"]):
        m = manifest["tensors"][k]
        print(f"  {k:20s} shape={m['shape']} rms={m['rms']:.4f} max={m['max']:.3f}")
    print("\nper-layer output rms:")
    for L in sorted(manifest["per_layer_rms"], key=int):
        print(f"  layer {int(L):2d}: rms={manifest['per_layer_rms'][L]:.4f}")
    print("\nwrote", args.outdir)


if __name__ == "__main__":
    main()
