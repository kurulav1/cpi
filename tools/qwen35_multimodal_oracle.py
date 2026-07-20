#!/usr/bin/env python3
"""End-to-end multimodal oracle: image + text in, TOKEN STREAM out.

Every other gate in this project stops at soft tokens, which means a wrong-but-finite answer is
indistinguishable from a right one. This runs HuggingFace's full Qwen3_5ForConditionalGeneration
on the same synthetic image and prompt the port uses, greedily, and dumps the tokens it emits.
Matching that stream is the only check that covers the splice, the text stack and the tower at
once.

Deliberately bypasses the image PROCESSOR: pixel_values is fed directly as an already-patchified
tensor, identical to the one qwen35_vision_oracle.py uses. That keeps this runnable without
torchvision (which transformers requires for any image processor and which is not installed
here), and it isolates the model from the resize/normalise step, which has its own oracle.

    python tools/qwen35_multimodal_oracle.py --model ~/models/qwen35-0.8b-hf \
                                             --out artifacts/q35_mm --max-new 8

WHAT THIS FOUND, first run, before any comparison: the text stack uses M-RoPE for multimodal
input and the Metal port does not. For <vision_start> + 16 image tokens + <vision_end> + 4 text
tokens the model assigns

    t: 0  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  5 6 7 8 9
    h: 0  1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4  5 6 7 8 9
    w: 0  1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4  5 6 7 8 9

against a naive 0..21. Two consequences, and the second is the one that bites:

  - image tokens carry a 4x4 (h, w) grid rather than a running index, with mrope_section
    [11, 11, 10] splitting each head's rotary lanes across the t, h and w axes;
  - the image block advances the counter by only 4 -- the MERGED grid width -- so every text
    token after the image sits at 5..9 where a 1-D counter puts it at 17..21.

So the splice can be perfectly correct and the whole remainder of the prompt still be rotated to
the wrong positions. Soft tokens alone are not enough.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


def build_patches(cfg, grid_h: int, grid_w: int) -> torch.Tensor:
    """The SAME synthetic patch tensor qwen35_vision_oracle.py builds.

    It has to be bit-identical, or this oracle and the tower oracle describe different images and
    a disagreement between them means nothing.
    """
    vc = cfg.vision_config
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    n = grid_h * grid_w
    idx = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(patch_dim, dtype=torch.float32).unsqueeze(0)
    return 0.5 + 0.5 * torch.sin(0.1 * idx + 0.037 * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-h", type=int, default=8)
    ap.add_argument("--grid-w", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(str(model_dir))
    torch.set_grad_enabled(False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.float32).eval()

    pixels = build_patches(cfg, args.grid_h, args.grid_w)
    grid_thw = torch.tensor([[1, args.grid_h, args.grid_w]], dtype=torch.long)

    merge = cfg.vision_config.spatial_merge_size
    n_soft = (args.grid_h * args.grid_w) // (merge * merge)

    # <vision_start> <image>*n_soft <vision_end> then a short question. The image tokens are
    # placeholders: the model replaces their embeddings with the tower's soft tokens, one per
    # token, so there must be exactly n_soft of them or the scatter fails.
    ids = ([cfg.vision_start_token_id] + [cfg.image_token_id] * n_soft +
           [cfg.vision_end_token_id] + [3838, 374, 419, 30])  # "What is this?"-ish filler
    input_ids = torch.tensor([ids], dtype=torch.long)

    # mm_token_type_ids marks which positions are image tokens (1) and which are text (0). The
    # model refuses multimodal input without it, because Qwen3.5's text stack uses M-RoPE: image
    # tokens get 3-D (t, h, w) position ids rather than one running index, and the mrope_section
    # [11, 11, 10] splits the head's rotary lanes between those axes.
    #
    # This is a REAL gap in the port, not a detail of the harness: PlanMetalEngine advances one
    # scalar position per token. The soft tokens can be spliced perfectly and still sit at the
    # wrong positions.
    mm_token_type_ids = torch.tensor(
        [[1 if t == cfg.image_token_id else 0 for t in ids]], dtype=torch.long)

    out = model.generate(input_ids=input_ids, pixel_values=pixels, image_grid_thw=grid_thw,
                         mm_token_type_ids=mm_token_type_ids,
                         max_new_tokens=args.max_new, do_sample=False, num_beams=1)
    generated = out[0, input_ids.shape[1]:].tolist()

    # First-step logits too: a token stream that matches proves agreement at the argmax, but the
    # logits show HOW close, which is what tells drift apart from a genuine disagreement.
    #
    # And the POST-SPLICE hidden state, which is what localises a disagreement. The logits are
    # downstream of four subsystems at once (tower, splice, M-RoPE, text stack); the input to
    # layer 0 is downstream of only two -- the embedding lookup and the image scatter. If ours
    # matches this, the splice is right and the defect is M-RoPE or the text stack; if it does
    # not, no amount of looking at the text stack will help.
    #
    # Captured with a pre-hook on the first decoder layer rather than by recomputing the scatter
    # here: a reimplementation of the thing under test proves only that two reimplementations
    # agree.
    captured = {}

    def grab_layer0_input(_mod, args, kwargs):
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        if h is not None and "h" not in captured:
            captured["h"] = h.detach()[0].to(torch.float32).contiguous()

    layers = model.model.language_model.layers
    handle = layers[0].register_forward_pre_hook(grab_layer0_input, with_kwargs=True)
    step = model(input_ids=input_ids, pixel_values=pixels, image_grid_thw=grid_thw,
                 mm_token_type_ids=mm_token_type_ids)
    handle.remove()

    logits = step.logits[0, -1].to(torch.float32).contiguous()
    (out_dir / "first_step_logits.f32").write_bytes(logits.numpy().astype("<f4").tobytes())
    if "h" in captured:
        (out_dir / "layer0_input.f32").write_bytes(captured["h"].numpy().astype("<f4").tobytes())
        print("[mm-oracle] layer0 input: %s" % (tuple(captured["h"].shape),))
    else:
        print("[mm-oracle] WARNING: layer-0 pre-hook captured nothing; no layer0_input.f32")

    manifest = {
        "input_ids": ids,
        "generated_ids": generated,
        "n_soft_tokens": n_soft,
        "grid_thw": grid_thw.tolist(),
        "image_token_id": cfg.image_token_id,
        "vision_start_token_id": cfg.vision_start_token_id,
        "vision_end_token_id": cfg.vision_end_token_id,
        "logits_file": "first_step_logits.f32",
        "layer0_input_file": "layer0_input.f32" if "h" in captured else "",
        "layer0_input_shape": list(captured["h"].shape) if "h" in captured else [],
        "vocab": int(logits.numel()),
        "mm_token_type_ids": mm_token_type_ids[0].tolist(),
        "mrope_section": list(getattr(cfg.text_config.rope_parameters, "get", lambda *_: None)(
            "mrope_section") or cfg.text_config.rope_parameters.get("mrope_section", [])),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print("[mm-oracle] prompt %d tokens (%d image placeholders)" % (len(ids), n_soft))
    print("[mm-oracle] generated: %s" % generated)
    print("[mm-oracle] first-step logits: vocab=%d max=%.4f" % (logits.numel(), logits.max()))
    print("[mm-oracle] wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
