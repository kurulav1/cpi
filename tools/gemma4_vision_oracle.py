#!/usr/bin/env python3
"""Reference oracle for the Gemma 4 VISION tower.

Runs HuggingFace's Gemma4VisionModel + the embed_vision projector on a DETERMINISTIC
synthetic image, and dumps both the input and the resulting soft tokens so the CUDA
vision encoder can be checked against them.

Only the vision weights are loaded (~100M params for E2B), not the whole model.

  python tools/gemma4_vision_oracle.py --model artifacts/hub/google__gemma-4-E2B-it/hf \
                                       --out artifacts/vision_oracle_e2b.bin
"""

import argparse
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel


def build_input(grid: int, patch_dim: int):
    """A fixed, reproducible 'image': grid x grid patches, values in [0, 1]."""
    p = grid * grid
    pixels = torch.zeros(1, p, patch_dim, dtype=torch.float32)
    pos = torch.zeros(1, p, 2, dtype=torch.long)
    for t in range(p):
        x, y = t % grid, t // grid
        pos[0, t, 0] = x
        pos[0, t, 1] = y
        for k in range(patch_dim):
            # deterministic, non-trivial, and bounded to [0,1]
            v = 0.5 + 0.5 * torch.sin(torch.tensor(0.1 * t + 0.037 * k))
            pixels[0, t, k] = v
    return pixels, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=12)
    args = ap.parse_args()

    model_dir = Path(args.model)
    cfg = Gemma4Config.from_pretrained(model_dir)
    vcfg, tcfg = cfg.vision_config, cfg.text_config
    patch_dim = 3 * vcfg.patch_size**2

    # Pull just the vision + projector tensors out of the shards.
    state, proj_w = {}, None
    for shard in sorted(model_dir.glob("*.safetensors")):
        for name, tensor in load_file(str(shard)).items():
            if name.startswith("model.vision_tower."):
                state[name[len("model.vision_tower.") :]] = tensor
            elif name == "model.embed_vision.embedding_projection.weight":
                proj_w = tensor
    if proj_w is None:
        raise SystemExit("no model.embed_vision.embedding_projection.weight in the checkpoint")

    torch.set_grad_enabled(False)
    vision = Gemma4VisionModel(vcfg).to(torch.float32).eval()
    missing, unexpected = vision.load_state_dict(
        {k: v.to(torch.float32) for k, v in state.items()}, strict=False
    )
    missing = [m for m in missing if "position_ids" not in m]
    if missing:
        raise SystemExit(f"missing vision weights: {missing[:6]}")

    pixels, pos = build_input(args.grid, patch_dim)

    # Stage taps, so a divergence can be localised instead of guessed at.
    padding = (pos == -1).all(dim=-1)
    stage_patch = vision.patch_embedder(pixels, pos, padding)          # [1, P, vhidden]
    stage_enc = vision.encoder(                                        # [1, P, vhidden]
        inputs_embeds=stage_patch, attention_mask=~padding, pixel_position_ids=pos
    ).last_hidden_state

    # after exactly ONE encoder layer: separates a structural bug from a compounding one
    from transformers.masking_utils import create_bidirectional_mask
    _mask = create_bidirectional_mask(
        config=vision.encoder.config, inputs_embeds=stage_patch, attention_mask=~padding
    )
    _pe = vision.encoder.rotary_emb(stage_patch, pos)
    stage_l1 = vision.encoder.layers[0](
        stage_patch, position_embeddings=_pe, attention_mask=_mask, position_ids=pos
    )

    out = vision(pixel_values=pixels, pixel_position_ids=pos)
    feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    # The vision model already POOLS internally (grid^2 patches -> grid^2/k^2 soft
    # tokens) and squeezes the batch dim, so feats is [soft_tokens, vision_hidden].
    feats = feats.reshape(-1, vcfg.hidden_size).float()

    # projector: weightless RMSNorm -> linear into text hidden size
    normed = feats * torch.rsqrt(feats.pow(2).mean(-1, keepdim=True) + tcfg.rms_norm_eps)
    soft = torch.nn.functional.linear(normed, proj_w.to(torch.float32))

    n_patches, out_tokens, text_hidden = pixels.shape[1], soft.shape[0], soft.shape[1]
    print(f"patches={n_patches} soft_tokens={out_tokens} text_hidden={text_hidden}")
    print(f"soft[0,:4]={soft[0, :4].tolist()}")

    # Binary blob the C++ gate reads: header, then input, then expected output.
    with open(args.out, "wb") as f:
        f.write(struct.pack("<5i", n_patches, patch_dim, out_tokens, text_hidden, args.grid))
        f.write(pixels[0].contiguous().numpy().astype("float32").tobytes())
        f.write(pos[0, :, 0].contiguous().numpy().astype("int32").tobytes())
        f.write(pos[0, :, 1].contiguous().numpy().astype("int32").tobytes())
        f.write(soft.contiguous().numpy().astype("float32").tobytes())
        f.write(stage_patch.reshape(-1).contiguous().numpy().astype("float32").tobytes())
        f.write(stage_enc.reshape(-1).contiguous().numpy().astype("float32").tobytes())
        f.write(stage_l1.reshape(-1).contiguous().numpy().astype("float32").tobytes())
    print(f"wrote {args.out}  (+ stage taps: patch_embed, post_encoder)")
    print(f"  patch[0,:4]={stage_patch.reshape(-1, vcfg.hidden_size)[0, :4].tolist()}")
    print(f"  enc[0,:4]={stage_enc.reshape(-1, vcfg.hidden_size)[0, :4].tolist()}")


if __name__ == "__main__":
    main()
