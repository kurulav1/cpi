#!/usr/bin/env python3
"""HF reference dumps for bringing LLaVA-1.5 up on CPI, stage by stage.

Each stage of the port is gated against one of these tensors, in isolation, so a
discrepancy points at exactly one component:

  image.png            the fixed input image (CPI reads THIS, not a re-encode)
  pixel_values.bin     CLIP preprocess output [1,3,336,336]  -> gates CPI preprocessing
  tower_hidden.bin     vision tower hidden_states[-2] [1,577,1024] (incl. CLS)
                                                        -> gates the CLIP ViT tower
  image_features.bin   selected features [1,576,1024] (CLS dropped, layer -2)
  projected.bin        multi_modal_projector output [1,576,4096]
                                                        -> gates the MLP connector
  logits_row0.bin      first-position logits for a fixed image+prompt [vocab]
                                                        -> gates end-to-end splice+LLM
  meta.json            shapes + the config knobs CPI must match

Everything is float32, C-contiguous, little-endian raw bytes -- a CPI parity test
mmaps the file and compares element-wise.

Run once after the checkpoint is downloaded:
  python oracle/llava/dump_oracle.py <checkpoint_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def make_test_image(path, w=336, h=336):
    # A deterministic, structured image: colour varies with position so a wrong
    # patch order or a flipped axis in preprocessing shows up as a large diff
    # rather than hiding in noise.
    ys, xs = np.mgrid[0:h, 0:w]
    r = (xs * 255 // w).astype(np.uint8)
    g = (ys * 255 // h).astype(np.uint8)
    b = (((xs + ys) * 255) // (w + h)).astype(np.uint8)
    Image.fromarray(np.dstack([r, g, b])).save(path)


def save(arr, path):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path)
    return list(arr.shape)


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)

    proc = AutoProcessor.from_pretrained(ckpt)
    model = LlavaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    feat_layer = getattr(cfg, "vision_feature_layer", -2)
    select = getattr(cfg, "vision_feature_select_strategy", "default")

    img_path = os.path.join(out, "image.png")
    make_test_image(img_path)
    image = Image.open(img_path).convert("RGB")

    prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(torch.float32)

    meta = {"feature_layer": int(feat_layer), "select_strategy": select}
    meta["pixel_values"] = save(pixel_values.numpy(), os.path.join(out, "pixel_values.bin"))

    with torch.no_grad():
        vt = model.vision_tower(pixel_values, output_hidden_states=True)
        hidden = vt.hidden_states[feat_layer]  # [1, 577, 1024]
        meta["tower_hidden"] = save(hidden.numpy(), os.path.join(out, "tower_hidden.bin"))

        feats = hidden[:, 1:] if select == "default" else hidden
        meta["image_features"] = save(feats.numpy(), os.path.join(out, "image_features.bin"))

        projected = model.multi_modal_projector(feats)  # [1, 576, 4096]
        meta["projected"] = save(projected.numpy(), os.path.join(out, "projected.bin"))

        out_full = model(**inputs)
        logits0 = out_full.logits[0, -1].to(torch.float32)  # next-token logits
        meta["logits_row0"] = save(logits0.numpy(), os.path.join(out, "logits_row0.bin"))
        meta["argmax_next"] = int(torch.argmax(logits0).item())

    # The config knobs CPI's tower/connector must reproduce exactly.
    vc = cfg.vision_config
    meta["vision_config"] = {
        "hidden_size": vc.hidden_size,
        "num_hidden_layers": vc.num_hidden_layers,
        "num_attention_heads": vc.num_attention_heads,
        "patch_size": vc.patch_size,
        "image_size": vc.image_size,
        "intermediate_size": vc.intermediate_size,
        "hidden_act": vc.hidden_act,
        "layer_norm_eps": getattr(vc, "layer_norm_eps", 1e-5),
    }
    meta["projector_hidden_act"] = getattr(cfg, "projector_hidden_act", "gelu")
    meta["image_token_index"] = getattr(cfg, "image_token_index", None)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("ORACLE_DONE", json.dumps(meta["vision_config"]))
    print("next-token argmax:", meta["argmax_next"])


if __name__ == "__main__":
    main()
