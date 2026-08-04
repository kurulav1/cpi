#!/usr/bin/env python3
"""Reference oracle for Qwen3.5's image PREPROCESSING.

The tower's gate (qwen35_vision_oracle.py) starts from an already-patchified tensor, which
leaves everything upstream of it, resize, normalisation, patch layout, ungated. This dumps
what HuggingFace's processor actually produces for a real image so the C++ side can be compared
against it rather than derived from reading the code.

That distinction matters here more than usual, because the resize is not a fixed target size: it
is a "smart resize" that picks dimensions from a PIXEL-AREA budget, subject to both axes being
multiples of patch_size * merge_size. Reimplementing it from the config alone is guesswork.

    python tools/qwen35_preproc_oracle.py --model ~/models/qwen35-0.8b-hf \
                                          --out artifacts/q35_preproc --width 200 --height 140

Writes <out>/pixels_<W>x<H>.f32 (the patch tensor), and manifest.json with the grid it chose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor


def synthetic_image(w: int, h: int) -> np.ndarray:
    """A deterministic RGB image that varies along both axes and differs per channel.

    A flat or single-channel-varying image would hide a transposed resize, a channel swap, or a
    row/column mix-up; all of which produce a tensor of the right shape.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.5 * np.sin(xs * 0.11 + ys * 0.03)
    g = 0.5 + 0.5 * np.sin(xs * 0.05 - ys * 0.09 + 1.7)
    b = 0.5 + 0.5 * np.sin(xs * 0.02 + ys * 0.13 + 3.1)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=140)
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = AutoImageProcessor.from_pretrained(str(Path(args.model).expanduser()))
    img = synthetic_image(args.width, args.height)

    # The raw uint8 image goes out too: the C++ side must start from exactly these bytes, or the
    # comparison measures two different images rather than two implementations.
    (out_dir / ("image_%dx%d.u8" % (args.width, args.height))).write_bytes(img.tobytes())

    enc = proc(images=[img], return_tensors="pt")
    pixels = enc["pixel_values"].to(torch.float32).contiguous()
    grid = enc["image_grid_thw"].tolist()

    name = "pixels_%dx%d.f32" % (args.width, args.height)
    (out_dir / name).write_bytes(pixels.numpy().astype("<f4").tobytes())

    manifest = {
        "source_image": {"width": args.width, "height": args.height, "channels": 3,
                         "file": "image_%dx%d.u8" % (args.width, args.height),
                         "layout": "HWC uint8"},
        "pixel_values": {"shape": list(pixels.shape), "file": name},
        "image_grid_thw": grid,
        "processor": {
            "class": type(proc).__name__,
            "patch_size": getattr(proc, "patch_size", None),
            "temporal_patch_size": getattr(proc, "temporal_patch_size", None),
            "merge_size": getattr(proc, "merge_size", None),
            "image_mean": list(getattr(proc, "image_mean", [])),
            "image_std": list(getattr(proc, "image_std", [])),
            "size": dict(getattr(proc, "size", {}) or {}),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    t, gh, gw = grid[0]
    print("[preproc] %dx%d -> grid t=%d h=%d w=%d  patches=%s" % (args.width, args.height, t, gh,
                                                                  gw, tuple(pixels.shape)))
    print("[preproc] pixel range: %.4f .. %.4f" % (float(pixels.min()), float(pixels.max())))
    print("[preproc] wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
