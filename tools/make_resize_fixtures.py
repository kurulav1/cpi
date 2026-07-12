#!/usr/bin/env python3
"""Reference fixtures for the hand-rolled bicubic resize, generated with PIL.

PIL is the resampler HF actually uses, so this is the right oracle. Resizing is the one
preprocessing step where a subtle mismatch (wrong bicubic `a`, or no antialias filter on
downscale) costs accuracy silently instead of failing loudly -- hence a real gate.

  python tools/make_resize_fixtures.py
"""

import os
import struct

from PIL import Image

OUT = os.path.join("artifacts", "resize_fixtures")


def make_source(w, h):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (
                (x * 7 + y * 3) % 256,
                (x * x + y) % 256,
                (x + y * y) % 256,
            )
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = [
        ("down_big", 200, 150, 96, 48),   # heavy downscale: needs the antialias filter
        ("down_small", 64, 64, 48, 48),   # mild downscale
        ("up", 20, 15, 96, 48),           # upscale
        ("same_ar", 120, 90, 48, 48),     # aspect change
    ]
    index = []
    for name, sw, sh, dw, dh in cases:
        src = make_source(sw, sh)
        dst = src.resize((dw, dh), Image.BICUBIC)
        with open(os.path.join(OUT, name + ".src"), "wb") as f:
            f.write(struct.pack("<2i", sw, sh))
            f.write(src.tobytes())
        with open(os.path.join(OUT, name + ".ref"), "wb") as f:
            f.write(struct.pack("<2i", dw, dh))
            f.write(dst.tobytes())
        index.append(f"{name} {sw} {sh} {dw} {dh}")
        print(f"  {name:12} {sw}x{sh} -> {dw}x{dh}")
    with open(os.path.join(OUT, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")
    print(f"\nwrote fixtures to {OUT}")


if __name__ == "__main__":
    print("writing resize fixtures (PIL BICUBIC):")
    main()
