#!/usr/bin/env python3
"""Writes PNG fixtures + their expected RGB bytes, for the hand-rolled decoder's gate.

Uses Python's zlib (a DEFLATE encoder we did not write), so the test is a real
cross-check rather than our code agreeing with itself. The cases are chosen to break
naive inflaters: dynamic Huffman, stored blocks, overlapping back-references, all five
scanline filters, and every supported colour type.

  python tools/make_png_fixtures.py
"""

import os
import struct
import zlib

OUT = os.path.join("artifacts", "png_fixtures")


def chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(name, width, height, color_type, raw_rows, expect_rgb, palette=None, level=6):
    """raw_rows: list of bytes, one per scanline, without the filter byte."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    # Cycle through all five filters so the decoder must implement each one.
    body = bytearray()
    prev = bytes(len(raw_rows[0]))
    for y, row in enumerate(raw_rows):
        f = y % 5
        body.append(f)
        if f == 0:
            body += row
        elif f == 1:  # Sub
            body += bytes((row[i] - (row[i - channels] if i >= channels else 0)) & 0xFF
                          for i in range(len(row)))
        elif f == 2:  # Up
            body += bytes((row[i] - prev[i]) & 0xFF for i in range(len(row)))
        elif f == 3:  # Average
            body += bytes(
                (row[i] - (((row[i - channels] if i >= channels else 0) + prev[i]) // 2)) & 0xFF
                for i in range(len(row))
            )
        else:  # Paeth
            def paeth(a, b, c):
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)

            body += bytes(
                (
                    row[i]
                    - paeth(
                        row[i - channels] if i >= channels else 0,
                        prev[i],
                        prev[i - channels] if i >= channels else 0,
                    )
                )
                & 0xFF
                for i in range(len(row))
            )
        prev = row

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
    if palette is not None:
        png += chunk(b"PLTE", palette)
    png += chunk(b"IDAT", zlib.compress(bytes(body), level))
    png += chunk(b"IEND", b"")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".png"), "wb") as f:
        f.write(png)
    with open(os.path.join(OUT, name + ".rgb"), "wb") as f:
        f.write(bytes(expect_rgb))
    print(f"  {name:14} {width}x{height}  color_type={color_type}  {len(png)} bytes")


def main():
    # 1. RGB gradient — compresses well, dynamic Huffman
    w, h = 64, 48
    rows, rgb = [], bytearray()
    for y in range(h):
        row = bytearray()
        for x in range(w):
            px = (x * 4 % 256, y * 5 % 256, (x + y) * 3 % 256)
            row += bytes(px)
            rgb += bytes(px)
        rows.append(bytes(row))
    write_png("rgb_gradient", w, h, 2, rows, rgb)

    # 2. RGBA pseudo-random noise at level 0 — incompressible, forces STORED blocks
    w, h = 32, 32
    rows, rgb = [], bytearray()
    state = 12345
    for y in range(h):
        row = bytearray()
        for x in range(w):
            vals = []
            for _ in range(4):
                state = (1103515245 * state + 12345) & 0x7FFFFFFF
                vals.append((state >> 16) & 0xFF)
            row += bytes(vals)
            rgb += bytes(vals[:3])  # alpha is dropped by the decoder
        rows.append(bytes(row))
    write_png("rgba_noise", w, h, 6, rows, rgb, level=0)

    # 3. Greyscale with long runs — overlapping LZ77 back-references
    w, h = 40, 20
    rows, rgb = [], bytearray()
    for y in range(h):
        v = (y // 4) * 50 % 256
        row = bytes([v] * w)
        rows.append(row)
        for x in range(w):
            rgb += bytes([v, v, v])
    write_png("gray_steps", w, h, 0, rows, rgb)

    # 4. Palette
    pal = bytes([255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 200, 0])
    w, h = 24, 12
    rows, rgb = [], bytearray()
    for y in range(h):
        row = bytearray()
        for x in range(w):
            idx = (x // 6) % 4
            row.append(idx)
            rgb += pal[idx * 3 : idx * 3 + 3]
        rows.append(bytes(row))
    write_png("palette_flag", w, h, 3, rows, rgb, palette=pal)

    # 5. Grey + alpha
    w, h = 16, 16
    rows, rgb = [], bytearray()
    for y in range(h):
        row = bytearray()
        for x in range(w):
            g = (x * 16 + y) % 256
            row += bytes([g, 128])
            rgb += bytes([g, g, g])
        rows.append(bytes(row))
    write_png("gray_alpha", w, h, 4, rows, rgb)

    # 6. 1x1 edge case
    write_png("tiny_1x1", 1, 1, 2, [bytes([17, 34, 51])], bytes([17, 34, 51]))


if __name__ == "__main__":
    print("writing PNG fixtures:")
    main()
    print(f"\nwrote fixtures to {OUT}")
