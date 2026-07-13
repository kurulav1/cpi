#!/usr/bin/env python3
"""Every Metal kernel must take its params block as its LAST buffer binding.

MetalContext::dispatch() binds the params block at index n_buffers -- i.e. after
every data buffer. A kernel that declares `constant XParams& p [[buffer(k)]]` with
another buffer at an index above k would therefore have its params written over
that buffer.

That mistake compiles perfectly. `metal -Werror` cannot see it, the CI job stays
green, and it only misbehaves on real hardware -- as garbage numbers, which read
like a kernel math bug rather than a binding bug. Three kernels had it (rope,
kv_store, attention_decode), which is exactly why this is enforced mechanically
instead of by care.

Also checks that buffer indices within a kernel are contiguous from 0, since
dispatch() binds them positionally.
"""

import re
import sys
from pathlib import Path

SRC = Path("src/kernels/metal/cpi_kernels.metal")

# `kernel void name( ... ) {`  -- capture the name and the parameter list.
KERNEL = re.compile(r"kernel\s+void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)
BUFFER = re.compile(r"\[\[buffer\((\d+)\)\]\]")


def main() -> int:
    if not SRC.exists():
        print(f"error: {SRC} not found")
        return 1

    text = SRC.read_text(encoding="utf-8")
    problems = []
    checked = 0

    for m in KERNEL.finditer(text):
        name, params = m.group(1), m.group(2)
        checked += 1

        # Split the parameter list on top-level commas.
        args = [a.strip() for a in params.split(",") if "[[buffer(" in a]
        indexed = []
        for a in args:
            b = BUFFER.search(a)
            if b:
                indexed.append((int(b.group(1)), a))

        if not indexed:
            continue

        idxs = sorted(i for i, _ in indexed)
        if idxs != list(range(len(idxs))):
            problems.append(f"{name}: buffer indices are not contiguous from 0: {idxs}")

        params_args = [(i, a) for i, a in indexed if "Params&" in a]
        if not params_args:
            continue  # a kernel may legitimately take no params block
        if len(params_args) > 1:
            problems.append(f"{name}: more than one params block")
            continue

        p_idx = params_args[0][0]
        highest = max(idxs)
        if p_idx != highest:
            over = [a.split("[[")[0].strip() for i, a in indexed if i > p_idx]
            problems.append(
                f"{name}: params block is at buffer({p_idx}) but buffer({highest}) exists "
                f"-- dispatch() binds params LAST, so it would overwrite: {', '.join(over)}"
            )

    if problems:
        print(f"metal binding check: {len(problems)} problem(s) in {checked} kernels\n")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"metal binding check: OK ({checked} kernels, params block is last in each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
