#!/usr/bin/env python3
"""Two mechanical checks on the Metal kernels and the C++ that dispatches them.

1. Every Metal kernel must take its params block as its LAST buffer binding.
2. Every tile constant the shader defines and the C++ mirrors must agree.

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

The second check exists because the tile constants live in three places -- the shader
#define, the engine's mirror, and metal_smoke's mirror -- and a caller whose tile size
disagrees with the shader does not fail to build. It computes a WRONG ANSWER, and only
in the shapes that particular caller exercises. That is not hypothetical: metal_smoke
restated GEMM_BN as 64 while the shader moved to 32, and every gemm_f16 check failed
with max_abs ~3.8 while the engine (which mirrors the constant correctly) stayed right
and fast. The comment above that dispatch claimed it derived the value rather than
hardcoding it. Comments do not hold; this does.
"""

import re
import sys
from pathlib import Path

SRC = Path("src/kernels/metal/cpi_kernels.metal")
SHADER_DIR = Path("src/kernels/metal")

# shader #define -> every C++ constant that must equal it, as (file, name).
#
# Add a file here the moment it restates a tile constant. metal_gemm_bench was missed on the
# first pass of this table and drifted to a wrong tile within the hour -- it caught itself only
# because it verifies against a reference, which the next such file might not.
ENGINE = "src/engine/plan_metal_engine.cpp"
SMOKE = "src/tests/metal_smoke.cpp"
GEMM_BENCH = "src/tests/metal_gemm_bench.cpp"
MIRRORS = {
    "GEMM_BN": [(ENGINE, "kGemmBN"), (SMOKE, "kSmokeGemmBN"), (GEMM_BENCH, "kBN")],
    "GEMM_FBM": [(ENGINE, "kGemmFBM"), (SMOKE, "kSmokeGemmFBM"), (GEMM_BENCH, "kFBM")],
    "GEMM_SPLITK": [(ENGINE, "kGemmSplitK")],
    "GEMM_QBN": [(ENGINE, "kGemmQBN")],
    "GEMM_QBK": [(ENGINE, "kGemmQBK")],
    "GEMM_RF": [(ENGINE, "kGemmRF")],
    "GEMM_CF": [(ENGINE, "kGemmCF")],
    "GEMV_TILE": [(ENGINE, "kGemvTile")],
    "Q_BLOCK": [(ENGINE, "kQBlock")],
    "QMM_BLOCK": [(ENGINE, "kQMMBlock")],
    "KEY_BLOCK": [(ENGINE, "kKeyBlock")],
    "MM_KEY_BLOCK": [(ENGINE, "kMMKeyBlock")],
    "DEC_KEY_BLOCK": [(ENGINE, "kDecKeyBlock")],
}

DEFINE = re.compile(r"^#define\s+(\w+)\s+(\d+)\s*(?://.*)?$", re.M)
CONST = re.compile(r"\b(?:constexpr|const)\s+[\w:]+(?:\s+[\w:]+)?\s+(\w+)\s*=\s*(\d+)\s*;")


def shader_defines() -> dict:
    """#define name -> int, across every shader family file."""
    out = {}
    for f in sorted(SHADER_DIR.glob("*.metal")):
        if f.name == SRC.name:
            continue  # the concatenated build artifact, if it is sitting here
        for m in DEFINE.finditer(f.read_text(encoding="utf-8")):
            out[m.group(1)] = int(m.group(2))
    return out


def cpp_constants(path: str) -> dict:
    f = Path(path)
    if not f.exists():
        return {}
    return {m.group(1): int(m.group(2)) for m in CONST.finditer(f.read_text(encoding="utf-8"))}


def check_mirrors() -> list:
    defines = shader_defines()
    cache = {}
    problems = []
    for shader_name, mirrors in MIRRORS.items():
        if shader_name not in defines:
            problems.append(f"{shader_name}: not #defined in any {SHADER_DIR}/*.metal")
            continue
        want = defines[shader_name]
        for path, cname in mirrors:
            if path not in cache:
                cache[path] = cpp_constants(path)
            got = cache[path].get(cname)
            if got is None:
                problems.append(f"{cname}: not found in {path} (mirrors {shader_name})")
            elif got != want:
                problems.append(
                    f"{cname} in {path} is {got} but the shader's {shader_name} is {want} "
                    f"-- a caller whose tile size disagrees computes a wrong answer, silently"
                )
    return problems

# `kernel void name( ... ) {`  -- capture the name and the parameter list.
KERNEL = re.compile(r"kernel\s+void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)
BUFFER = re.compile(r"\[\[buffer\((\d+)\)\]\]")


def main() -> int:
    # The kernels live as family files (00_common, 10_dense, ...); the single-file
    # cpi_kernels.metal is a BUILD ARTIFACT and is absent from a fresh checkout. This used to
    # read only that path, so on CI -- which runs against a checkout -- it printed "not found"
    # and returned 1 without checking a single kernel. Read the family files.
    parts = [f for f in sorted(SHADER_DIR.glob("*.metal")) if f.name != SRC.name]
    if not parts and SRC.exists():
        parts = [SRC]
    if not parts:
        print(f"error: no shader sources under {SHADER_DIR}")
        return 1

    text = "\n".join(f.read_text(encoding="utf-8") for f in parts)
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

    mirror_problems = check_mirrors()

    if problems or mirror_problems:
        if problems:
            print(f"metal binding check: {len(problems)} problem(s) in {checked} kernels\n")
            for p in problems:
                print(f"  {p}")
        if mirror_problems:
            print(f"metal tile-constant check: {len(mirror_problems)} mismatch(es)\n")
            for p in mirror_problems:
                print(f"  {p}")
        return 1

    print(f"metal binding check: OK ({checked} kernels, params block is last in each)")
    print(f"metal tile-constant check: OK ({len(MIRRORS)} constants agree with their mirrors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
