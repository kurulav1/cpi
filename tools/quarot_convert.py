#!/usr/bin/env python3
"""quarot_convert.py - apply QuaRot R1 (hidden) + R2 (head_dim) rotations to a .ll2c model, offline.

Computational invariance: for orthogonal R the rotated model's fp16 logits match the original's
(proven in scratchpad/quarot_invariance.py). The point is that the rotated weights (and the
activations that exit each plain RMSNorm) are outlier-suppressed, so a later int4 quantize rounds
much cleaner. R1/R2 are fully offline, so the CPI runtime is unchanged except that the norm vectors
become all-ones (their gamma is absorbed into the following linear). R4 (online, MLP intermediate) is
a separate step, not applied here.

Rotation = randomized Hadamard D*H/sqrt(n) (QuaRot's construction). hidden and head_dim must be
powers of two here (Llama-3.1-8B: 4096 and 128). Transposes match the verified invariance test:
  input projs (wq,wk,wv,w1,w3): W' = (W * gamma) @ R1        (rotate input columns)
  output projs (wo,w2):         W' = R1^T @ W                (rotate output rows)
  wv also (R2, per KV head, output side):  W' = (blockdiag_KV(R2) @ (wv*gamma)) @ R1
  wo also (R2, per query head, input side): W' = R1^T @ (wo @ blockdiag_NH(R2)^T)
  tok_embeddings: E @ R1 ;  output(lm_head): (lm * gamma_final) @ R1 ;  every norm.weight -> 1
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turbo_quant_convert import read_ll2c_mmap, write_ll2c_streaming, get_tensor  # noqa: E402


def dense_rand_hadamard(n: int, rng: np.random.Generator) -> np.ndarray:
    """R = D * H / sqrt(n) as a dense [n,n] fp32 matrix (orthogonal). n must be a power of two."""
    if n & (n - 1):
        raise ValueError(f"dense_rand_hadamard needs power-of-two n, got {n}")
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H /= math.sqrt(n)
    signs = (rng.integers(0, 2, n) * 2 - 1).astype(np.float32)
    return signs[:, None] * H  # D @ H / sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-r2", action="store_true", help="apply only R1 (skip the head_dim rotation)")
    args = ap.parse_args()
    if args.input.resolve() == args.output.resolve():
        sys.exit("[quarot] input and output must differ")

    fields, table, mm, fh = read_ll2c_mmap(args.input.resolve())
    try:
        H = fields["hidden_size"]
        nh = fields["num_heads"]
        kv = fields["num_kv_heads"]
        hd = H // nh
        n_layers = fields["num_layers"]
        print(f"[quarot] hidden={H} heads={nh} kv={kv} head_dim={hd} layers={n_layers}")

        rng = np.random.default_rng(args.seed)
        R1 = dense_rand_hadamard(H, rng)                      # [H,H]
        R1T = R1.T.copy()
        R2 = None if args.no_r2 else dense_rand_hadamard(hd, rng)  # [hd,hd]

        idx = {name: (off, bc) for name, off, bc in table}

        def fp16(name):
            off, bc = idx[name]
            return get_tensor(mm, off, bc, dtype=np.float16)

        # Preload the small gamma vectors (norms) so input projs can absorb them.
        def gamma(name):
            return fp16(name).astype(np.float32)  # [H]

        NORMS = {"norm.weight"}
        for L in range(n_layers):
            NORMS.add(f"layers.{L}.attention_norm.weight")
            NORMS.add(f"layers.{L}.ffn_norm.weight")

        def rot_in(W, g):        # (W * gamma) @ R1 : rotate input columns
            return (W.astype(np.float32) * g[None, :]) @ R1

        def rot_out(W):          # R1^T @ W : rotate output rows
            return R1T @ W.astype(np.float32)

        def r2_out_rows(W):      # blockdiag_KV(R2) @ W : per-KV-head rotate output rows [kv*hd, H]
            Wr = W.reshape(kv, hd, -1)
            return np.einsum("ij,kjm->kim", R2, Wr).reshape(kv * hd, -1)

        def r2_in_cols(W):       # W @ blockdiag_NH(R2)^T : per-query-head rotate input cols [H, nh*hd]
            Wr = W.reshape(H, nh, hd)
            return np.einsum("hnj,ij->hni", Wr, R2).reshape(H, nh * hd)

        def provider(name):
            # norms -> all ones (gamma absorbed into the following linear).
            if name in NORMS:
                return np.ones(H, dtype=np.float16).view(np.uint8)
            if name == "tok_embeddings.weight":
                W = fp16(name).reshape(-1, H)
                return (W.astype(np.float32) @ R1).astype(np.float16).view(np.uint8)
            if name == "output.weight":
                W = fp16(name).reshape(-1, H)
                out = (W.astype(np.float32) * gamma("norm.weight")[None, :]) @ R1
                return out.astype(np.float16).view(np.uint8)
            # layer tensors
            if ".attention.wq" in name or ".attention.wk" in name:
                L = name.split(".")[1]
                W = fp16(name).reshape(-1, H)
                return rot_in(W, gamma(f"layers.{L}.attention_norm.weight")).astype(np.float16).view(np.uint8)
            if ".attention.wv" in name:
                L = name.split(".")[1]
                W = (fp16(name).reshape(-1, H).astype(np.float32)
                     * gamma(f"layers.{L}.attention_norm.weight")[None, :])
                if R2 is not None:
                    W = r2_out_rows(W)
                return (W @ R1).astype(np.float16).view(np.uint8)
            if ".attention.wo" in name:
                W = fp16(name).reshape(H, -1).astype(np.float32)
                if R2 is not None:
                    W = r2_in_cols(W)
                return rot_out(W).astype(np.float16).view(np.uint8)
            if ".feed_forward.w1" in name or ".feed_forward.w3" in name:
                L = name.split(".")[1]
                W = fp16(name).reshape(-1, H)
                return rot_in(W, gamma(f"layers.{L}.ffn_norm.weight")).astype(np.float16).view(np.uint8)
            if ".feed_forward.w2" in name:
                W = fp16(name).reshape(H, -1)
                return rot_out(W).astype(np.float16).view(np.uint8)
            # passthrough (should not hit for this model)
            off, bc = idx[name]
            return bytes(mm[off:off + bc])

        out_names = [n for n, _, _ in table]
        sizes = {n: bc for n, _, bc in table}
        print(f"[quarot] writing {args.output} ({len(out_names)} tensors)...")
        write_ll2c_streaming(args.output.resolve(), fields, out_names, sizes, provider)
        print("[quarot] done")
        return 0
    finally:
        mm.close()
        fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
