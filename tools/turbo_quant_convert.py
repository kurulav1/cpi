#!/usr/bin/env python3
"""
turbo_quant_convert.py - Convert a .ll2c weight file to TurboQuant tensors.

Two objectives are supported:
  - mse  : TQ3 (Hadamard-rotated Lloyd-Max 3-bit scalar quantization)
  - prod : TQ3 + 1-bit residual correction metadata/tensors (Qprod path)

For each eligible tensor (wq, wk, wv, wo, w1, w3) whose input dimension is
power-of-two, this script:
  1. Applies a random Hadamard rotation: D * H / sqrt(n)
  2. Uses a shared 8-level Lloyd-Max scalar codebook (fit for N(0,1))
  3. Quantizes each coordinate to 3 bits
  4. Packs 10 indices into one uint32
  5. Optionally computes residual 1-bit projected signatures for prod mode
  6. Stores additive tensors while preserving original fp16 tensors
"""

from __future__ import annotations

import argparse
import ctypes
import math
import mmap
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# LL2C container constants
# ---------------------------------------------------------------------------

MAGIC = b"LL2CUDA\x00"
HEADER_V3_FMT = "<8siiiiiiiiiiQffiii"  # HeaderV3
HEADER_V4_FMT = "<8siiiiiiiiiiQffiiiiii"  # HeaderV4
HEADER_V2_FMT = "<8siiiiiiiiiiQ"  # HeaderV2
HEADER_V1_FMT = "<8siiiiiiiiiQ"  # HeaderV1
ENTRY_FMT = "<64sqq"  # TensorEntry: name[64], offset int64, bytes int64

HEADER_SIZE = struct.calcsize(HEADER_V4_FMT)
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)


def _pad_name(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) >= 64:
        raise ValueError(f"Tensor name too long ({len(raw)} chars): {name!r}")
    return raw + b"\x00" * (64 - len(raw))


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _get_available_ram_bytes() -> int | None:
    """Best-effort available system RAM bytes."""
    # Try psutil first if available.
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Windows fallback without extra deps.
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)

    return None


def _estimate_chunk_peak_bytes(chunk_rows: int, in_features: int) -> int:
    """
    Conservative peak temporary RAM for one quantization chunk.
    """
    # W float32 + hadamard copy buffers + idx uint8 + misc overhead.
    base = chunk_rows * in_features
    return int(base * (4 + 4 + 4 + 1) + 32 * 1024 * 1024)


def _choose_safe_chunk_rows(
    requested_chunk_rows: int,
    in_features: int,
    ram_budget_bytes: int | None,
) -> int:
    """
    Return a chunk size that fits the RAM budget. If budget unknown, return requested.
    """
    if requested_chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    if ram_budget_bytes is None:
        return requested_chunk_rows

    safe = requested_chunk_rows
    while safe > 1 and _estimate_chunk_peak_bytes(safe, in_features) > ram_budget_bytes:
        safe //= 2
    return max(1, safe)


def _get_windows_memory_percent() -> float | None:
    if os.name != "nt":
        return None
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return float(st.dwMemoryLoad)
    return None


class ResourceGuard:
    """
    Host-level CPU/RAM guard with throttle-then-abort behavior.
    """

    def __init__(
        self,
        max_cpu_percent: float,
        max_memory_percent: float,
        sample_ms: int,
        sustain_ms: int,
        throttle_ms: int,
    ) -> None:
        self.max_cpu_percent = float(max_cpu_percent)
        self.max_memory_percent = float(max_memory_percent)
        self.sample_ms = max(0, int(sample_ms))
        self.sustain_ms = max(1, int(sustain_ms))
        self.throttle_ms = max(0, int(throttle_ms))

        self._last_sample_t = 0.0
        self._over_since_t: float | None = None
        self._psutil = None
        self._last_cpu: float | None = None
        self._last_mem: float | None = None

        try:
            import psutil  # type: ignore
            self._psutil = psutil
            # Prime CPU counters so next sample is meaningful.
            self._psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

    def _sample(self) -> tuple[float | None, float | None]:
        cpu = None
        mem = None
        if self._psutil is not None:
            try:
                cpu = float(self._psutil.cpu_percent(interval=None))
                mem = float(self._psutil.virtual_memory().percent)
            except Exception:
                cpu = None
                mem = None
        if mem is None:
            mem = _get_windows_memory_percent()
        self._last_cpu = cpu
        self._last_mem = mem
        return cpu, mem

    def enforce(self, stage: str, chunk_rows: int) -> int:
        """
        Returns the chunk_rows to use (may reduce under pressure).
        Raises RuntimeError when sustained over-limit.
        """
        now = time.monotonic()
        if self.sample_ms > 0 and (now - self._last_sample_t) * 1000.0 < self.sample_ms:
            return chunk_rows
        self._last_sample_t = now

        cpu, mem = self._sample()
        over_cpu = cpu is not None and cpu > self.max_cpu_percent
        over_mem = mem is not None and mem > self.max_memory_percent
        over = over_cpu or over_mem

        if not over:
            self._over_since_t = None
            return chunk_rows

        if self._over_since_t is None:
            self._over_since_t = now
        over_ms = int((now - self._over_since_t) * 1000.0)

        next_chunk_rows = chunk_rows
        if over_mem and chunk_rows > 1:
            next_chunk_rows = max(1, chunk_rows // 2)
            if next_chunk_rows != chunk_rows:
                print(
                    f"[tq3] resource pressure ({stage}): cpu={cpu} mem={mem} -> "
                    f"reducing chunk_rows {chunk_rows} -> {next_chunk_rows}"
                )

        if over_ms >= self.sustain_ms:
            raise RuntimeError(
                f"resource_limit_exceeded stage={stage} cpu={cpu} "
                f"(limit={self.max_cpu_percent}) mem={mem} "
                f"(limit={self.max_memory_percent}) sustained_ms={over_ms}"
            )

        print(
            f"[tq3] throttle stage={stage} cpu={cpu} mem={mem} "
            f"sustained_ms={over_ms}/{self.sustain_ms}"
        )
        if self.throttle_ms > 0:
            time.sleep(self.throttle_ms / 1000.0)
        return next_chunk_rows


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_ll2c_mmap(path: Path):
    """
    Parse a .ll2c file using mmap without full-file RAM load.

    Returns:
      fields: decoded header values in v3-compatible dict form
      table:  list[(name, offset, bytes)]
      mm:     mmap handle
      fh:     file handle
    """
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    if mm[:8] != MAGIC:
        sys.exit(f"[error] {path}: not a LL2CUDA file")

    version = struct.unpack_from("<i", mm, 8)[0]

    if version >= 4:
        hdr = struct.unpack_from(HEADER_V4_FMT, mm, 0)
        fields = {
            "magic": hdr[0],
            "version": 4,
            "vocab_size": hdr[2],
            "hidden_size": hdr[3],
            "intermediate_size": hdr[4],
            "num_layers": hdr[5],
            "num_heads": hdr[6],
            "num_kv_heads": hdr[7],
            "max_seq_len": hdr[8],
            "tensor_parallel": hdr[9],
            "tensor_count": hdr[10],
            "table_offset": hdr[11],
            "rope_theta": hdr[12],
            "norm_eps": hdr[13],
            "sliding_window": hdr[14],
            "flags": hdr[15],
            "model_family_id": hdr[16],
            "num_local_experts": hdr[17],
            "num_experts_per_tok": hdr[18],
            "expert_intermediate_size": hdr[19],
        }
    elif version >= 3:
        hdr = struct.unpack_from(HEADER_V3_FMT, mm, 0)
        fields = {
            "magic": hdr[0],
            "version": 4,
            "vocab_size": hdr[2],
            "hidden_size": hdr[3],
            "intermediate_size": hdr[4],
            "num_layers": hdr[5],
            "num_heads": hdr[6],
            "num_kv_heads": hdr[7],
            "max_seq_len": hdr[8],
            "tensor_parallel": hdr[9],
            "tensor_count": hdr[10],
            "table_offset": hdr[11],
            "rope_theta": hdr[12],
            "norm_eps": hdr[13],
            "sliding_window": hdr[14],
            "flags": hdr[15],
            "model_family_id": hdr[16],
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "expert_intermediate_size": 0,
        }
    elif version == 2:
        hdr = struct.unpack_from(HEADER_V2_FMT, mm, 0)
        fields = {
            "magic": hdr[0],
            "version": 4,
            "vocab_size": hdr[2],
            "hidden_size": hdr[3],
            "intermediate_size": hdr[4],
            "num_layers": hdr[5],
            "num_heads": hdr[6],
            "num_kv_heads": hdr[7],
            "max_seq_len": hdr[8],
            "tensor_parallel": hdr[9],
            "tensor_count": hdr[10],
            "table_offset": hdr[11],
            "rope_theta": 0.0,
            "norm_eps": 1e-5,
            "sliding_window": 0,
            "flags": 0,
            "model_family_id": 0,
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "expert_intermediate_size": 0,
        }
    else:
        hdr = struct.unpack_from(HEADER_V1_FMT, mm, 0)
        fields = {
            "magic": hdr[0],
            "version": 4,
            "vocab_size": hdr[2],
            "hidden_size": hdr[3],
            "intermediate_size": hdr[4],
            "num_layers": hdr[5],
            "num_heads": hdr[6],
            "num_kv_heads": hdr[6],  # v1 has no separate kv heads
            "max_seq_len": hdr[7],
            "tensor_parallel": hdr[8],
            "tensor_count": hdr[9],
            "table_offset": hdr[10],
            "rope_theta": 0.0,
            "norm_eps": 1e-5,
            "sliding_window": 0,
            "flags": 0,
            "model_family_id": 0,
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "expert_intermediate_size": 0,
        }

    print(f"[tq3] input version={version}, writing output as header v4")

    table_offset = fields["table_offset"]
    tensor_count = fields["tensor_count"]
    table = []
    for i in range(tensor_count):
        raw_name, offset, byte_count = struct.unpack_from(ENTRY_FMT, mm, table_offset + i * ENTRY_SIZE)
        name = raw_name.rstrip(b"\x00").decode("utf-8")
        table.append((name, offset, byte_count))

    return fields, table, mm, fh


def get_tensor(mm: mmap.mmap, offset: int, byte_count: int, dtype=np.uint8) -> np.ndarray:
    """Return a numpy view into mmap data at the given offset."""
    item_sz = np.dtype(dtype).itemsize
    return np.frombuffer(mm, dtype=dtype, count=byte_count // item_sz, offset=offset)


# ---------------------------------------------------------------------------
# Hadamard rotation
# ---------------------------------------------------------------------------

def hadamard_batch(W: np.ndarray, signs: np.ndarray, block_size: int | None = None) -> np.ndarray:
    """
    Apply normalized block-diagonal Hadamard rotation D*H/sqrt(block_size) row-wise in-place.

    W shape:     [rows, n], float32
    signs:       [n], int8 in {-1, +1}
    block_size:  size of each WHT sub-block (must be power-of-2 and divide n).
                 Defaults to n (single block, requires n to be power-of-2).

    When n is not a power of 2, set block_size = n & -n (largest power-of-2 factor
    of n) to apply WHT independently to each sub-block of that size.
    """
    rows, n = W.shape
    if block_size is None:
        block_size = n
    if not _is_power_of_2(block_size):
        raise ValueError(f"WHT block_size must be power-of-2, got block_size={block_size}")
    if n % block_size != 0:
        raise ValueError(f"n={n} must be divisible by block_size={block_size}")

    n_blocks = n // block_size
    for blk in range(n_blocks):
        s = blk * block_size
        e = s + block_size

        # Diagonal random signs for this block.
        W[:, s:e] *= signs[s:e].astype(np.float32)[None, :]

        # In-place Walsh-Hadamard butterfly on this block.
        h = 1
        while h < block_size:
            segs = block_size // (2 * h)
            Wr = W[:, s:e].reshape(rows, segs, 2, h)
            a = Wr[:, :, 0, :].copy()
            b = Wr[:, :, 1, :].copy()
            Wr[:, :, 0, :] = a + b
            Wr[:, :, 1, :] = a - b
            h <<= 1

        W[:, s:e] *= (1.0 / math.sqrt(block_size))

    return W


# ---------------------------------------------------------------------------
# Lloyd-Max codebook
# ---------------------------------------------------------------------------

def lloyd_max_gaussian(n_levels: int = 8, n_iter: int = 100) -> np.ndarray:
    """
    Compute scalar Lloyd-Max reconstruction points for N(0,1).
    Returns sorted float32 array of shape [n_levels].
    """
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(200_000).astype(np.float32)
    samples.sort()

    boundaries = np.percentile(samples, np.linspace(0, 100, n_levels + 1)[1:-1])
    pts = np.zeros(n_levels, dtype=np.float32)

    for k in range(n_levels):
        lo = boundaries[k - 1] if k > 0 else -1e9
        hi = boundaries[k] if k < n_levels - 1 else 1e9
        mask = (samples >= lo) & (samples < hi)
        pts[k] = samples[mask].mean() if mask.any() else (lo + hi) / 2

    for _ in range(n_iter):
        dists = np.abs(samples[:, None] - pts[None, :])
        labels = dists.argmin(axis=1)
        new_pts = np.array(
            [samples[labels == k].mean() if (labels == k).any() else pts[k] for k in range(n_levels)],
            dtype=np.float32,
        )
        if np.max(np.abs(new_pts - pts)) < 1e-7:
            break
        pts = new_pts

    return np.sort(pts)


# ---------------------------------------------------------------------------
# TQ3 quantization
# ---------------------------------------------------------------------------

def quantise_matrix_tq3(
    W_fp16: np.ndarray,
    signs: np.ndarray,
    codebook: np.ndarray,
    chunk_rows: int = 128,
    block_size: int | None = None,
    qjl_indices: np.ndarray | None = None,
    qjl_signs: np.ndarray | None = None,
    resource_guard: ResourceGuard | None = None,
):
    """
    Quantize fp16 matrix to TQ3 with bounded chunk memory.

    Returns:
      packed: uint32[out, ceil(in/10)]
      scales: float16[out]
      residual_bits: uint32[out, ceil(qjl_dim/32)] when qjl_* provided, else None
      residual_scales: float16[out] when qjl_* provided, else None
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")

    out_f, in_f = W_fp16.shape
    words_per_row = (in_f + 9) // 10

    packed = np.zeros((out_f, words_per_row), dtype=np.uint32)
    scales = np.zeros((out_f,), dtype=np.float16)
    qjl_dim = 0 if qjl_indices is None else int(qjl_indices.shape[0])
    residual_words = (qjl_dim + 31) // 32
    residual_bits = None
    residual_scales = None
    if qjl_dim > 0:
        residual_bits = np.zeros((out_f, residual_words), dtype=np.uint32)
        residual_scales = np.zeros((out_f,), dtype=np.float16)

    boundaries = ((codebook[:-1] + codebook[1:]) / 2).astype(np.float32)  # 7 midpoints
    word_pos = np.arange(words_per_row)

    r0 = 0
    active_chunk_rows = int(chunk_rows)
    while r0 < out_f:
        if resource_guard is not None:
            active_chunk_rows = max(1, int(resource_guard.enforce("quantize_chunk", active_chunk_rows)))
        r1 = min(r0 + active_chunk_rows, out_f)
        W = W_fp16[r0:r1].astype(np.float32, copy=True)

        hadamard_batch(W, signs, block_size=block_size)

        # Normalize row-wise to approximately match N(0,1) codebook assumptions.
        row_rms = np.sqrt(np.mean(W**2, axis=1, keepdims=True)).clip(min=1e-9)
        W /= row_rms

        idx = np.searchsorted(boundaries, W.ravel()).reshape(r1 - r0, in_f).astype(np.uint8)

        for k in range(10):
            col = word_pos * 10 + k
            valid = col < in_f
            if valid.any():
                packed[r0:r1, valid] |= idx[:, col[valid]].astype(np.uint32) << (k * 3)

        scales[r0:r1] = row_rms.squeeze(1).astype(np.float16)

        if qjl_dim > 0 and residual_bits is not None and residual_scales is not None:
            recon = codebook[idx]
            residual = W - recon
            residual_scales[r0:r1] = np.sqrt(np.mean(residual**2, axis=1)).astype(np.float16)
            proj = residual[:, qjl_indices] * qjl_signs.astype(np.float32)[None, :]
            sign_bits = proj >= 0.0
            for b in range(qjl_dim):
                if not np.any(sign_bits[:, b]):
                    continue
                word = b // 32
                bit = b & 31
                residual_bits[r0:r1, word] |= sign_bits[:, b].astype(np.uint32) << bit

        r0 = r1

    return packed, scales, residual_bits, residual_scales


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_ll2c_streaming(
    path: Path,
    fields: dict,
    out_names: list[str],
    tensor_sizes: dict[str, int],
    tensor_provider,
) -> int:
    """
    Write v3 ll2c file in a single streaming pass.
    Returns total aligned output bytes.
    """
    tensor_count = len(out_names)

    table_base = _align8(HEADER_SIZE)
    data_start = _align8(table_base + tensor_count * ENTRY_SIZE)

    offsets = {}
    cursor = data_start
    for name in out_names:
        offsets[name] = cursor
        cursor = _align8(cursor + tensor_sizes[name])

    hdr = struct.pack(
        HEADER_V4_FMT,
        fields["magic"],
        fields["version"],
        fields["vocab_size"],
        fields["hidden_size"],
        fields["intermediate_size"],
        fields["num_layers"],
        fields["num_heads"],
        fields["num_kv_heads"],
        fields["max_seq_len"],
        fields["tensor_parallel"],
        tensor_count,
        table_base,
        fields["rope_theta"],
        fields["norm_eps"],
        fields["sliding_window"],
        fields["flags"],
        fields["model_family_id"],
        int(fields.get("num_local_experts", 0)),
        int(fields.get("num_experts_per_tok", 0)),
        int(fields.get("expert_intermediate_size", 0)),
    )

    table = bytearray()
    for name in out_names:
        table += struct.pack(ENTRY_FMT, _pad_name(name), offsets[name], tensor_sizes[name])

    tmp_path = path.with_name(path.name + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    with open(tmp_path, "wb") as out:
        out.write(hdr)
        pad = table_base - len(hdr)
        if pad > 0:
            out.write(b"\x00" * pad)

        out.write(table)
        pad = data_start - (table_base + len(table))
        if pad > 0:
            out.write(b"\x00" * pad)

        for name in out_names:
            data = tensor_provider(name)
            raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
            expect = tensor_sizes[name]
            if len(raw) != expect:
                raise ValueError(f"{name}: expected {expect} bytes, got {len(raw)}")
            out.write(raw)
            pad = _align8(len(raw)) - len(raw)
            if pad > 0:
                out.write(b"\x00" * pad)

    # Atomic replace so interrupted runs do not leave a corrupt final file.
    os.replace(tmp_path, path)
    return cursor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TQ3_ELIGIBLE = [
    "attention.wq",
    "attention.wk",
    "attention.wv",
    "attention.wo",
    "feed_forward.w1",
    "feed_forward.w3",
]

LAYER_TQ3_RE = re.compile(r"^layers\.(\d+)\..*\.(tq3|tq3s|tq3r|tq3rs)$")


def _estimate_output_bytes(out_names: list[str], tensor_sizes: dict[str, int]) -> int:
    total = _align8(_align8(HEADER_SIZE) + len(out_names) * ENTRY_SIZE)
    for name in out_names:
        total += _align8(tensor_sizes[name])
    return total


def _dequant_int8_matrix(
    mm: mmap.mmap,
    in_index: dict[str, tuple[int, int]],
    base_name: str,
    hidden: int,
) -> tuple[np.ndarray, int] | None:
    q_name = f"{base_name}.int8"
    s_name = f"{base_name}.scale"
    if q_name not in in_index or s_name not in in_index:
        return None

    q_off, q_bc = in_index[q_name]
    if q_bc % hidden != 0:
        return None
    out_f = q_bc // hidden
    q_i8 = get_tensor(mm, q_off, q_bc, dtype=np.int8).reshape(out_f, hidden).astype(np.float32, copy=True)

    s_off, s_bc = in_index[s_name]
    if s_bc % 4 != 0:
        return None
    scales = get_tensor(mm, s_off, s_bc, dtype=np.float32).copy()
    if scales.size == 1:
        q_i8 *= float(scales[0])
    elif scales.size == out_f:
        q_i8 *= scales[:, None]
    else:
        return None

    return q_i8.astype(np.float16), out_f


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Input .ll2c file")
    parser.add_argument("output", type=Path, help="Output .ll2c file with TQ3 tensors")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for Hadamard sign generation (default: 42)")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=128,
        help="Rows per quantization chunk (smaller uses less RAM, default: 128)",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=0.0,
        help="Hard RAM budget in GiB for quantization temp memory (0=auto from system available RAM)",
    )
    parser.add_argument(
        "--ram-safety-frac",
        type=float,
        default=0.60,
        help="Use at most this fraction of available RAM for temporary quantization memory (default: 0.60)",
    )
    parser.add_argument(
        "--strict-ram",
        action="store_true",
        help="Fail if requested --chunk-rows exceeds RAM budget instead of auto-reducing it",
    )
    parser.add_argument(
        "--objective",
        choices=["mse", "prod"],
        default="mse",
        help="TurboQuant objective: mse (TQ3 only) or prod (TQ3 + residual 1-bit stage)",
    )
    parser.add_argument(
        "--qjl-seed",
        type=int,
        default=17,
        help="Seed for residual 1-bit projection metadata in --objective prod mode",
    )
    parser.add_argument(
        "--qjl-dim",
        type=int,
        default=256,
        help="Residual projection dimension in --objective prod mode (<= hidden_size; 0 means hidden_size)",
    )
    parser.add_argument(
        "--require-full-tq",
        action="store_true",
        help="Fail when any eligible layer tensor cannot be converted to TurboQuant tensors",
    )
    parser.add_argument(
        "--max-cpu-percent",
        type=float,
        default=85.0,
        help="Host CPU utilization hard limit percentage (default: 85)",
    )
    parser.add_argument(
        "--max-memory-percent",
        type=float,
        default=85.0,
        help="Host physical memory utilization hard limit percentage (default: 85)",
    )
    parser.add_argument(
        "--resource-sample-ms",
        type=int,
        default=250,
        help="Resource sampling interval in milliseconds (default: 250)",
    )
    parser.add_argument(
        "--resource-sustain-ms",
        type=int,
        default=5000,
        help="Abort after this long sustained over-limit pressure in milliseconds (default: 5000)",
    )
    parser.add_argument(
        "--resource-throttle-ms",
        type=int,
        default=50,
        help="Throttle sleep while over limit in milliseconds (default: 50)",
    )
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        sys.exit("[error] --chunk-rows must be > 0")
    if not (0.05 <= args.ram_safety_frac <= 0.95):
        sys.exit("[error] --ram-safety-frac must be in [0.05, 0.95]")
    if args.max_ram_gb < 0:
        sys.exit("[error] --max-ram-gb must be >= 0")
    if args.qjl_dim < 0:
        sys.exit("[error] --qjl-dim must be >= 0")
    if not (0.0 < args.max_cpu_percent <= 100.0):
        sys.exit("[error] --max-cpu-percent must be in (0, 100]")
    if not (0.0 < args.max_memory_percent <= 100.0):
        sys.exit("[error] --max-memory-percent must be in (0, 100]")
    if args.resource_sample_ms < 0:
        sys.exit("[error] --resource-sample-ms must be >= 0")
    if args.resource_sustain_ms <= 0:
        sys.exit("[error] --resource-sustain-ms must be > 0")
    if args.resource_throttle_ms < 0:
        sys.exit("[error] --resource-throttle-ms must be >= 0")

    in_path = args.input.resolve()
    out_path = args.output.resolve()
    if in_path == out_path:
        sys.exit("[error] input and output must be different files")

    print(f"[tq3] reading {in_path} ...")
    fields, in_table, mm, fh = read_ll2c_mmap(in_path)
    try:
        hidden = fields["hidden_size"]
        n_layers = fields["num_layers"]
        # Block-diagonal WHT: block_size is the largest power-of-2 factor of hidden.
        # For power-of-2 hidden this equals hidden (single block, same as before).
        # For non-power-of-2 (e.g. 5120=5×1024, 3584=7×512) WHT runs per block.
        block_size = hidden & -hidden
        if block_size < 64:
            sys.exit(f"[error] hidden_size={hidden} has too-small power-of-2 factor "
                     f"({block_size}); minimum supported block_size is 64")

        objective_id = 0 if args.objective == "mse" else 1
        qjl_dim = 0
        qjl_indices = None
        qjl_signs = None
        if objective_id == 1:
            qjl_dim = hidden if args.qjl_dim == 0 else args.qjl_dim
            if qjl_dim <= 0:
                sys.exit("[error] --qjl-dim resolved to <= 0")
            if qjl_dim > hidden:
                sys.exit(f"[error] --qjl-dim ({qjl_dim}) exceeds hidden_size ({hidden})")
            qrng = np.random.default_rng(args.qjl_seed)
            qjl_indices = qrng.permutation(hidden).astype(np.int32)[:qjl_dim]
            qjl_signs = qrng.choice([-1, 1], size=qjl_dim).astype(np.int8)

        n_wht_blocks = hidden // block_size
        print(f"[tq3] model: layers={n_layers}, hidden={hidden}, "
              f"block_size={block_size}, wht_blocks={n_wht_blocks}, objective={args.objective}")
        print(f"[tq3] requested chunk_rows={args.chunk_rows}")
        if objective_id == 1:
            print(f"[tq3] residual projection: qjl_dim={qjl_dim}, qjl_seed={args.qjl_seed}")

        # Global Hadamard signs.
        rng = np.random.default_rng(args.seed)
        signs = rng.choice([-1, 1], size=hidden).astype(np.int8)

        # Shared codebook.
        print("[tq3] computing Lloyd-Max codebook ...")
        codebook_f32 = lloyd_max_gaussian(n_levels=8)
        codebook_f16 = codebook_f32.astype(np.float16)
        print(f"[tq3] codebook: {codebook_f16}")

        in_index = {name: (off, bc) for name, off, bc in in_table}

        avail_ram = _get_available_ram_bytes()
        if args.max_ram_gb > 0:
            ram_budget = int(args.max_ram_gb * (1024**3))
            print(f"[tq3] RAM budget forced by flag: {ram_budget/1024**3:.2f} GiB")
        elif avail_ram is not None:
            ram_budget = int(avail_ram * args.ram_safety_frac)
            print(
                f"[tq3] RAM budget from system: avail={avail_ram/1024**3:.2f} GiB, "
                f"budget={ram_budget/1024**3:.2f} GiB (safety frac={args.ram_safety_frac:.2f})"
            )
        else:
            ram_budget = None
            print("[tq3] RAM availability unknown; using requested chunk size as-is")

        safe_chunk_rows = _choose_safe_chunk_rows(args.chunk_rows, hidden, ram_budget)
        est_peak = _estimate_chunk_peak_bytes(safe_chunk_rows, hidden)
        if safe_chunk_rows != args.chunk_rows:
            msg = (
                f"[tq3] lowering chunk_rows {args.chunk_rows} -> {safe_chunk_rows} "
                f"to satisfy RAM guardrail (est peak temp {est_peak/1024**3:.2f} GiB)"
            )
            if args.strict_ram:
                sys.exit("[error] " + msg.replace("[tq3] ", ""))
            print(msg)
        else:
            print(f"[tq3] using chunk_rows={safe_chunk_rows} (est peak temp {est_peak/1024**3:.2f} GiB)")

        resource_guard = ResourceGuard(
            max_cpu_percent=args.max_cpu_percent,
            max_memory_percent=args.max_memory_percent,
            sample_ms=args.resource_sample_ms,
            sustain_ms=args.resource_sustain_ms,
            throttle_ms=args.resource_throttle_ms,
        )
        print(
            f"[tq3] host limits: cpu<={args.max_cpu_percent:.1f}% "
            f"mem<={args.max_memory_percent:.1f}% sample_ms={args.resource_sample_ms} "
            f"sustain_ms={args.resource_sustain_ms} throttle_ms={args.resource_throttle_ms}"
        )

        # Determine quantizable tensor shapes.
        tq3_info: dict[str, tuple[int, int, int, str]] = {}  # full_name -> (out_f, in_f, words_per_row, src_kind)
        missing_eligible: list[str] = []
        for layer in range(n_layers):
            for base in TQ3_ELIGIBLE:
                full = f"layers.{layer}.{base}"
                out_f = 0
                src_kind = ""
                if full in in_index:
                    _, bc = in_index[full]
                    if (bc % 2) == 0:
                        elems = bc // 2  # fp16
                        if elems % hidden == 0:
                            out_f = elems // hidden
                            src_kind = "fp16"
                else:
                    q_name = f"{full}.int8"
                    s_name = f"{full}.scale"
                    if q_name in in_index and s_name in in_index:
                        _, q_bc = in_index[q_name]
                        _, s_bc = in_index[s_name]
                        if q_bc % hidden == 0:
                            cand_out = q_bc // hidden
                            if s_bc in (4, cand_out * 4):
                                out_f = cand_out
                                src_kind = "int8"
                if out_f <= 0:
                    missing_eligible.append(full)
                    continue
                wpr = (hidden + 9) // 10
                tq3_info[full] = (out_f, hidden, wpr, src_kind)

        if not tq3_info:
            sys.exit("[error] no eligible tensors found for TurboQuant conversion")
        src_fp16 = sum(1 for _, _, _, k in tq3_info.values() if k == "fp16")
        src_int8 = sum(1 for _, _, _, k in tq3_info.values() if k == "int8")
        print(f"[tq3] source coverage: fp16={src_fp16} int8={src_int8}")
        if missing_eligible:
            print(f"[tq3] coverage: converted {len(tq3_info)} tensors, missing {len(missing_eligible)} eligible tensors")
            if args.require_full_tq:
                preview = ", ".join(missing_eligible[:8])
                sys.exit(f"[error] --require-full-tq failed; missing eligible tensors (sample): {preview}")

        # Build output tensor manifest.
        out_names: list[str] = []
        tensor_sizes: dict[str, int] = {}

        for name, _, bc in in_table:
            out_names.append(name)
            tensor_sizes[name] = bc

        out_names.append("tq3_codebook")
        tensor_sizes["tq3_codebook"] = 8 * 2  # float16[8]
        out_names.append("tq3_signs_hidden")
        tensor_sizes["tq3_signs_hidden"] = hidden  # int8[hidden]
        out_names.append("tq3_block_size")
        tensor_sizes["tq3_block_size"] = 4         # int32 scalar
        out_names.append("tq_objective")
        tensor_sizes["tq_objective"] = 4  # int32 scalar
        if objective_id == 1:
            out_names.append("tq_qjl_dim")
            tensor_sizes["tq_qjl_dim"] = 4
            out_names.append("tq_qjl_seed")
            tensor_sizes["tq_qjl_seed"] = 4
            out_names.append("tq_qjl_indices_hidden")
            tensor_sizes["tq_qjl_indices_hidden"] = qjl_dim * 4
            out_names.append("tq_qjl_signs_hidden")
            tensor_sizes["tq_qjl_signs_hidden"] = qjl_dim

        for layer in range(n_layers):
            for base in TQ3_ELIGIBLE:
                full = f"layers.{layer}.{base}"
                if full not in tq3_info:
                    continue
                out_f, _, wpr, _ = tq3_info[full]
                packed_name = f"{full}.tq3"
                scales_name = f"{full}.tq3s"
                out_names.append(packed_name)
                tensor_sizes[packed_name] = out_f * wpr * 4
                out_names.append(scales_name)
                tensor_sizes[scales_name] = out_f * 2
                if objective_id == 1:
                    residual_words = (qjl_dim + 31) // 32
                    residual_name = f"{full}.tq3r"
                    residual_scales_name = f"{full}.tq3rs"
                    out_names.append(residual_name)
                    tensor_sizes[residual_name] = out_f * residual_words * 4
                    out_names.append(residual_scales_name)
                    tensor_sizes[residual_scales_name] = out_f * 2

        # Report conversion size ratio.
        total_fp16 = 0
        total_tq3 = 0
        total_residual = 0
        for layer in range(n_layers):
            for base in TQ3_ELIGIBLE:
                full = f"layers.{layer}.{base}"
                if full not in tq3_info:
                    continue
                out_f, in_f, wpr, _ = tq3_info[full]
                total_fp16 += out_f * in_f * 2
                total_tq3 += (out_f * wpr * 4) + (out_f * 2)
                if objective_id == 1:
                    residual_words = (qjl_dim + 31) // 32
                    total_residual += (out_f * residual_words * 4) + (out_f * 2)

        ratio = (total_fp16 / total_tq3) if total_tq3 else 0.0
        if objective_id == 1:
            combined = total_tq3 + total_residual
            ratio_combined = (total_fp16 / combined) if combined else 0.0
            print(
                f"[tq3] {total_fp16/1024**2:.0f} MiB fp16 -> "
                f"{total_tq3/1024**2:.0f} MiB TQ3 + {total_residual/1024**2:.0f} MiB residual "
                f"({ratio_combined:.2f}x)"
            )
        else:
            print(f"[tq3] {total_fp16/1024**2:.0f} MiB fp16 -> {total_tq3/1024**2:.0f} MiB TQ3 ({ratio:.2f}x)")

        # Disk-space sanity check before expensive run.
        est_bytes = _estimate_output_bytes(out_names, tensor_sizes)
        out_parent = out_path.parent if out_path.parent != Path("") else Path(".")
        out_parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(out_parent).free
        safety_margin = 2 * 1024**3
        if free_bytes < est_bytes + safety_margin:
            need = est_bytes + safety_margin - free_bytes
            sys.exit(f"[error] insufficient disk in {out_parent} (need ~{need/1024**3:.2f} GiB more)")

        # One-layer cache: keeps memory bounded while writing.
        current_layer_id: int | None = None
        current_layer_cache: dict[str, np.ndarray] = {}

        def quantise_layer_tensors(layer: int) -> dict[str, np.ndarray]:
            t0 = time.time()
            resource_guard.enforce(f"quantize_layer_{layer}_begin", safe_chunk_rows)
            cache: dict[str, np.ndarray] = {}
            for base in TQ3_ELIGIBLE:
                full = f"layers.{layer}.{base}"
                if full not in tq3_info:
                    continue
                out_f, in_f, _, src_kind = tq3_info[full]
                if src_kind == "fp16":
                    off, bc = in_index[full]
                    W_fp16 = get_tensor(mm, off, bc, dtype=np.float16).reshape(out_f, in_f).copy()
                else:
                    loaded = _dequant_int8_matrix(mm, in_index, full, hidden)
                    if loaded is None:
                        raise ValueError(f"failed to dequantize source int8 tensor for {full}")
                    W_fp16, decoded_out = loaded
                    if decoded_out != out_f:
                        raise ValueError(f"{full}: expected {out_f} rows, got {decoded_out} after int8 decode")
                packed, scales, residual_bits, residual_scales = quantise_matrix_tq3(
                    W_fp16=W_fp16,
                    signs=signs,
                    codebook=codebook_f32,
                    chunk_rows=safe_chunk_rows,
                    block_size=block_size,
                    qjl_indices=qjl_indices,
                    qjl_signs=qjl_signs,
                    resource_guard=resource_guard,
                )
                cache[f"{full}.tq3"] = packed
                cache[f"{full}.tq3s"] = scales
                if objective_id == 1 and residual_bits is not None and residual_scales is not None:
                    cache[f"{full}.tq3r"] = residual_bits
                    cache[f"{full}.tq3rs"] = residual_scales
                resource_guard.enforce(f"quantize_tensor_{full}", safe_chunk_rows)
            elapsed = time.time() - t0
            print(f"[tq3] layer {layer:2d}/{n_layers-1} quantized ({elapsed:.1f}s)")
            return cache

        def tensor_provider(name: str):
            nonlocal current_layer_id, current_layer_cache
            m = LAYER_TQ3_RE.match(name)
            if m:
                layer = int(m.group(1))
                if current_layer_id != layer:
                    current_layer_cache = quantise_layer_tensors(layer)
                    current_layer_id = layer
                if name not in current_layer_cache:
                    raise KeyError(f"internal error: missing computed tensor {name}")
                return current_layer_cache[name].view(np.uint8)

            if name == "tq3_codebook":
                return codebook_f16.view(np.uint8)
            if name == "tq3_signs_hidden":
                return signs.view(np.uint8)
            if name == "tq3_block_size":
                return np.asarray([block_size], dtype=np.int32).view(np.uint8)
            if name == "tq_objective":
                return np.asarray([objective_id], dtype=np.int32).view(np.uint8)
            if name == "tq_qjl_dim":
                return np.asarray([qjl_dim], dtype=np.int32).view(np.uint8)
            if name == "tq_qjl_seed":
                return np.asarray([args.qjl_seed], dtype=np.int32).view(np.uint8)
            if name == "tq_qjl_indices_hidden":
                return qjl_indices.view(np.uint8)
            if name == "tq_qjl_signs_hidden":
                return qjl_signs.view(np.uint8)

            # Original tensor passthrough.
            off, bc = in_index[name]
            return bytes(mm[off : off + bc])

        fields["tensor_count"] = len(out_names)
        print(f"[tq3] writing {out_path} ...")
        total_bytes = write_ll2c_streaming(out_path, fields, out_names, tensor_sizes, tensor_provider)
        print(f"[tq3] wrote {out_path} ({total_bytes / 1024**3:.2f} GiB)")
        return 0
    finally:
        mm.close()
        fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
