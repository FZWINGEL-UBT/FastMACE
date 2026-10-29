"""CUTLASS tensor-core FP32 GEMMs (``libfm_tc_gemm.so``, ``build.sh``) for the tensor-core GEMM screen.

Screen code, not library code: ``install(variant)`` reroutes the opt-in ``tf32x3_gemms`` record's two launch
functions from Triton's ``tf32x3`` to CUTLASS, in this process only:

- ``fastmace.kernels.triton.tf32x3_gemm._launch`` (dense products and their input gradients);
- ``fastmace.kernels.triton.segment_gemm._gemm`` with ``precision="tf32x3"`` (the joined products and their input
  gradients, written straight into the joined layout as a strided batched GEMM over the components).

Sub-tile products (K or N < 32) stay on SIMT IEEE as the record routes them (``uses_tf32x3_tile``, OPT5-T05). The
launch functions run at capture time, so the CUDA graphs hold the CUTLASS kernels; nothing is decided per replay.
Tools call ``install_from_env()``: ``FASTMACE_SCREEN_TC_GEMM`` = ``cutlass-rz3`` | ``cutlass-rn3`` |
``cutlass-rn4`` | ``cutlass-tf32`` (unset or ``triton``: no change), with the library at
``FASTMACE_SCREEN_TC_GEMM_LIB``. The environment variable reaches the gate's and the NVE tool's child processes.

Variants: ``rz3`` = CUTLASS's own 3xTF32 (big rounded toward zero, small half-ulp truncated); ``rn3`` = both
parts rounded to nearest (Ootomo and Yokota's split); ``rn4`` = ``rn3`` plus small * small; ``tf32`` = 1xTF32,
a rate reference that is not FP32-accurate.
"""

from __future__ import annotations

import ctypes
import math
import os
from functools import cache

VARIANTS = {"rz3": 0, "rn3": 1, "rn4": 2, "tf32": 3}
TILES = ((64, 64), (64, 128))  # cutlass_tc.cu's Tile<0>, Tile<1>
ENV, ENV_LIB = "FASTMACE_SCREEN_TC_GEMM", "FASTMACE_SCREEN_TC_GEMM_LIB"
_ARGS = [ctypes.c_int] * 7 + [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong] * 3 + [ctypes.c_void_p]


@cache
def library(path=None):
    lib = ctypes.CDLL(path or os.environ[ENV_LIB])
    for v in VARIANTS.values():
        fn = getattr(lib, f"fm_tc_gemm_v{v}")
        fn.argtypes, fn.restype = _ARGS, ctypes.c_int
    return lib


@cache
def _sms(index):
    import torch

    return torch.cuda.get_device_properties(index).multi_processor_count


def tile_for(m, n, batch, sms):
    """The tile rule: 64x128 only where N >= 128 and its grid still fills two waves of the SMs, 64x64 otherwise."""
    if n >= 128 and math.ceil(m / 64) * math.ceil(n / 128) * batch >= 2 * sms:
        return 1
    return 0


def batched(variant, m, n, k, batch, a, lda, sa, b, b_col_major, ldb, sb, c, ldc, sc, tile=None, stream=None):
    """C[i] = A[i] @ B[i] (i < batch) from raw pointers into ``a``, ``b``, ``c`` (byte offsets already applied
    as ``(tensor, element_offset)`` pairs); A, C row-major. Raises on a CUDA or CUTLASS failure."""
    import torch

    (ta, oa), (tb, ob), (tc, oc) = a, b, c
    device = tc.device
    if tile is None:
        tile = tile_for(m, n, batch, _sms(device.index if device.index is not None else torch.cuda.current_device()))
    if stream is None:
        stream = torch.cuda.current_stream(device).cuda_stream
    fn = getattr(library(), f"fm_tc_gemm_v{VARIANTS[variant]}")
    pa, pb, pc = (ta.data_ptr() + 4 * oa, tb.data_ptr() + 4 * ob, tc.data_ptr() + 4 * oc)
    status = 1
    if all(p % 16 == 0 for p in (pa, pb, pc)) and all(int(x) % 4 == 0 for x in (lda, sa, ldb, sb, ldc, sc, k, n)):
        status = fn(tile, int(b_col_major), 4, m, n, k, batch, pa, lda, sa, pb, ldb, sb, pc, ldc, sc, stream)
    if status == 1:
        status = fn(tile, int(b_col_major), 1, m, n, k, batch, pa, lda, sa, pb, ldb, sb, pc, ldc, sc, stream)
    if status:
        raise RuntimeError(f"CUTLASS {variant} GEMM failed with status {status} (m={m} n={n} k={k} batch={batch})")


def _b_layout(b):
    """(tensor, column-major, leading dimension) for B = ``b`` ([K, N]) without a copy where its strides allow."""
    if b.stride(1) == 1 and b.stride(0) >= max(1, b.shape[1]):
        return b, False, b.stride(0)
    if b.stride(0) == 1 and b.stride(1) >= max(1, b.shape[0]):
        return b, True, b.stride(1)
    b = b.contiguous()
    return b, False, b.shape[1]


def dense(variant, a, b, out, tile=None):
    """out = a @ b ([M, K] @ [K, N], out row-major with unit column stride)."""
    m, k = a.shape
    n = b.shape[1]
    if a.stride(1) != 1 or a.stride(0) < max(1, k):
        a = a.contiguous()
    b, col, ldb = _b_layout(b)
    batched(variant, m, n, k, 1, (a, 0), a.stride(0), 0, (b, 0), col, ldb, 0, (out, 0), out.stride(0), 0, tile)


def joined(variant, a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, tile=None):
    """segment_gemm's ``_gemm`` contract: C rows (atom, m) <- A rows (atom, m) @ B, A and C addressed from their
    flat storage (``a`` and ``c`` contiguous)."""
    k, n = b.shape
    atoms = rows // dim
    b, col, ldb = _b_layout(b)
    batched(variant, atoms, n, k, dim, (a, a_offset), a_n, a_m, (b, 0), col, ldb, 0, (c, c_offset), c_n, c_m, tile)


def install(variant):
    """Reroute the ``tf32x3_gemms`` record's launches to CUTLASS ``variant`` (see the module docstring)."""
    import torch

    from fastmace.kernels.triton import segment_gemm, tf32x3_gemm

    if variant not in VARIANTS:
        raise ValueError(f"unknown CUTLASS variant {variant!r}; one of {sorted(VARIANTS)}")
    library()
    triton_gemm = segment_gemm._gemm

    def launch(a, b, out):
        if not tf32x3_gemm.uses_tf32x3_tile(a.shape[1], b.shape[1]):
            torch.mm(a.contiguous(), b.contiguous(), out=out)
            return
        dense(variant, a, b, out)

    def gemm(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
        if precision != "tf32x3":
            return triton_gemm(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision)
        joined(variant, a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim)

    tf32x3_gemm._launch, segment_gemm._gemm = launch, gemm
    return variant


def install_from_env():
    """``install`` the variant named by ``FASTMACE_SCREEN_TC_GEMM`` (``cutlass-<variant>``); None if unset."""
    value = os.environ.get(ENV, "").strip()
    if not value or value == "triton":
        return None
    if not value.startswith("cutlass-"):
        raise ValueError(f"{ENV}={value!r}: expected cutlass-<variant> or triton")
    return install(value.removeprefix("cutlass-"))
