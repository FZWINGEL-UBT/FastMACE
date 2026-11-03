"""3xTF32 GEMMs from CUTLASS on compute capability 8.0 (the A100), the backend of ``tf32x3_mm`` there.

CUTLASS's ``OpMultiplyAddFastF32`` splits each float32 operand into a TF32 big and small part and sums three
tensor-core products (big * big + big * small + small * big) in FP32, the same split as Triton's ``tf32x3``, but
with a CUTLASS sm_80 mainloop: 40.8 TFLOPS at 4096³ on an A100 against cuBLAS IEEE's 19.0 and Triton's 16.8, and
the summed MH-1 GEMM time per force call 0.75 (NaCl) and 0.55 (NbCl5) of cuBLAS IEEE's
(``docs/validation/opt-T01-tensor-cores.md`` §5; ``docs/validation/cutlass-tf32x3.md``).

The kernels come from ``tf32x3_sm80.cu``, built offline into ``tf32x3_sm80.cubin`` (``tools/build_cutlass_tf32x3.sh``;
the build record is ``tf32x3_sm80.build.txt``). An sm_80 cubin loads only on compute capability 8.x, and the tile
was chosen on the A100 alone, so ``available`` admits capability 8.0 only. ``mm_into`` runs ``C = A @ B`` with A
row-major and B row-major or column-major, all leading dimensions multiples of 4 and the pointers 16-byte aligned
(``eligible``); the caller falls back otherwise. One fixed tile (64x64x16), no split-K: deterministic.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

CUBIN = Path(__file__).with_name("tf32x3_sm80.cubin")
CAPABILITY = (8, 0)
ALIGNMENT = 4  # elements: 128-bit loads
_KERNELS = {}  # device index -> _Kernels


class _Kernels:
    """The cubin's two GEMM entries on one device, with their launch sizes (``fm_tf32x3_info``)."""

    def __init__(self, index):
        import numpy as np
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        with torch.cuda.device(index):
            torch.zeros(1, device=f"cuda:{index}")  # a runtime call: the device's primary context is current
            self.context = _check(driver.cuCtxGetCurrent())
            self.module = _check(driver.cuModuleLoadData(CUBIN.read_bytes()))
            self.functions = {layout: _check(driver.cuModuleGetFunction(self.module, f"fm_tf32x3_{layout}_64x64"
                                                                        .encode()))
                              for layout in ("rr", "rc")}
            pointer, size = _check(driver.cuModuleGetGlobal(self.module, b"fm_tf32x3_info"))
            info = np.zeros(5, dtype=np.int32)
            _check(driver.cuMemcpyDtoH(info, pointer, size))
        self.threads, self.tile_m, self.tile_n = int(info[0]), int(info[3]), int(info[4])
        self.shared = {"rr": int(info[1]), "rc": int(info[2])}
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        for layout, function in self.functions.items():
            if self.shared[layout] > 48 * 1024:
                _check(driver.cuFuncSetAttribute(function, attribute, self.shared[layout]))


def available(device) -> bool:
    """Whether the CUTLASS kernels run on ``device``: CUDA, compute capability 8.0, the cubin present and loadable."""
    device = torch.device(device)
    if device.type != "cuda" or not CUBIN.is_file():
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    if tuple(torch.cuda.get_device_capability(index)) != CAPABILITY:
        return False
    try:
        _kernels(index)
    except (ImportError, RuntimeError):
        return False
    return True


def _kernels(index):
    if index not in _KERNELS:
        _KERNELS[index] = _Kernels(index)
    return _KERNELS[index]


def _layout(t):
    """"r" (row-major, unit column stride), "c" (column-major, unit row stride) or None, with its leading
    dimension."""
    rows, columns = t.shape
    if t.stride(1) == 1 and (rows == 1 or t.stride(0) >= columns):
        return "r", t.stride(0) if rows > 1 else max(columns, 1)
    if t.stride(0) == 1 and (columns == 1 or t.stride(1) >= rows):
        return "c", t.stride(1) if columns > 1 else max(rows, 1)
    return None, 0


def eligible(a, b, out) -> bool:
    """Whether ``mm_into(a, b, out)`` takes these operands: float32 on one CUDA device, A row-major, B row- or
    column-major, ``out`` row-major, leading dimensions multiples of ``ALIGNMENT``, pointers 16-byte aligned, and
    sizes below 2**31."""
    tensors = (a, b, out)
    if any(t.dtype != torch.float32 or t.dim() != 2 for t in tensors):
        return False
    (la, lda), (lb, ldb), (lo, ldo) = _layout(a), _layout(b), _layout(out)
    if la != "r" or lo != "r" or lb is None:
        return False
    if any(ld % ALIGNMENT or t.data_ptr() % 16 for ld, t in ((lda, a), (ldb, b), (ldo, out))):
        return False
    return max(a.numel(), b.numel(), out.numel()) < 2**31


def takes(m, k, n) -> bool:
    """Whether ``tf32x3_mm`` runs ``[m, k] @ [k, n]`` here: K and N multiples of ``ALIGNMENT`` (the kernels' 128-bit
    loads; any strides are made to fit by ``fit``), at least ``MIN_TILES`` output tiles or K at most ``LONG_K``, and
    below 2**31 elements per operand. A rule on the shape alone, so a product's arithmetic does not depend on its
    operands' strides."""
    tiles = -(-m // 64) * -(-n // 64)
    return (k % ALIGNMENT == 0 and n % ALIGNMENT == 0 and (tiles >= MIN_TILES or k <= LONG_K)
            and max(m * k, k * n, m * n) < 2**31)


# Without split-K a product with few 64x64 output tiles and a long K leaves most SMs idle: on the A100 (108 SMs,
# CUDA-graph timings) 16 tiles at K 512 and 2048 took 1.2-2.3x cuBLAS IEEE's time, while every product with 16
# tiles at K <= 256 or with 160 tiles or more took 0.54-1.00x (docs/validation/cutlass-tf32x3.md). Those stay on
# cuBLAS IEEE (an MD step's products have atom rows, so many tiles; a weight gradient's K is the atom count).
MIN_TILES = 54  # half the A100's SMs
LONG_K = 256


def fit(a, b):
    """``(a, b)`` in layouts ``mm_into`` takes: A row-major, B row- or column-major, aligned (copies otherwise)."""
    (la, lda), (lb, ldb) = _layout(a), _layout(b)
    if la != "r" or lda % ALIGNMENT or a.data_ptr() % 16:
        a = a.contiguous() if la != "r" or a.stride(0) != a.shape[1] else a.clone()
    if lb is None or ldb % ALIGNMENT or b.data_ptr() % 16:
        b = b.contiguous() if lb != "r" or b.stride(0) != b.shape[1] else b.clone()
    return a, b


def mm_into(a, b, out):
    """``out = a @ b`` (``[M, K] @ [K, N]``) through the CUTLASS kernel for B's layout; ``eligible`` holds."""
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    index = a.device.index if a.device.index is not None else torch.cuda.current_device()
    kernels = _kernels(index)
    current = _check(driver.cuCtxGetCurrent())
    if int(current) != int(kernels.context):  # autograd's device threads may have no context current yet
        _check(driver.cuCtxSetCurrent(kernels.context))
    m, k = a.shape
    n = b.shape[1]
    _, lda = _layout(a)
    lb, ldb = _layout(b)
    _, ldo = _layout(out)
    layout = "r" + lb
    p, i = ctypes.c_void_p, ctypes.c_int
    values = (m, n, k, a.data_ptr(), lda, b.data_ptr(), ldb, out.data_ptr(), ldo)
    types = (i, i, i, p, i, p, i, p, i)
    stream = driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    grid = (-(-m // kernels.tile_m), -(-n // kernels.tile_n))
    _check(driver.cuLaunchKernel(kernels.functions[layout], grid[0], grid[1], 1, kernels.threads, 1, 1,
                                 kernels.shared[layout], stream, (values, types), 0))


__all__ = ["ALIGNMENT", "CAPABILITY", "CUBIN", "LONG_K", "MIN_TILES", "available", "eligible", "fit", "mm_into", "takes"]
