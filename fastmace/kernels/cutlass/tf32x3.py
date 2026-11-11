"""3xTF32 GEMMs from CUTLASS on compute capability 8.0 (the A100), the backend of ``tf32x3_mm`` there.

CUTLASS's ``OpMultiplyAddFastF32`` splits each float32 operand into a TF32 big and small part and sums three
tensor-core products (big * big + big * small + small * big) in FP32, the same split as Triton's ``tf32x3``, but
with a CUTLASS sm_80 mainloop: 40.8 TFLOPS at 4096³ on an A100 against cuBLAS IEEE's 19.0 and Triton's 16.8, and
the summed MH-1 GEMM time per force call 0.75 (NaCl) and 0.55 (NbCl5) of cuBLAS IEEE's
(``docs/validation/speed/opt-T01-tensor-cores.md`` §5; ``docs/validation/omat0/cutlass-tf32x3.md``).

The kernels come from ``tf32x3_sm80.cu``, built offline into ``tf32x3_sm80.cubin`` (``tools/build_cutlass_tf32x3.sh``;
the build record is ``tf32x3_sm80.build.txt``). An sm_80 cubin loads only on compute capability 8.x, and the tile
was chosen on the A100 alone, so ``available`` admits capability 8.0 only. ``mm_into`` runs ``C = A @ B`` with A
row-major and B row-major or column-major, all leading dimensions multiples of 4 and the pointers 16-byte aligned
(``eligible``); the caller falls back otherwise. Tile 64x64x16 (``TILE`` selects 32x64 or 64x32 for A/Bs, bitwise the
same), no split-K: deterministic.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from fastmace.kernels.gpu_models import GPUModels

CUBIN = Path(__file__).with_name("tf32x3_sm80.cubin")
CAPABILITY = (8, 0)
# One cubin per compute capability, built from the same source (5 November 2026: sm_90 for the H100 and H200; the
# mainloop is the same mma.sync code, so every element's sum is the same as on the A100).
CUBINS = {CAPABILITY: CUBIN, (9, 0): Path(__file__).with_name("tf32x3_sm90.cubin")}
ALIGNMENT = 4  # elements: 128-bit loads
# The output tile: "64x64" (four warps), or "32x64" / "64x32" (two warps, twice the blocks; 4 November 2026, for
# Benchmark_loup's 5,000-row products of 128 columns, 158 blocks of 64x64 on the A100's 108 SMs). The same
# 3xTF32 split; the K loop and so each element's sum are the same in every tile. A module constant for the A/B.
TILE = "64x64"
_TILES = ("64x64", "32x64", "64x32")
# HYBRID (11 November 2026, review retry of the A100 screen's idea 4; results/hybrid-gemm-20261111): every entry from
# the cubins built with -DFM_HYBRID_F16 (hybrid_f16.h: big * big on TF32 tensor cores, both cross products in one FP16
# mma; error at or below 3xTF32's, kernel 0.85-0.90 of it on the A100). Operands must stay below 32752 in magnitude
# (float16 cross parts). None: on for HYBRID_GPUS; True / False: an A/B side's choice, read at call time like TILE.
# On the A100 family (decision 66, 11 November 2026, Alex A100-SXM4-80GB, OMAT-0 medium on the default plan: MD step
# 0.9896 / 0.9877 SrTiO3 10x10x10, 0.9832 / 0.9846 NbCl5 3x3x3; gate passes). The H200 (sm_90 cubin) is not measured.
HYBRID = None
HYBRID_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB",))
HYBRID_CUBINS = {CAPABILITY: Path(__file__).with_name("tf32x3_hyb_sm80.cubin"),
                 (9, 0): Path(__file__).with_name("tf32x3_hyb_sm90.cubin")}
_KERNELS = {}  # (device index, hybrid) -> _Kernels


class _Kernels:
    """The cubin's GEMM entries on one device (both B layouts, every tile), with their launch sizes
    (``fm_tf32x3_info``)."""

    def __init__(self, index, hybrid=False):
        import numpy as np
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        with torch.cuda.device(index):
            torch.zeros(1, device=f"cuda:{index}")  # a runtime call: the device's primary context is current
            self.context = _check(driver.cuCtxGetCurrent())
            self.module = _check(driver.cuModuleLoadData(_cubin(index, hybrid).read_bytes()))
            self.functions = {(layout, tile): _check(driver.cuModuleGetFunction(
                self.module, f"fm_tf32x3_{layout}_{tile}".encode())) for layout in ("rr", "rc") for tile in _TILES}
            pointer, size = _check(driver.cuModuleGetGlobal(self.module, b"fm_tf32x3_info"))
            info = np.zeros(size // 4, dtype=np.int32)
            _check(driver.cuMemcpyDtoH(info, pointer, size))
            self.species_function = _check(driver.cuModuleGetFunction(self.module, b"fm_tf32x3_species_rr_64x64"))
            pointer, size = _check(driver.cuModuleGetGlobal(self.module, b"fm_tf32x3_species_info"))
            species_info = np.zeros(size // 4, dtype=np.int32)
            _check(driver.cuMemcpyDtoH(species_info, pointer, size))
            # idea 31 (linear_div): the 64x64 kernels with the remapped, divided store
            self.remap_functions = {layout: _check(driver.cuModuleGetFunction(
                self.module, f"fm_tf32x3_remap_div_{layout}_64x64".encode())) for layout in ("rr", "rc")}
            # idea 31 (skip_select): the same with the own-species selection in the store (the same shared memory)
            self.select_functions = {layout: _check(driver.cuModuleGetFunction(
                self.module, f"fm_tf32x3_remap_select_{layout}_64x64".encode())) for layout in ("rr", "rc")}
            pointer, size = _check(driver.cuModuleGetGlobal(self.module, b"fm_tf32x3_remap_info"))
            remap_info = np.zeros(size // 4, dtype=np.int32)
            _check(driver.cuMemcpyDtoH(remap_info, pointer, size))
        self.remap_threads = int(remap_info[0])
        self.remap_shared = {"rr": int(remap_info[1]), "rc": int(remap_info[2])}
        two_warps = int(info[5])
        self.threads = {"64x64": int(info[0]), "32x64": two_warps, "64x32": two_warps}
        self.tile = {t: tuple(int(x) for x in t.split("x")) for t in _TILES}
        self.shared = {("rr", "64x64"): int(info[1]), ("rc", "64x64"): int(info[2]), ("rr", "32x64"): int(info[6]),
                       ("rc", "32x64"): int(info[7]), ("rr", "64x32"): int(info[8]), ("rc", "64x32"): int(info[9])}
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        for key, function in self.functions.items():
            if self.shared[key] > 48 * 1024:
                _check(driver.cuFuncSetAttribute(function, attribute, self.shared[key]))
        self.species_threads, self.species_shared = int(species_info[0]), int(species_info[1])
        if self.species_shared > 48 * 1024:
            _check(driver.cuFuncSetAttribute(self.species_function, attribute, self.species_shared))
        for functions in (self.remap_functions, self.select_functions):
            for layout, function in functions.items():
                if self.remap_shared[layout] > 48 * 1024:
                    _check(driver.cuFuncSetAttribute(function, attribute, self.remap_shared[layout]))


def available(device, family=False) -> bool:
    """Whether the CUTLASS kernels run on ``device``: CUDA, compute capability 8.0 (``family``: any 8.x, where the
    sm_80 cubin also runs; for tests), the cubin present and loadable."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    capability = tuple(torch.cuda.get_device_capability(index))
    if capability not in CUBINS and not (family and capability[0] == CAPABILITY[0]):
        return False
    if not _cubin(index, hybrid_default(index)).is_file():
        return False
    try:
        _kernels(index)
    except (ImportError, RuntimeError):
        return False
    return True


def _cubin(index, hybrid=False):
    """The cubin for device ``index``: its capability's own, else the sm_80 one (8.x family, for tests); with
    ``hybrid`` the HYBRID build."""
    cubins = HYBRID_CUBINS if hybrid else CUBINS
    return cubins.get(tuple(torch.cuda.get_device_capability(index)), cubins[CAPABILITY])


def hybrid_default(index):
    """``HYBRID`` on device ``index``: the constant when set, else on for ``HYBRID_GPUS``."""
    if HYBRID is not None:
        return bool(HYBRID)
    return torch.cuda.get_device_name(index) in HYBRID_GPUS


def _kernels(index):
    key = (index, hybrid_default(index))
    if key not in _KERNELS:
        _KERNELS[key] = _Kernels(*key)
    return _KERNELS[key]


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
# tiles at K <= 256 or with 160 tiles or more took 0.54-1.00x (docs/validation/omat0/cutlass-tf32x3.md). Those stay on
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
    key = ("r" + lb, TILE)
    p, i = ctypes.c_void_p, ctypes.c_int
    values = (m, n, k, a.data_ptr(), lda, b.data_ptr(), ldb, out.data_ptr(), ldo)
    types = (i, i, i, p, i, p, i, p, i)
    stream = driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    tile_m, tile_n = kernels.tile[TILE]
    grid = (-(-m // tile_m), -(-n // tile_n))
    _check(driver.cuLaunchKernel(kernels.functions[key], grid[0], grid[1], 1, kernels.threads[TILE], 1, 1,
                                 kernels.shared[key], stream, (values, types), 0))


def remap_div_into(a, b, out, rows_per_atom, atom_stride, row_stride, density=None):
    """Row r of ``a @ b`` ([M, K] @ [K, N]; A row-major, B row- or column-major) stored at ``out``'s pointer plus
    ``(r // rows_per_atom) * atom_stride + (r % rows_per_atom) * row_stride`` floats, each element divided by
    ``density[r // rows_per_atom] + 1`` (float32, contiguous, one per atom) unless ``density`` is None
    (``fm_tf32x3_remap_div_*_64x64``; idea 31, ``linear_div``). The same K loop as ``mm_into``, any ``TILE``: each
    element's accumulator is the same. A and B as ``eligible``; N, both strides multiples of ``ALIGNMENT`` and
    ``out`` 16-byte aligned; the caller checks and keeps every written element inside ``out``."""
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    index = a.device.index if a.device.index is not None else torch.cuda.current_device()
    kernels = _kernels(index)
    current = _check(driver.cuCtxGetCurrent())
    if int(current) != int(kernels.context):
        _check(driver.cuCtxSetCurrent(kernels.context))
    m, k = a.shape
    n = b.shape[1]
    _, lda = _layout(a)
    lb, ldb = _layout(b)
    p, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    values = (m, n, k, a.data_ptr(), lda, b.data_ptr(), ldb, out.data_ptr(), rows_per_atom, atom_stride, row_stride,
              0 if density is None else density.data_ptr())
    types = (i, i, i, p, i, p, i, p, i, q, q, p)
    stream = driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    layout = "r" + lb
    _check(driver.cuLaunchKernel(kernels.remap_functions[layout], -(-m // 64), -(-n // 64), 1, kernels.remap_threads,
                                 1, 1, kernels.remap_shared[layout], stream, (values, types), 0))


def remap_select_into(a, b, out, rows_per_atom, atom_stride, row_stride, kind, select_columns):
    """``remap_div_into`` without the division and with the species skip's selection: ``b`` holds one block of
    ``select_columns`` columns per species ([K, species * select_columns]); element (r, j) is stored only if
    ``kind[r // rows_per_atom] == j // select_columns`` (``kind`` int32, contiguous, one per atom), at column
    ``j % select_columns`` of the remapped row (``fm_tf32x3_remap_select_*_64x64``; idea 31, ``skip_select``).
    ``select_columns`` a multiple of ``ALIGNMENT``; otherwise as ``remap_div_into``."""
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    index = a.device.index if a.device.index is not None else torch.cuda.current_device()
    kernels = _kernels(index)
    current = _check(driver.cuCtxGetCurrent())
    if int(current) != int(kernels.context):
        _check(driver.cuCtxSetCurrent(kernels.context))
    m, k = a.shape
    n = b.shape[1]
    _, lda = _layout(a)
    lb, ldb = _layout(b)
    p, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    values = (m, n, k, a.data_ptr(), lda, b.data_ptr(), ldb, out.data_ptr(), rows_per_atom, atom_stride, row_stride,
              0, kind.data_ptr(), select_columns)
    types = (i, i, i, p, i, p, i, p, i, q, q, p, p, i)
    stream = driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    layout = "r" + lb
    _check(driver.cuLaunchKernel(kernels.select_functions[layout], -(-m // 64), -(-n // 64), 1, kernels.remap_threads,
                                 1, 1, kernels.remap_shared[layout], stream, (values, types), 0))


def species_into(a, lda, gather, w, out, scatter, counts, size, grid_rows):
    """``out[scatter[r]] = a[gather[r]] @ w[s]`` for every species s's rows r (rows sorted by species, ``counts[s] *
    size`` of them; ``grid_rows`` = ``species_gemm.tiles``): ``a`` the base of rows ``lda`` floats apart, ``w`` [S, K,
    N] contiguous, ``out`` [rows, N] contiguous, ``gather`` / ``scatter`` int32 row indices
    (``fm_tf32x3_species_rr_64x64``). K, N, lda multiples of ``ALIGNMENT``, pointers 16-byte aligned."""
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    index = out.device.index if out.device.index is not None else torch.cuda.current_device()
    kernels = _kernels(index)
    current = _check(driver.cuCtxGetCurrent())
    if int(current) != int(kernels.context):
        _check(driver.cuCtxSetCurrent(kernels.context))
    _, k, n = w.shape
    p, i = ctypes.c_void_p, ctypes.c_int
    values = (counts.shape[0], counts.data_ptr(), size, n, k, a.data_ptr(), lda, gather.data_ptr(), w.data_ptr(),
              out.data_ptr(), n, scatter.data_ptr())
    types = (i, p, i, i, i, p, i, p, p, p, i, p)
    stream = driver.CUstream(torch.cuda.current_stream(out.device).cuda_stream)
    _check(driver.cuLaunchKernel(kernels.species_function, grid_rows, -(-n // 64), 1, kernels.species_threads, 1, 1,
                                 kernels.species_shared, stream, (values, types), 0))


__all__ = ["ALIGNMENT", "CAPABILITY", "CUBIN", "CUBINS", "HYBRID", "HYBRID_CUBINS", "HYBRID_GPUS", "LONG_K", "MIN_TILES", "TILE", "available", "eligible", "fit",
           "mm_into", "remap_div_into", "remap_select_into", "species_into", "takes"]
