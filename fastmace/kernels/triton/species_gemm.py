"""Per-species row products of the generic species skip (``SpeciesSkip``, ``SPECIES_GEMM``).

``species_mm(x, w, plan)`` is ``y[a, j] = x[a, j] @ w[species[a]]`` for ``x [atoms, size, K]`` (any strides) and the
stacked matrices ``w [species, K, N]``: every atom's rows through its own species' matrix only, instead of through
all present species' concatenated matrices followed by the selection of the atom's block (a third of the products
for three species). Its input gradient is the same product with the transposed matrices.

The grid is fixed for a structure's atom count, so the call can be captured in CUDA graphs and survives atoms
changing species: ``species_plan`` sorts the atoms by species on the device (stable) and counts them, once for every
product of a call. The kernel cuts each species' rows (atom, j) into tiles of ``BLOCK_M`` rows (``tile_rows``), at
most ``ceil(atoms * size / BLOCK_M) + species`` of them; the tiles past the last are empty. A tile gathers its rows
from ``x`` and scatters its results to ``y``; every row is written once.

Each output element is the same K-long dot product as in the concatenated product, reduced in ``BLOCK_K`` steps;
``precision`` "ieee" (FP32 FMA) or "tf32x3" (``tf32x3_gemms``). On the CPU the product runs in PyTorch.

``TENSOR_CORES`` (tf32x3 only, compute capability 8.0+, N a multiple of 16 up to 256): the product in a CUDA
kernel of our own instead of Triton's tf32x3 dot, which on the A100 runs at about a third of the CUTLASS kernels'
rate. A tile of ``TC_ROWS`` rows reduces K in chunks of ``TC_CHUNK``: the chunk of its species' matrix and of its
gathered rows in shared memory, 8 warps (4 along the rows, 2 along the columns) running 3xTF32 mma.sync m16n8k8 with
the fused forward's split (``fused_forward.tc_helpers``).
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
NUM_WARPS, NUM_STAGES = 4, 2
TENSOR_CORES = False  # A/B switch (4 November 2026): the tf32x3 product in the mma.sync kernel
TC_ROWS = BLOCK_M  # the tensor-core kernel's tile rows: the same tiles as the Triton kernel's
TC_CHUNK = 16  # its reduction chunk: shared memory for several blocks per SM, loads overlapping other blocks' MMAs
TC_PRESPLIT = False  # lab switch: chunks split into high and low parts once, in shared memory (5060 Ti: slower)

_TC_SOURCE = """
using int64_t = long long;
#define PRESPLIT {presplit}
constexpr int K = {k}, N = {n}, BM = {bm}, KC = {kc}, KS = KC + 4, NS = N + 8, NT = N / 16;
{helpers}
extern "C" __global__ void __launch_bounds__(256) species_mm_tc(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, const int* __restrict__ order,
    const int* __restrict__ counts, int S, int size, int64_t sxa, int64_t sxj, int64_t sxk, int64_t sya, int64_t syj,
    int64_t syn) {{
#if PRESPLIT
    extern __shared__ unsigned smem[];
    unsigned* Wh = smem;  // [KC][NS]: the species' matrix, rows k0 .. k0 + KC, split once (high, low)
    unsigned* Wl = Wh + KC * NS;
    unsigned* Ah = Wl + KC * NS;  // [BM][KS]: the tile's gathered rows, columns k0 .. k0 + KC, split once
    unsigned* Al = Ah + BM * KS;
#else
    extern __shared__ float smem[];
    float* Ws = smem;  // [KC][NS]: the species' matrix, rows k0 .. k0 + KC
    float* As = smem + KC * NS;  // [BM][KS]: the tile's gathered rows, columns k0 .. k0 + KC
#endif
    const int t = blockIdx.x;
    int species = 0, start = 0, end = 0, tiles_before = 0, rows_before = 0;
    for (int s = 0; s < S; ++s) {{  // species_gemm.tile_rows
        const int rows = counts[s] * size, n_tiles = (rows + BM - 1) / BM;
        if (t >= tiles_before && t < tiles_before + n_tiles) {{
            species = s;
            start = rows_before + (t - tiles_before) * BM;
            end = min(start + BM, rows_before + rows);
        }}
        tiles_before += n_tiles;
        rows_before += rows;
    }}
    if (start >= end) return;
    const float* ws = w + (int64_t)species * K * N;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tq = lane & 3;
    const int m0 = (warp & 3) * 16, n0 = (warp >> 2) * (N / 2);
    float acc[NT][4];
    #pragma unroll
    for (int q = 0; q < NT; ++q) acc[q][0] = acc[q][1] = acc[q][2] = acc[q][3] = 0.f;
    for (int kc = 0; kc < K; kc += KC) {{
        if (kc) __syncthreads();
        for (int i = threadIdx.x; i < KC * N; i += 256) {{
            const int k = i / N, n = i - k * N;
            const float v = kc + k < K ? ws[(int64_t)(kc + k) * N + n] : 0.f;
#if PRESPLIT
            split_tf32(v, Wh[k * NS + n], Wl[k * NS + n]);
#else
            Ws[k * NS + n] = v;
#endif
        }}
        for (int i = threadIdx.x; i < BM * KC; i += 256) {{
            const int m = i / KC, k = i - m * KC, r = start + m;
            float v = 0.f;
            if (r < end && kc + k < K) {{
                const int a = r / size, j = r - a * size;
                v = x[(int64_t)order[a] * sxa + j * sxj + (kc + k) * sxk];
            }}
#if PRESPLIT
            split_tf32(v, Ah[m * KS + k], Al[m * KS + k]);
#else
            As[m * KS + k] = v;
#endif
        }}
        __syncthreads();
        #pragma unroll
        for (int k0 = 0; k0 < KC; k0 += 8) {{
            unsigned ah[4], al[4];  // A fragment: rows m0 + gid (+8), columns k0 + tq (+4)
#if PRESPLIT
            const int ao = (m0 + gid) * KS + k0 + tq;
            ah[0] = Ah[ao], ah[1] = Ah[ao + 8 * KS], ah[2] = Ah[ao + 4], ah[3] = Ah[ao + 8 * KS + 4];
            al[0] = Al[ao], al[1] = Al[ao + 8 * KS], al[2] = Al[ao + 4], al[3] = Al[ao + 8 * KS + 4];
#else
            const float* ap = As + (m0 + gid) * KS + k0 + tq;
            split_tf32(ap[0], ah[0], al[0]);
            split_tf32(ap[8 * KS], ah[1], al[1]);
            split_tf32(ap[4], ah[2], al[2]);
            split_tf32(ap[8 * KS + 4], ah[3], al[3]);
#endif
            #pragma unroll
            for (int q = 0; q < NT; ++q) {{  // B fragment: rows k0 + tq (+4), column n0 + 8 q + gid
#if PRESPLIT
                const int bo = (k0 + tq) * NS + n0 + 8 * q + gid;
                const unsigned bh[2] = {{Wh[bo], Wh[bo + 4 * NS]}}, bl[2] = {{Wl[bo], Wl[bo + 4 * NS]}};
#else
                const float* bp = Ws + (k0 + tq) * NS + n0 + 8 * q + gid;
                unsigned bh[2], bl[2];
                split_tf32(bp[0], bh[0], bl[0]);
                split_tf32(bp[4 * NS], bh[1], bl[1]);
#endif
                mma_tf32(acc[q], al, bh);
                mma_tf32(acc[q], ah, bl);
                mma_tf32(acc[q], ah, bh);
            }}
        }}
    }}
    #pragma unroll
    for (int h = 0; h < 2; ++h) {{  // C fragment: rows m0 + gid (+8), columns n0 + 8 q + 2 tq (+1)
        const int r = start + m0 + gid + 8 * h;
        if (r < end) {{
            const int a = r / size, j = r - a * size;
            float* yp = y + (int64_t)order[a] * sya + j * syj;
            #pragma unroll
            for (int q = 0; q < NT; ++q) {{
                const int c = n0 + 8 * q + 2 * tq;
                yp[c * syn] = acc[q][2 * h];
                yp[(c + 1) * syn] = acc[q][2 * h + 1];
            }}
        }}
    }}
}}
"""
_TC = {}  # (device index, K, N, chunk, presplit) -> (module, function, shared bytes)


def _tc_shared(k, n):
    return 4 * (2 if TC_PRESPLIT else 1) * (TC_CHUNK * (n + 8) + TC_ROWS * (TC_CHUNK + 4))


def tc_supported(device, k, n):
    """Whether the tensor-core kernel takes a ``[*, k] @ [k, n]`` product on ``device``."""
    return (torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0)
            and n % 16 == 0 and 16 <= n <= 256
            and _tc_shared(k, n) <= torch.cuda.get_device_properties(device).shared_memory_per_block_optin)


def _tc_function(device, k, n):
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.fused_forward import tc_helpers
    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, k, n, TC_CHUNK, TC_PRESPLIT)
    if key not in _TC:
        source = _TC_SOURCE.format(k=k, n=n, bm=TC_ROWS, kc=TC_CHUNK, presplit=int(TC_PRESPLIT), helpers=tc_helpers())
        cubin = compile_cubin(source, torch.cuda.get_device_capability(index))
        module = _check(driver.cuModuleLoadData(cubin))
        function = _check(driver.cuModuleGetFunction(module, b"species_mm_tc"))
        shared = _tc_shared(k, n)
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(driver.cuFuncSetAttribute(function, attribute, shared))
        _TC[key] = (module, function, shared)
    return _TC[key]


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(x_ptr, w_ptr, y_ptr, order_ptr, counts_ptr, S, size, N, K, sxa, sxj, sxk, swk, swn, sya, syj, syn,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        t, pn = tl.program_id(0), tl.program_id(1)
        species, start, end = 0, 0, 0  # tile_rows: species s's tiles follow those of the species before it
        tiles_before, rows_before = 0, 0
        for s in range(S):
            rows = tl.load(counts_ptr + s) * size
            n_tiles = (rows + BLOCK_M - 1) // BLOCK_M
            mine = (t >= tiles_before) & (t < tiles_before + n_tiles)
            species = tl.where(mine, s, species)
            start = tl.where(mine, rows_before + (t - tiles_before) * BLOCK_M, start)
            end = tl.where(mine, rows_before + rows, end)
            tiles_before += n_tiles
            rows_before += rows
        if start >= end:
            return
        r = start + tl.arange(0, BLOCK_M)
        live = r < end
        i = tl.where(live, r // size, 0)
        j = r - i * size
        atom = tl.load(order_ptr + i, mask=live, other=0).to(tl.int64)
        xrow = atom * sxa + j.to(tl.int64) * sxj
        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        wbase = w_ptr + species.to(tl.int64) * K * N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + rk
            a = tl.load(x_ptr + xrow[:, None] + kk[None, :] * sxk, mask=live[:, None] & (kk < K)[None, :], other=0.0)
            b = tl.load(wbase + kk[:, None] * swk + rn[None, :] * swn, mask=(kk < K)[:, None] & (rn < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        yrow = atom * sya + j.to(tl.int64) * syj
        tl.store(y_ptr + yrow[:, None] + rn[None, :] * syn, acc, mask=live[:, None] & (rn < N)[None, :])

    return gemm


def tiles(atoms, size, species):
    """The fixed tile count of ``atoms`` atoms with ``size`` rows each over ``species`` species."""
    return -(-atoms * size // BLOCK_M) + species


def species_plan(index, species):
    """(order [atoms] int32: the atoms sorted by species, stable; counts [species] int32) of the atoms' species
    ``index`` ([atoms], values in ``range(species)``); fixed shapes, on ``index``'s device, no host sync."""
    order = torch.argsort(index, stable=True).to(torch.int32)
    counts = torch.zeros(species, dtype=torch.int32, device=index.device)  # no bincount: it syncs (max)
    return order, counts.index_add(0, index.long(), torch.ones_like(index, dtype=torch.int32))


def tile_rows(t, counts, size):
    """(species, first row, end row) of tile ``t`` in the sorted rows, the kernel's rule in Python (``counts``: a
    list); first == end past the last tile."""
    tiles_before = rows_before = 0
    for s, count in enumerate(counts):
        rows = count * size
        n_tiles = -(-rows // BLOCK_M)
        if tiles_before <= t < tiles_before + n_tiles:
            first = rows_before + (t - tiles_before) * BLOCK_M
            return s, first, min(first + BLOCK_M, rows_before + rows)
        tiles_before += n_tiles
        rows_before += rows
    return 0, 0, 0


def _launch_tc(x, w, order, counts, out):
    import ctypes

    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    atoms, size, k = x.shape
    n = w.shape[2]
    w = w.contiguous()
    _, function, shared = _tc_function(x.device, k, n)
    p = ctypes.c_void_p
    values = (x.data_ptr(), w.data_ptr(), out.data_ptr(), order.data_ptr(), counts.data_ptr(), counts.shape[0], size,
              x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1), out.stride(2))
    types = (p, p, p, p, p, ctypes.c_int, ctypes.c_int) + (ctypes.c_longlong,) * 6
    stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    _check(driver.cuLaunchKernel(function, tiles(atoms, size, counts.shape[0]), 1, 1, 256, 1, 1, shared, stream,
                                 (values, types), 0))


def _launch(x, w, order, counts, out, precision):
    import triton

    atoms, size, k = x.shape
    n = w.shape[2]
    if TENSOR_CORES and precision == "tf32x3" and tc_supported(x.device, k, n):
        _launch_tc(x, w, order, counts, out)
        return
    grid = (tiles(atoms, size, counts.shape[0]), triton.cdiv(n, BLOCK_N))
    _kernel()[grid](x, w, out, order, counts, counts.shape[0], size, n, k, x.stride(0), x.stride(1), x.stride(2),
                    w.stride(1), w.stride(2), out.stride(0), out.stride(1), out.stride(2), BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, PRECISION=precision, num_warps=NUM_WARPS,
                    num_stages=NUM_STAGES)


def _reference(x, w, index):
    return torch.einsum("ajk,akn->ajn", x, w[index.long()])


@torch.library.custom_op("fastmace::species_mm", mutates_args=())
def species_mm(x: torch.Tensor, w: torch.Tensor, w_t: torch.Tensor, index: torch.Tensor, order: torch.Tensor,
               counts: torch.Tensor, precision: str) -> torch.Tensor:
    """``x[a] @ w[index[a]]`` for ``x [atoms, size, K]`` and ``w [species, K, N]`` (``w_t``: w transposed, for the
    gradient; ``order``, ``counts``: ``species_plan``'s); PyTorch on the CPU and for other dtypes."""
    if x.device.type != "cuda" or x.dtype != torch.float32:
        return _reference(x, w, index)
    out = x.new_empty((x.shape[0], x.shape[1], w.shape[2]))
    if out.numel():
        _launch(x, w, order, counts, out, precision)
    return out


@species_mm.register_fake
def _(x, w, w_t, index, order, counts, precision):
    return x.new_empty((x.shape[0], x.shape[1], w.shape[2]))


def _setup_context(ctx, inputs, output):
    x, w, w_t, index, order, counts, precision = inputs
    ctx.save_for_backward(w, w_t, index, order, counts)
    ctx.precision = precision


def _backward(ctx, grad):
    w, w_t, index, order, counts = ctx.saved_tensors
    grad_x = species_mm(grad, w_t, w, index, order, counts, ctx.precision) if ctx.needs_input_grad[0] else None
    return grad_x, None, None, None, None, None, None


species_mm.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "TENSOR_CORES", "species_mm", "species_plan", "tc_supported",
           "tile_rows", "tiles"]
