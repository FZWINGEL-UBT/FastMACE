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
``precision`` "ieee" (FP32 FMA), "tf32x3" (``tf32x3_gemms``) or "tf32x3-cutlass" (tf32x3 through the CUTLASS
launch where the operands fit, whatever ``CUTLASS`` says). On the CPU the product runs in PyTorch.

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
# The tf32x3 product as one CUTLASS 3xTF32 launch with gathered rows and scattered results
# (``cutlass.tf32x3.species_into``) on compute capability 8.0 (4 November 2026; on the A100 Triton's tf32x3 dot runs
# at about a third of CUTLASS's rate). A/B switch; ``CUTLASS_FAMILY`` also allows any 8.x (tests).
CUTLASS = False
CUTLASS_FAMILY = False
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


def species_plan(index, species, counting=False):
    """(order [atoms] int32: the atoms sorted by species, stable; counts [species] int32) of the atoms' species
    ``index`` ([atoms], values in ``range(species)``); fixed shapes, on ``index``'s device, no host sync.
    ``counting`` (``compose_skip.COUNT_PLAN``): the same order by a counting sort over the few species (one-hot
    prefix sums and a scatter, which a compiled caller fuses) instead of a radix sort of the int64 index."""
    if counting:
        onehot = (index[:, None] == torch.arange(species, device=index.device)[None, :]).to(torch.int32)
        counts = onehot.sum(dim=0, dtype=torch.int32)
        rank = (torch.cumsum(onehot, dim=0, dtype=torch.int32) - 1).gather(1, index.long()[:, None])[:, 0]
        start = torch.cumsum(counts, dim=0, dtype=torch.int32) - counts
        position = (start[index.long()] + rank).long()
        atoms = torch.arange(index.shape[0], dtype=torch.int32, device=index.device)
        return torch.empty_like(atoms).scatter(0, position, atoms), counts
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


def _cutlass_rows(x, w, out):
    """The leading dimension of ``x``'s rows for ``species_into`` (row (a, j) at a * sxa + j * sxj = index * lda),
    or 0 when the operands do not fit."""
    import math

    from fastmace.kernels.cutlass import tf32x3
    atoms, size, k = x.shape
    n = w.shape[2]
    if x.stride(2) != 1 or k % 4 or n % 4 or not out.is_contiguous() or not w.is_contiguous():
        return 0
    lda = math.gcd(x.stride(0), x.stride(1)) if size > 1 else x.stride(0)
    if lda % 4 or lda <= 0 or x.data_ptr() % 16 or w.data_ptr() % 16 or out.data_ptr() % 16:
        return 0
    if not tf32x3.available(x.device, family=CUTLASS_FAMILY):
        return 0
    return lda


def _launch_cutlass(x, w, order, counts, out, lda):
    from fastmace.kernels.cutlass import tf32x3
    atoms, size, _ = x.shape
    j = torch.arange(size, device=x.device)
    a = order.long()[:, None]
    gather = ((a * x.stride(0) + j * x.stride(1)) // lda).to(torch.int32).flatten()
    scatter = (a * size + j).to(torch.int32).flatten()
    tf32x3.species_into(x, lda, gather, w, out, scatter, counts, size, tiles(atoms, size, counts.shape[0]))


def _launch(x, w, order, counts, out, precision):
    import triton

    atoms, size, k = x.shape
    n = w.shape[2]
    cutlass = CUTLASS
    if precision == "tf32x3-cutlass":  # the caller's choice of the CUTLASS launch (compose_skip), else as "tf32x3"
        precision, cutlass = "tf32x3", True
    if cutlass and precision == "tf32x3":
        lda = _cutlass_rows(x, w, out)
        if lda:
            _launch_cutlass(x, w, order, counts, out, lda)
            return
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


def joined_rows(order, sizes):
    """(gathers, scatters) of ``species_mm_joined``: per path of ``sizes`` rows per atom, the int32 rows of its
    contiguous ``[atoms, size, K]`` input in species order (``a size + j``) and of the joined ``[atoms, sum(sizes),
    N]`` output (``a total + offset + j``), flattened. Plain tensor operations, so a compiled caller fuses them
    instead of building them inside the opaque product."""
    a = order.long()[:, None]
    total = sum(sizes)
    gathers, scatters, offset = [], [], 0
    for size in sizes:
        j = torch.arange(size, device=order.device)
        gathers.append((a * size + j).to(torch.int32).flatten())
        scatters.append((a * total + (offset + j)).to(torch.int32).flatten())
        offset += size
    return gathers, scatters


def _joined_cutlass(xs, ws, out):
    """Whether every path of ``species_mm_joined`` fits the CUTLASS launch with rows ``K`` floats apart."""
    from fastmace.kernels.cutlass import tf32x3
    n = out.shape[2]
    if n % 4 or out.data_ptr() % 16 or not out.is_contiguous():
        return False
    for x, w in zip(xs, ws):
        if not x.is_contiguous() or not w.is_contiguous() or x.shape[2] % 4 or x.data_ptr() % 16 or w.data_ptr() % 16:
            return False
    return tf32x3.available(out.device, family=CUTLASS_FAMILY)


@torch.library.custom_op("fastmace::species_mm_joined", mutates_args=())
def species_mm_joined(xs: list[torch.Tensor], ws: list[torch.Tensor], w_ts: list[torch.Tensor], index: torch.Tensor,
                      order: torch.Tensor, counts: torch.Tensor, gathers: list[torch.Tensor],
                      scatters: list[torch.Tensor], precision: str) -> torch.Tensor:
    """``species_mm`` of every path ``x [atoms, size_p, K]`` with its ``w [species, K, N]``, joined along the rows
    as ``[atoms, sum of sizes, N]`` (``torch.cat`` of the paths' outputs along dim 1). Precision "tf32x3-cutlass"
    with fitting operands: one CUTLASS launch per path writes its rows straight into the joined output
    (``joined_rows``' gathers and scatters, no index building and no join here); otherwise per path as
    ``species_mm`` and copied in. Each output row is the same product as ``species_mm``'s (bitwise)."""
    if xs[0].device.type != "cuda" or xs[0].dtype != torch.float32:
        return torch.cat([_reference(x, w, index) for x, w in zip(xs, ws)], dim=1)
    atoms, n = xs[0].shape[0], ws[0].shape[2]
    out = xs[0].new_empty((atoms, sum(x.shape[1] for x in xs), n))
    if not out.numel():
        return out
    if precision == "tf32x3-cutlass" and _joined_cutlass(xs, ws, out):
        from fastmace.kernels.cutlass import tf32x3
        for x, w, gather, scatter in zip(xs, ws, gathers, scatters):
            size = x.shape[1]
            tf32x3.species_into(x, x.shape[2], gather, w, out, scatter, counts, size,
                                tiles(atoms, size, counts.shape[0]))
        return out
    offset = 0
    for x, w in zip(xs, ws):
        size = x.shape[1]
        block = x.new_empty((atoms, size, n))
        _launch(x, w, order, counts, block, precision)
        out[:, offset:offset + size] = block
        offset += size
    return out


@species_mm_joined.register_fake
def _(xs, ws, w_ts, index, order, counts, gathers, scatters, precision):
    return xs[0].new_empty((xs[0].shape[0], sum(x.shape[1] for x in xs), ws[0].shape[2]))


@torch.library.custom_op("fastmace::species_mm_joined_grad", mutates_args=())
def species_mm_joined_grad(grad: torch.Tensor, w_ts: list[torch.Tensor], index: torch.Tensor, order: torch.Tensor,
                           counts: torch.Tensor, gathers: list[torch.Tensor], scatters: list[torch.Tensor],
                           sizes: list[int], precision: str) -> list[torch.Tensor]:
    """The input gradients of ``species_mm_joined``: per path ``grad[:, rows of the path] @ w_t[species]`` as a
    contiguous ``[atoms, size, K]``; the forward's gathers and scatters swap roles (CUTLASS)."""
    if grad.device.type != "cuda" or grad.dtype != torch.float32:
        out, offset = [], 0
        for size, w_t in zip(sizes, w_ts):
            out.append(_reference(grad[:, offset:offset + size], w_t, index))
            offset += size
        return out
    atoms, n = grad.shape[0], grad.shape[2]
    outs = [grad.new_empty((atoms, size, w_t.shape[2])) for size, w_t in zip(sizes, w_ts)]
    if not grad.numel():
        return outs
    if precision == "tf32x3-cutlass" and grad.is_contiguous() and _joined_cutlass(outs, w_ts, grad):
        from fastmace.kernels.cutlass import tf32x3
        for out, w_t, gather, scatter, size in zip(outs, w_ts, gathers, scatters, sizes):
            tf32x3.species_into(grad, n, scatter, w_t, out, gather, counts, size, tiles(atoms, size, counts.shape[0]))
        return outs
    offset = 0
    for out, w_t, size in zip(outs, w_ts, sizes):
        _launch(grad[:, offset:offset + size], w_t, order, counts, out, precision)
        offset += size
    return outs


@species_mm_joined_grad.register_fake
def _(grad, w_ts, index, order, counts, gathers, scatters, sizes, precision):
    return [grad.new_empty((grad.shape[0], size, w_t.shape[2])) for size, w_t in zip(sizes, w_ts)]


def _joined_setup(ctx, inputs, output):
    xs, ws, w_ts, index, order, counts, gathers, scatters, precision = inputs
    ctx.paths = len(xs)
    ctx.sizes = [x.shape[1] for x in xs]
    ctx.precision = precision
    ctx.save_for_backward(*w_ts, index, order, counts, *gathers, *scatters)


def _joined_backward(ctx, grad):
    p = ctx.paths
    saved = ctx.saved_tensors
    w_ts, (index, order, counts) = list(saved[:p]), saved[p:p + 3]
    gathers, scatters = list(saved[p + 3:2 * p + 3]), list(saved[2 * p + 3:])
    grads = None
    if ctx.needs_input_grad[0]:
        grads = species_mm_joined_grad(grad.contiguous(), w_ts, index, order, counts, gathers, scatters, ctx.sizes,
                                       ctx.precision)
    nothing = [None] * p
    return grads, list(nothing), list(nothing), None, None, None, list(nothing), list(nothing), None


species_mm_joined.register_autograd(_joined_backward, setup_context=_joined_setup)


def _rows_cutlass(x, ws, out):
    """Whether ``species_mm_rows`` fits the CUTLASS launch (one contiguous row base, rows ``K`` floats apart)."""
    from fastmace.kernels.cutlass import tf32x3
    if out.shape[-1] % 4 or out.data_ptr() % 16 or not out.is_contiguous():
        return False
    if not x.is_contiguous() or x.shape[1] % 4 or x.data_ptr() % 16:
        return False
    if any(not w.is_contiguous() or w.data_ptr() % 16 for w in ws):
        return False
    return tf32x3.available(out.device, family=CUTLASS_FAMILY)


def _rows_views(x, sizes):
    """Path p's ``[atoms, size_p, K]`` view of ``species_mm_rows``' row base ``x [atoms * sum(sizes), K]``."""
    total = sum(sizes)
    rows = x.view(-1, total, x.shape[1])
    views, offset = [], 0
    for size in sizes:
        views.append(rows[:, offset:offset + size])
        offset += size
    return views


@torch.library.custom_op("fastmace::species_mm_rows", mutates_args=())
def species_mm_rows(x: torch.Tensor, sizes: list[int], ws: list[torch.Tensor], w_ts: list[torch.Tensor],
                    index: torch.Tensor, order: torch.Tensor, counts: torch.Tensor, rows: list[torch.Tensor],
                    precision: str) -> torch.Tensor:
    """``species_mm_joined`` of paths that are row groups of one input: ``x [atoms * R, K]`` (R = sum of
    ``sizes``) holds atom a's path-p rows at ``a R + offset_p + j``, the layout of the joined output, so ``rows``
    (``joined_rows``' scatters) index the input and the output alike (``compose_skip.JOINED_LAYOUT``: the first
    interaction's per-slot sums in the d-major ``inner`` layout). No per-path copy of the input; the gradient is one
    ``[atoms * R, K]`` tensor. Each output row is the same product as ``species_mm``'s (bitwise)."""
    total = sum(sizes)
    atoms, n = x.shape[0] // total, ws[0].shape[2]
    if x.device.type != "cuda" or x.dtype != torch.float32:
        return torch.cat([_reference(v, w, index) for v, w in zip(_rows_views(x, sizes), ws)], dim=1)
    out = x.new_empty((atoms, total, n))
    if not out.numel():
        return out
    if precision == "tf32x3-cutlass" and _rows_cutlass(x, ws, out):
        from fastmace.kernels.cutlass import tf32x3
        for w, row, size in zip(ws, rows, sizes):
            tf32x3.species_into(x, x.shape[1], row, w, out, row, counts, size, tiles(atoms, size, counts.shape[0]))
        return out
    offset = 0
    for v, w, size in zip(_rows_views(x, sizes), ws, sizes):
        block = x.new_empty((atoms, size, n))
        _launch(v.contiguous(), w, order, counts, block, precision)
        out[:, offset:offset + size] = block
        offset += size
    return out


@species_mm_rows.register_fake
def _(x, sizes, ws, w_ts, index, order, counts, rows, precision):
    return x.new_empty((x.shape[0] // sum(sizes), sum(sizes), ws[0].shape[2]))


@torch.library.custom_op("fastmace::species_mm_rows_grad", mutates_args=())
def species_mm_rows_grad(grad: torch.Tensor, sizes: list[int], w_ts: list[torch.Tensor], index: torch.Tensor,
                         order: torch.Tensor, counts: torch.Tensor, rows: list[torch.Tensor],
                         precision: str) -> torch.Tensor:
    """The input gradient of ``species_mm_rows``: ``[atoms * R, K]``, path p's rows ``grad[rows of p] @
    w_t[species]`` written in place (CUTLASS: the same ``rows`` index both sides)."""
    total = sum(sizes)
    atoms, k = grad.shape[0], w_ts[0].shape[2]
    if grad.device.type != "cuda" or grad.dtype != torch.float32:
        out, offset = [], 0
        for size, w_t in zip(sizes, w_ts):
            out.append(_reference(grad[:, offset:offset + size], w_t, index))
            offset += size
        return torch.cat(out, dim=1).reshape(atoms * total, k)
    gx = grad.new_empty((atoms * total, k))
    if not grad.numel():
        return gx
    if precision == "tf32x3-cutlass" and grad.is_contiguous() and _rows_cutlass(gx, w_ts, grad.view(-1, grad.shape[2])):
        from fastmace.kernels.cutlass import tf32x3
        for w_t, row, size in zip(w_ts, rows, sizes):
            tf32x3.species_into(grad, grad.shape[2], row, w_t, gx, row, counts, size,
                                tiles(atoms, size, counts.shape[0]))
        return gx
    views, offset = _rows_views(gx, sizes), 0
    for v, w_t, size in zip(views, w_ts, sizes):
        block = grad.new_empty((atoms, size, k))
        _launch(grad[:, offset:offset + size].contiguous(), w_t, order, counts, block, precision)
        v.copy_(block)
        offset += size
    return gx


@species_mm_rows_grad.register_fake
def _(grad, sizes, w_ts, index, order, counts, rows, precision):
    return grad.new_empty((grad.shape[0] * sum(sizes), w_ts[0].shape[2]))


def _rows_setup(ctx, inputs, output):
    x, sizes, ws, w_ts, index, order, counts, rows, precision = inputs
    ctx.sizes, ctx.paths, ctx.precision = list(sizes), len(ws), precision
    ctx.save_for_backward(*w_ts, index, order, counts, *rows)


def _rows_backward(ctx, grad):
    p = ctx.paths
    saved = ctx.saved_tensors
    w_ts, (index, order, counts), rows = list(saved[:p]), saved[p:p + 3], list(saved[p + 3:])
    gx = None
    if ctx.needs_input_grad[0]:
        gx = species_mm_rows_grad(grad.contiguous(), ctx.sizes, w_ts, index, order, counts, rows, ctx.precision)
    nothing = [None] * p
    return gx, None, list(nothing), list(nothing), None, None, None, list(nothing), None


species_mm_rows.register_autograd(_rows_backward, setup_context=_rows_setup)


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "CUTLASS", "TENSOR_CORES", "joined_rows", "species_mm",
           "species_mm_joined", "species_mm_rows", "species_plan", "tc_supported", "tile_rows", "tiles"]
