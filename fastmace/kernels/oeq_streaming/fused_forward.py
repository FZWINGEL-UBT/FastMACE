"""The folded radial last layer fused into the receiver-streaming forward (``fused_folded_forward``, opt-8 OPT8-T02).

Today's chain builds the per-edge weights ``w = a @ W.T`` with cuBLAS (``W [Q, K]``, ``Q = paths × channels``) and
the streaming forward reads them back. The fused forward builds each batch's ``w`` rows on chip from ``a`` and a
shared-memory slice of ``W``, uses them for the message at once and also stores them, so the chunked backward
(``chunked_conv_backward``) runs unchanged on them. It saves the cuBLAS product and the forward's ``w`` read.
The stored ``w`` and the message are bitwise those of the chain on the measured GPUs: each ``w`` element is the
same k-ordered FP32 sum of fused multiply-adds, and each receiver's message sums its edges in CSR order with the
same angular terms. Results: ``docs/validation/opt8-T01.md`` (T01b) and ``docs/validation/opt8-T02.md``.

Generated, like the vendored kernels, from the streaming schedule (``FactorizedComputationSchedule``) of any uvu
problem whose paths have one edge channel and whose channel count is a multiple of 32; float32 only. Work split:

- a block owns one slice: 32 channels (``blockIdx.y``) × one part of the paths (``blockIdx.z``), and keeps that
  slice of ``W`` in shared memory (row stride 33 floats, conflict-free);
- one warp (lane = channel) takes chunks of ``RECEIVERS_PER_CHUNK`` receivers from a per-slice counter and walks
  their edges in batches of ``BATCH`` across receiver boundaries; the batch's ``a`` rows are staged transposed in
  warp-private shared memory (four edges per float4 broadcast);
- per batch: ``w[part paths][BATCH]`` by a k-loop that reads each ``W`` value once for the batch, stored to ``w``;
  then per edge the angular terms into the receiver's outputs (registers), written when the receiver changes;
- edges past the last CSR row (the neighbour list's padding) get their ``w`` rows the same way, without messages,
  so the backward sees what the chain would have built.

Launch rule (``launch_rule``), from the device and the problem only:
- **paths per part** at most ``W_TILE_REGISTERS / BATCH`` (5): the part's ``w`` tile is at most 40 registers
  per lane;
- **shared memory:** the fewest parts whose ``W`` slice plus ``MIN_WARPS`` warps' staging fit the device's
  opt-in shared memory per block;
- **warps per block:** as many as fit beside the slice, at most ``MAX_WARPS`` (12, the warps for which 168
  registers per thread fit the 64K-register file of every sm_70+ GPU);
- **grid:** ``ceil(2 × SMs / slices)`` blocks per slice.
"""

from __future__ import annotations

import ctypes
import itertools
from dataclasses import dataclass

import torch

BATCH = 8  # edges per batch: two float4 broadcasts of a per k
RECEIVERS_PER_CHUNK = 2  # receivers per work chunk (OPT8-T01 screen: 2 balanced best on NbCl5 and NaNbO3)
TAIL_EDGES = 64  # padding edges (past the last CSR row) per work chunk
W_TILE_REGISTERS = 40  # paths per part × BATCH
MAX_WARPS = 12
MIN_WARPS = 4
CHANNEL_BLOCK = 32
BLOCKS_PER_SM = 2  # blocks per SM over all slices of the grid (the persistent grid's size)

_REGISTRY = {}
_IDS = itertools.count()


def _lit(value):
    return f"{float(value).hex()}f"


@dataclass(frozen=True)
class LaunchRule:
    parts: int
    warps: int
    blocks_per_slice: int
    shared_bytes: int


def slice_floats(paths_per_part, k):
    """Floats of a block's W slice (row stride CHANNEL_BLOCK + 1), rounded up to keep the staging 16-byte aligned."""
    return -(-paths_per_part * k * (CHANNEL_BLOCK + 1) // 4) * 4


def staging_floats(k, edge_dim):
    """Floats of one warp's staging: a [K][BATCH] and sh [BATCH][edge_dim]."""
    return BATCH * (k + edge_dim)


def launch_rule(paths, k, edge_dim, channels, shared_optin, sms):
    """The kernel's parts, warps and grid for ``paths`` uvu paths of ``k``-wide folded inputs on a device with
    ``shared_optin`` bytes of opt-in shared memory per block and ``sms`` SMs (module docstring)."""
    staging = 4 * staging_floats(k, edge_dim)
    parts = max(1, -(-paths * BATCH // W_TILE_REGISTERS))
    while parts <= paths:
        per = -(-paths // parts)
        if 4 * slice_floats(per, k) + MIN_WARPS * staging <= shared_optin:
            break
        parts += 1
    else:
        raise ValueError(f"fused_folded_forward: no W slice of {k}-wide rows fits {shared_optin} bytes of "
                         "shared memory")
    per = -(-paths // parts)
    parts = -(-paths // per)
    wbytes = 4 * slice_floats(per, k)
    warps = min(MAX_WARPS, (shared_optin - wbytes) // staging)
    slices = (channels // CHANNEL_BLOCK) * parts
    return LaunchRule(parts, warps, max(1, -(-BLOCKS_PER_SM * sms // slices)), wbytes + warps * staging)


def balanced_parts(costs, parts):
    """Paths split into ``parts`` groups of equal size (the last may be smaller); the angular terms (``costs``)
    balanced greedily, heaviest first; each group sorted."""
    per = -(-len(costs) // parts)
    groups, load = [[] for _ in range(parts)], [0] * parts
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        q = min((q for q in range(parts) if len(groups[q]) < per), key=lambda q: (load[q], q))
        groups[q].append(index)
        load[q] += costs[index]
    return [sorted(g) for g in groups if g]


def check_problem(schedule):
    """Raises ValueError unless the fused forward can run this schedule."""
    if schedule.channels % CHANNEL_BLOCK:
        raise ValueError(f"fused_folded_forward needs channels divisible by {CHANNEL_BLOCK}, "
                         f"not {schedule.channels}")
    if any(p.edge_mul != 1 for p in schedule.paths):
        raise ValueError("fused_folded_forward needs one edge channel per path")
    if schedule.layout not in ("ir_mul", "mul_ir"):
        raise ValueError(f"unknown layout {schedule.layout!r}")


def _node(layout, channels, start, component, dim):
    """The node-feature offset expression of the vendored template (``node_offset``) for channel ``ch``."""
    if layout == "ir_mul":
        return f"{start + component * channels} + ch"
    return f"{start + component} + ch * {dim}"


def render(schedule, k, rule, store_w=True):
    """CUDA source of ``fused_folded_forward`` for a schedule, the folded input width ``k`` and a launch rule.
    ``store_w=False``: the message alone; ``w`` is neither stored nor built for the padding edges (``wout`` unused;
    plans/capacity.md, the w-rebuild falsifier)."""
    check_problem(schedule)
    C, layout, paths = schedule.channels, schedule.layout, schedule.paths
    costs = [sum(len(o.terms) for o in p.outputs) for p in paths]
    parts = balanced_parts(costs, rule.parts)
    owner = {}
    for q, part in enumerate(parts):
        for pi in part:
            for o in paths[pi].outputs:
                if owner.setdefault(o.accumulator.name, q) != q:
                    raise ValueError("fused_folded_forward: an output slot is shared by two path parts")
    slots = {s.name: s for s in schedule.output_slots}
    npp = max(len(p) for p in parts)
    cases = "\n".join(f"    case {q}: {{{_part(schedule, part, slots, C, layout)}    }} break;"
                      for q, part in enumerate(parts))
    return f"""// fused_folded_forward (FastMACE opt-8): folded last layer + receiver-streaming forward, w stored.
using int64_t = long long;
constexpr bool STORE_W = {"true" if store_w else "false"};
constexpr int K = {k};
constexpr int CB = {CHANNEL_BLOCK};
constexpr int WS = {CHANNEL_BLOCK + 1};
constexpr int ED = {schedule.edge_dim};
constexpr int IN_DIM = {schedule.input_dim};
constexpr int OUT_DIM = {schedule.output_dim};
constexpr int64_t Q = {len(paths) * C};
constexpr int B = {BATCH};
constexpr int RC = {RECEIVERS_PER_CHUNK};
constexpr int TAIL = {TAIL_EDGES};
constexpr unsigned FULL = 0xffffffffu;

extern "C" __global__ void __launch_bounds__({rule.warps * 32}, 1) fused_folded_forward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr, int* __restrict__ counters,
    float* __restrict__ out, float* __restrict__ wout) {{
    extern __shared__ float smem[];
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* a_s = Ws + {slice_floats(npp, k)} + warp * {staging_floats(k, schedule.edge_dim)};
    float* sh_s = a_s + K * B;
    const int cb = blockIdx.y;
    const int ch = cb * CB + lane;
    int* ctr = counters + blockIdx.y * gridDim.z + blockIdx.z;
    // Receiver chunks first, then chunks of the padding edges past the last row.
    const int64_t receiver_chunks = (N + RC - 1) / RC;
    int64_t tail_begin = row_ptr[N] < 0 ? 0 : row_ptr[N];
    tail_begin = tail_begin > E ? E : tail_begin;
    switch (blockIdx.z) {{
{cases}
    }}
}}
"""


def _part(schedule, part, slots, C, layout):
    paths = schedule.paths
    xkeys = sorted({(paths[pi].input_start, t.input_component, paths[pi].input_irrep_dim)
                    for pi in part for o in paths[pi].outputs for t in o.terms})
    xi = {key: i for i, key in enumerate(xkeys)}
    nx = len(xkeys)
    names = list(dict.fromkeys(o.accumulator.name for pi in part for o in paths[pi].outputs))
    si = {n: i for i, n in enumerate(names)}
    n = len(names)
    npl = len(part)
    wst = ", ".join(str(paths[pi].weight_start) for pi in part)
    xloads = "\n".join(f"                    xv[b][{i}] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + "
                       f"{_node(layout, C, s, c, d)}] : 0.f;" for (s, c, d), i in xi.items())
    outs = "\n".join(
        f"                out[cur * OUT_DIM + "
        f"{_node(layout, C, slots[name].start, slots[name].component, slots[name].irrep_dim)}] = acc[{i}];"
        for name, i in si.items())
    gemm = []
    for q in range(BATCH // 4):
        gemm.append(f"                    const float4 av{q} ="
                    f" *reinterpret_cast<const float4*>(&a_s[k * B + {4 * q}]);")
    for g in range(npl):
        gemm.append(f"                    {{ const float wk = Ws[({g} * K + k) * WS + lane];")
        for q in range(BATCH // 4):
            for c, comp in enumerate("xyzw"):
                b = 4 * q + c
                gemm.append(f"                      w[{g}][{b}] = fmaf(av{q}.{comp}, wk, w[{g}][{b}]);")
        gemm.append("                    }")
    tp = []
    for g, pi in enumerate(part):
        p = paths[pi]
        tp.append("                        {")
        for j in range(p.edge_irrep_dim):
            tp.append(f"                        const float e{j} = sh_s[b * ED + {p.edge_start + j}];")
        for o in p.outputs:
            terms = " + ".join(f"xv[b][{xi[(p.input_start, t.input_component, p.input_irrep_dim)]}]"
                               f" * e{t.edge_component} * {_lit(t.coefficient)}" for t in o.terms) or "0.f"
            tp.append(f"                        acc[{si[o.accumulator.name]}] = fmaf(w[{g}][b], {terms},"
                      f" acc[{si[o.accumulator.name]}]);")
        tp.append("                        }")
    flush = f"""{{
{outs}
                #pragma unroll
                for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
                ++cur;
            }}"""
    return f"""
        {{
            const int wst[{npl}] = {{{wst}}};
            for (int i = threadIdx.x; i < {npl} * K * CB; i += blockDim.x) {{
                const int p = i / (K * CB), rem = i - p * (K * CB), k = rem / CB, l = rem - k * CB;
                Ws[(p * K + k) * WS + l] = W[(int64_t)(wst[p] + cb * CB + l) * K + k];
            }}
        }}
        __syncthreads();
        while (true) {{
            int cc = 0;
            if (lane == 0) cc = atomicAdd(ctr, 1);
            cc = __shfl_sync(FULL, cc, 0);
            int64_t r0, r1, ebeg, eend;
            const bool tail = cc >= receiver_chunks;
            if (!tail) {{
                r0 = (int64_t)cc * RC;
                r1 = r0 + RC < N ? r0 + RC : N;
                ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                eend = eend > E ? E : eend;
            }} else {{
                r0 = r1 = N;
                ebeg = tail_begin + (cc - receiver_chunks) * TAIL;
                if (ebeg >= E || !STORE_W) break;
                eend = ebeg + TAIL < E ? ebeg + TAIL : E;
            }}
            float acc[{n}];
            #pragma unroll
            for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
            int64_t cur = r0;
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
                for (int i = lane; i < B * K; i += 32) {{
                    const int b = i / K, k = i - b * K;
                    a_s[k * B + b] = (b < nb) ? a[(e0 + b) * K + k] : 0.f;
                }}
                for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;
                float xv[B][{nx}];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {{
                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;
                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;
                    if (snd < 0 || snd >= N) snd = -1;
{xloads}
                }}
                __syncwarp();
                float w[{npl}][B];
                #pragma unroll
                for (int g = 0; g < {npl}; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = 0.f;
                #pragma unroll 5
                for (int k = 0; k < K; ++k) {{
{chr(10).join(gemm)}
                }}
                if (STORE_W) {{
                    const int wst[{npl}] = {{{wst}}};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {{
                            #pragma unroll
                            for (int g = 0; g < {npl}; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                        }}
                }}
                if (!tail) {{
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {{
                        if (b >= nb) break;
                        while (cur < rv[b]) {flush}
{chr(10).join(tp)}
                    }}
                }}
            }}
            while (cur < r1) {flush}
        }}
"""


class FusedFoldedForward:
    """Compiled ``fused_folded_forward`` for the streaming kernels of one interaction (``StreamingKernels``) and a
    folded input width ``k``; float32 only."""

    def __init__(self, kernels, k, device=None, store_w=True):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
        if "using scalar_t = float;" not in kernels.source:
            raise ValueError("fused_folded_forward needs float32 streaming kernels")
        schedule = kernels.schedule
        check_problem(schedule)
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        props = torch.cuda.get_device_properties(device)
        self.rule = launch_rule(len(schedule.paths), int(k), schedule.edge_dim, schedule.channels,
                                int(props.shared_memory_per_block_optin), int(props.multi_processor_count))
        self.k, self.channels, self.paths = int(k), schedule.channels, len(schedule.paths)
        self.output_dim, self.weight_dim = schedule.output_dim, kernels.weight_dim
        self.store_w = bool(store_w)
        self.source = render(schedule, self.k, self.rule, self.store_w)
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(device))
        torch.cuda.current_stream(device)
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._function = _check(driver.cuModuleGetFunction(self._module, b"fused_folded_forward"))
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(driver.cuFuncSetAttribute(self._function, attribute, self.rule.shared_bytes))
        self.slices = (self.channels // CHANNEL_BLOCK) * self.rule.parts
        self.id = next(_IDS)
        _REGISTRY[self.id] = self

    def __call__(self, a, weight, x, sh, senders, receivers, row_ptr, out, w):
        """Writes the message into ``out [N, output_dim]`` and ``a @ weight.T`` into ``w [E, Q]``."""
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        nodes, edges = x.shape[0], sh.shape[0]
        counters = torch.zeros(self.slices, dtype=torch.int32, device=x.device)
        p = ctypes.c_void_p
        values = (nodes, edges, a.data_ptr(), weight.data_ptr(), x.data_ptr(), sh.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), row_ptr.data_ptr(), counters.data_ptr(), out.data_ptr(), w.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p, p)
        stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        _check(driver.cuLaunchKernel(self._function, self.rule.blocks_per_slice, self.channels // CHANNEL_BLOCK,
                                     self.rule.parts, self.rule.warps * 32, 1, 1, self.rule.shared_bytes, stream,
                                     (values, types), 0))


@torch.library.custom_op("fastmace::fused_folded_forward", mutates_args=())
def fused_folded_forward(a: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, sh: torch.Tensor,
                         senders: torch.Tensor, receivers: torch.Tensor, row_ptr: torch.Tensor,
                         fused_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(message [N, output_dim], w = a @ weight.T [E, Q]) through the fused kernel ``fused_id``."""
    fused = _REGISTRY[fused_id]
    a, weight, x, sh = a.contiguous(), weight.contiguous(), x.contiguous(), sh.contiguous()
    out = x.new_empty((x.shape[0], fused.output_dim))
    w = x.new_empty((sh.shape[0], weight.shape[0]))
    if sh.shape[0] or x.shape[0]:
        fused(a, weight, x, sh, senders.contiguous(), receivers.contiguous(), row_ptr.contiguous(), out, w)
    return out, w


@fused_folded_forward.register_fake
def _(a, weight, x, sh, senders, receivers, row_ptr, fused_id):
    return x.new_empty((x.shape[0], _REGISTRY[fused_id].output_dim)), x.new_empty((sh.shape[0], weight.shape[0]))

