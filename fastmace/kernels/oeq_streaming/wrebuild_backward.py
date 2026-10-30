"""``wprime_backward`` without stored per-edge weights: the backward builds ``w`` on chip (the w-rebuild falsifier).

``wprime_backward`` (opt-9 OPT9-T03) reads ``w = a @ W.T`` (``[E, Q]``) that ``fused_folded_forward`` stored; at
32,400 NbCl5 atoms that is 5.1 GiB, 93 % of what the force call saves per edge (docs/validation/
capacity-T08-inventory.md). Here the forward stores nothing (``FusedFoldedForward(store_w=False)``) and this kernel
builds each batch's ``w`` rows again, exactly as the fused forward does: a block owns one slice (32 channels ×
one part of the paths) and keeps that slice of ``W`` in shared memory; a warp (lane = channel) takes chunks of
edges and builds ``w[part paths][BATCH]`` by the same k-ordered fused multiply-adds from the same staged ``a``, so
each ``w`` element is bitwise the forward's. Per edge it then computes the vendored backward's sums for its slice:

- ``dx``: per (sender, slot, channel) over the part's paths, added atomically (as today, over the slices);
- ``dsh`` and ``dL/dr``: reduced over the warp's 32 channels and the part's paths, added atomically over the
  slices (today one 32-lane group owns an edge and writes them once).

So ``w`` and the message are unchanged, while ``dx``, ``dsh`` and ``dr`` are rounding-class (another summation order).
Generated from the streaming schedule like ``fused_forward``; float32, one edge channel per path, channel counts
divisible by 32. Edges are taken in chunks of ``EDGE_CHUNK`` from a per-slice counter. Edges whose sender or
receiver is outside ``[0, N)`` (the neighbour list's padding) contribute nothing; ``dsh`` and ``dr`` start at zero.
"""

from __future__ import annotations

import ctypes

import torch

from fastmace.kernels.oeq_streaming.fused_forward import (
    BATCH,
    CHANNEL_BLOCK,
    MAX_WARPS,
    LaunchRule,
    _lit,
    _node,
    balanced_parts,
    check_problem,
    launch_rule,
    slice_floats,
)

EDGE_CHUNK = 64  # edges per work chunk (eight batches)

_REGISTRY = {}


def staging_floats(k, edge_dim, batch=BATCH):
    """Floats of one warp's staging: a [K][batch] and sh [batch][edge_dim]."""
    return batch * (k + edge_dim)


def rule_for(paths, k, edge_dim, channels, shared_optin, sms, parts=None, batch=BATCH):
    """The fused forward's launch rule, or one with ``parts`` path parts and ``batch`` edges per batch (tuning)."""
    if parts is None and batch == BATCH:
        return launch_rule(paths, k, edge_dim, channels, shared_optin, sms)
    parts = parts or 1
    per = -(-paths // parts)
    wbytes = 4 * slice_floats(per, k)
    staging = 4 * staging_floats(k, edge_dim, batch)
    warps = min(MAX_WARPS, (shared_optin - wbytes) // staging)
    if warps < 1:
        raise ValueError(f"wrebuild_backward: {parts} part(s) of {k}-wide rows do not fit {shared_optin} bytes")
    slices = (channels // CHANNEL_BLOCK) * parts
    return LaunchRule(parts, warps, max(1, -(-2 * sms // slices)), wbytes + warps * staging)


def render(schedule, k, rule, terms, batch=BATCH):
    """CUDA source of ``wrebuild_backward`` for a schedule, the folded input width ``k``, a launch rule (the fused
    forward's, or ``rule_for``'s), the derivative table's ``terms`` and ``batch`` edges per batch (4 or 8)."""
    check_problem(schedule)
    if batch % 4:
        raise ValueError("wrebuild_backward: the batch is a multiple of 4 (float4 broadcasts of a)")
    paths = schedule.paths
    costs = [sum(len(o.terms) for o in p.outputs) for p in paths]
    parts = balanced_parts(costs, rule.parts)
    npp = max(len(p) for p in parts)
    cases = "\n".join(f"    case {q}: {{{_part(schedule, part, batch)}    }} break;" for q, part in enumerate(parts))
    return f"""// wrebuild_backward (FastMACE): wprime_backward building w on chip (the fused forward's w, bitwise).
using int64_t = long long;
constexpr int K = {k};
constexpr int CB = {CHANNEL_BLOCK};
constexpr int WS = {CHANNEL_BLOCK + 1};
constexpr int ED = {schedule.edge_dim};
constexpr int IN_DIM = {schedule.input_dim};
constexpr int OUT_DIM = {schedule.output_dim};
constexpr int64_t Q = {len(paths) * schedule.channels};
constexpr int B = {batch};
constexpr int CHUNK = {EDGE_CHUNK};
constexpr int TERMS = {int(terms)};
constexpr unsigned FULL = 0xffffffffu;

extern "C" __global__ void __launch_bounds__({rule.warps * 32}, 1) wrebuild_backward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const float* __restrict__ dout, float* __restrict__ dx,
    float* __restrict__ dsh, const float* __restrict__ lengths, const int* __restrict__ pairs,
    const float* __restrict__ gprime, const float* __restrict__ starts, const float* __restrict__ scales,
    const int* __restrict__ bins, float r_lo, float bin_scale, float r_max, int n_bins, int pieces,
    int* __restrict__ counters, float* __restrict__ dr) {{
    extern __shared__ float smem[];
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* a_s = Ws + {slice_floats(npp, k)} + warp * {staging_floats(k, schedule.edge_dim, batch)};
    float* sh_s = a_s + K * B;
    const int cb = blockIdx.y;
    const int ch = cb * CB + lane;
    int* ctr = counters + blockIdx.y * gridDim.z + blockIdx.z;
    switch (blockIdx.z) {{
{cases}
    }}
}}
"""


def _part(schedule, part, batch):
    paths = schedule.paths
    C, layout = schedule.channels, schedule.layout
    npl = len(part)
    wst = ", ".join(str(paths[pi].weight_start) for pi in part)
    slots = list(dict.fromkeys(t.input_accumulator.name for pi in part for o in paths[pi].outputs for t in o.terms))
    slot_of = {t.input_accumulator.name: t.input_accumulator
               for pi in part for o in paths[pi].outputs for t in o.terms}
    gemm = []
    for q in range(batch // 4):
        gemm.append(f"                    const float4 av{q} ="
                    f" *reinterpret_cast<const float4*>(&a_s[k * B + {4 * q}]);")
    for g in range(npl):
        gemm.append(f"                    {{ const float wk = Ws[({g} * K + k) * WS + lane];")
        for q in range(batch // 4):
            for c, comp in enumerate("xyzw"):
                b = 4 * q + c
                gemm.append(f"                      w[{g}][{b}] = fmaf(av{q}.{comp}, wk, w[{g}][{b}]);")
        gemm.append("                    }")
    body = []
    for g, pi in enumerate(part):
        p = paths[pi]
        body.append("                    {")
        body.append(f"                        const float radial_weight = w[{g}][b];")
        body.append("                        float weight_gradient = 0.f;")
        for o in p.outputs:
            slot = o.accumulator
            body.append(f"                        {{ const float output_gradient = dout[(int64_t)rcv * OUT_DIM + "
                        f"{_node(layout, C, slot.start, slot.component, slot.irrep_dim)}];")
            for t in o.terms:
                edge = p.edge_start + t.edge_component
                coef = _lit(t.coefficient)
                body.append(f"                          {{ const float input_value = x[(int64_t)snd * IN_DIM + "
                            f"{_node(layout, C, p.input_start, t.input_component, p.input_irrep_dim)}];")
                body.append(f"                            const float edge_value = sh_s[b * ED + {edge}];")
                body.append(f"                            weight_gradient += output_gradient * input_value"
                            f" * edge_value * {coef};")
                body.append(f"                            {t.input_accumulator.name} += output_gradient"
                            f" * radial_weight * edge_value * {coef};")
                body.append(f"                            edge_gradient[{edge}] += output_gradient * radial_weight"
                            f" * input_value * {coef}; }}")
            body.append("                        }")
        body.append("                        if (sloped) {")
        body.append("                            float slope = 0.f;")
        body.append("                            #pragma unroll")
        body.append(f"                            for (int t = 0; t < TERMS; ++t) slope += cheb[t] * "
                    f"__ldg(gp + t * Q + {p.weight_start} + ch);")
        body.append("                            radial_gradient += weight_gradient * slope;")
        body.append("                        }")
        body.append("                    }")
    zero = "\n".join(f"                    float {name} = 0.f;" for name in slots)
    adds = "\n".join(
        f"                    atomicAdd(dx + (int64_t)snd * IN_DIM + "
        f"{_node(layout, C, slot_of[name].start, slot_of[name].component, slot_of[name].irrep_dim)}, {name});"
        for name in slots)
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
            const int64_t ebeg = (int64_t)cc * CHUNK;
            if (ebeg >= E) break;
            const int64_t eend = ebeg + CHUNK < E ? ebeg + CHUNK : E;
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
                for (int i = lane; i < B * K; i += 32) {{
                    const int b = i / K, k = i - b * K;
                    a_s[k * B + b] = (b < nb) ? a[(e0 + b) * K + k] : 0.f;
                }}
                for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;
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
                #pragma unroll
                for (int b = 0; b < B; ++b) {{
                    if (b >= nb) break;
                    const int64_t e = e0 + b;
                    const int snd = senders[e], rcv = receivers[e];
                    if (snd < 0 || snd >= N || rcv < 0 || rcv >= N) continue;
                    const float r = lengths[e];
                    const bool sloped = r < r_max && r >= r_lo;
                    const float clamped = r > r_lo ? r : r_lo;
                    int bin = (int)((clamped - r_lo) * bin_scale);
                    bin = bin < n_bins - 1 ? bin : n_bins - 1;
                    const int piece = bins[bin];
                    const float tt = (clamped - starts[piece]) * scales[piece] - 1.f;
                    const float* gp = gprime + ((int64_t)pairs[e] * pieces + piece) * TERMS * Q;
                    float cheb[TERMS];
                    cheb[0] = 1.f;
                    if (TERMS > 1) cheb[1] = tt;
                    #pragma unroll
                    for (int t = 2; t < TERMS; ++t) cheb[t] = 2.f * tt * cheb[t - 1] - cheb[t - 2];
                    float edge_gradient[ED];
                    #pragma unroll
                    for (int c = 0; c < ED; ++c) edge_gradient[c] = 0.f;
                    float radial_gradient = 0.f;
{zero}
{chr(10).join(body)}
{adds}
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {{
                        #pragma unroll
                        for (int c = 0; c < ED; ++c)
                            edge_gradient[c] += __shfl_down_sync(FULL, edge_gradient[c], offset);
                        radial_gradient += __shfl_down_sync(FULL, radial_gradient, offset);
                    }}
                    if (lane == 0) {{
                        #pragma unroll
                        for (int c = 0; c < ED; ++c) atomicAdd(dsh + e * ED + c, edge_gradient[c]);
                        atomicAdd(dr + e, radial_gradient);
                    }}
                }}
            }}
        }}
"""


class WRebuildBackward:
    """Compiled ``wrebuild_backward`` for the streaming kernels of one interaction (``StreamingKernels``), a folded
    input width ``k`` and a derivative table of ``terms`` terms; float32 only."""

    def __init__(self, kernels, k, terms, device=None, parts=None, batch=BATCH):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        if "using scalar_t = float;" not in kernels.source:
            raise ValueError("wrebuild_backward needs float32 streaming kernels")
        schedule = kernels.schedule
        check_problem(schedule)
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        props = torch.cuda.get_device_properties(device)
        self.rule = rule_for(len(schedule.paths), int(k), schedule.edge_dim, schedule.channels,
                             int(props.shared_memory_per_block_optin), int(props.multi_processor_count), parts, batch)
        self.k, self.terms, self.channels, self.batch = int(k), int(terms), schedule.channels, int(batch)
        self.source = render(schedule, self.k, self.rule, self.terms, self.batch)
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(device))
        torch.cuda.current_stream(device)
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._function = _check(driver.cuModuleGetFunction(self._module, b"wrebuild_backward"))
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(driver.cuFuncSetAttribute(self._function, attribute, self.rule.shared_bytes))
        self.slices = (self.channels // CHANNEL_BLOCK) * self.rule.parts
        self.id = content_id("WRebuildBackward", self.source, self.rule, _device_key(device))
        _REGISTRY[self.id] = self

    def __call__(self, a, weight, x, sh, senders, receivers, dout, r, pair, gprime, starts, scales, bins, r_lo,
                 bin_scale, r_max):
        """(dx, dsh, dL/dr [E]) of the streaming convolution with ``w = a @ weight.T`` built on chip."""
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        a, weight, x, sh, dout = a.contiguous(), weight.contiguous(), x.contiguous(), sh.contiguous(), dout.contiguous()
        senders, receivers, r = senders.contiguous(), receivers.contiguous(), r.reshape(-1).contiguous()
        dx, dsh, dr = torch.zeros_like(x), torch.zeros_like(sh), sh.new_zeros((sh.shape[0],))
        nodes, edges = x.shape[0], sh.shape[0]
        if edges:
            counters = torch.zeros(self.slices, dtype=torch.int32, device=x.device)
            p = ctypes.c_void_p
            values = (nodes, edges, a.data_ptr(), weight.data_ptr(), x.data_ptr(), sh.data_ptr(),
                      senders.data_ptr(), receivers.data_ptr(), dout.data_ptr(), dx.data_ptr(), dsh.data_ptr(),
                      r.data_ptr(), pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(),
                      bins.data_ptr(), r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], counters.data_ptr(),
                      dr.data_ptr())
            types = (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p,
                     ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, p, p)
            stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
            _check(driver.cuLaunchKernel(self._function, self.rule.blocks_per_slice, self.channels // CHANNEL_BLOCK,
                                         self.rule.parts, self.rule.warps * 32, 1, 1, self.rule.shared_bytes,
                                         stream, (values, types), 0))
        return dx, dsh, dr


@torch.library.custom_op("fastmace::wrebuild_backward", mutates_args=())
def wrebuild_backward(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor,
                      dout: torch.Tensor, r: torch.Tensor, pair: torch.Tensor, gprime: torch.Tensor,
                      coef: torch.Tensor, weight: torch.Tensor, starts: torch.Tensor, scales: torch.Tensor,
                      bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                      rebuild_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dsh, dL/dr [E] of the radial path) with ``w`` built on chip (``recompute_conv_weights``): the radial
    basis ``a`` again from the edge lengths (the forward's table, bitwise), then the kernel ``rebuild_id``."""
    from fastmace.kernels.triton.radial_table import radial_table_forward

    r = r.reshape(-1).contiguous()
    a, _ = radial_table_forward(r, pair.contiguous(), coef, starts, scales, bins, r_lo, bin_scale, r_max)
    return _REGISTRY[rebuild_id](a, weight, x, sh, senders, receivers, dout, r, pair, gprime, starts, scales, bins,
                                 r_lo, bin_scale, r_max)


@wrebuild_backward.register_fake
def _(x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts, scales, bins, r_lo, bin_scale, r_max,
      rebuild_id):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


__all__ = ["EDGE_CHUNK", "WRebuildBackward", "render", "rule_for", "staging_floats", "wrebuild_backward"]
