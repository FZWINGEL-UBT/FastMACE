"""SPLINE_W: a generic interaction's message with ``w`` from a cubic spline in r, in Symmetrix-XL's layout.

sx-layout lab (7 November 2026, results/sx-layout-lab-20261107): one thread per (receiver, channel), the receiver's
CSR edges walked serially, every output accumulator in registers and stored once; ``w`` per edge and path from the
piecewise-uniform cubic spline of ``rewrites.radial.spline`` (power basis of t, one float4 per weight from the
q-major table ``[pairs * intervals, Q, 4]`` that the w' backward reads too) instead of the a.W tensor-core product of
``fused_forward``. The angular code is generated from the schedule's paths as ``fused_forward._part``'s, with the
w' table's per-path FOLD_ALPHA factors divided out of the CG coefficients. The message is written in the block
layout (``wprime_backward.BLOCK_MESSAGE``).

Lab (H200, OMAT-0 medium, SrTiO3 10x10x10): forward 0.875 of today's, w' 0.894; MD step about -310 us.
"""
from __future__ import annotations

import ctypes

import torch

from fastmace.kernels.gpu_models import GPUModels

# None: on for SPLINE_W_GPUS (decision 66); True / False: forced (an A/B side).
SPLINE_W = None
# NVIDIA H200 (7 November 2026, results/spline-w-h200-20261107): MD step b/a 0.8789 / 0.8775 (b = on; OMAT-0 medium,
# SrTiO3 10x10x10, Benchmark_loup settings, ab / ba), force/stress gate passes. The A100 is not measured.
SPLINE_W_GPUS = GPUModels(("NVIDIA H200",))
# Launch shape (SX improvement screen, 7 November 2026; plans/notes/sx-layout-opportunities-20261107). THREADS: one
# owner per (receiver, channel); 128 threads with 128 channels is one receiver per block, so receivers of different
# degree retire independently instead of in pairs. MIN_BLOCKS: __launch_bounds__' minimum blocks per SM (a register
# cap: 128 x 6 and 256 x 3 both give 24 warps at 80 registers), None for none. PARTS: one kernel per input irrep
# dimension (scalar-input and vector-input paths; DEGREE_PARTS' split in this layout), each walking every receiver's
# edges for its own output slots: fewer registers per kernel, the edge walk and lookup twice. All three bitwise (the
# same terms, per output slot in the same edge order). A/B switches; THREADS 256, MIN_BLOCKS None, PARTS False today.
THREADS = 256
MIN_BLOCKS = None
PARTS = False  # H200 on top of SPLINE_W with MIN_BLOCKS 4: 0.9938 / 0.9904 (under 1 %; off, not tried with R128)
# R128: THREADS 128 with MIN_BLOCKS 6 (one receiver per block, 80-register cap) as the launch shape; it overrides
# THREADS and MIN_BLOCKS where on. None: on for R128_GPUS (decision 66); True / False: an A/B side's choice.
R128 = None
# NVIDIA H200 (7 November 2026, results/spline-w-h200-20261107): MD step b/a 0.9838 / 0.9848 on top of SPLINE_W
# (80 registers, 8 bytes local memory).
R128_GPUS = GPUModels(("NVIDIA H200",))


def r128_default(device):
    """``R128`` on ``device``: the constant when set, else on for ``R128_GPUS``."""
    if R128 is not None:
        return bool(R128)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in R128_GPUS

_REGISTRY = {}


def spline_w_default(device):
    """``SPLINE_W`` on ``device``: the constant when set, else on for ``SPLINE_W_GPUS``."""
    if SPLINE_W is not None:
        return bool(SPLINE_W)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in SPLINE_W_GPUS


def lookup_code(grid, r, scalar="float", indent="    "):
    """C code defining ``iv`` (the global interval), ``t`` in [0, 1) and ``sx_inv`` (1 / h) from length ``r``: one
    compare per band edge (an edge is one warp's for every channel, so the branch is uniform), then one multiply."""
    def lit(v):
        return f"{float(v).hex()}f" if scalar == "float" else f"{scalar}({float(v).hex()})"

    lines = [f"{indent}int iv; {scalar} t, sx_inv;"]
    offset = 0
    for k, (a, b, n) in enumerate(grid):
        head = "if" if k == 0 else "else if"
        cond = f"{head} ({r} < {lit(b)}) " if k < len(grid) - 1 else ("else " if k else "")
        body = (f"{{ sx_inv = {lit(n / (b - a))}; {scalar} u = ({r} - {lit(a)}) * sx_inv; "
                f"u = u > {lit(0)} ? u : {lit(0)}; int il = (int)u; il = il < {n - 1} ? il : {n - 1}; "
                f"t = u - ({scalar})il; iv = {offset} + il; }}")
        lines.append(f"{indent}{cond}{body}")
        offset += n
    return "\n".join(lines)


def spline_parts(schedule):
    """``PARTS``' path groups (``fused_forward.degree_parts``), or None where there is one input degree or two
    groups share an output slot (each part must own its slots)."""
    from fastmace.kernels.oeq_streaming.fused_forward import degree_parts

    groups = degree_parts(schedule)
    if groups is None:
        return None
    owned = [{o.accumulator.name for g in group for o in schedule.paths[g].outputs} for group in groups]
    if sum(len(o) for o in owned) != len(set().union(*owned)):
        return None
    return groups


def render(schedule, blocks, grid, r_max, path_scale=None, part=None, threads=None, min_blocks=None,
           name="spline_forward"):
    """CUDA source of ``spline_forward`` for a schedule (ir_mul), its block layout ``blocks`` and the spline's
    ``grid``; ``path_scale``: per path the factor the table's rows carry (FOLD_ALPHA), divided out here. ``part``:
    the path indices this kernel computes (``PARTS``; None: all); ``threads`` / ``min_blocks``: its launch bounds."""
    from fastmace.kernels.oeq_streaming.fused_forward import _lit, _node, block_map

    if schedule.layout != "ir_mul":
        raise ValueError("spline_forward writes the block layout: it needs ir_mul")
    paths, C = schedule.paths, schedule.channels
    if any(p.edge_mul != 1 for p in paths):
        raise ValueError("spline_forward: one weight per path and channel (edge_mul 1)")
    threads = THREADS if threads is None else int(threads)
    own = set(range(len(paths)) if part is None else part)
    mine = [p for g, p in enumerate(paths) if g in own]
    scale = [1.0] * len(paths) if path_scale is None else [float(v) for v in path_scale]
    bmap = block_map(blocks, C)
    slots = {s.name: s for s in schedule.output_slots}
    names = list(dict.fromkeys(o.accumulator.name for p in mine for o in p.outputs))
    si = {n: i for i, n in enumerate(names)}
    xkeys = sorted({(p.input_start, t.input_component, p.input_irrep_dim) for p in mine for o in p.outputs
                    for t in o.terms})
    xi = {key: i for i, key in enumerate(xkeys)}
    body = [f"            const float x{i} = __ldg(x + (int64_t)snd * IN_DIM + {_node('ir_mul', C, s, c, d)});"
            for (s, c, d), i in xi.items()]
    for g, p in enumerate(paths):
        if g not in own:
            continue
        body.append("            {")
        for j in range(p.edge_irrep_dim):
            body.append(f"                const float e{j} = __ldg(sh + e * ED + {p.edge_start + j});")
        body.append(f"                const float4 c = __ldg(tq + {p.weight_start});")
        body.append("                const float w = fmaf(t, fmaf(t, fmaf(t, c.w, c.z), c.y), c.x);")
        for o in p.outputs:
            terms = " + ".join(f"x{xi[(p.input_start, t.input_component, p.input_irrep_dim)]} * e{t.edge_component}"
                               f" * {_lit(t.coefficient / scale[g])}" for t in o.terms) or "0.f"
            body.append(f"                acc[{si[o.accumulator.name]}] = fmaf(w, {terms}, acc[{si[o.accumulator.name]}]);")
        body.append("            }")
    stores = []
    for slot_name, i in si.items():
        slot = slots[slot_name]
        group, width, base, stride = bmap[slot.start]
        stores.append(f"        out{group}[(int64_t)rcv * {width} + {base + slot.component * stride} + ch] = acc[{i}];")
    outs = ", ".join(f"float* __restrict__ out{g}" for g in range(len(blocks)))
    n = len(names)
    nl = "\n"
    bounds = f"{threads}" if min_blocks is None else f"{threads}, {int(min_blocks)}"
    return f"""// {name} (SPLINE_W, sx-layout lab 7 November 2026): Symmetrix-XL's layout, w from the cubic spline.
using int64_t = long long;
constexpr int C = {C};
constexpr int64_t Q = {len(paths) * C};
constexpr int IN_DIM = {schedule.input_dim};
constexpr int ED = {schedule.edge_dim};
constexpr float R_MAX = {_lit(r_max)};

extern "C" __global__ void __launch_bounds__({bounds}) {name}(
    int64_t N, int64_t E, const float* __restrict__ x, const float* __restrict__ sh,
    const int* __restrict__ senders, const int* __restrict__ row_ptr, const float* __restrict__ lengths,
    const int* __restrict__ pairs, const float4* __restrict__ table, int intervals, {outs}) {{
    const int64_t owners = N * C;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t owner = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; owner < owners; owner += stride) {{
        const int rcv = (int)(owner / C), ch = (int)(owner % C);
        float acc[{n}];
        #pragma unroll
        for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
        int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
        eb = eb < 0 ? 0 : eb;
        ee = ee > E ? E : ee;
        for (int64_t e = eb; e < ee; ++e) {{
            const float re = lengths[e];
            if (!(re < R_MAX)) continue;  // w = 0 at and beyond the cutoff (and the padding edges)
            const int snd = senders[e];
            if (snd < 0 || snd >= N) continue;
{lookup_code(grid, "re", "float", "            ")}
            const float4* tq = table + ((int64_t)pairs[e] * intervals + iv) * Q + ch;
{nl.join(body)}
        }}
{nl.join(stores)}
    }}
}}
"""


class SplineForward:
    """The compiled ``spline_forward`` of one interaction (its schedule, block layout and spline grid)."""

    def __init__(self, schedule, blocks, grid, r_max, device, path_scale=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        device = torch.device(device)
        self.blocks, self.channels, self.grid = blocks, schedule.channels, [tuple(g) for g in grid]
        self.intervals = sum(int(g[2]) for g in self.grid)
        self.threads, self.min_blocks = (128, 6) if r128_default(device) else (int(THREADS), MIN_BLOCKS)
        self.parts = (spline_parts(schedule) if PARTS else None) or [None]
        self.sources = [render(schedule, blocks, self.grid, r_max, path_scale, part, self.threads, self.min_blocks,
                               f"spline_forward{k}") for k, part in enumerate(self.parts)]
        self.source = "\n".join(self.sources)
        capability = torch.cuda.get_device_capability(device)
        attr = driver.CUfunction_attribute
        self._modules, self._functions, self.registers, self.local_bytes = [], [], [], []
        for k, source in enumerate(self.sources):
            module = _check(driver.cuModuleLoadData(compile_cubin(source, capability, ())))
            function = _check(driver.cuModuleGetFunction(module, f"spline_forward{k}".encode()))
            self._modules.append(module)
            self._functions.append(function)
            self.registers.append(int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))))
            self.local_bytes.append(int(_check(driver.cuFuncGetAttribute(
                attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function))))
        self.id = content_id("SplineForward", self.source, _device_key(device))
        _REGISTRY[self.id] = self

    def __call__(self, x, sh, senders, row_ptr, lengths, pairs, table, outs):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (nodes * self.channels + self.threads - 1) // self.threads
        if not blocks:
            return
        p = ctypes.c_void_p
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), senders.data_ptr(), row_ptr.data_ptr(),
                  lengths.data_ptr(), pairs.data_ptr(), table.data_ptr(), self.intervals, *[o.data_ptr() for o in outs])
        types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * 7 + (ctypes.c_int,) + (p,) * len(outs)
        stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        for function in self._functions:  # PARTS: each part writes its own output slots
            _check(driver.cuLaunchKernel(function, blocks, 1, 1, self.threads, 1, 1, 0, stream, (values, types), 0))


@torch.library.custom_op("fastmace::spline_message_blocks", mutates_args=())
def spline_message_blocks(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor, row_ptr: torch.Tensor,
                          lengths: torch.Tensor, pairs: torch.Tensor, table: torch.Tensor,
                          spline_id: int) -> list[torch.Tensor]:
    """The message groups ``[N * dim, P * C]`` (block layout) of a ``SplineForward``; edges receiver-sorted
    (``row_ptr``), ``lengths`` ``[E]``, ``pairs`` ``[E]`` int32, ``table`` the q-major spline table."""
    from fastmace.kernels.oeq_streaming.fused_forward import block_shapes

    kernel = _REGISTRY[spline_id]
    outs = [x.new_empty(shape) for shape in block_shapes(kernel.blocks, kernel.channels, x.shape[0])]
    kernel(x.contiguous(), sh.contiguous(), senders, row_ptr, lengths.reshape(-1).contiguous(), pairs, table, outs)
    return outs


@spline_message_blocks.register_fake
def _(x, sh, senders, row_ptr, lengths, pairs, table, spline_id):
    from fastmace.kernels.oeq_streaming.fused_forward import block_shapes

    kernel = _REGISTRY[spline_id]
    return [x.new_empty(shape) for shape in block_shapes(kernel.blocks, kernel.channels, x.shape[0])]


__all__ = ["MIN_BLOCKS", "PARTS", "R128", "R128_GPUS", "SPLINE_W", "SPLINE_W_GPUS", "SplineForward", "THREADS",
           "lookup_code", "r128_default", "render", "spline_message_blocks", "spline_parts", "spline_w_default"]
