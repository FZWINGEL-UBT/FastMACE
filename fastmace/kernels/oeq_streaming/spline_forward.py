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
# SHARED_LOCATOR (Codex H200 report, item 5): one producer kernel (spline_locator, NVRTC, the same lookup_code text)
# writes per edge a 16-byte record int4 (slab = pair * intervals + iv, t bits, 1 / h bits, r < r_max); the forward
# and the w' kernel read it instead of locating r themselves. Bitwise: the same IEEE operations on the same float r
# (see ``shared_locator_supported`` for the w' kernel's clamp at r_lo). None: on for SHARED_LOCATOR_GPUS;
# True / False: an A/B side's choice. Not measured.
SHARED_LOCATOR = None
SHARED_LOCATOR_GPUS = GPUModels(())


def shared_locator_default(device):
    """``SHARED_LOCATOR`` on ``device``: the constant when set, else on for ``SHARED_LOCATOR_GPUS``."""
    if SHARED_LOCATOR is not None:
        return bool(SHARED_LOCATOR)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in SHARED_LOCATOR_GPUS


def shared_locator_supported(grid, r_lo):
    """Whether the w' kernel's own locator equals the record's for every r: it locates max(r, r_lo) in float32, the
    record r. ``lookup_code`` maps every r below the grid's first start a to t = 0 in interval 0, as it maps a (u is
    clamped at 0), so the two agree for every r (NaN included: no weight, no slope there) when float32 r_lo <= a."""
    import struct

    def f32(v):
        return struct.unpack("f", struct.pack("f", float(v)))[0]

    return f32(r_lo) <= f32(grid[0][0])


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
           name="spline_forward", shared_locator=False):
    """CUDA source of ``spline_forward`` for a schedule (ir_mul), its block layout ``blocks`` and the spline's
    ``grid``; ``path_scale``: per path the factor the table's rows carry (FOLD_ALPHA), divided out here. ``part``:
    the path indices this kernel computes (``PARTS``; None: all); ``threads`` / ``min_blocks``: its launch bounds.
    ``shared_locator`` (``SHARED_LOCATOR``): the edge's slab and t read from ``spline_locator``'s record."""
    source = _render(schedule, blocks, grid, r_max, path_scale, part, threads, min_blocks, name)
    return located_source(source, grid) if shared_locator else source


# SHARED_LOCATOR in the forward: the lines of _render's edge head it replaces, and their replacements. The record
# replaces the lengths argument (same position, so SplineForward's launch is unchanged); pairs and intervals stay in
# the signature, unread.
_LOCATED_EDGE = "            const int4 loc = __ldg(locator + e);  // SHARED_LOCATOR: spline_locator's record\n" \
                "            if (!loc.w) continue;  // w = 0 at and beyond the cutoff (and the padding edges)\n"


def located_source(source, grid):
    """``spline_forward``'s source with the per-edge lookup replaced by ``spline_locator``'s record (SHARED_LOCATOR);
    raises where the source does not hold each replaced line exactly once."""
    lookup = lookup_code(grid, "re", "float", "            ") + "\n"
    swaps = [
        ("const float* __restrict__ lengths,", "const int4* __restrict__ locator,"),
        ("            const float re = lengths[e];\n"
         "            if (!(re < R_MAX)) continue;  // w = 0 at and beyond the cutoff (and the padding edges)\n",
         _LOCATED_EDGE),
        (lookup, "            const float t = __int_as_float(loc.y);\n"),
        ("table + ((int64_t)pairs[e] * intervals + iv) * Q + ch;", "table + (int64_t)loc.x * Q + ch;"),
    ]
    for old, new in swaps:
        if source.count(old) != 1:
            raise ValueError(f"SHARED_LOCATOR: the forward source no longer holds {old.strip()[:60]!r} once")
        source = source.replace(old, new)
    return source


def render_locator(grid, r_max, name="spline_locator", threads=256):
    """CUDA source of ``spline_locator`` (SHARED_LOCATOR): per edge ``e`` the record ``int4(pair * intervals + iv,
    t bits, 1 / h bits, r < r_max)`` from ``lookup_code``, the same text and operations both spline kernels ran."""
    from fastmace.kernels.oeq_streaming.fused_forward import _lit

    return f"""// {name} (SHARED_LOCATOR): the spline locator of every edge, once, for spline_forward and the w' kernel.
using int64_t = long long;
constexpr float R_MAX = {_lit(r_max)};

extern "C" __global__ void __launch_bounds__({int(threads)}) {name}(
    int64_t E, const float* __restrict__ lengths, const int* __restrict__ pairs, int intervals,
    int4* __restrict__ locator) {{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {{
        const float re = lengths[e];
{lookup_code(grid, "re", "float", "        ")}
        locator[e] = make_int4((int)((int64_t)pairs[e] * intervals + iv), __float_as_int(t), __float_as_int(sx_inv),
                               re < R_MAX ? 1 : 0);
    }}
}}
"""


def locator_reference(r, pair, grid, r_max):
    """CPU mirror of ``spline_locator`` (float32, the kernel's operations in its order): ``int32 [E, 4]``."""
    import struct

    def f32(v):
        return struct.unpack("f", struct.pack("f", float(v)))[0]

    def bits(v):
        return struct.unpack("i", struct.pack("f", v))[0]

    def mul(a, b):
        return f32(a * b)  # one rounding: a float32 product is exact in float64

    intervals = sum(int(n) for _, _, n in grid)
    rows = []
    for re, p in zip(r, pair):
        re = f32(re)
        offset = 0
        for k, (a, b, n) in enumerate(grid):
            if k < len(grid) - 1 and not re < f32(b):
                offset += n
                continue
            sx_inv = f32(n / (b - a))
            u = mul(f32(re - f32(a)), sx_inv)
            u = u if u > 0 else 0.0
            il = n - 1 if u >= n - 1 else int(u)  # the device's (int) saturates; then the clamp to n - 1
            t = f32(u - il)
            iv = offset + il
            break
        rows.append([int(p) * intervals + iv, bits(t), bits(sx_inv), 1 if re < f32(r_max) else 0])
    return torch.tensor(rows, dtype=torch.int32).reshape(-1, 4)


def _render(schedule, blocks, grid, r_max, path_scale=None, part=None, threads=None, min_blocks=None,
            name="spline_forward"):
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
    """The compiled ``spline_forward`` of one interaction (its schedule, block layout and spline grid);
    ``shared_locator`` (SHARED_LOCATOR): it reads ``spline_locator``'s records, compiled alongside (``locate``)."""

    def __init__(self, schedule, blocks, grid, r_max, device, path_scale=None, shared_locator=False):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        device = torch.device(device)
        self.blocks, self.channels, self.grid = blocks, schedule.channels, [tuple(g) for g in grid]
        self.intervals = sum(int(g[2]) for g in self.grid)
        self.threads, self.min_blocks = (128, 6) if r128_default(device) else (int(THREADS), MIN_BLOCKS)
        self.parts = (spline_parts(schedule) if PARTS else None) or [None]
        self.shared_locator = bool(shared_locator)
        self.sources = [render(schedule, blocks, self.grid, r_max, path_scale, part, self.threads, self.min_blocks,
                               f"spline_forward{k}", self.shared_locator) for k, part in enumerate(self.parts)]
        if self.shared_locator:  # the producer's source is part of the content id
            self.sources.append(render_locator(self.grid, r_max))
        self.source = "\n".join(self.sources)
        capability = torch.cuda.get_device_capability(device)
        attr = driver.CUfunction_attribute
        self._modules, self._functions, self.registers, self.local_bytes = [], [], [], []
        names = [f"spline_forward{k}" for k in range(len(self.parts))] + ["spline_locator"] * self.shared_locator
        for name, source in zip(names, self.sources):
            module = _check(driver.cuModuleLoadData(compile_cubin(source, capability, ())))
            function = _check(driver.cuModuleGetFunction(module, name.encode()))
            self._modules.append(module)
            self._functions.append(function)
            self.registers.append(int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))))
            self.local_bytes.append(int(_check(driver.cuFuncGetAttribute(
                attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function))))
        self._locator = self._functions.pop() if self.shared_locator else None  # registers / local stay listed last
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

    def locate(self, lengths, pairs):
        """SHARED_LOCATOR: ``spline_locator``'s records ``int32 [E, 4]`` of ``lengths`` ``[E]`` and ``pairs``."""
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        edges = lengths.shape[0]
        locator = torch.empty((edges, 4), dtype=torch.int32, device=lengths.device)
        if edges:
            p = ctypes.c_void_p
            values = (edges, lengths.data_ptr(), pairs.data_ptr(), self.intervals, locator.data_ptr())
            types = (ctypes.c_longlong, p, p, ctypes.c_int, p)
            stream = driver.CUstream(torch.cuda.current_stream(lengths.device).cuda_stream)
            _check(driver.cuLaunchKernel(self._locator, (edges + 255) // 256, 1, 1, 256, 1, 1, 0, stream,
                                         (values, types), 0))
        return locator


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


@torch.library.custom_op("fastmace::spline_locate", mutates_args=())
def spline_locate(lengths: torch.Tensor, pairs: torch.Tensor, spline_id: int) -> torch.Tensor:
    """SHARED_LOCATOR: the per-edge records ``int32 [E, 4]`` (slab, t bits, 1 / h bits, r < r_max) of a
    ``SplineForward`` built with ``shared_locator``."""
    return _REGISTRY[spline_id].locate(lengths.reshape(-1).contiguous(), pairs.contiguous())


@spline_locate.register_fake
def _(lengths, pairs, spline_id):
    return lengths.new_empty((lengths.numel(), 4), dtype=torch.int32)


@torch.library.custom_op("fastmace::spline_message_blocks_located", mutates_args=())
def spline_message_blocks_located(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor, row_ptr: torch.Tensor,
                                  locator: torch.Tensor, pairs: torch.Tensor, table: torch.Tensor,
                                  spline_id: int) -> list[torch.Tensor]:
    """``spline_message_blocks`` of a ``SHARED_LOCATOR`` kernel: ``locator`` (``spline_locate``'s records) in place
    of the lengths."""
    from fastmace.kernels.oeq_streaming.fused_forward import block_shapes

    kernel = _REGISTRY[spline_id]
    outs = [x.new_empty(shape) for shape in block_shapes(kernel.blocks, kernel.channels, x.shape[0])]
    kernel(x.contiguous(), sh.contiguous(), senders, row_ptr, locator.contiguous(), pairs, table, outs)
    return outs


@spline_message_blocks_located.register_fake
def _(x, sh, senders, row_ptr, locator, pairs, table, spline_id):
    from fastmace.kernels.oeq_streaming.fused_forward import block_shapes

    kernel = _REGISTRY[spline_id]
    return [x.new_empty(shape) for shape in block_shapes(kernel.blocks, kernel.channels, x.shape[0])]


__all__ = ["MIN_BLOCKS", "PARTS", "R128", "R128_GPUS", "SHARED_LOCATOR", "SHARED_LOCATOR_GPUS", "SPLINE_W",
           "SPLINE_W_GPUS", "SplineForward", "THREADS", "located_source", "locator_reference", "lookup_code",
           "r128_default", "render", "render_locator", "shared_locator_default", "shared_locator_supported",
           "spline_locate",
           "spline_message_blocks", "spline_message_blocks_located", "spline_parts", "spline_w_default"]
