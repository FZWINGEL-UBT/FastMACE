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
# A100 family (8 November 2026, results/a100-omat0-best-20261108, NHR Alex A100-SXM4-80GB; OMAT-0 medium,
# Benchmark_loup, ab / ba): MD step b/a 0.8654 / 0.8661 against the A100 defaults; all ten H200 switches together
# 0.7471 / 0.7469 (9.85 -> 7.36 ms), force/stress gate passes.
# NVIDIA RTX 4000 Ada Generation (8 November 2026, results/omat0-ada-capacity-20261108, leto; OMAT-0 medium, Benchmark_loup, ab / ba): with
# wprime_backward.HORNER, which the spline's w' needs (WPrime.attach_spline), 0.8396 / 0.8304 against the Ada defaults;
# SPLINE_W and HORNER off again from the Ada set 1.2003 / 1.2015 (b = off); gate passes.
SPLINE_W_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX 4000 Ada Generation"))
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
# (with STAGE_EDGES, H200, 8 November 2026: the same shape with MIN_BLOCKS 7 1.0097 / 1.0103, 8 1.0515 / 1.0507; 6 stays)
# THREADS and MIN_BLOCKS where on. None: on for R128_GPUS (decision 66); True / False: an A/B side's choice.
R128 = None
# NVIDIA H200 (7 November 2026, results/spline-w-h200-20261107): MD step b/a 0.9838 / 0.9848 on top of SPLINE_W
# (80 registers, 8 bytes local memory).
# A100 family (8 November 2026, results/a100-omat0-best-20261108, NHR Alex A100-SXM4-80GB; OMAT-0 medium,
# Benchmark_loup, ab / ba): on top of SPLINE_W 0.981 / 0.978; off again from all ten (with STAGE_EDGES) 1.0325 /
# 1.0311 (b = off).
# NVIDIA RTX 4000 Ada Generation (8 November 2026, results/omat0-ada-capacity-20261108, leto): on top of SPLINE_W 0.8328 / 0.8312 against the
# Ada defaults (SPLINE_W alone 0.8396 / 0.8304); off again from the Ada set, with STAGE_EDGES, 1.0017 / 1.0004.
R128_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX 4000 Ada Generation"))
# Codex H200 follow-up, item 4 (7 November 2026). FACTOR_W: a path whose terms all share one input component x_k
# (scalar-input paths) or one edge component e_j (scalar-edge paths) multiplies w by it once (wf = w * x_k) and
# accumulates fmaf(wf, rest, acc): fewer multiplies, rounding-class (the products associate differently; gate).
# CURSOR32: the owner and edge-cursor loop state in int (the pointer offsets widened where they multiply), where
# N * C and E fit (checked at launch); bitwise. Both A/B switches: False today; None would mean the GPU lists.
FACTOR_W = False
FACTOR_W_GPUS = GPUModels(())
CURSOR32 = False
CURSOR32_GPUS = GPUModels(())


def _switch(value, gpus, device):
    if value is not None:
        return bool(value)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in gpus


def factor_w_default(device):
    """``FACTOR_W`` on ``device`` (None: on for ``FACTOR_W_GPUS``)."""
    return _switch(FACTOR_W, FACTOR_W_GPUS, device)


def cursor32_default(device):
    """``CURSOR32`` on ``device`` (None: on for ``CURSOR32_GPUS``)."""
    return _switch(CURSOR32, CURSOR32_GPUS, device)


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


# STAGE_EDGES (Codex H200 report item 2, 8 November 2026; bitwise): where one block owns one receiver with one thread
# per channel (R128 with C = 128), the block walks the receiver's edges in tiles of 32: all threads copy the tile's
# edge features into shared memory (coalesced), warp 0 reads the lengths, senders and pairs of the 32 edges at once,
# runs the lookup and compacts the live edges in order (ballot), and the channel loop then reads sender, t and slab
# offset from shared memory: no dependent per-edge load chain and no branch in it. Same edges in the same order, same
# arithmetic. None: on for STAGE_EDGES_GPUS; True / False: an A/B side's choice.
STAGE_EDGES = None
# NVIDIA H200 (8 November 2026, results/h200-fastpath-20261108): MD step b/a 0.9864 / 0.9856 against the H200 defaults with COMPOSE_FAST,
# CUEQ_PROJECTED and GENERIC_FOLD_READOUT_TAIL (ab / ba); forces and energy within the build's spread (decision 66).
# A100 family (8 November 2026, results/a100-omat0-best-20261108, NHR Alex A100-SXM4-80GB; OMAT-0 medium,
# Benchmark_loup, ab / ba): off again from all ten H200 switches 1.0050 / 1.0044 (b = off).
# NVIDIA RTX 4000 Ada Generation (8 November 2026, results/omat0-ada-capacity-20261108, leto): off again from the Ada set 1.0018 / 1.0028.
STAGE_EDGES_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX 4000 Ada Generation"))
STAGE_TILE = 32
# PREFETCH_AHEAD (Codex H200 round-6 report item 2, 8 November 2026; bitwise): with STAGE_EDGES the channel loop hints
# the table rows of the edge PREFETCH_AHEAD positions later (its first PREFETCH_PATHS paths) into L1 with
# prefetch.global.L1, from the slab offsets the tile already holds in shared memory: no extra lookup, no register
# held across edges. ptxas sinks a hint for the next edge to the end of the current one, so the distance is 2. The
# values and their arithmetic are unchanged. None: PREFETCH_AHEAD_DISTANCE on PREFETCH_AHEAD_GPUS, else 0 (off);
# an int: an A/B side's distance (0 off).
PREFETCH_AHEAD = None
# NVIDIA H200 (8 November 2026, round 6): MD step b/a 1.0046 / 1.0067 (b = 2): off.
PREFETCH_AHEAD_GPUS = GPUModels(())
PREFETCH_AHEAD_DISTANCE = 2
PREFETCH_PATHS = 4


def prefetch_ahead_default(device):
    """``PREFETCH_AHEAD`` on ``device``: the constant when set, else ``PREFETCH_AHEAD_DISTANCE`` on
    ``PREFETCH_AHEAD_GPUS`` and 0."""
    if PREFETCH_AHEAD is not None:
        return int(PREFETCH_AHEAD)
    device = torch.device(device)
    listed = device.type == "cuda" and torch.cuda.get_device_name(device) in PREFETCH_AHEAD_GPUS
    return int(PREFETCH_AHEAD_DISTANCE) if listed else 0


def stage_edges_default(device):
    """``STAGE_EDGES`` on ``device``: the constant when set, else on for ``STAGE_EDGES_GPUS``."""
    if STAGE_EDGES is not None:
        return bool(STAGE_EDGES)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in STAGE_EDGES_GPUS


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


def _factor(p, xi):
    """``FACTOR_W``'s common factor of path ``p``: ("x", k) when every term reads input value x_k, ("e", j) when
    every term reads edge component e_j (the input key first), else None."""
    terms = [t for o in p.outputs for t in o.terms]
    if not terms:
        return None
    xs = {xi[(p.input_start, t.input_component, p.input_irrep_dim)] for t in terms}
    if len(xs) == 1:
        return "x", xs.pop()
    es = {t.edge_component for t in terms}
    if len(es) == 1:
        return "e", es.pop()
    return None


def render(schedule, blocks, grid, r_max, path_scale=None, part=None, threads=None, min_blocks=None,
           name="spline_forward", factor_w=False, cursor32=False, shared_locator=False, stage_edges=False,
           prefetch=0, block_order=False):
    """CUDA source of ``spline_forward`` for a schedule (ir_mul), its block layout ``blocks`` and the spline's
    ``grid``; ``path_scale``: per path the factor the table's rows carry (FOLD_ALPHA), divided out here. ``part``:
    the path indices this kernel computes (``PARTS``; None: all); ``threads`` / ``min_blocks``: its launch bounds.
    ``factor_w``: ``FACTOR_W``; ``cursor32``: ``CURSOR32`` (the launch checks the ranges). ``shared_locator``
    (``SHARED_LOCATOR``): the edge's slab and t read from ``spline_locator``'s record. ``stage_edges``
    (``STAGE_EDGES``): the edges staged per tile, for a launch of one block per receiver with ``threads`` = C.
    ``prefetch`` (``PREFETCH_AHEAD``; with ``stage_edges``): hint the table rows of the edge that many positions on.
    ``block_order`` (``BLOCK_ORDER``): the receiver of owner row k is ``order[k]`` (an int32 permutation argument)."""
    if stage_edges and shared_locator:
        raise ValueError("STAGE_EDGES and SHARED_LOCATOR are alternatives")
    if prefetch and not stage_edges:
        raise ValueError("PREFETCH_AHEAD reads STAGE_EDGES' staged slab offsets: it needs stage_edges")
    source = _render(schedule, blocks, grid, r_max, path_scale, part, threads, min_blocks, name, factor_w, cursor32,
                     stage_edges, int(prefetch))
    source = located_source(source, grid) if shared_locator else source
    return ordered_source(source) if block_order else source


# BLOCK_ORDER (8 November 2026): the owner rows walk the receivers in a given order (an int32 permutation of [0, N),
# argument ``order`` after ``intervals``), so the blocks resident at one time hold receivers of one species and read
# the same spline-table slabs (results/omat0-breakthroughs-assessment-20261108: modelled L1 hits 0.54 -> 0.70 with
# SORT_BY_LENGTH). The same arithmetic per receiver: bitwise. None: on for BLOCK_ORDER_GPUS (decision 66).
BLOCK_ORDER = None
# NVIDIA H200 (8 November 2026, results/slab-sorted-h200-20261108; OMAT-0 medium, Benchmark_loup, Modal `sixth`, on
# top of SORT_BY_LENGTH): MD step b/a 0.9753 / 0.9724 (3.514 -> 3.42 ms); forces 2.6e-6 against the same build's
# 3.3e-6 eV/A (bitwise forward; w' atomics).
BLOCK_ORDER_GPUS = GPUModels(("NVIDIA H200",))


def block_order_default(device):
    """``BLOCK_ORDER`` on ``device`` (None: on for ``BLOCK_ORDER_GPUS``)."""
    return _switch(BLOCK_ORDER, BLOCK_ORDER_GPUS, device)


_ORDER_SWAPS = (
    ("const float4* __restrict__ table, int intervals, ",
     "const float4* __restrict__ table, int intervals, const int* __restrict__ order, "),
)
_ORDER_RECEIVER = (  # one of these per source: staged, cursor (int64), cursor (CURSOR32)
    ("const int rcv = blockIdx.x, ch = threadIdx.x;\n    if (rcv >= N) return;  // the whole block",
     "if (blockIdx.x >= N) return;  // the whole block\n    const int rcv = order[blockIdx.x], ch = threadIdx.x;"),
    ("const int rcv = (int)(owner / C), ch = (int)(owner % C);",
     "const int rcv = order[owner / C], ch = (int)(owner % C);"),
    ("const int rcv = owner / C, ch = owner % C;", "const int rcv = order[owner / C], ch = owner % C;"),
)


def ordered_source(source):
    """``spline_forward``'s source with BLOCK_ORDER: the ``order`` argument and the owner's receiver read from it."""
    for old, new in _ORDER_SWAPS:
        if old not in source:
            raise ValueError(f"BLOCK_ORDER: the forward source no longer holds {old.strip()[:60]!r}")
        source = source.replace(old, new)
    found = [(old, new) for old, new in _ORDER_RECEIVER if old in source]
    if len(found) != 1:
        raise ValueError(f"BLOCK_ORDER: expected one receiver line, found {len(found)}")
    old, new = found[0]
    return source.replace(old, new)


def receiver_order(species):
    """BLOCK_ORDER's permutation: the receivers grouped by species (stable), int32."""
    return torch.argsort(species.to(torch.int64), stable=True).to(torch.int32)


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
            name="spline_forward", factor_w=False, cursor32=False, stage_edges=False, prefetch=0):
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
            if stage_edges:  # STAGE_EDGES: the tile's copy in shared memory
                body.append(f"                const float e{j} = she[{p.edge_start + j}];")
                continue
            edge = "(int64_t)e" if cursor32 else "e"  # CURSOR32: the offset widened before it multiplies
            body.append(f"                const float e{j} = __ldg(sh + {edge} * ED + {p.edge_start + j});")
        body.append(f"                const float4 c = __ldg(tq + {p.weight_start});")
        body.append("                const float w = fmaf(t, fmaf(t, fmaf(t, c.w, c.z), c.y), c.x);")
        common = _factor(p, xi) if factor_w else None
        if common is not None:
            body.append(f"                const float wf = w * {common[0]}{common[1]};  // FACTOR_W")
        for o in p.outputs:
            def factors(t):
                x = f"x{xi[(p.input_start, t.input_component, p.input_irrep_dim)]}"
                e = f"e{t.edge_component}"
                if common is None:
                    return f"{x} * {e} * "
                return f"{e} * " if common[0] == "x" else f"{x} * "
            terms = " + ".join(f"{factors(t)}{_lit(t.coefficient / scale[g])}" for t in o.terms) or "0.f"
            factor = "w" if common is None else "wf"
            body.append(f"                acc[{si[o.accumulator.name]}] = fmaf({factor}, {terms}, "
                        f"acc[{si[o.accumulator.name]}]);")
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
    if cursor32:  # CURSOR32: the loop state in int; N * C and E < 2^31 (the launch checks)
        cursor = """    const int owners = (int)(N * C);
    const int stride = blockDim.x * gridDim.x;
    for (int owner = blockIdx.x * blockDim.x + threadIdx.x; owner < owners; owner += stride) {
        const int rcv = owner / C, ch = owner % C;"""
        edges = """        int eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
        eb = eb < 0 ? 0 : eb;
        ee = ee > (int)E ? (int)E : ee;
        for (int e = eb; e < ee; ++e) {"""
    else:
        cursor = """    const int64_t owners = N * C;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t owner = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; owner < owners; owner += stride) {
        const int rcv = (int)(owner / C), ch = (int)(owner % C);"""
        edges = """        int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
        eb = eb < 0 ? 0 : eb;
        ee = ee > E ? E : ee;
        for (int64_t e = eb; e < ee; ++e) {"""
    if stage_edges:
        if threads != C:
            raise ValueError(f"STAGE_EDGES needs one thread per channel ({threads} threads, C = {C})")
        hints = [p.weight_start for p in mine][:PREFETCH_PATHS] if prefetch else []
        return _staged(name, bounds, C, len(paths) * C, schedule, r_max, outs, n, grid, body, stores, prefetch, hints)
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
{cursor}
        float acc[{n}];
        #pragma unroll
        for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
{edges}
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


def _staged(name, bounds, C, q, schedule, r_max, outs, n, grid, body, stores, prefetch=0, hints=()):
    """``spline_forward``'s source with STAGE_EDGES: one block per receiver, thread = channel; ``prefetch`` > 0
    (PREFETCH_AHEAD): the rows at the offsets ``hints`` of the edge ``prefetch`` positions on hinted into L1."""
    from fastmace.kernels.oeq_streaming.fused_forward import _lit

    nl = "\n"
    tile = STAGE_TILE
    hint = ""
    if prefetch:  # PREFETCH_AHEAD: no value is read, so nothing is held across edges
        rows = "".join(f'                asm volatile("prefetch.global.L1 [%0];" :: "l"(ahead + {h}));\n' for h in hints)
        hint = (f"            if (k + {prefetch} < count) {{  // PREFETCH_AHEAD\n"
                f"                const float4* ahead = table + s_tq[k + {prefetch}] + ch;\n{rows}            }}\n")
    return f"""// {name} (SPLINE_W + STAGE_EDGES, 8 November 2026): one receiver per block, its edges staged per tile.
using int64_t = long long;
constexpr int C = {C};
constexpr int64_t Q = {q};
constexpr int IN_DIM = {schedule.input_dim};
constexpr int ED = {schedule.edge_dim};
constexpr int TILE = {tile};
constexpr float R_MAX = {_lit(r_max)};

extern "C" __global__ void __launch_bounds__({bounds}) {name}(
    int64_t N, int64_t E, const float* __restrict__ x, const float* __restrict__ sh,
    const int* __restrict__ senders, const int* __restrict__ row_ptr, const float* __restrict__ lengths,
    const int* __restrict__ pairs, const float4* __restrict__ table, int intervals, {outs}) {{
    __shared__ int s_snd[TILE], s_k[TILE], s_count;
    __shared__ float s_t[TILE];
    __shared__ int64_t s_tq[TILE];
    __shared__ __align__(16) float s_sh[TILE * ED];
    const int rcv = blockIdx.x, ch = threadIdx.x;
    if (rcv >= N) return;  // the whole block
    float acc[{n}];
    #pragma unroll
    for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
    int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
    eb = eb < 0 ? 0 : eb;
    ee = ee > E ? E : ee;
    for (int64_t base = eb; base < ee; base += TILE) {{
        const int count_in = (int)(ee - base < TILE ? ee - base : TILE);
        for (int i = ch; i < count_in * ED; i += C) s_sh[i] = __ldg(sh + base * ED + i);
        if (ch < 32) {{  // warp 0: the tile's edges at once, the live ones compacted in order
            bool ok = false;
            int snd_k = 0;
            float t_k = 0.f;
            int64_t tq_k = 0;
            if (ch < count_in) {{
                const int64_t e = base + ch;
                const float re = lengths[e];
                if (re < R_MAX) {{  // w = 0 at and beyond the cutoff (and the padding edges)
                    snd_k = senders[e];
                    if (snd_k >= 0 && snd_k < N) {{
{lookup_code(grid, "re", "float", "                        ")}
                        t_k = t;
                        tq_k = ((int64_t)pairs[e] * intervals + iv) * Q;
                        ok = true;
                    }}
                }}
            }}
            const unsigned live = __ballot_sync(0xffffffffu, ok);
            if (ok) {{
                const int slot = __popc(live & ((1u << ch) - 1u));
                s_snd[slot] = snd_k;
                s_t[slot] = t_k;
                s_tq[slot] = tq_k;
                s_k[slot] = ch;
            }}
            if (ch == 0) s_count = __popc(live);
        }}
        __syncthreads();
        const int count = s_count;
        for (int k = 0; k < count; ++k) {{
            const int snd = s_snd[k];
            const float t = s_t[k];
            const float* she = s_sh + s_k[k] * ED;
            const float4* tq = table + s_tq[k] + ch;
{hint}{nl.join(body)}
        }}
        __syncthreads();
    }}
{nl.join(line[4:] for line in stores)}
}}
"""


class SplineForward:
    """The compiled ``spline_forward`` of one interaction (its schedule, block layout and spline grid);
    ``shared_locator`` (SHARED_LOCATOR): it reads ``spline_locator``'s records, compiled alongside (``locate``)."""

    def __init__(self, schedule, blocks, grid, r_max, device, path_scale=None, shared_locator=False,
                 block_order=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        device = torch.device(device)
        self.blocks, self.channels, self.grid = blocks, schedule.channels, [tuple(g) for g in grid]
        self.intervals = sum(int(g[2]) for g in self.grid)
        self.threads, self.min_blocks = (128, 6) if r128_default(device) else (int(THREADS), MIN_BLOCKS)
        self.parts = (spline_parts(schedule) if PARTS else None) or [None]
        self.factor_w, self.cursor32 = factor_w_default(device), cursor32_default(device)
        self.shared_locator = bool(shared_locator)
        # STAGE_EDGES: only where a block is one receiver (thread = channel) and nothing else changes the edge head
        self.stage_edges = (stage_edges_default(device) and self.threads == self.channels and not self.shared_locator
                            and not self.cursor32)
        self.prefetch = prefetch_ahead_default(device) if self.stage_edges else 0  # PREFETCH_AHEAD
        self.block_order = block_order_default(device) if block_order is None else bool(block_order)
        # BLOCK_ORDER: the receivers' order the owning force field binds before its calls (``bind_order``; any
        # permutation of [0, N) gives the same message); a call with another row count walks them in index order.
        self.order, self._identity = None, {}
        self.init_args = (schedule, blocks, grid, r_max, device, path_scale, shared_locator)
        self.sources = [render(schedule, blocks, self.grid, r_max, path_scale, part, self.threads, self.min_blocks,
                               f"spline_forward{k}", self.factor_w, self.cursor32, self.shared_locator,
                               self.stage_edges, self.prefetch, self.block_order)
                        for k, part in enumerate(self.parts)]
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

    def __call__(self, x, sh, senders, row_ptr, lengths, pairs, table, outs, order=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        nodes, edges = x.shape[0], sh.shape[0]
        if self.cursor32 and (nodes * self.channels >= 2 ** 31 or edges >= 2 ** 31):
            raise RuntimeError(f"CURSOR32: {nodes} x {self.channels} owners or {edges} edges exceed int")
        blocks = (nodes * self.channels + self.threads - 1) // self.threads
        if not blocks:
            return
        p = ctypes.c_void_p
        ordered = ()
        if self.block_order:  # BLOCK_ORDER: the receivers' order, a permutation of [0, N)
            order = self.order if order is None else order
            if order is None or order.numel() != nodes:
                order = self._identity.get(nodes)
                if order is None:
                    order = self._identity[nodes] = torch.arange(nodes, dtype=torch.int32, device=x.device)
            if order.dtype != torch.int32 or order.device != x.device:
                raise ValueError("BLOCK_ORDER: the order must be int32 on the kernel's device")
            ordered = (order.data_ptr(),)
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), senders.data_ptr(), row_ptr.data_ptr(),
                  lengths.data_ptr(), pairs.data_ptr(), table.data_ptr(), self.intervals, *ordered,
                  *[o.data_ptr() for o in outs])
        types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * 7 + (ctypes.c_int,) + (p,) * (len(ordered) + len(outs))
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


__all__ = ["PREFETCH_AHEAD", "PREFETCH_AHEAD_DISTANCE", "PREFETCH_AHEAD_GPUS", "PREFETCH_PATHS",
           "prefetch_ahead_default", "STAGE_EDGES", "STAGE_EDGES_GPUS", "CURSOR32", "CURSOR32_GPUS", "FACTOR_W", "FACTOR_W_GPUS", "MIN_BLOCKS", "PARTS", "R128", "R128_GPUS",
           "SHARED_LOCATOR", "SHARED_LOCATOR_GPUS", "SPLINE_W", "SPLINE_W_GPUS", "SplineForward", "THREADS",
           "cursor32_default", "factor_w_default", "located_source", "locator_reference", "lookup_code", "r128_default",
           "render", "render_locator", "shared_locator_default", "shared_locator_supported", "spline_locate",
           "spline_message_blocks", "spline_message_blocks_located", "spline_parts", "spline_w_default"]
