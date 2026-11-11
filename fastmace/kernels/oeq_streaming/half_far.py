"""HALF_FAR: the w' kernel's far edges in packed half precision (idea 1 of the A100 screen, 11 November 2026;
results/a100-colab-ideas-20261111, results/half-far-wprime-20261111).

Interaction edges longer than ``radius`` (and inside the cutoff) are taken out of the float32 w' kernel two at a time:
consecutive far edges of one receiver form a pair, whose channel loop runs once with every channel value an ``h2``
(two float16 in 32 bits, edge ``e0`` in the low half): HFMA2 on the second FMA pipe of sm_80, with half the
instructions per edge. An odd far edge of a receiver stays in the float32 kernel. The forward is unchanged.

Three pieces, generated from the float32 kernel's own source (so the paths, CG terms, their order, the dout blocks
and FOLD_ALPHA's constants are the float32 kernel's):

* ``wprime_half_pairs``: one thread per receiver run pairs the run's far edges (in edge order), appends ``(e0, e1)``
  to the pair list (an atomic counter) and writes every edge's skip flag (1 for paired edges);
* the float32 kernel skips flagged edges (``half_skip``, a source transform: ``with_skip``);
* ``wprime_half_far``: a warp per pair. x of the two senders and dout (one receiver, broadcast) are packed from
  float32 by one conversion each; the harmonics as h2 in shared memory; the spline table from a float16 copy (two
  8-byte loads and four byte permutes per path); w and dw/dt by the dual Horner pass in HFMA2; dx by two float32
  atomics per value; the 32 per-lane edge-gradient values (16 per edge) reduce-scattered in float32.

The rounding (each statement one float16 rounding, as the oracle ``half_wprime_oracle.py`` emulates) needs the
force/stress gate; the switch is ``wprime_backward.HALF_FAR`` with an empty ``HALF_FAR_GPUS``.
"""
from __future__ import annotations

import re

import numpy as np

PRELUDE = r"""
// HALF_FAR: packed half precision by PTX (no cuda_fp16.h: NVRTC finds no headers here). h2: edge e0 in the low half.
typedef unsigned int h2;
__device__ __forceinline__ h2 hfma(h2 a, h2 b, h2 c) {
    h2 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ h2 hfms(h2 a, h2 b, h2 c) {  // c - a b
    h2 d; asm("{.reg .b32 n; neg.f16x2 n, %1; fma.rn.f16x2 %0, n, %2, %3;}" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d; }
__device__ __forceinline__ h2 hmul(h2 a, h2 b) {
    h2 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ h2 pack(float lo, float hi) {
    h2 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ float lo2f(h2 a) {
    float f; asm("{.reg .f16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, l;}" : "=f"(f) : "r"(a)); return f; }
__device__ __forceinline__ float hi2f(h2 a) {
    float f; asm("{.reg .f16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, h;}" : "=f"(f) : "r"(a)); return f; }
"""

PAIRS = r"""
// HALF_FAR: one thread per receiver run (edges in order) pairs consecutive far edges; every edge's skip flag, and
// each receiver's flag (1: it has a pair; zeroed by the caller).
extern "C" __global__ void wprime_half_pairs(
    int64_t edge_count, int64_t node_count, const int32_t* __restrict__ senders,
    const int32_t* __restrict__ receivers, const float* __restrict__ lengths, float r_half, float r_max,
    unsigned char* __restrict__ skip, int2* __restrict__ pairs, int* __restrict__ count,
    unsigned char* __restrict__ paired) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= edge_count) return;
    const int32_t rc = receivers[e];
    if (e > 0 && receivers[e - 1] == rc) return;  // not the start of a run
    const bool valid_rc = rc >= 0 && rc < node_count;
    int64_t open = -1;
    for (int64_t k = e; k < edge_count && receivers[k] == rc; ++k) {
        const int32_t s = senders[k];
        const float r = lengths[k];
        unsigned char flag = 0;
        if (valid_rc && s >= 0 && s < node_count && r > r_half && r < r_max) {
            if (open < 0) {
                open = k;
            } else {
                const int slot = atomicAdd(count, 1);
                pairs[slot] = make_int2(int(open), int(k));
                paired[rc] = 1;
                skip[open] = 1;
                flag = 1;
                open = -1;
            }
        }
        if (k != open) skip[k] = flag;
        else skip[k] = 0;
    }
}
"""


REDUCE_SCATTER = r"""        // the 32 per-lane values (16 per edge) reduce-scattered in float32: lane l ends with value l
        float v[32];
        #pragma unroll
        for (int j = 0; j < EDGE_DIM; ++j) { v[j] = lo2f(edge_gradient[j]); v[EDGE_DIM + j] = hi2f(edge_gradient[j]); }
        #pragma unroll
        for (int step = 0; step < 5; ++step) {
            const int half = 16 >> step;
            const bool upper = (lane & half) != 0;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < half) {
                    const float send = upper ? v[i] : v[i + half];
                    const float keep = upper ? v[i + half] : v[i];
                    v[i] = keep + __shfl_xor_sync(FULL_MASK, send, half, 32);
                }
            }
        }
        dsh[(lane < EDGE_DIM ? e0 : e1) * EDGE_DIM + (lane & (EDGE_DIM - 1))] = v[0] * (lane < EDGE_DIM ? inv0 : inv1);"""
REDUCE_EACH = r"""        // fewer than 16 edge features: each value reduced over the warp in float32
        #pragma unroll
        for (int j = 0; j < EDGE_DIM; ++j) {
            float a = lo2f(edge_gradient[j]), b = hi2f(edge_gradient[j]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                a += __shfl_xor_sync(FULL_MASK, a, offset, 32);
                b += __shfl_xor_sync(FULL_MASK, b, offset, 32);
            }
            if (lane == 0) { dsh[e0 * EDGE_DIM + j] = a * inv0; dsh[e1 * EDGE_DIM + j] = b * inv1; }
        }"""


# ---------------------------------------------------------------------------------------------------------------
# the float32 kernel's channel loop, statement by statement

def channel_body(source):
    """The float32 kernel's channel loop (after the per-edge set-up, before the reduction)."""
    head = "    acc_t radial_gradient = acc_t(0);\n"
    a = source.index(head) + len(head)
    loop = source.index("for (int channel = lane;", a)
    depth, i = 0, source.index("{", loop)
    while True:  # the loop's closing brace
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            break
        i += 1
    return re.sub(r"//[^\n]*", "", source[a:i + 1])


def statements(body):
    """The loop as ``(kind, ...)`` tuples in the kernel's order: ``path`` (weight base), ``horner``, ``zero`` (name),
    ``let`` (name, atoms), ``acc`` (lhs, sign, a, b), ``slope``, ``radial``, ``atomic`` (x offset, name). Atoms:
    ``("x", k)``, ``("d", index)``, ``("sh", j)``, ``("c", value)``, ``("v", name)``, ``("eg", j)``."""
    def atom(e):
        e = re.sub(r"\s+", " ", e).strip()
        if m := re.fullmatch(r"x\[input_base \+ (\d+) \+ channel\]", e):
            return ("x", int(m.group(1)))
        if m := re.fullmatch(r"stage_row\[(\d+) \+ (\d+) \+ channel\]", e):
            return ("d", int(m.group(1)) + int(m.group(2)))
        if m := re.fullmatch(r"sh_edge\[(\d+)\]", e):
            return ("sh", int(m.group(1)))
        if m := re.fullmatch(r"static_cast<scalar_t>\(([-0-9a-fx.p+]+)f\)", e):
            return ("c", float.fromhex(m.group(1)))
        if m := re.fullmatch(r"edge_gradient\[(\d+)\]", e):
            return ("eg", int(m.group(1)))
        if re.fullmatch(r"\w+", e):
            return ("v", e)
        raise ValueError(f"HALF_FAR: unknown operand {e!r}")

    out = []
    for raw in re.split(r"[;{}]", body):
        t = re.sub(r"\s+", " ", raw).strip()
        if not t:
            continue
        if m := re.fullmatch(r"const int weight_index = (\d+) \+ channel", t):
            out.append(("path", int(m.group(1))))
        elif t == "scalar_t radial_weight = gv[TERMS - 1]":
            out.append(("horner",))
        elif m := re.fullmatch(r"acc_t (\w+) = acc_t\(0\)", t):
            if m.group(1) != "slope":
                out.append(("zero", m.group(1)))
        elif m := re.fullmatch(r"const scalar_t (\w+) = (.*)", t):
            out.append(("let", m.group(1), tuple(atom(p) for p in m.group(2).split(" * "))))
        elif t == "slope = radial_dt * sc":
            out.append(("slope",))
        elif t == "radial_gradient += weight_gradient * slope":
            out.append(("radial",))
        elif (m := re.fullmatch(r"(\S+) ([+-])= (.+)", t)) and " * " in m.group(3):
            a, b = m.group(3).split(" * ", 1)
            out.append(("acc", atom(m.group(1)), m.group(2), atom(a), atom(b)))
        elif m := re.fullmatch(r"atomicAdd\( dx \+ input_base \+ (\d+) \+ channel, scalar_t\((\w+)\)\)", t):
            out.append(("atomic", int(m.group(1)), m.group(2)))
        elif re.match(r"(const int64_t radial_weight_index|scalar_t gv|if \(inside\)|const float4|#pragma|k [<>]|"
                      r"\+\+k|--k|gv\[|else|radial_dt = radial_dt|radial_weight = radial_weight|scalar_t radial_dt|"
                      r"if \(sloped\)|for \(int channel|channel [<+])", t):
            continue
        else:
            raise ValueError(f"HALF_FAR: untranslated statement {t!r}")
    return out


def dout_blocks(source):
    """``[(block, start in the staged row, row stride)]`` of the float32 kernel's STAGE_DOUT copies."""
    found = re.findall(r"dout(\d+) \+ int64_t\(stage_r\) \* (\d+);.*?stage_row \+ (\d+)\)", source, flags=re.S)
    blocks = sorted({(int(k), int(start), int(stride)) for k, stride, start in found}, key=lambda b: b[1])
    if not blocks:
        raise ValueError("HALF_FAR: no STAGE_DOUT blocks in the w' source")
    return blocks


def half_bits(value):
    bits = int(np.array([value], dtype=np.float16).view(np.uint16)[0])
    return f"0x{bits:04x}{bits:04x}u"


def half_body(stmts, blocks):
    """C of the h2 channel loop body for ``statements``."""
    def a(t):
        kind, v = t
        if kind == "x":
            return f"pack(__ldg(xa + {v}), __ldg(xb + {v}))"
        if kind == "d":
            k, start, stride = max((b for b in blocks if b[1] <= v), key=lambda b: b[1])
            return f"dbc(__ldg(dc{k} + {v - start}) * dsc)"
        if kind == "sh":
            return f"sh2[{v}]"
        if kind == "c":
            return half_bits(v)
        if kind == "eg":
            return f"edge_gradient[{v}]"
        return v

    names = sorted({s[1] for s in stmts if s[0] in ("zero", "let")})
    lines = [f"h2 {', '.join(names)};", "h2 radial_weight, radial_dt;", "float slope0, slope1;",
             "const float* xa = x + ib0 + channel;", "const float* xb = x + ib1 + channel;",
             "float* dxa = dx + ib0 + channel;", "float* dxb = dx + ib1 + channel;",
             "const uint2* gca = reinterpret_cast<const uint2*>(g0) + channel;",
             "const uint2* gcb = reinterpret_cast<const uint2*>(g1) + channel;"]
    lines += [f"const float* dc{k} = drow{k} + channel;" for k, _, _ in blocks]
    for s in stmts:
        if s[0] == "path":
            lines.append(f"// path at weight row {s[1]}")
            wb = s[1]
        elif s[0] == "horner":
            lines += [
                "{",
                f"    const uint2 ga = __ldg(gca + {wb});",
                f"    const uint2 gb = __ldg(gcb + {wb});",
                "    radial_weight = __byte_perm(ga.y, gb.y, 0x7632);",
                "    radial_dt = radial_weight;",
                "    radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));",
                "    radial_dt = hfma(radial_dt, t2, radial_weight);",
                "    radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));",
                "    radial_dt = hfma(radial_dt, t2, radial_weight);",
                "    radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));",
                "}"]
        elif s[0] == "zero":
            lines.append(f"{s[1]} = 0u;")
        elif s[0] == "let":
            ops = [a(t) for t in s[2]]
            lines.append(f"{s[1]} = {ops[0]};" if len(ops) == 1 else f"{s[1]} = hmul({ops[0]}, {ops[1]});")
        elif s[0] == "acc":
            lhs = a(s[1])
            fn = "hfma" if s[2] == "+" else "hfms"
            lines.append(f"{lhs} = {fn}({a(s[3])}, {a(s[4])}, {lhs});")
        elif s[0] == "slope":  # float32 (the oracle: float16 slope and sum double the radial-gradient error)
            lines.append("slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;")
        elif s[0] == "radial":
            lines.append("rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); "
                         "rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);")
        elif s[0] == "atomic":
            lines.append(f"atomicAdd(dxa + {s[1]}, lo2f({s[2]}) * inv0);")
            lines.append(f"atomicAdd(dxb + {s[1]}, hi2f({s[2]}) * inv1);")
    return "\n".join("            " + line for line in lines)


def kernel_source(source, lookup, min_blocks=3):
    """The pairing and h2 kernels for the float32 w' kernel ``source`` (STAGE_EDGES, SPLINE_W, FACTORED, HORNER,
    BUTTERFLY, 32-lane groups, 16 edge features). ``lookup``: the spline locator's C (``spline_forward.lookup_code``
    on ``clamped``)."""
    head = source[:source.index('extern "C"')]
    for need in ("constexpr int PADDED_TERMS = 4;", "LOGICAL_GROUP_SIZE =\n    32;"):
        if need not in head:
            raise ValueError(f"HALF_FAR: the w' kernel needs {need!r}")
    edge_dim = int(re.search(r"constexpr int EDGE_DIM = (\d+);", head).group(1))
    if edge_dim > 16:
        raise ValueError("HALF_FAR: at most 16 edge features")
    blocks = dout_blocks(source)
    stmts = statements(channel_body(source))
    dout_params = "".join(f"    const float* __restrict__ dout{k},\n" for k, _, _ in blocks)
    drows = "\n".join(f"        const float* drow{k} = dout{k} + int64_t(rcv) * {stride};"
                      for k, _, stride in blocks)
    kernel = f"""
extern "C" __global__ void __launch_bounds__(256, {min_blocks}) wprime_half_far(
    int64_t node_count,
    const int2* __restrict__ pairs,
    const int* __restrict__ count,
    const float* __restrict__ x,
    const float* __restrict__ sh,
    const int32_t* __restrict__ senders,
    const int32_t* __restrict__ receivers,
{dout_params}    float* dx,
    float* dsh,
    const float* __restrict__ lengths,
    const int32_t* __restrict__ species,
    const unsigned short* __restrict__ gh,
    const float* __restrict__ ginv,
    int32_t pieces,
    const float* __restrict__ dscale,
    float* dr) {{
    __shared__ h2 sh2s[8][EDGE_DIM];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = *count;
    for (int pidx = blockIdx.x * 8 + w; pidx < n; pidx += gridDim.x * 8) {{
        const int2 pr = pairs[pidx];
        const int64_t e0 = pr.x, e1 = pr.y;
        const int32_t rcv = receivers[e0];
        const int64_t ib0 = int64_t(senders[e0]) * INPUT_DIM, ib1 = int64_t(senders[e1]) * INPUT_DIM;
        if (lane < EDGE_DIM) sh2s[w][lane] = pack(sh[e0 * EDGE_DIM + lane], sh[e1 * EDGE_DIM + lane]);
        float t0, t1, sx0, sx1;
        int iv0, iv1;
        {{
            const float clamped = lengths[e0];
{lookup}
            t0 = t; sx0 = sx_inv; iv0 = iv;
        }}
        {{
            const float clamped = lengths[e1];
{lookup}
            t1 = t; sx1 = sx_inv; iv1 = iv;
        }}
        const h2 t2 = pack(t0, t1);
        const int64_t slab0 = int64_t(species[e0]) * pieces + iv0, slab1 = int64_t(species[e1]) * pieces + iv1;
        // the receiver's dout row times a power of two (dscale) and each slab's table scale: undone in float32
        const float dsc = dscale[rcv];
        const float inv0 = ginv[slab0] / dsc, inv1 = ginv[slab1] / dsc;
        const unsigned short* g0 = gh + slab0 * int64_t(WEIGHT_DIM) * 4;
        const unsigned short* g1 = gh + slab1 * int64_t(WEIGHT_DIM) * 4;
{drows}
        __syncwarp();
        const h2* sh2 = sh2s[w];
        h2 edge_gradient[EDGE_DIM];
        #pragma unroll
        for (int j = 0; j < EDGE_DIM; ++j) edge_gradient[j] = 0u;
        float rg0 = 0.f, rg1 = 0.f;
        for (int channel = lane; channel < CHANNEL_DIM; channel += LOGICAL_GROUP_SIZE) {{
{half_body(stmts, blocks)}
        }}
{REDUCE_SCATTER if edge_dim == 16 else REDUCE_EACH}
        float r0 = rg0, r1 = rg1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {{
            r0 += __shfl_xor_sync(FULL_MASK, r0, offset, 32);
            r1 += __shfl_xor_sync(FULL_MASK, r1, offset, 32);
        }}
        if (lane == 0) {{ dr[e0] = r0 * inv0; dr[e1] = r1 * inv1; }}
        __syncwarp();
    }}
}}
"""
    dbc = ("__device__ __forceinline__ h2 dbc(float v) {  // one value in both halves\n"
           "    h2 d; asm(\"cvt.rn.f16x2.f32 %0, %1, %1;\" : \"=r\"(d) : \"f\"(v)); return d; }\n")
    rows = "\n".join(f"        const float* row{k} = dout{k} + int64_t(node) * {stride};\n"
                     f"        for (int j = lane; j < {stride}; j += 32) top = fmaxf(top, fabsf(__ldg(row{k} + j)));"
                     for k, _, stride in blocks)
    scale = f"""
// HALF_FAR: each paired receiver's dout row scale, a power of two taking its largest |value| into [2^DOUT_TOP,
// 2^(DOUT_TOP + 1)) (a third of the far rows' values are otherwise float16-subnormal). A warp per node.
extern "C" __global__ void wprime_half_scale(
    int64_t node_count,
    const unsigned char* __restrict__ paired,
{dout_params}    float* __restrict__ dscale) {{
    const int lane = threadIdx.x & 31;
    const int64_t node = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (node >= node_count || !paired[node]) return;
    float top = 0.f;
{rows}
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) top = fmaxf(top, __shfl_xor_sync(FULL_MASK, top, offset, 32));
    int e;
    frexpf(top, &e);  // top = f 2^e, f in [0.5, 1)
    if (lane == 0) dscale[node] = top > 0.f ? ldexpf(1.f, {DOUT_TOP} + 1 - e) : 1.f;
}}
"""
    return head + PRELUDE + dbc + PAIRS + scale + kernel


def with_skip(source):
    """The float32 w' kernel skipping the edges the pairing flags (``half_skip``, its last parameter)."""
    params = "    scalar_t* dr) {"
    anchor = "    const int64_t input_base = int64_t(sender) * INPUT_DIM;\n"
    if source.count(params) != 1 or source.count(anchor) != 1:
        raise ValueError("HALF_FAR: the w' source has no unique parameter end / edge set-up to transform")
    source = source.replace(params, "    scalar_t* dr,\n    const unsigned char* __restrict__ half_skip) {")
    return source.replace(anchor, "    if (half_skip[e]) continue;  // HALF_FAR: a paired far edge\n" + anchor)


SLAB_TOP = 4  # each slab's largest |coefficient| scaled into [2^SLAB_TOP, 2^(SLAB_TOP + 1)) before rounding
DOUT_TOP = 4  # each paired receiver's largest |dout| likewise (wprime_half_scale)


def half_table(table):
    """``(bits, inverse)``: the float16 copy (as int16 bits) of SPLINE_W's table ``[slabs, Q, 4]`` float32 with each
    slab times a power of two (its largest |coefficient| into [2^SLAB_TOP, 2^(SLAB_TOP + 1)): near the cutoff the far
    slabs' coefficients are otherwise float16-subnormal; 2^10 overflowed float16 on NbCl5 in the oracle), and the
    ``[slabs]`` float32 inverse scales the kernel multiplies its outputs by (exact: powers of two)."""
    import torch

    top = table.abs().reshape(table.shape[0], -1).amax(dim=1).double()
    exponent = torch.where(top > 0, SLAB_TOP - torch.floor(torch.log2(top.clamp_min(1e-300))), torch.zeros_like(top))
    scale = torch.exp2(exponent)
    bits = (table.double() * scale[:, None, None]).to(torch.float16).view(torch.int16).contiguous()
    return bits, (1.0 / scale).to(torch.float32).contiguous()
