"""OMAT-0 kernel lab (owner, 3 November 2026: "come up with more ideas to speed up omat-0 on a100").

The A100 profile of Benchmark_loup (results/omat0-a100-profile-20261103) puts 65 % of the force call in two kernels:
the radial r-gradient backward ``oeq_projected_backward_wprime`` (6.9 ms) and the generic fused forward
``fused_folded_forward`` (5.1 ms). This tool captures one force call's arguments of both from an eager call of the
default plan (SrTiO3 10x10x10, 5,000 atoms), frees the engine, and times variants of the two kernels' rendered
sources on those arguments, round-robin in one process (CUDA events, median per variant). Since 5 November 2026 the
capture follows --equilibrate-steps MD steps (default 300): the first steps from the ideal lattice have few distinct
edge lengths, so the radial tables' reads stay in L2 and memory-bound kernels look compute-bound (the w' REDUX
variant: 0.72 of the kernel there, about 0.99 in MD):

- exact variants (cache hints, register caps, K = 64 without the zero cutoff column, unordered species pairs): their
  outputs are compared with the plan's kernel bit for bit (``dx`` of the backward is summed atomically, so it is
  compared with a tolerance);
- falsifiers (timing only, wrong results): one part of the work dropped (the derivative-table loads, the atomics,
  the stored-weight loads, the w store, the a @ W product, the angular terms), to locate what bounds each kernel
  on a GPU.

Usage (hybrid environment, FastMACE on PYTHONPATH):
  python tools/omat0_kernel_lab.py --checkpoint mace-omat-0-medium.model --output DIR [--variants NAME ...]
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402

FAKE_ADD = """
__device__ __forceinline__ void fake_add(float* p, float v) { if (v == 1.2345e-38f) *p = v; }
"""
POLICY = """
__device__ __forceinline__ unsigned long long fm_policy_last() {
    unsigned long long p; asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ unsigned long long fm_policy_first() {
    unsigned long long p; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ float fm_ld_hint(const float* a, unsigned long long p) {
    float v; asm volatile("ld.global.nc.L2::cache_hint.f32 %0, [%1], %2;" : "=f"(v) : "l"(a), "l"(p)); return v; }
"""


def _sub(source, old, new, count=None, regex=False):
    """``source`` with ``old`` replaced by ``new``; raises unless it occurs (``count`` times if given)."""
    found = len(re.findall(old, source)) if regex else source.count(old)
    if not found or (count is not None and found != count):
        raise ValueError(f"variant pattern found {found} times: {old[:80]!r}")
    return re.sub(old, new, source) if regex else source.replace(old, new)


def _after_constants(source, text):
    """``text`` inserted after the ``using``/``constexpr`` header (before the first kernel)."""
    index = source.index('extern "C"')
    return source[:index] + text + source[index:]


# ---------------------------------------------------------------------------------------------------------- wprime
def wprime_variants(species, kernel=None):
    """name -> (kind, source transform, extra NVRTC options); kind "exact", "rounding" (compared with relative
    errors) or "falsifier"."""
    load_w = "const scalar_t radial_weight = weights[radial_weight_index];"
    slope = "slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);"
    group = "const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;"

    def unordered(s):
        s = _after_constants(s, f"""
__device__ __forceinline__ int fm_unordered(int pr) {{
    const int a = pr / {species}, b = pr - a * {species}; return a <= b ? pr : b * {species} + a; }}
""")
        return _sub(s, "int64_t(pairs[e])", "int64_t(fm_unordered(pairs[e]))", 1)

    def hints(s, w=True, g=True):
        s = _after_constants(s, POLICY)
        if w:
            s = _sub(s, load_w, "const scalar_t radial_weight = __ldcs(weights + radial_weight_index);")
        if g:
            s = _sub(s, "acc_t radial_gradient = acc_t(0);",
                     "acc_t radial_gradient = acc_t(0);\n    const unsigned long long g_policy = fm_policy_last();", 1)
            s = _sub(s, slope, "slope += cheb[k] * fm_ld_hint(g + k * WEIGHT_DIM + weight_index, g_policy);")
        return s

    def qmajor(s, vector):
        """G' read q-major ([group, Q, terms(, padded)]) with float2 or float4 loads; the same k-ordered sum."""
        if vector == 2:
            body = ("{ const float2* gq = reinterpret_cast<const float2*>(g + weight_index * TERMS);\n"
                    "                    #pragma unroll\n"
                    "                    for (int k2 = 0; k2 < TERMS / 2; ++k2) { const float2 v = __ldg(gq + k2);\n"
                    "                        slope += cheb[2 * k2] * v.x; slope += cheb[2 * k2 + 1] * v.y; } }")
        else:
            body = ("{ const float4* gq = reinterpret_cast<const float4*>(g + weight_index * TPAD);\n"
                    "                    #pragma unroll\n"
                    "                    for (int k4 = 0; k4 < TPAD / 4; ++k4) { const float4 v = __ldg(gq + k4);\n"
                    "                        if (4 * k4 < TERMS) slope += cheb[4 * k4] * v.x;\n"
                    "                        if (4 * k4 + 1 < TERMS) slope += cheb[4 * k4 + 1] * v.y;\n"
                    "                        if (4 * k4 + 2 < TERMS) slope += cheb[4 * k4 + 2] * v.z;\n"
                    "                        if (4 * k4 + 3 < TERMS) slope += cheb[4 * k4 + 3] * v.w; } }")
            s = _sub(s, r"(constexpr int TERMS = \d+;)", r"\1\nconstexpr int TPAD = (TERMS + 3) / 4 * 4;", 1,
                     regex=True)
            s = _sub(s, group, "const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * TPAD * WEIGHT_DIM;",
                     1)
        return _sub(s, r"for \(int k = 0; k < TERMS; \+\+k\)\s*" + re.escape(slope), body.replace("\\", "\\\\"),
                    regex=True)

    def qmajor_args(vector):
        def transform(args):
            import torch
            args = list(args)
            g = args[10]  # [groups, terms, Q]
            g = g.transpose(1, 2)  # [groups, Q, terms]
            if vector == 4:
                pad = -g.shape[2] % 4
                g = torch.nn.functional.pad(g, (0, pad))
            args[10] = g.contiguous()
            return args
        return transform

    def edge_order(key):
        """An argument transform: every per-edge input permuted by ``key(r, pair)`` (stable sort). The outputs come
        out in that order too, so this is a timing bound, not an exact variant."""
        def transform(args):
            import torch
            args = list(args)
            sh, r, pair = args[1], args[8], args[9]
            edges = sh.shape[0]
            order = torch.sort(key(r, pair), stable=True).indices
            for i in (1, 2, 3, 4, 7, 8, 9, 17):  # sh, weights, senders, receivers, dsh, r, pair, dr
                t = args[i]
                if t.dim() >= 1 and t.shape[0] == edges:
                    args[i] = t.index_select(0, order).contiguous()
                elif t.dim() == 1 and t.numel() % edges == 0:
                    args[i] = t.view(edges, -1).index_select(0, order).flatten().contiguous()
            return args
        return transform

    by_piece = edge_order(lambda r, pair: pair.double() * 1e3 + r.double())  # within a pair, r orders the pieces

    # Mirror pairs (H200 research answers, 5 November 2026: share w and w' between an edge and its reverse). The edges
    # ordered as (lo <- hi, hi <- lo) pairs by their atoms (lo < hi), the receiver-sorted order kept within a key, so
    # edge 2k and 2k + 1 are mirrors with the same table row (SYMMETRIC_PAIRS) and length. Reports how many edges have
    # exactly one mirror and how many mirror lengths are bitwise equal.
    def mirror_order(args):
        import torch
        args = list(args)
        nodes = args[0].shape[0]
        senders, receivers = args[3].long(), args[4].long()
        lo, hi = torch.minimum(senders, receivers), torch.maximum(senders, receivers)
        key = lo * nodes + hi
        order = torch.sort(key, stable=True).indices
        edges = key.shape[0]
        sk = key[order]
        paired = bool(edges % 2 == 0 and torch.equal(sk[0::2], sk[1::2])
                      and (edges < 4 or bool((sk[2::2] != sk[1:-1:2]).all())))
        valid = (senders >= 0) & (senders < nodes) & (receivers >= 0) & (receivers < nodes)
        multiplicity = torch.bincount(torch.unique(key[valid], return_counts=True)[1])
        print(f"mirror_order: invalid edges {int((~valid).sum())}, self edges {int((senders == receivers).sum())}, "
              f"keys by multiplicity {multiplicity.tolist()}", flush=True)
        r = args[8][order]
        equal_r = float((r[0::2] == r[1::2]).float().mean()) if paired else float("nan")
        both = sk[1:] == sk[:-1]  # adjacent edges of one atom pair (every key has at most two edges)
        if bool(both.any()):
            same = r[1:][both] == r[:-1][both]
            largest = float((r[1:][both] - r[:-1][both]).abs().max())
            print(f"mirror_order: {int(both.sum())} mirror pairs, bitwise-equal lengths "
                  f"{float(same.float().mean()):.6f}, largest |dr| {largest:.3e}", flush=True)
        print(f"mirror_order: {edges} edges, every edge paired: {paired}, bitwise-equal mirror lengths: "
              f"{equal_r:.4f}", flush=True)
        for i in (1, 2, 3, 4, 7, 8, 9, 17):  # sh, weights, senders, receivers, dsh, r, pair, dr
            t = args[i]
            if t.dim() >= 1 and t.shape[0] == edges:
                args[i] = t[order].contiguous()
        return args

    def sender_order(args):
        import torch
        args = list(args)
        edges = args[3].shape[0]
        order = torch.sort(args[3].long(), stable=True).indices
        for i in (1, 2, 3, 4, 7, 8, 9, 17):
            t = args[i]
            if t.dim() >= 1 and t.shape[0] == edges:
                args[i] = t[order].contiguous()
        return args

    # The exact form of mirror order: the data stays in receiver order and warp k processes edge perm[k], perm the
    # mirror order of the edges (built once per neighbour list). Outputs land at their own edges, so the result is
    # the base kernel's up to the dx atomics' order.
    def mirror_perm(args):
        import torch
        args = list(args)
        nodes = args[0].shape[0]
        senders, receivers = args[3].long(), args[4].long()
        key = torch.minimum(senders, receivers) * nodes + torch.maximum(senders, receivers)
        args.append(torch.sort(key, stable=True).indices.to(torch.int32).contiguous())
        return args

    def perm_source(s):
        s, n = re.subn(r"(scalar_t\* dr)\) \{", r"\1,\n    const int32_t* __restrict__ edge_perm) {", s, count=1)
        s, m = re.subn(r"const int64_t e =\s*int64_t\(blockIdx\.x\) \* \(blockDim\.x / LOGICAL_GROUP_SIZE\) "
                       r"\+ logical_group;\s*if \(e >= edge_count\) return;",
                       "const int64_t slot =\n        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + "
                       "logical_group;\n    if (slot >= edge_count) return;\n    const int64_t e = edge_perm[slot];", s,
                       count=1)
        if not (n and m):
            raise ValueError("perm_source: the kernel signature or edge index was not found")
        return s

    # The hybrid: the 4-byte per-edge arrays (senders, receivers, r, pair, and dr written) in mirror order, a cheap
    # gather per call (a scatter back for dr), and only the 64-byte sh/dsh rows (and stored w) read through the
    # permutation, so no edge scalar costs a whole sector. dsh exact; dr comes out in mirror order here.
    def mirror_hybrid(args):
        import torch
        args = list(args)
        nodes = args[0].shape[0]
        senders, receivers = args[3].long(), args[4].long()
        key = torch.minimum(senders, receivers) * nodes + torch.maximum(senders, receivers)
        order = torch.sort(key, stable=True).indices
        edges = order.shape[0]
        for i in (3, 4, 8, 9, 17):  # senders, receivers, r, pair, dr
            t = args[i]
            if t.dim() >= 1 and t.shape[0] == edges:
                args[i] = t[order].contiguous()
        args.append(order.to(torch.int32).contiguous())
        return args

    def hybrid_source(s):
        s = re.sub(r"(scalar_t\* dr)\) \{", r"\1,\n    const int32_t* __restrict__ edge_perm) {", s, count=1)
        for old, new in (("const int64_t edge_base = e * EDGE_DIM;",
                          "const int64_t row_edge = edge_perm[e];\n    const int64_t edge_base = row_edge * EDGE_DIM;"),
                         ("const int64_t weight_base = e * WEIGHT_DIM;",
                          "const int64_t weight_base = row_edge * WEIGHT_DIM;")):
            if s.count(old) != 1:
                raise ValueError(f"hybrid_source: {old!r} not found once")
            s = s.replace(old, new)
        return s

    horner_block = (r"scalar_t gv\[PADDED_TERMS\];.*?radial_weight = radial_weight \* t \+ gv\[k\];\s*\}")

    def skip_table(condition):
        """The value-table load and the dual Horner replaced by constants where ``condition`` holds (warp-uniform:
        one warp per edge). Timing only."""
        def transform(s):
            def repl(m):
                body = (m.group(0).replace("scalar_t radial_weight = gv[TERMS - 1];", "radial_weight = gv[TERMS - 1];")
                        .replace("scalar_t radial_dt = scalar_t(0);", "radial_dt = scalar_t(0);"))
                return (f"scalar_t radial_weight = t, radial_dt = sc;\n                if (!({condition})) {{\n"
                        f"                {body}\n                }}")
            out, n = re.subn(horner_block, repl, s, flags=re.S)
            if not n:
                raise ValueError("skip_table: the HORNER value-table block was not found")
            return out
        return transform

    # The tensor product's per-term arithmetic (OMAT-0 TP backward screen, 4 November 2026). The kernel is generated
    # from the model's schedule: its 10 paths, 40 outputs and 86 Clebsch-Gordan terms are unrolled with the
    # coefficients as literals, so a model-specialised backward can only save arithmetic (fewer operations per term,
    # by factoring) or loads (sh is per edge, yet read once per channel). These variants bound that.
    dx_term = r"\w+ \+= output_gradient\s*\* radial_weight \* edge_value \* static_cast<scalar_t>\([^)]*\);"
    dsh_term = (r"edge_gradient\[edge_component_index\] \+=\s*output_gradient \* radial_weight \* input_value"
                r"\s*\* static_cast<scalar_t>\([^)]*\);")
    wg_term = r"weight_gradient \+= output_gradient \* input_value\s*\* edge_value \* static_cast<scalar_t>\([^)]*\);"
    coefficient = r"\s*\* static_cast<scalar_t>\([^)]*\)(?=;)"
    load_sh = "const scalar_t edge_value = sh[edge_index];"

    def wg_only(s):
        """Only the weight gradient per term (the input and edge gradients' updates dropped): about 3 of the 7
        floating-point operations per term remain, as many as any factoring of the three trilinear sums keeps."""
        return _sub(_sub(s, dx_term, "", 86, regex=True), dsh_term, "", 86, regex=True)

    def no_tp(s):
        """No per-term arithmetic at all; each output gradient still loaded (into the weight gradient), x and sh die:
        the outer bound of any rewrite of the tensor product."""
        s = _sub(_sub(_sub(s, dx_term, "", 86, regex=True), dsh_term, "", 86, regex=True), wg_term, "", 86, regex=True)
        return _sub(s, r"(const scalar_t output_gradient = [^;]*;)", r"\1 weight_gradient += output_gradient;",
                    regex=True)

    def sh_hoist(s):
        """sh's EDGE_DIM values loaded once per edge into registers instead of once per channel and term (exact)."""
        s = _sub(s, "acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};", """scalar_t shv[EDGE_DIM];
    #pragma unroll
    for (int c = 0; c < EDGE_DIM; ++c) shv[c] = sh[edge_base + c];
    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};""", 1)
        return _sub(s, load_sh, "const scalar_t edge_value = shv[edge_component_index];")

    def gslab(s):
        """w and its slope from one slab of the value table G [groups, TERMS + 1, Q]: w = sum_k T_k(x) G_k and
        w' = sum_k T_k'(x) G_k (T_k' = k U_(k-1)(x) scale), so neither the stored w nor G' is read. A timing bound:
        the argument transform passes G' with one extra term, so w and dr are not the real ones."""
        s = _sub(s, r"(constexpr int TERMS = \d+;)", r"\1\nconstexpr int GT = TERMS + 1;", 1, regex=True)
        s = _sub(s, group, "const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * GT * WEIGHT_DIM;", 1)
        basis = """scalar_t wcheb[GT], dcheb[GT];
    {
        const scalar_t sc = scales[piece];
        scalar_t um1 = scalar_t(0), u = scalar_t(1);  // U_(k-2), U_(k-1)
        wcheb[0] = scalar_t(1);
        dcheb[0] = scalar_t(0);
        #pragma unroll
        for (int k = 1; k < GT; ++k) {
            wcheb[k] = k == 1 ? t : scalar_t(2) * t * wcheb[k == 1 ? 0 : k - 1] - wcheb[k < 2 ? 0 : k - 2];
            dcheb[k] = scalar_t(k) * u * sc;
            const scalar_t un = scalar_t(2) * t * u - um1;
            um1 = u;
            u = un;
        }
    }
    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};"""
        s = _sub(s, "acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};", basis, 1)
        s = _sub(s, load_w, """scalar_t gv[GT];
                scalar_t radial_weight = scalar_t(0);
                #pragma unroll
                for (int k = 0; k < GT; ++k) {
                    gv[k] = __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_weight += wcheb[k] * gv[k];
                }""")
        return _sub(s, r"for \(int k = 0; k < TERMS; \+\+k\)\s*" + re.escape(slope),
                    "for (int k = 1; k < GT; ++k) slope += dcheb[k] * gv[k];", regex=True)

    def gslab_args(args, qmajor=False):
        import torch
        args = list(args)
        g = args[10]  # [groups, terms, Q] -> [groups, terms + 1, Q] (q-major: [groups, Q, terms + 1])
        g = torch.cat([g, g[:, :1]], dim=1)
        args[10] = (g.transpose(1, 2) if qmajor else g).contiguous()
        return args

    def gslab_q(s, vector):
        """``gslab`` with the slab q-major, each weight's GT terms loaded as float2/float4 (GT a multiple of it)."""
        s = gslab(s)
        loads = "\n".join(
            f"                    {{ const float{vector} v = __ldg(gq + {i});"
            + "".join(f" gv[{vector * i + c}] = v.{comp};" for c, comp in enumerate("xyzw"[:vector])) + " }"
            for i in range(6 // vector))
        return _sub(s, """                #pragma unroll
                for (int k = 0; k < GT; ++k) {
                    gv[k] = __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_weight += wcheb[k] * gv[k];
                }""", f"""                static_assert(GT == 6, "the q-major slab bound assumes 6 terms");
                const float{vector}* gq = reinterpret_cast<const float{vector}*>(g + weight_index * GT);
{loads}
                #pragma unroll
                for (int k = 0; k < GT; ++k) radial_weight += wcheb[k] * gv[k];""")


    def rerender(**options):
        """The plan's kernel rendered again from its schedule with ``options`` (wprime_backward.render)."""
        def transform(s):
            from fastmace.kernels.oeq_streaming import wprime_backward as wp
            return wp.render(kernel.schedule, kernel.scalar, kernel.terms, vector=kernel.vector,
                             table_w=kernel.table_w, blocks=kernel.blocks, **options)
        return transform

    def power_args(a):
        """The argument list with the value table in the power basis of t (``horner``)."""
        from fastmace.kernels.oeq_streaming import wprime_backward as wp
        a = list(a)
        a[10] = wp.power_table(a[10], kernel.terms)
        return a

    def pair():
        """The channel-pair TABLE_W kernel (wprime_pair): float2 / float4 loads, the factored product."""
        def transform(s):
            from fastmace.kernels.oeq_streaming import wprime_pair
            if not kernel.table_w:
                raise ValueError("pair needs the TABLE_W kernel")
            return wprime_pair.render(kernel.schedule, kernel.terms, blocks=kernel.blocks)
        return transform

    # The end-of-edge reduction (round 5): 16 edge-gradient components and dr are shuffled down at 5 offsets each,
    # 85 shuffles per lane and edge. A butterfly reduce-scatter swaps half of the remaining components with the xor
    # partner at each step (8 + 4 + 2 + 1 shuffles, then 1 for the last pair), so lane 2c ends with component c and
    # 16 lanes write dsh coalesced; dr keeps its 5 shuffles: 21 in all. Summation order differs: "rounding".
    reduction = (r"for \(int offset = 16; offset > 0; offset >>= 1\) \{.*?"
                 r"dr\[e\] = scalar_t\(radial_gradient\);\s*\}")
    butterfly_code = """static_assert(EDGE_DIM == 16 && LOGICAL_GROUP_SIZE == 32, "butterfly: 16 components, 32 lanes");
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int half = 8 >> step, offset = 16 >> step;
        const bool upper = (lane & offset) != 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < half) {
                const acc_t send = upper ? edge_gradient[i] : edge_gradient[i + half];
                const acc_t keep = upper ? edge_gradient[i + half] : edge_gradient[i];
                edge_gradient[i] = keep + __shfl_xor_sync(FULL_MASK, send, offset, 32);
            }
        }
    }
    edge_gradient[0] += __shfl_xor_sync(FULL_MASK, edge_gradient[0], 1, 32);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        radial_gradient += __shfl_xor_sync(FULL_MASK, radial_gradient, offset, 32);
    if ((lane & 1) == 0) dsh[edge_base + (lane >> 1)] = scalar_t(edge_gradient[0]);
    if (lane == 0) dr[e] = scalar_t(radial_gradient);"""

    def butterfly(s):
        return re.sub(reduction, lambda m: butterfly_code, s, count=1, flags=re.S) if re.search(
            reduction, s, flags=re.S) else _sub(s, reduction, "", 1, regex=True)

    # Round 5 follow-ups (5 November 2026, external ideas 2 and 5). F-cart3: each lane projects its 16 dsh partials
    # (and dr) onto three Cartesian components with a 16x3 Jacobian before the reduction, then 3 values are reduced
    # (the Jacobian is a stand-in read from sh: timing only). F-redux16: the 17 sums as fixed point through the
    # hardware integer reduction (redux.sync.add.s32), with one redux.sync.max for the warp's exponent; deterministic,
    # about 2^-25 of the warp's largest partial per term. F-cart3-redux: both.
    jacobian = """acc_t g3[3] = {acc_t(0), acc_t(0), acc_t(0)};
    #pragma unroll
    for (int c = 0; c < EDGE_DIM; ++c) {
        #pragma unroll
        for (int k = 0; k < 3; ++k)
            g3[k] += edge_gradient[c] * acc_t(sh[edge_base + ((c + k + 1) & (EDGE_DIM - 1))]);
    }
    #pragma unroll
    for (int k = 0; k < 3; ++k) g3[k] += radial_gradient * acc_t(sh[edge_base + 1 + k]);
"""
    cart3_code = jacobian + """    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) g3[k] += __shfl_xor_sync(FULL_MASK, g3[k], offset, 32);
    }
    if (lane < 3) dsh[edge_base + lane] = scalar_t(lane == 0 ? g3[0] : (lane == 1 ? g3[1] : g3[2]));
    if (lane == 0) dr[e] = scalar_t(0);"""

    def redux(values, count, store):
        return f"""float big = 0.f;
    #pragma unroll
    for (int c = 0; c < {count}; ++c) big = fmaxf(big, fabsf(float({values}[c])));
    unsigned ex = __reduce_max_sync(FULL_MASK, __float_as_uint(big) >> 23);
    ex = ex < 30u ? 30u : ex;
    const float scale = __uint_as_float((278u - ex) << 23), inverse = __uint_as_float((ex - 24u) << 23);
    #pragma unroll
    for (int c = 0; c < {count}; ++c) {{
        const int total = __reduce_add_sync(FULL_MASK, __float2int_rn(float({values}[c]) * scale));
        if (lane == 0) {store} = scalar_t(float(total) * inverse);
    }}"""

    redux16_code = """acc_t all[EDGE_DIM + 1];
    #pragma unroll
    for (int c = 0; c < EDGE_DIM; ++c) all[c] = edge_gradient[c];
    all[EDGE_DIM] = radial_gradient;
    """ + redux("all", "EDGE_DIM + 1", "(c < EDGE_DIM ? dsh[edge_base + c] : dr[e])")
    cart3_redux_code = jacobian + "    " + redux("g3", 3, "dsh[edge_base + c]") + """
    if (lane == 0) dr[e] = scalar_t(0);"""

    def redux_per_component(values, count, store):
        """Fixed point with each component's own exponent (one redux.sync.max per component): small components keep
        their bits next to large ones."""
        return f"""#pragma unroll
    for (int c = 0; c < {count}; ++c) {{
        const float v = float({values}[c]);
        unsigned ex = __reduce_max_sync(FULL_MASK, __float_as_uint(fabsf(v)) >> 23);
        ex = ex < 30u ? 30u : ex;
        const float scale = __uint_as_float((278u - ex) << 23), inverse = __uint_as_float((ex - 24u) << 23);
        const int total = __reduce_add_sync(FULL_MASK, __float2int_rn(v * scale));
        if (lane == 0) {store} = scalar_t(float(total) * inverse);
    }}"""

    redux_pc_code = """acc_t all[EDGE_DIM + 1];
    #pragma unroll
    for (int c = 0; c < EDGE_DIM; ++c) all[c] = edge_gradient[c];
    all[EDGE_DIM] = radial_gradient;
    """ + redux_per_component("all", "EDGE_DIM + 1", "(c < EDGE_DIM ? dsh[edge_base + c] : dr[e])")

    def _reduced(s, code):
        if not re.search(reduction, s, flags=re.S):
            raise ValueError("the shfl_down reduction is not in the source (BUTTERFLY=False?)")
        return re.sub(reduction, lambda m: code, s, count=1, flags=re.S)

    def no_reduce(s):
        """Falsifier: no cross-lane reduction; lane 0 writes its own partial sums (the outer bound)."""
        if not re.search(reduction, s, flags=re.S):
            raise ValueError("no-reduce: the shfl_down reduction is not in the source")
        return re.sub(reduction, lambda m: """if (lane == 0) {
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
        dr[e] = scalar_t(radial_gradient);
    }""", s, count=1, flags=re.S)

    def split_edge(split):
        """Timing bound (5 November 2026, external review idea 1): ``split`` warps per edge, each a 1/split share of
        the channels; the partial dsh and dr are added with atomics (the argument transform doubles the edge rows so
        the launcher starts split times as many warps; dsh and dr come out in the first rows, not exact)."""
        def transform(s):
            s = _sub(s, """    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;""", f"""    const int64_t gid =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    const int64_t e = gid / {split};
    const int part = int(gid % {split});
    if (e >= edge_count / {split}) return;""", 1)
            s = _sub(s, """    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {""", f"""    for (int channel = lane + part * LOGICAL_GROUP_SIZE;
         channel < CHANNEL_DIM;
         channel += {split} * LOGICAL_GROUP_SIZE) {{""", 1)
            if not re.search(reduction, s, flags=re.S):
                raise ValueError("split: the shfl_down reduction is not in the source (BUTTERFLY on? pass "
                                 "--constant fastmace.kernels.oeq_streaming.wprime_backward:BUTTERFLY=False)")
            return re.sub(reduction, lambda m: butterfly_code.replace(
                "if ((lane & 1) == 0) dsh[edge_base + (lane >> 1)] = scalar_t(edge_gradient[0]);",
                "if ((lane & 1) == 0) atomicAdd(dsh + edge_base + (lane >> 1), scalar_t(edge_gradient[0]));").replace(
                "if (lane == 0) dr[e] = scalar_t(radial_gradient);",
                "if (lane == 0) atomicAdd(dr + e, scalar_t(radial_gradient));"), s, count=1, flags=re.S)

        def args(a):
            import torch
            a = list(a)
            a[1] = torch.cat([a[1]] * split)  # sh: edge_count = split x edges, so split x the warps
            a[7] = torch.zeros((a[1].shape[0], a[7].shape[1]), dtype=a[7].dtype, device=a[7].device)  # dsh
            a[17] = torch.zeros(a[1].shape[0], dtype=a[17].dtype, device=a[17].device)  # dr
            return a
        return transform, args

    return {
        "butterfly": ("rounding", butterfly, ()),
        "F-split2": ("falsifier", split_edge(2)[0], (), split_edge(2)[1]),
        "F-split4": ("falsifier", split_edge(4)[0], (), split_edge(4)[1]),
        "F-split1-atomic": ("falsifier", split_edge(1)[0], (), split_edge(1)[1]),
        "F-no-reduce": ("falsifier", no_reduce, ()),
        "F-reduce-dr-only": ("falsifier", lambda s: re.sub(reduction, lambda m: """
    for (int offset = 16; offset > 0; offset >>= 1)
        radial_gradient += __shfl_xor_sync(FULL_MASK, radial_gradient, offset, 32);
    if (lane == 0) {
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
        dr[e] = scalar_t(radial_gradient);
    }""", s, count=1, flags=re.S), ()),
        "redux": ("rounding", lambda s: _reduced(s, redux16_code), ()),
        "redux-pc": ("rounding", lambda s: _reduced(s, redux_pc_code), ()),
        "F-cart3": ("falsifier", lambda s: _reduced(s, cart3_code), ()),
        "F-redux16": ("falsifier", lambda s: _reduced(s, redux16_code), ()),
        "F-cart3-redux": ("falsifier", lambda s: _reduced(s, cart3_redux_code), ()),
        "F-reduce-1level": ("falsifier", lambda s: _sub(s, "for (int offset = 16; offset > 0; offset >>= 1) {",
                                                        "for (int offset = 16; offset > 8; offset >>= 1) {", 1), ()),
        "F-reduce-3level": ("falsifier", lambda s: _sub(s, "for (int offset = 16; offset > 0; offset >>= 1) {",
                                                        "for (int offset = 16; offset > 2; offset >>= 1) {", 1), ()),
        "F-reduce-sh-only": ("falsifier", lambda s: re.sub(r"radial_gradient \+= __shfl_down_sync\([^;]*;", "",
                                                         s, count=1), ()),
        "B-piece-sorted": ("falsifier", lambda s: s, (), by_piece),
        "B-mirror-order": ("falsifier", lambda s: s, (), mirror_order),
        "B-mirror-share": ("falsifier", skip_table("e & 1"), (), mirror_order),
        "F-no-table-horner": ("falsifier", skip_table("true"), ()),
        "F-half-table": ("falsifier", skip_table("e & 1"), ()),
        "M-mirror-perm": ("rounding", perm_source, (), mirror_perm),
        "M-mirror-hybrid": ("falsifier", hybrid_source, (), mirror_hybrid),
        "B-sender-order": ("falsifier", lambda s: s, (), sender_order),
        "pair": ("rounding", pair(), ()),
        "pair-maxreg128": ("rounding", pair(), ("--maxrregcount=128",)),
        "pair-maxreg96": ("rounding", pair(), ("--maxrregcount=96",)),
        "factored": ("rounding", rerender(factored=True), ()),
        "factored-scaled": ("rounding", rerender(factored=True, scale=True), ()),
        "factored+unordered": ("rounding", lambda s: unordered(rerender(factored=True)(s)), ()),
        # Where the factored kernel's time goes (timing bounds on top of it).
        "FF-one-slab": ("falsifier",
                        lambda s: _sub(rerender(factored=True)(s), group, "const scalar_t* g = gprime;", 1), ()),
        "FF-no-atomics": ("falsifier", lambda s: _sub(_after_constants(rerender(factored=True)(s), FAKE_ADD),
                                                      "atomicAdd(", "fake_add("), ()),
        "FF-no-table": ("falsifier", lambda s: _sub(rerender(factored=True)(s), "const float2 v = __ldg(gq + k);",
                                                    "const float2 v = make_float2(t, t * t);"), ()),
        "FF-no-xload": ("falsifier", lambda s: _sub(rerender(factored=True)(s), r"x\[input_base\s*\+", "sh[edge_base +",
                                                    regex=True), ()),
        # Ideas session idea 12 (5 November 2026): receiver-staged dout. Bound: every edge reads receiver 0's dout
        # row, which then stays in L1.
        "FF-dout-one-row": ("falsifier", lambda s: _sub(
            s, "const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;", "const int64_t output_base = 0;"), ()),
        "FF-no-dout": ("falsifier", lambda s: _sub(rerender(factored=True)(s),
                                                   r"const scalar_t output_gradient = [^;]*;",
                                                   "const scalar_t output_gradient = radial_weight;", regex=True), ()),
        "factored-maxreg80": ("rounding", rerender(factored=True), ("--maxrregcount=80",)),
        # Round 2 (5 November 2026): dual Horner on a power-basis table, l1 = 0 paths without v_j.
        "horner": ("rounding", rerender(factored=True, horner=True), (), power_args),
        "scalar-l1": ("rounding", rerender(factored=True, scalar_paths=True), ()),
        "horner+scalar-l1": ("rounding", rerender(factored=True, horner=True, scalar_paths=True), (), power_args),
        "horner-maxreg80": ("rounding", rerender(factored=True, horner=True), ("--maxrregcount=80",), power_args),
        "horner+scalar-l1-maxreg80": ("rounding", rerender(factored=True, horner=True, scalar_paths=True),
                                      ("--maxrregcount=80",), power_args),
        # Horner on the plan's own kernel (its FACTORED and BLOCK_LDG defaults), e.g. the RTX 5060 Ti's unfactored one.
        "horner-default": ("rounding", lambda s: rerender(factored=kernel.factored, horner=True,
                                                          block_ldg=_block_ldg())(s), (), power_args),
        "horner-default-maxreg80": ("rounding", lambda s: rerender(factored=kernel.factored, horner=True,
                                                                   block_ldg=_block_ldg())(s),
                                    ("--maxrregcount=80",), power_args),
        "factored-noldg": ("rounding", rerender(factored=True, block_ldg=False), ()),
        "factored-noldg-horner-maxreg80": ("rounding", rerender(factored=True, horner=True, block_ldg=False),
                                           ("--maxrregcount=80",), power_args),
        "factored-maxreg72": ("rounding", rerender(factored=True), ("--maxrregcount=72",)),
        "factored-maxreg64": ("rounding", rerender(factored=True), ("--maxrregcount=64",)),
        "sh-hoist": ("exact", sh_hoist, ()),
        "F-tp-no-coef": ("falsifier", lambda s: _sub(s, coefficient, "", 258, regex=True), ()),
        "F-tp-wg-only": ("falsifier", wg_only, ()),
        "F-no-tp": ("falsifier", no_tp, ()),
        "B-gslab": ("falsifier", gslab, (), gslab_args),
        "B-gslab-f2": ("falsifier", lambda s: gslab_q(s, 2), (), lambda a: gslab_args(a, True)),
        "base": ("exact", lambda s: s, ()),
        "qmajor-f2": ("exact", lambda s: qmajor(s, 2), (), qmajor_args(2)),
        "qmajor-f4": ("exact", lambda s: qmajor(s, 4), (), qmajor_args(4)),
        "qmajor-f2+ldcs-w": ("exact", lambda s: qmajor(hints(s, g=False), 2), (), qmajor_args(2)),
        "maxreg64": ("exact", lambda s: s, ("--maxrregcount=64",)),
        "maxreg80": ("exact", lambda s: s, ("--maxrregcount=80",)),
        "maxreg-none": ("exact", lambda s: s, ("--maxrregcount=255",)),
        "maxreg72": ("exact", lambda s: s, ("--maxrregcount=72",)),
        "ldcs-w": ("exact", lambda s: hints(s, g=False), ()),
        "evict-last-g": ("exact", lambda s: hints(s, w=False), ()),
        "hints": ("exact", hints, ()),
        "unordered": ("exact", unordered, ()),
        "unordered+hints": ("exact", lambda s: hints(unordered(s)), ()),
        "F-no-slope": ("falsifier", lambda s: _sub(s, "if (sloped) {", "if (false) {"), ()),
        "F-half-terms": ("falsifier", lambda s: _sub(s, slope, "if (k < TERMS / 2) " + slope), ()),
        "F-one-slab": ("falsifier", lambda s: _sub(s, group, "const scalar_t* g = gprime;", 1), ()),
        "F-no-atomics": ("falsifier", lambda s: _sub(_after_constants(s, FAKE_ADD), "atomicAdd(", "fake_add("), ()),
        "F-no-wload": ("falsifier", lambda s: _sub(s, load_w, "const scalar_t radial_weight = scalar_t(1);"), ()),
        # Kernel review (5 November 2026): every pointer argument __restrict__, so loads of x and sh can be kept
        # across the dx atomics (exact).
        "restrict": ("exact", _restrict_all, ()),
        # Receiver-local forward + backward of the last interaction (6 November 2026): a kernel that owns a receiver
        # tile would hold the receiver's output gradient (dout) and its edges' sh in shared memory; the sender
        # features x would still be gathered (2 KB per edge does not fit). These bounds read them from a 4 KB shared
        # array instead of global memory (values garbage, timing only). F-smem-dout-sh is the realistic one,
        # F-smem-all (x too) the optimistic one. Pair with the forward's F-no-out (no message store).
        **{name: ("falsifier", lambda s, keys=keys: _smem_reads(s, keys), ())
           for name, keys in (("F-smem-dout", ("dout",)), ("F-smem-sh", ("sh",)), ("F-smem-x", ("x",)),
                              ("F-smem-dout-sh", ("dout", "sh")), ("F-smem-all", ("dout", "sh", "x")))},
        # dout staging (6 November 2026, after the conductor's FF-dout-one-row 1.001): the block's first and last
        # receivers' dout rows copied to shared memory once, then a __syncthreads; warps whose receiver is one of
        # them read shared memory, any other receiver reads global memory. Exact (the same values, the same order).
        # stage-dout-r0 stages the first receiver only. F-stage-dout-nofill reads the same shared layout (real
        # per-receiver offsets, so bank conflicts show) without the copy and the barrier: the staging's read bound.
        "stage-dout": ("exact", lambda s: _stage_dout(s, 2), ()),
        "stage-dout-r0": ("exact", lambda s: _stage_dout(s, 1), ()),
        "F-stage-dout-nofill": ("falsifier", lambda s: _stage_dout(s, 2, fill=False), ()),
    }


def _restrict_all(s):
    """The w' kernel with __restrict__ on every pointer argument that lacks it (kernel review, 5 November 2026)."""
    head, sep, body = s.partition("oeq_projected_backward_wprime(")
    args, close, rest = body.partition(") {")
    if not sep or not close:
        raise ValueError("restrict: kernel signature not found")
    args = re.sub(r"(\b(?:scalar_t|int32_t|float|int)\*) (?!__restrict__)", r"\1 __restrict__ ", args)
    return head + sep + args + close + rest


def _stage_dout(source, rows, fill=True):
    """The w' kernel with the block's receivers' dout rows staged in shared memory (see ``stage-dout``)."""
    pattern = r"\bdout(\d)\[int64_t\(receiver\) \* (\d+)([^\[\]]*)\]"
    dims = {}
    for n, d, _ in re.findall(pattern, source):
        dims[int(n)] = int(d)
    if not dims:
        raise ValueError("variant pattern found 0 times: dout[int64_t(receiver) * D ...]")
    offsets, total = {}, 0
    for n in sorted(dims):
        offsets[n], total = total, total + dims[n]
    copy = "".join(
        f"        for (int i = threadIdx.x; i < {dims[n]}; i += blockDim.x) fm_d[q][{offsets[n]} + i] = "
        f"dout{n}[int64_t(r) * {dims[n]} + i];\n" for n in sorted(dims))
    stage = f"""    __shared__ scalar_t fm_d[{rows}][{total}];
    const int64_t fm_e0 = int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE);
    int64_t fm_e1 = fm_e0 + blockDim.x / LOGICAL_GROUP_SIZE - 1;
    fm_e1 = fm_e1 < edge_count ? fm_e1 : edge_count - 1;
    const int32_t fm_r0 = receivers[fm_e0];
    const int32_t fm_r1 = {"receivers[fm_e1]" if rows == 2 else "fm_r0"};
"""
    if fill:
        stage += f"""    for (int q = 0; q < {rows}; ++q) {{
        const int32_t r = q == 0 ? fm_r0 : fm_r1;
        if (r < 0 || r >= node_count || (q == 1 && r == fm_r0)) continue;
{copy}    }}
    __syncthreads();
"""
    s = _sub(source, "    if (e >= edge_count) return;", stage + "    if (e >= edge_count) return;", 1)
    # One generic pointer per dout block, chosen once per warp (a three-way select per read spilled: 40-248 bytes on
    # the RTX 5060 Ti at its 80-register cap).
    pointers = "".join(
        f"    const scalar_t* fm_p{n} = receiver == fm_r0 ? &fm_d[0][{offsets[n]}] : receiver == fm_r1 ? "
        f"&fm_d[{rows - 1}][{offsets[n]}] : dout{n} + int64_t(receiver) * {dims[n]};\n" for n in sorted(dims))
    s = _sub(s, "    const int64_t input_base = int64_t(sender) * INPUT_DIM;",
             pointers + "    const int64_t input_base = int64_t(sender) * INPUT_DIM;", 1)
    return re.sub(pattern, lambda m: f"fm_p{m.group(1)}[0 {m.group(3)}]", s)


def _smem_reads(source, keys):
    """The w' kernel with the global reads of ``keys`` (``dout``, ``sh``, ``x``) taken from a shared array."""
    s = _sub(source, "const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;",
             "__shared__ scalar_t fm_s[1024];\n    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;", 1)
    patterns = {"dout": (r"\bdout(\d)\[([^\[\]]*)\]", r"fm_s[(\2) & 1023]"),
                "sh": (r"\bsh\[([^\[\]]*)\]", r"fm_s[(\1) & 1023]"),
                "x": (r"\bx\[([^\[\]]*)\]", r"fm_s[(\1) & 1023]")}
    for key in keys:
        s = _sub(s, *patterns[key], regex=True)
    return s


def plan_options(kernel, extra):
    """``extra`` with the plan kernel's own register cap (``WPrimeBackward.max_registers``, a per-GPU default) unless
    the variant sets one. Lab runs before 5 November 2026 about 13:30 UTC compiled every variant without it: on the
    RTX 5060 Ti (cap 80) their base was the uncapped kernel, 7.88 against the plan's 6.77 ms."""
    cap = int(getattr(kernel, "max_registers", 0) or 0)
    if cap and not any(str(o).startswith("--maxrregcount") for o in extra):
        return (f"--maxrregcount={cap}",) + tuple(extra)
    return tuple(extra)


def wprime_launcher(kernel, source, extra):
    """A launch function for a variant of ``kernel`` (a WPrimeBackward) compiled from ``source``."""
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    extra = plan_options(kernel, extra)
    cubin = compile_cubin(source, torch.cuda.get_device_capability(), extra)
    module = _check(driver.cuModuleLoadData(cubin))
    function = _check(driver.cuModuleGetFunction(module, b"oeq_projected_backward_wprime"))
    registers = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    spills = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                              function))

    def launch(args):
        (x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins, r_lo, bin_scale,
         r_max, dr, *perm) = args
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (edges * kernel.cohort + kernel.num_threads - 1) // kernel.num_threads
        p = ctypes.c_void_p
        douts = list(dout) if isinstance(dout, (list, tuple)) else [dout]  # BLOCK_MESSAGE: one pointer per group
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), *[d.data_ptr() for d in douts], dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr(), *[t.data_ptr() for t in perm])
        types = ((ctypes.c_longlong, ctypes.c_longlong) + (p,) * (13 + len(douts))
                 + (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, p) + (p,) * len(perm))
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(function, blocks, 1, 1, kernel.num_threads, 1, 1, kernel.shared_memory, stream,
                                     (values, types), 0))
    return launch, dict(registers=int(registers), local_bytes=int(spills))


# --------------------------------------------------------------------------------------------------- fused forward
TC_SPLIT_RNA_HI = 'asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(hi) : "f"(x));'
TC_SPLIT_MASK_HI = "hi = __float_as_uint(x) & 0xffffe000u;"
TC_SPLIT_RNA_LO = 'asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(lo) : "f"(rest));'
TC_SPLIT_RAW_LO = "lo = __float_as_uint(rest);"


def _tc_no_wsplit(s):
    """The tensor-core product with W's fragments used unsplit (the high part twice): no split instructions for W."""
    for i, offset in enumerate(("0", "8", "4 * WS", "4 * WS + 8")):
        s = _sub(s, f"split_tf32(wp[{offset}], ah[{i}], al[{i}]);",
                 f"ah[{i}] = __float_as_uint(wp[{offset}]); al[{i}] = ah[{i}];")
    return s


_RULES = {}  # a re-rendered variant's source -> its launch rule


def _block_ldg():
    import torch

    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    return wp.block_ldg_default(torch.device("cuda", torch.cuda.current_device()))


def _tc_rerender(kernel, tc, name, warps=None, prefetch=False, tctp=False):
    """The tensor-core kernel re-rendered in mode ``tc`` (fused_forward.render) with its own launch rule;
    ``warps``: that many warps per block (the launch bound, so more registers per thread with fewer). The plan
    kernel's VEC_SH / VEC_A / TC_WFRAG carry over where the mode allows them; ``tctp``: the TP on tensor cores."""
    def transform(s):
        import torch

        from fastmace.kernels.oeq_streaming import fused_forward as ff
        if not kernel.tc and tc:
            raise ValueError(f"{name} needs the tensor-core kernel (--constant ...:TENSOR_CORES=True)")
        props = torch.cuda.get_device_properties(0)
        rule = ff.launch_rule(kernel.paths, kernel.used if tc else kernel.k, kernel.schedule.edge_dim,
                              kernel.channels, int(props.shared_memory_per_block_optin),
                              int(props.multi_processor_count), tc=tc, prefetch=prefetch,
                              x_slots=kernel.schedule.input_dim // kernel.channels)
        if warps is not None:
            import dataclasses
            per = -(-kernel.paths // rule.parts)
            width = kernel.used if tc else kernel.k
            rule = dataclasses.replace(rule, warps=warps, shared_bytes=4 * ff.slice_floats(per, width, tc) + warps * 4
                                       * ff.staging_floats(width, kernel.schedule.edge_dim, tc, per, prefetch,
                                                           kernel.schedule.input_dim // kernel.channels))
        plain = tc is True and not prefetch
        source = ff.render(kernel.schedule, kernel.k, rule, kernel.store_w, kernel.used, tc=tc,
                           blocks=kernel.blocks, prefetch=prefetch,
                           vec_sh=getattr(kernel, "vec_sh", False) and not prefetch,
                           vec_a=getattr(kernel, "vec_a", False) and plain,
                           wfrag=getattr(kernel, "wfrag", False) and plain, tctp=tctp)
        _RULES[source] = rule
        print(f"{name} rule {rule} (plan {kernel.rule})", flush=True)
        return source
    return transform


def _grid(kernel, blocks_per_sm):
    """The plan's kernel with a persistent grid of ``blocks_per_sm`` blocks per SM over all slices (the source gets
    a comment so that its launch rule has its own key)."""
    def transform(s):
        import dataclasses

        import torch

        sms = int(torch.cuda.get_device_properties(0).multi_processor_count)
        slices = (kernel.channels // 32) * kernel.rule.parts
        source = s + f"\n// lab grid: {blocks_per_sm} blocks per SM\n"
        _RULES[source] = dataclasses.replace(kernel.rule, blocks_per_slice=max(1, -(-blocks_per_sm * sms // slices)))
        print(f"grid{blocks_per_sm} rule {_RULES[source]} (plan {kernel.rule})", flush=True)
        return source
    return transform


def _tc_presplit(kernel):
    """The tensor-core kernel re-rendered with the W slice split once per block (fused_forward.TC_PRESPLIT): bitwise
    the plan's tensor-core kernel, its own launch rule (twice the slice's shared memory)."""
    def transform(s):
        import torch

        from fastmace.kernels.oeq_streaming import fused_forward as ff
        if not kernel.tc:
            raise ValueError("tc-presplit needs the tensor-core kernel (--constant ...:TENSOR_CORES=True)")
        props = torch.cuda.get_device_properties(0)
        rule = ff.launch_rule(kernel.paths, kernel.used, kernel.schedule.edge_dim, kernel.channels,
                              int(props.shared_memory_per_block_optin), int(props.multi_processor_count),
                              tc="presplit")
        source = ff.render(kernel.schedule, kernel.k, rule, kernel.store_w, kernel.used, tc="presplit")
        _RULES[source] = rule
        print(f"tc-presplit rule {rule} (plan {kernel.rule})", flush=True)
        return source
    return transform


def _tc_wfrag_split(kernel):
    """The plan's wfrag kernel re-rendered with its W slice stored split (fused_forward.TC_WFRAG_SPLIT): bitwise, its
    own launch rule (the presplit slice size), the plan's parts, staging and flags otherwise."""
    def transform(s):
        import torch

        from fastmace.kernels.oeq_streaming import fused_forward as ff
        if not getattr(kernel, "wfrag", False):
            raise ValueError("tc-wfrag-split needs the plan's wfrag kernel (TC_WFRAG)")
        props = torch.cuda.get_device_properties(0)
        shape = (kernel.paths, kernel.used, kernel.schedule.edge_dim, kernel.channels,
                 int(props.shared_memory_per_block_optin), int(props.multi_processor_count))
        options = dict(tc="presplit", prefetch=False, x_slots=kernel.schedule.input_dim // kernel.schedule.channels)
        groups = kernel.groups
        rule = ff.launch_rule(*shape, groups=groups, **options) if groups else None
        if rule is None:
            groups, rule = None, ff.launch_rule(*shape, **options)
        source = ff.render(kernel.schedule, kernel.k, rule, kernel.store_w, kernel.used, tc="presplit",
                           blocks=kernel.blocks, prefetch=False, vec_sh=kernel.vec_sh, vec_a=kernel.vec_a, wfrag=True,
                           groups=groups)
        _RULES[source] = rule
        print(f"tc-wfrag-split rule {rule} (plan {kernel.rule})", flush=True)
        return source
    return transform


def _w_group(kernel):
    """The plan's wfrag kernel re-rendered with warp pairs sharing each W fragment (fused_forward.W_GROUP, idea 17 at
    G = 2): bitwise, the plan's launch rule, parts, staging and flags."""
    def transform(s):
        from fastmace.kernels.oeq_streaming import fused_forward as ff
        if not getattr(kernel, "wfrag", False) or kernel.tc is not True or kernel.tctp or kernel.wgmma:
            raise ValueError("w-group needs the plan's wfrag kernel (TC_WFRAG, not presplit, TC_TP or WGMMA)")
        source = ff.render(kernel.schedule, kernel.k, kernel.rule, kernel.store_w, kernel.used, tc=True,
                           blocks=kernel.blocks, prefetch=False, vec_sh=kernel.vec_sh, vec_a=kernel.vec_a, wfrag=True,
                           groups=kernel.groups, sh_lds4=kernel.sh_lds4, lane_idx=kernel.lane_idx, w_group=True)
        print(f"w-group rule {kernel.rule} (the plan's)", flush=True)
        return source
    return transform


def _ns_slab(s):
    """No store, and w from a 6-term slab with float2 loads instead of the a.W product (wprime_backward.TABLE_W in the
    forward): w[g][b] = sum_k a_k G_k, the a row's first 6 values standing in for the Chebyshev basis and W's own
    buffer for the table (three groups by edge index: an L2-resident timing bound, not the real values)."""
    return _ns_slab_body(_sub(s, "constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;", 1))


def _ns_slab_body(s):
    def slab(m):
        npl = int(m.group(1))
        return m.group(0) + f"""
                {{
                    const float2* slab = reinterpret_cast<const float2*>(W);
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {{
                        const int grp = (int)((e0 + b) % 3);
                        const float t0 = a_s[0 * B + b], t1 = a_s[1 * B + b], t2 = a_s[2 * B + b];
                        const float t3 = a_s[3 * B + b], t4 = a_s[4 * B + b], t5 = a_s[5 * B + b];
                        #pragma unroll
                        for (int g = 0; g < {npl}; ++g) {{
                            const float2* q = slab + (grp * 1280 + g * 128 + lane) * 3;
                            const float2 v0 = __ldg(q), v1 = __ldg(q + 1), v2 = __ldg(q + 2);
                            w[g][b] = t0 * v0.x + t1 * v0.y + t2 * v1.x + t3 * v1.y + t4 * v2.x + t5 * v2.y;
                        }}
                    }}
                }}"""
    s = _sub(s, r"float w\[(\d+)\]\[B\];\s*#pragma unroll\s*for \(int g = 0; g < \d+; \+\+g\)\s*#pragma unroll\s*"
                r"for \(int b = 0; b < B; \+\+b\) w\[g\]\[b\] = 0\.f;", slab, regex=True)
    return _sub(s, "for (int k = 0; k < K; ++k) {", "for (int k = 0; k < 0; ++k) {")


SLAB_GROUPS = 5340  # round 5: the value table's slabs (164 MB of 1,280 x 6 floats, Benchmark_loup interaction 2)


def _slab_real(kernel, groups):
    """Round 5 (5 November 2026): ``NS-slab`` on the FP32-SIMT kernel with ``groups`` slabs of 1,280 x 6 floats
    (``_slab_args`` passes a buffer of that size as W), each edge on a hashed slab: the forward evaluating w from
    the value table (wprime_backward.TABLE_W) with the table's real footprint, a timing bound."""
    def transform(s):
        s = _tc_rerender(kernel, False, f"simt-slab{groups}")(s)
        s = s.replace("constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;")
        s = _ns_slab_body(s)
        return _sub(s, "const int grp = (int)((e0 + b) % 3);",
                    f"const int grp = (int)(((unsigned)(e0 + b) * 2654435761u) % {groups}u);")
    return transform


_WPRIME_CAPTURE = {}  # main() puts the captured w' call's arguments here (the real slabs of simt-slab-edges)


def _edge_slabs():
    """Per edge: pair * pieces + piece of the value table, as the w' kernel computes it; the q-major table."""
    import torch
    a = _WPRIME_CAPTURE["args"]
    r, pair, g, starts, bins, r_lo, bin_scale = a[8], a[9], a[10], a[11], a[13], a[14], a[15]
    pieces = starts.numel()
    clamped = torch.clamp(r.float(), min=float(r_lo))
    b = torch.clamp(((clamped - float(r_lo)) * float(bin_scale)).int(), max=bins.numel() - 1).long()
    slabs = (pair.long() * pieces + bins[b].long()).int()
    return slabs, g


def _slab_edges(kernel, tensor_cores):
    def transform(s):
        s = _tc_rerender(kernel, tensor_cores, "slab-edges")(s)
        s = s.replace("constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;")
        s = _ns_slab_body(s)
        slabs, g = _edge_slabs()
        offset = g.numel()
        return _sub(s, "const int grp = (int)((e0 + b) % 3);",
                    f"const int grp = reinterpret_cast<const int*>(W)[{offset}LL + (long long)min((int)(e0 + b), {slabs.numel() - 1})];")
    return transform


def _slab_edges_args(args):
    import torch
    a, weight, *rest = args
    slabs, g = _edge_slabs()
    if g.dim() != 3 or g.shape[1] != 1280:
        raise ValueError(f"simt-slab-edges expects a q-major [groups, 1280, terms] table, got {tuple(g.shape)}")
    big = torch.cat([g.reshape(-1).float(), slabs.view(torch.float32)]).contiguous()
    print(f"slab-edges: {g.shape[0]} slabs, {g.numel() * 4 / 1e6:.0f} MB table, "
          f"{torch.unique(slabs).numel()} slabs touched by {slabs.numel()} edges "
          f"({torch.unique(slabs).numel() * 1280 * 6 * 4 / 1e6:.1f} MB)", flush=True)
    return (a, big, *rest)


def _slab_args(groups):
    def transform(args):
        import torch
        a, weight, *rest = args
        big = torch.full((groups * 1280 * 6 + weight.numel(),), 0.01, dtype=weight.dtype, device=weight.device)
        big[:weight.numel()] = weight.reshape(-1)
        return (a, big, *rest)
    return transform


A4_STRIDE = 68  # tc-a4: a's padded row stride (a multiple of 4 floats: 16-byte rows)


def _tc_a4(s):
    """The tensor-core kernel with a staged by float4 loads from rows padded to A4_STRIDE floats (the argument
    transform ``_a4_args``) into a batch-major tile, XOR-swizzled by 4-float group so the fragment reads stay
    conflict-free (bank (group ^ edge) * 4 + tq), and sh by float4 loads. Bitwise: the same values reach the product.
    Needs K == KP (every staged column used) and ED % 4 == 0."""
    k = int(re.search(r"constexpr int K = (\d+);", s).group(1))
    kp = int(re.search(r"constexpr int KP = (\d+);", s).group(1))
    ed = int(re.search(r"constexpr int ED = (\d+);", s).group(1))
    if k != kp or ed % 4:
        raise ValueError(f"tc-a4 needs K == KP and ED % 4 == 0 (K {k}, KP {kp}, ED {ed})")
    s = _sub(s, r"constexpr int AS = \d+;", f"constexpr int AS = {A4_STRIDE};", 1, regex=True)
    s = _sub(s, """                for (int i = lane; i < B * KP; i += 32) {
                    const int b = i / KP, k = i - b * KP;
                    a_s[k * B + b] = (b < nb && k < K) ? a[(e0 + b) * KA + k] : 0.f;
                }""", """                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);
                    *reinterpret_cast<float4*>(&a_s[b * KP + ((q ^ b) << 2)]) = v;
                }""")
    s = _sub(s, "split_tf32(a_s[(k0 + tq) * B + gid], bh[0], bl[0]);",
             "split_tf32(a_s[gid * KP + (((k0 >> 2) ^ gid) << 2) + tq], bh[0], bl[0]);")
    s = _sub(s, "split_tf32(a_s[(k0 + tq + 4) * B + gid], bh[1], bl[1]);",
             "split_tf32(a_s[gid * KP + ((((k0 >> 2) + 1) ^ gid) << 2) + tq], bh[1], bl[1]);")
    return _sub(s, "for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;",
                """for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i)
                        : make_float4(0.f, 0.f, 0.f, 0.f);""")


def _simt_a4(s):
    """The FP32-SIMT kernel with a staged by float4 loads from rows padded to A4_STRIDE floats (``_a4_args``),
    scattered into the k-major tile as before. Bitwise."""
    k = int(re.search(r"constexpr int K = (\d+);", s).group(1))
    if -(-k // 4) * 4 > A4_STRIDE:
        raise ValueError(f"simt-a4: K {k} exceeds the padded stride {A4_STRIDE}")
    s = _sub(s, r"constexpr int AS = \d+;", f"constexpr int AS = {A4_STRIDE};\nconstexpr int KQ = (K + 3) / 4;", 1,
             regex=True)
    return _sub(s, """                for (int i = lane; i < B * K; i += 32) {
                    const int b = i / K, k = i - b * K;
                    a_s[k * B + b] = (b < nb) ? a[(e0 + b) * KA + k] : 0.f;
                }""", """                for (int i = lane; i < B * KQ; i += 32) {
                    const int b = i / KQ, q = i - b * KQ, k = 4 * q;
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);
                    a_s[k * B + b] = v.x;
                    if (k + 1 < K) a_s[(k + 1) * B + b] = v.y;
                    if (k + 2 < K) a_s[(k + 2) * B + b] = v.z;
                    if (k + 3 < K) a_s[(k + 3) * B + b] = v.w;
                }""")


def _sh4(s):
    """sh staged by float4 loads (fused_forward.VEC_SH). Bitwise."""
    return _sub(s, "for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;",
                """for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i)
                        : make_float4(0.f, 0.f, 0.f, 0.f);""")


CACHE_HINTS = r"""
__device__ __forceinline__ float ld_x_el(const float* p) {  // x: kept in L1 (re-read by the next edges' senders)
    float v;
    asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float4 ld_na4(const float4* p) {  // a and sh: read once, not allocated in L1
    float4 v;
    asm("ld.global.nc.L1::no_allocate.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
"""


def _hints(s, x=None, stream=False):
    """Cache hints (round 4, after the x tile did not fit the A100's shared memory): ``x`` "ldg" (``__ldg``) or "el"
    (L1 evict_last) on the sender-feature gathers; ``stream`` the float4 a / sh loads with L1::no_allocate, so L1
    holds x. Bitwise (the same values)."""
    s = _sub(s, 'extern "C" __global__', CACHE_HINTS + 'extern "C" __global__', 1)
    if x == "ldg":
        s = _sub(s, r"\(snd >= 0\) \? x\[([^\]]*)\]", r"(snd >= 0) ? __ldg(&x[\1])", regex=True)
    elif x == "el":
        s = _sub(s, r"\(snd >= 0\) \? x\[([^\]]*)\]", r"(snd >= 0) ? ld_x_el(&x[\1])", regex=True)
    if stream:
        s = _sub(s, "__ldg(reinterpret_cast<const float4*>(", "ld_na4(reinterpret_cast<const float4*>(")
    return s


def _wfrag(s):
    """The tensor-core W slice stored in mma.sync fragment order, [npl][KP / 8][2][32 lanes][4], so each lane loads its
    four W values of a (path, k-step, m-tile) with one float4 shared load (conflict-free: consecutive 16 B per lane)
    instead of four scalar loads. The same values reach the same mma: bitwise. Needs CB == 32 (two m-tiles)."""
    if "constexpr int CB = 32;" not in s:
        raise ValueError("wfrag needs CB == 32")
    s = _sub(s, r"""                Ws\[\(p \* KP \+ k\) \* WS \+ l\] = (k < K \? W\[[^;]*\] : 0\.f);""",
             r"""                const int fm = l >> 4, fr = l & 15, fk = k & 7;
                Ws[((p * (KP / 8) + (k >> 3)) * 2 + fm) * 128 + ((fr & 7) * 4 + (fk & 3)) * 4
                   + (fr >> 3) + 2 * (fk >> 2)] = \1;""", regex=True)
    return _sub(s, """                                const float* wp = Ws + (g * KP + k0 + tq) * WS + m * 16 + gid;
                                unsigned ah[4], al[4];
                                split_tf32(wp[0], ah[0], al[0]);
                                split_tf32(wp[8], ah[1], al[1]);
                                split_tf32(wp[4 * WS], ah[2], al[2]);
                                split_tf32(wp[4 * WS + 8], ah[3], al[3]);""",
                """                                const float4 wv = reinterpret_cast<const float4*>(Ws)[
                                    ((g * (KP / 8) + (k0 >> 3)) * 2 + m) * 32 + lane];
                                unsigned ah[4], al[4];
                                split_tf32(wv.x, ah[0], al[0]);
                                split_tf32(wv.y, ah[1], al[1]);
                                split_tf32(wv.z, ah[2], al[2]);
                                split_tf32(wv.w, ah[3], al[3]);""")


def _s_tile(s):
    """Falsifier of the H200 ideas plan's idea 4 (an S tile: CG coefficient times edge harmonic summed per output and
    input component once per edge, shared by the 32 channel lanes): each TP output becomes one product per distinct
    input component, ``xv[b][i] * S``, with S read from shared memory at a distinct address per (output, input) pair
    (outputs of a single term unchanged).
    The S tile's own cooperative computation is not included (a lower bound of the per-lane TP cost)."""
    line = re.compile(r"acc\[(\d+)\] = fmaf\(w\[(\d+)\]\[b\], (.*), acc\[\1\]\);")
    term = re.compile(r"xv\[b\]\[(\d+)\] \* e\d+(?: \* -?0x[0-9a-f.]+p[-+]\d+f)?")
    out, k = [], 0
    for text in s.split("\n"):
        m = line.search(text)
        if m:
            if m.group(1) == "0":
                k = 0
            found = term.findall(m.group(3))
            inputs = sorted({int(i) for i in found})
            if len(found) == 1:  # a single term reads its harmonic as before
                out.append(text)
                continue
            if not inputs:
                raise ValueError(f"s-tile: no terms in {text.strip()[:80]!r}")
            parts = []
            for i in inputs:
                parts.append(f"xv[b][{i}] * sh_s[b * ED + {k}]")
                k += 1
            text = text[:m.start(3)] + " + ".join(parts) + text[m.end(3):]
        out.append(text)
    return "\n".join(out)


def _xs(kernel):
    """The sender features x kept in a per-warp shared tile [B][NX][32] (after the plan's shared memory) instead of
    32 registers live across the product; each lane writes and reads its own column (no sync). Bitwise. Own launch
    rule (more shared memory, the same warps)."""
    def transform(s):
        import dataclasses

        nx = max(int(v) for v in re.findall(r"float xv\[B\]\[(\d+)\];", s))
        rule = kernel.rule
        base = rule.shared_bytes // 4
        tile = BATCH_LAB * nx * 32
        s = _sub(s, "    const int cb = blockIdx.y;\n", f"    constexpr int NX = {nx};\n"
                 f"    float* x_s = smem + {base} + warp * {tile};  // the warp's x tile [B][NX][32]\n"
                 "    const int cb = blockIdx.y;\n", 1)
        s = _sub(s, r"xv\[b\]\[(\d+)\] = \(snd >= 0\) \? (x\[[^\]]*\]) : 0\.f;",
                 r"x_s[(b * NX + \1) * 32 + lane] = (snd >= 0) ? \2 : 0.f;", regex=True)
        s = _sub(s, r"xv\[b\]\[(\d+)\]", r"x_s[(b * NX + \1) * 32 + lane]", regex=True)
        s = _sub(s, r"float xv\[B\]\[\d+\];", "", regex=True)
        _RULES[s] = dataclasses.replace(rule, shared_bytes=rule.shared_bytes + 4 * tile * rule.warps)
        return s
    return transform


BATCH_LAB = 8  # fused_forward.BATCH


def _a4_args(args):
    import torch

    a, *rest = args
    padded = torch.nn.functional.pad(a, (0, A4_STRIDE - a.shape[1])).contiguous()
    return (padded, *rest)


def _tc_split_accumulators(source, count):
    """The 3xTF32 projection with the small products in their own accumulator(s): ``count`` 2: al*bh and ah*bl into
    wacc1; 3: al*bh into wacc1, ah*bl into wacc2. Each is added to wacc before the redistribution."""
    decl = re.search(r"float wacc\[(\d+)\]\[2\]\[4\];", source)
    if decl is None:
        raise ValueError("variant pattern found 0 times: float wacc[..][2][4];")
    shape = f"[{decl.group(1)}][2][4]"
    extra = ["wacc1"] + (["wacc2"] if count == 3 else [])
    source = _sub(source, decl.group(0), decl.group(0) + "".join(f" float {a}{shape};" for a in extra))
    source = _sub(source, "wacc[g][m][i] = 0.f;",
                  "{ wacc[g][m][i] = 0.f;" + "".join(f" {a}[g][m][i] = 0.f;" for a in extra) + " }")
    source = _sub(source, "mma_tf32(wacc[g][m], al, bh);", "mma_tf32(wacc1[g][m], al, bh);")
    source = _sub(source, "mma_tf32(wacc[g][m], ah, bl);", f"mma_tf32({extra[-1]}[g][m], ah, bl);")
    total = " + ".join(f"{a}[g][m][i]" for a in extra)
    return _sub(source, "float* rp = r_s + (g * B + 2 * tq) * RS + m * 16 + gid;",
                f"#pragma unroll\n for (int i = 0; i < 4; ++i) wacc[g][m][i] += {total};\n"
                "                            float* rp = r_s + (g * B + 2 * tq) * RS + m * 16 + gid;")


HYBRID16_HELPERS = """// TF32 + FP16 hybrid (round 5 lab): x = hi + lo as for 3xTF32; the small products in FP16 m16n8k16.
__device__ __forceinline__ unsigned pack_h2(float k_low, float k_high) {
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(k_high), "f"(k_low));
    return r;
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
"""

HYBRID16_LOOP = """#pragma unroll 1
                    for (int k0 = 0; k0 < KP; k0 += 16) {  // two TF32 k-steps: the k16 FP16 MMA's slots
                        // slot 2tq <- k0 + 8s + tq, slot 2tq + 1 <- k0 + 8s + tq + 4 (the same in A and B: exact)
                        unsigned bs[2][2], bh16[2], bl16[2];
                        #pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            float hi[2], lo[2];
                            #pragma unroll
                            for (int j = 0; j < 2; ++j) {
                                const float v = a_s[gid * KP + ((((k0 >> 2) + 2 * s + j) ^ gid) << 2) + tq];
                                hi[j] = __uint_as_float(__float_as_uint(v) & 0xffffe000u);
                                lo[j] = (v - hi[j]) * 2048.f;
                                bs[s][j] = __float_as_uint(hi[j] * 2048.f);  // TF32-exact: the main product x 2^11
                            }
                            bh16[s] = pack_h2(hi[0], hi[1]);
                            bl16[s] = pack_h2(lo[0], lo[1]);
                        }
                        #pragma unroll
                        for (int g = 0; g < NPL_H; ++g)
                            #pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                const float4 w0 = reinterpret_cast<const float4*>(Ws)[
                                    ((g * (KP / 8) + (k0 >> 3)) * 2 + m) * 32 + lane];
                                const float4 w1 = reinterpret_cast<const float4*>(Ws)[
                                    ((g * (KP / 8) + (k0 >> 3) + 1) * 2 + m) * 32 + lane];
                                const float wv[2][4] = {{w0.x, w0.y, w0.z, w0.w}, {w1.x, w1.y, w1.z, w1.w}};
                                float wh[2][4], wl[2][4];
                                unsigned ah[2][4];
                                #pragma unroll
                                for (int s = 0; s < 2; ++s)
                                    #pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        ah[s][i] = __float_as_uint(wv[s][i]) & 0xffffe000u;
                                        wh[s][i] = __uint_as_float(ah[s][i]);
                                        wl[s][i] = (wv[s][i] - wh[s][i]) * 2048.f;
                                    }
                                // A rows gid (x, z) and gid + 8 (y, w); k slots as in B
                                const unsigned alo[4] = {pack_h2(wl[0][0], wl[0][2]), pack_h2(wl[0][1], wl[0][3]),
                                                         pack_h2(wl[1][0], wl[1][2]), pack_h2(wl[1][1], wl[1][3])};
                                const unsigned ahi[4] = {pack_h2(wh[0][0], wh[0][2]), pack_h2(wh[0][1], wh[0][3]),
                                                         pack_h2(wh[1][0], wh[1][2]), pack_h2(wh[1][1], wh[1][3])};
                                mma_f16(wacc[g][m], alo, bh16);  // 2^11 W_lo a_hi
                                mma_f16(wacc[g][m], ahi, bl16);  // 2^11 W_hi a_lo
                                mma_tf32(wacc[g][m], ah[0], bs[0]);  // 2^11 W_hi a_hi
                                mma_tf32(wacc[g][m], ah[1], bs[1]);
                            }
                    }
"""


def _tc_hybrid16(source):
    """The 3xTF32 projection as TF32 big*big plus FP16 m16n8k16 for both small products (lab, round 5): 4 MMAs per
    16 k instead of 6, the FP16 MMA at twice the TF32 rate. Every product carries 2^11 (exact), undone on the store.
    Needs TC_WFRAG's float4 W fragments and VEC_A's staging (the A100 default)."""
    start = "#pragma unroll 2\n                    for (int k0 = 0; k0 < KP; k0 += 8) {"
    end = "                    #pragma unroll\n                    for (int g = 0; g < "
    npl = re.search(r"float wacc\[(\d+)\]\[2\]\[4\];", source)
    if npl is None or start not in source or "const float4 wv = reinterpret_cast<const float4*>(Ws)[" not in source:
        raise ValueError("variant pattern found 0 times: the TC_WFRAG product loop")
    kp = int(re.search(r"constexpr int KP = (\d+);", source).group(1))
    if kp % 16:
        raise ValueError(f"tc-hybrid16 needs KP % 16 == 0 (KP = {kp}; SKIP_ZERO_COLUMN gives 64 for OMAT-0)")
    loop = HYBRID16_LOOP.replace("NPL_H", npl.group(1))
    out, pos = [], 0
    while True:
        i = source.find(start, pos)
        if i < 0:
            break
        j = source.index(end, i)
        out += [source[pos:i], loop]
        pos = j
    source = "".join(out) + source[pos:]
    for k, v in (("rp[0] = wacc[g][m][0];", "rp[0] = wacc[g][m][0] * 0x1p-11f;"),
                 ("rp[RS] = wacc[g][m][1];", "rp[RS] = wacc[g][m][1] * 0x1p-11f;"),
                 ("rp[8] = wacc[g][m][2];", "rp[8] = wacc[g][m][2] * 0x1p-11f;"),
                 ("rp[RS + 8] = wacc[g][m][3];", "rp[RS + 8] = wacc[g][m][3] * 0x1p-11f;")):
        source = _sub(source, k, v)
    return _after_constants(source, HYBRID16_HELPERS)


# ------------------------------------------------------------------- H200 ideas plan, revision 7 (6 November 2026)
def _then(first, *after):
    """``first``, then the source edits ``after``; a launch rule ``first`` registered (``_RULES``) carries over."""
    def transform(s):
        out = first(s)
        rule = _RULES.get(out)
        for edit in after:
            out = edit(out)
        if rule is not None:
            _RULES[out] = rule
        return out
    return transform


def _store_w_on(s):
    """STORE_W on (w stored for every edge; unchanged where the plan stores it already)."""
    return s.replace("constexpr bool STORE_W = false;", "constexpr bool STORE_W = true;", 1)


def _split_mode(mode):
    """The source with ``split_tf32``'s body set to ``fused_forward._TC_SPLITS[mode]`` ("mask", "round", "rna")."""
    def edit(s):
        from fastmace.kernels.oeq_streaming import fused_forward as ff
        body = ff._TC_SPLITS[mode]
        return _sub(s, r"(?s)(void split_tf32\(float x, unsigned& hi, unsigned& lo\) \{\n).*?(\n\})",
                    lambda m: m.group(1) + body + m.group(2), count=1, regex=True)
    return edit


B1_HELPERS = """// Idea 20 B1 (lab): W rounded to 22 significant bits (to nearest even) as the slice is filled,
// so the mask split keeps it exactly; a split with its high part rounded to nearest and the low part the raw
// remainder (random sign).
__device__ __forceinline__ float round22(float x) {
    const unsigned u = __float_as_uint(x);
    return __uint_as_float((u + 1u + ((u >> 2) & 1u)) & ~3u);
}
__device__ __forceinline__ void split_tf32_a(float x, unsigned& hi, unsigned& lo) {
    hi = (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
    lo = __float_as_uint(x - __uint_as_float(hi));
}
"""


def _b1(s):
    """Idea 20's B1: the mask split for W, whose slice is filled rounded to 22 bits (``round22``), and a split with a
    rounded high part for a (``split_tf32_a``; one integer add per a value). Rounding class."""
    s = _split_mode("mask")(s)
    s = _sub(s, r"= k < K \? (W\[[^;]*?\]) : 0\.f;", r"= k < K ? round22(\1) : 0.f;", regex=True)
    s = _sub(s, r"split_tf32\((?=(?:at|a_s)\[)", "split_tf32_a(", regex=True)
    return _after_constants(s, B1_HELPERS)


_W_BUFFER = {}  # (edges, weight_dim) -> one [E, Q] w buffer the revision-7 variants share (the plan may store no w)


def _shared_w_args(kernel):
    """A w buffer [E, Q] shared by the variants that store w (``poison`` zeroes it before each); the plan's own
    buffer where the plan stores w."""
    def transform(args):
        import torch
        args = list(args)
        edges = args[3].shape[0]
        if tuple(args[8].shape) == (edges, kernel.weight_dim):
            return args
        key = (edges, kernel.weight_dim)
        if key not in _W_BUFFER:
            _W_BUFFER.clear()
            _W_BUFFER[key] = torch.zeros(edges, kernel.weight_dim, device=args[3].device, dtype=args[3].dtype)
        args[8] = _W_BUFFER[key]
        return args
    return transform


_W_STORE = "wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];"


def _stream_store(s):
    """w stored with ``__stcs`` (evict-first; ``render(..., stream_store=True)``, ``STREAM_STORE_W``)."""
    return _sub(s, _W_STORE, "__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);")


def _store_v4(s):
    """14(A)'s pass-1 stores (``store-w-v4``): after the r_s writes, each lane reads four channels of one (path, edge)
    row with an LDS.128 and stores them with an STG.128 (2 npl LDS.128 and STG.128 per batch instead of 8 npl
    STG.32). Bitwise: the same values. Needs the tensor-core product's r_s and B = 8."""
    if "constexpr int B = 8;" not in s or "w[g][b] = r_s[(g * B + b) * RS + lane];" not in s:
        raise ValueError("store-w-v4 needs the tensor-core product's r_s and B = 8")
    pattern = (r"(const int wst\[(\d+)\] = \{[^}]*\};\n)(\s*)#pragma unroll\n\s*for \(int b = 0; b < B; \+\+b\)\n"
               r"\s*if \(b < nb\) \{\n\s*#pragma unroll\n\s*for \(int g = 0; g < \2; \+\+g\) "
               + re.escape(_W_STORE) + r"\n\s*\}")

    def v4(m):
        i = m.group(3)
        return (f"{m.group(1)}{i}#pragma unroll\n"
                f"{i}for (int it = 0; it < 2 * {m.group(2)}; ++it) {{  // store-w-v4: rows (g, b), 4 per step\n"
                f"{i}    const int g = it >> 1, b = (it & 1) * 4 + (lane >> 3), q = lane & 7;\n"
                f"{i}    const float4 v = *reinterpret_cast<const float4*>(&r_s[(g * B + b) * RS + 4 * q]);\n"
                f"{i}    if (b < nb) *reinterpret_cast<float4*>(&wout[(e0 + b) * Q + wst[g] + cb * CB + 4 * q]) = v;\n"
                f"{i}}}")
    return _sub(s, pattern, v4, regex=True)


def _w_from_buffer(source):
    """F-bias's message reference: the plan's kernel with w read from ``wout`` (the float64 product rounded to float32,
    filled by ``main``) instead of the product's r_s, STORE_W off. The TP and its sums are the plan's."""
    old = "w[g][b] = r_s[(g * B + b) * RS + lane];"
    if old not in source:
        raise ValueError("F-bias message reference: needs the tensor-core product's r_s")
    out = []
    for block in re.split(r"(?=\n    case \d+: \{)", source):
        m = re.search(r"const int wst\[(\d+)\] = (\{[^}]*\});", block)
        if m and old in block:
            block = _sub(block, f"float w[{m.group(1)}][B];",
                         f"const int wsl[{m.group(1)}] = {m.group(2)};\n                float w[{m.group(1)}][B];", 1)
            block = block.replace(old, "w[g][b] = (b < nb) ? __ldg(wout + (e0 + b) * Q + wsl[g] + ch) : 0.f;")
        out.append(block)
    return "".join(out).replace("constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;", 1)


def _wg_own_a(offset):
    """Idea 17's a path on the W_GROUP render (timing bounds, wrong w): each warp reads its own a_s for the partner's
    n-tile too. ``offset`` "0": the same address, so the compiler may drop the duplicated loads and splits (the
    partner's a loads disappear); "1": its own slice one float on (the same instructions, no reads from the
    partner's slice)."""
    def transform(s):
        old = "const float* at = a_s + (t - gr) * A_STRIDE;"
        if "group_sync(gbar);" not in s:
            raise ValueError("F-wg-own-a needs the W_GROUP render (fused_forward.W_GROUP)")
        return _sub(s, old, "const float* at = a_s;" if offset == "0" else "const float* at = a_s + (t != gr);")
    return transform


def _wg_no_bar(s):
    """Idea 17's a path: the W_GROUP render without its group barriers (timing bound, wrong w: the partner's a_s and
    r_s race). A done warp leaves at once."""
    if "group_sync(gbar);" not in s:
        raise ValueError("F-wg-no-bar needs the W_GROUP render (fused_forward.W_GROUP)")
    s = _sub(s, "group_sync(gbar);  // the group's r_s rows written", "")
    return _sub(s, "if (!group_any(!done, gbar)) break;", "if (done) break;")


A100_SXM4_OPTIN = 166912  # bytes of opt-in shared memory per block (163 KB of the 164 KB per SM)


def _wfrag_region(kernel, name, warps=12, w_group=False):
    """Idea 21: the plan's wfrag kernel re-rendered with the W region sized by the fragment layout
    (``fused_forward.slice_floats(..., wfrag=True)``) and launched at ``warps`` warps (``w_group``: with W_GROUP;
    None: as many as fit here, ``launch_rule(..., wfrag=True)``, even with W_GROUP). Bitwise. Its rule's shared
    bytes are printed against this device's opt-in and the A100-SXM4's."""
    def transform(s):
        import dataclasses

        import torch

        from fastmace.kernels.oeq_streaming import fused_forward as ff
        if not getattr(kernel, "wfrag", False) or kernel.tc is not True or kernel.tctp or kernel.wgmma:
            raise ValueError(f"{name} needs the plan's wfrag kernel (TC_WFRAG, not presplit, TC_TP or WGMMA)")
        props = torch.cuda.get_device_properties(0)
        optin = int(props.shared_memory_per_block_optin)
        ed, x_slots = kernel.schedule.edge_dim, kernel.schedule.input_dim // kernel.channels
        shape = (kernel.paths, kernel.used, ed, kernel.channels, optin, int(props.multi_processor_count))
        options = dict(tc=True, prefetch=False, x_slots=x_slots, wfrag=True)
        groups = kernel.groups
        rule = ff.launch_rule(*shape, groups=groups, **options) if groups else None
        if rule is None:
            groups, rule = None, ff.launch_rule(*shape, **options)
        per = max(len(g) for g in groups) if groups else -(-kernel.paths // rule.parts)
        wbytes = 4 * ff.slice_floats(per, kernel.used, True, wfrag=True)
        staging = 4 * ff.staging_floats(kernel.used, ed, True, per, False, x_slots)
        count = warps if warps is not None else rule.warps - (rule.warps % 2 if w_group else 0)
        rule = dataclasses.replace(rule, warps=count, shared_bytes=wbytes + count * staging)
        source = ff.render(kernel.schedule, kernel.k, rule, kernel.store_w, kernel.used, tc=True, blocks=kernel.blocks,
                           prefetch=False, vec_sh=kernel.vec_sh, vec_a=kernel.vec_a, wfrag=True, groups=groups,
                           sh_lds4=kernel.sh_lds4, lane_idx=kernel.lane_idx, w_group=w_group, wfrag_region=True)
        _RULES[source] = rule
        print(f"{name} rule {rule} (plan {kernel.rule}): W region {wbytes} B + {count} x {staging} B staging = "
              f"{rule.shared_bytes} B; opt-in here {optin} B, A100-SXM4 {A100_SXM4_OPTIN} B "
              f"({'fits' if rule.shared_bytes <= A100_SXM4_OPTIN else 'does NOT fit'} there)", flush=True)
        return source
    return transform


def _part_only(part):
    """Idea 25's falsifier: only part ``part``'s blocks work (the others return at once); wrong outputs for the
    other parts. Times one part's phase of the forward."""
    def transform(s):
        anchor = "    const int cb = blockIdx.y;\n"
        return _sub(s, anchor, f"    if (blockIdx.z != {part}) return;\n" + anchor, 1)
    return transform


def _l1_pad(kernel, total):
    """F-l1-28 (idea 22(ii)): the plan's kernel with its dynamic shared request padded to ``total`` bytes, so the SM
    takes the largest carve-out and keeps less L1 (224 KB: 28 KB of L1 on the H200). Bitwise."""
    def transform(s):
        import dataclasses
        source = s + f"\n// lab shared request padded to {total} B\n"
        _RULES[source] = dataclasses.replace(kernel.rule, shared_bytes=max(total, kernel.rule.shared_bytes))
        print(f"l1-pad rule {_RULES[source]} (plan {kernel.rule})", flush=True)
        return source
    return transform


_NO_XLOAD = (r"\(snd >= 0\) \? x\[[^\]]*\] : 0\.f", "(snd >= 0) ? 0.5f : 0.f")


def _no_xload(s):
    return _sub(s, *_NO_XLOAD, regex=True)


# F-bias (idea 20): the signed mean error of w against the float64 product, and of the messages against the plan's
# TP fed that product rounded to float32. Relative measures over the real edges (those in the CSR rows) and all Q:
# slope = sum(err ref) / sum(ref^2) (the plan's bias, the relative shrink: negative toward zero),
# toward0 = mean(err sign(ref)) / mean|ref|, mean = mean(err) / mean|ref| (the plain signed mean),
# rms = rms(err) / rms(ref).
FORWARD_REFERENCES = {"store-w-cs": "store-w", "store-w-v4": "store-w"}  # variant -> the variant it must match


class _Moments:
    def __init__(self):
        self.er = self.rr = self.es = self.ar = self.ee = self.e = 0.0
        self.n = 0

    def add(self, err, ref):
        self.er += float((err * ref).sum())
        self.rr += float((ref * ref).sum())
        self.es += float((err * ref.sign()).sum())
        self.ar += float(ref.abs().sum())
        self.ee += float((err * err).sum())
        self.e += float(err.sum())
        self.n += ref.numel()

    def result(self):
        if not self.n or not self.rr:
            return dict(n=self.n)
        return dict(slope=self.er / self.rr, toward0=self.es / self.ar, mean=self.e / self.ar,
                    rms=(self.ee / self.rr) ** 0.5, n=self.n)


def _real_edges(args):
    return max(0, min(int(args[6][-1]), int(args[3].shape[0])))


def w_reference(kernel, args, rows):
    """The float64 product ``a @ W.T`` over the used columns, edges ``rows`` (a slice)."""
    used = kernel.used
    return args[0][rows, :used].double() @ args[1][:, :used].double().T


def bias_of(kernel, args, message_reference=None, chunk=4096):
    """F-bias statistics of a launched variant's w (``args[8]``) and messages (``args[7]``)."""
    import torch
    w = args[8]
    result = {}
    if tuple(w.shape) == (args[3].shape[0], kernel.weight_dim):
        moments = _Moments()
        for e in range(0, _real_edges(args), chunk):
            rows = slice(e, min(e + chunk, _real_edges(args)))
            ref = w_reference(kernel, args, rows)
            moments.add(w[rows].double() - ref, ref)
        result["w"] = moments.result()
    if message_reference is not None:
        moments = _Moments()
        outs = args[7] if isinstance(args[7], list) else [args[7]]
        for o, r in zip(outs, message_reference):
            for i in range(0, o.shape[0], chunk):
                ref = r[i:i + chunk].double()
                moments.add(o[i:i + chunk].double() - ref, ref)
        result["messages"] = moments.result()
    torch.cuda.synchronize()
    return result


def message_reference(kernel, args, chunk=4096):
    """The plan's TP fed the float64 product rounded to float32 (``_w_from_buffer``): the messages without the
    product's rounding. Fills ``args[8]`` (the shared w buffer) on the way; the caller poisons it after."""
    import torch
    w = args[8]
    for e in range(0, w.shape[0], chunk):
        rows = slice(e, min(e + chunk, w.shape[0]))
        w[rows] = w_reference(kernel, args, rows).float()
    launch, _ = forward_launcher(kernel, _w_from_buffer(kernel.source), ())
    ref = list(args)
    ref[7] = [torch.zeros_like(o) for o in args[7]] if isinstance(args[7], list) else torch.zeros_like(args[7])
    launch(ref)
    torch.cuda.synchronize()
    return ref[7] if isinstance(ref[7], list) else [ref[7]]


def gemm_bias(rows=65536, k=128, n=128, seed=20261106):
    """F-bias for one GEMM at K = ``k``: CUTLASS's 3xTF32 (``fastmace.kernels.cutlass.tf32x3.mm_into``; the error
    where this GPU has no cubin) and cuBLAS IEEE float32 (TF32 off), against float64; A = randn and A = |randn|."""
    import torch

    from fastmace.kernels.cutlass import tf32x3
    generator = torch.Generator(device="cuda").manual_seed(seed)
    results = {}
    allow = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for label in ("randn", "abs-randn"):
            a = torch.randn(rows, k, generator=generator, device="cuda")
            a = a.abs() if label == "abs-randn" else a
            b = torch.randn(k, n, generator=generator, device="cuda")
            ref = a.double() @ b.double()
            products = {"ieee": lambda: a @ b}

            def cutlass():
                out = torch.empty(rows, n, device="cuda")
                if not tf32x3.eligible(a, b, out):
                    raise RuntimeError("tf32x3.eligible is False")
                tf32x3.mm_into(a, b, out)
                torch.cuda.synchronize()
                return out
            products["cutlass-3xtf32"] = cutlass
            for name, product in products.items():
                try:
                    moments = _Moments()
                    moments.add(product().double() - ref, ref)
                    results[f"{name} K={k} A={label}"] = moments.result()
                except Exception as error:  # no cubin for this GPU (sm_120), or a load error
                    results[f"{name} K={k} A={label}"] = dict(error=f"{type(error).__name__}: {str(error)[-300:]}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow
    return results


def _fmt_bias(result):
    def one(m):
        if "error" in m or "slope" not in m:
            return str(m)
        return (f"slope {m['slope']:+.2e} toward0 {m['toward0']:+.2e} mean {m['mean']:+.2e} rms {m['rms']:.2e} "
                f"(n {m['n']})")
    return "; ".join(f"{key}: {one(m)}" for key, m in result.items())


def forward_variants(kernel=None):
    store = "wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];"

    def k64(s):
        k = int(re.search(r"constexpr int K = (\d+);", s).group(1))
        s = _sub(s, f"constexpr int K = {k};", f"constexpr int K = {k - 1};\nconstexpr int KA = {k};", 1)
        s = _sub(s, "a[(e0 + b) * K + k]", "a[(e0 + b) * KA + k]")
        return _sub(s, "W[(int64_t)(wst[p] + cb * CB + l) * K + k]", "W[(int64_t)(wst[p] + cb * CB + l) * KA + k]")

    tp = r"if \(!tail\) \{(\s*#pragma unroll\s*for \(int b = 0; b < B; \+\+b\) \{\s*if \(b >= nb\) break;)"
    return {
        "base": ("exact", lambda s: s, ()),
        "stcs": ("exact", lambda s: _sub(s, store, "__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);"), ()),
        "ldcs-a": ("exact", lambda s: _sub(s, "a[(e0 + b) * K + k]", "__ldcs(&a[(e0 + b) * K + k])"), ()),
        "k64": ("exact", k64, ()),
        "k64+stcs": ("exact", lambda s: _sub(k64(s), store, "__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);"),
                     ()),
        "F-no-store": ("falsifier", lambda s: _sub(s, "constexpr bool STORE_W = true;",
                                                   "constexpr bool STORE_W = false;", 1), ()),
        "F-no-gemm": ("falsifier", lambda s: _sub(s, "for (int k = 0; k < K; ++k) {", "for (int k = 0; k < 0; ++k) {"),
                      ()),
        "F-no-tp": ("falsifier", lambda s: _sub(s, tp, r"if (false) {\1", regex=True), ()),
        # Without the store (the message-only kernel of wprime_backward.TABLE_W): the product's share.
        "NS-no-gemm": ("falsifier", lambda s: _sub(_sub(s, "for (int k = 0; k < K; ++k) {",
                                                        "for (int k = 0; k < 0; ++k) {"),
                                                   "constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;",
                                                   1), ()),
        "NS-slab": ("falsifier", _ns_slab, ()),
        "simt-slab3": ("falsifier", _slab_real(kernel, 3), (), _slab_args(3)),
        "simt-slab-real": ("falsifier", _slab_real(kernel, SLAB_GROUPS), (), _slab_args(SLAB_GROUPS)),
        # 6 November 2026 (Symmetrix-XL comparison): the same bound with each edge on its REAL table slab (pair and
        # piece from the captured w' call's lengths, pairs and bins) and the real value table as the buffer, so the
        # footprint is what MD touches, not all 5,340 slabs.
        "simt-slab-edges": ("falsifier", _slab_edges(kernel, False), (), _slab_edges_args),
        "simt-ns-no-gemm": ("falsifier", lambda s: _sub(_tc_rerender(kernel, False, "simt-ns-no-gemm")(s).replace(
            "constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;"), "for (int k = 0; k < K; ++k) {",
            "for (int k = 0; k < 0; ++k) {"), ()),
        "F-gemm-only": ("falsifier", lambda s: _sub(_sub(s, tp, r"if (false) {\1", regex=True),
                                                    "constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;",
                                                    1), ()),
        # Tensor-core base (--constant ...fused_forward:TENSOR_CORES=True): where the 3xTF32 product's time goes.
        "F-tc-1pass": ("falsifier", lambda s: _sub(_sub(s, "mma_tf32(wacc[g][m], al, bh);", ""),
                                                   "mma_tf32(wacc[g][m], ah, bl);", ""), ()),
        "F-tc-no-mma": ("falsifier", lambda s: _sub(_sub(_sub(s, "mma_tf32(wacc[g][m], al, bh);", ""),
                                                         "mma_tf32(wacc[g][m], ah, bl);", ""),
                                                    "mma_tf32(wacc[g][m], ah, bh);", "wacc[g][m][0] += __uint_as_float("
                                                    "ah[0] ^ ah[1] ^ ah[2] ^ ah[3] ^ bh[0] ^ bh[1]);"), ()),
        "F-tc-no-wsplit": ("falsifier", _tc_no_wsplit, ()),
        # Ideas session census (5 November 2026): the W split in the wfrag kernel (TC_WFRAG), each fragment value
        # used unsplit. Bounds a pre-split fragment-order W slice (two float4 per lane, no split instructions).
        "tc-wfrag-split": ("exact", _tc_wfrag_split(kernel), ()),
        "F-wfrag-no-wsplit": ("falsifier", lambda s: _sub(s, r"split_tf32\(wv\.([xyzw]), ah\[(\d)\], al\[\d\]\);",
                                                          r"ah[\2] = __float_as_uint(wv.\1); al[\2] = ah[\2];",
                                                          regex=True), ()),
        # Round 5 (A100 stage bounds, 5 November 2026): one MMA pass costs 42 us, three dependent passes 1.09 ms, so
        # the passes wait on each other. tc-mma-free: the MMA asm without volatile (the compiler may interleave the
        # (g, m) chains; per accumulator the order is the same: bitwise). tc-2acc / tc-3acc: the small products into
        # their own accumulator(s), added once per batch (shorter chains, more registers; rounding).
        "tc-mma-free": ("exact", lambda s: _sub(s, 'asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32',
                                                'asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32'), ()),
        "tc-2acc": ("rounding", lambda s: _tc_split_accumulators(s, 2), ()),
        "tc-3acc": ("rounding", lambda s: _tc_split_accumulators(s, 3), ()),
        # TF32 + FP16 hybrid (round 5): W_hi a_hi in TF32, both small products in FP16 m16n8k16 (4 MMAs per 16 k).
        "tc-hybrid16": ("rounding", _tc_hybrid16, ()),
        # Cheaper splits than rounding both parts; they need the rounded split in the capture
        # (--constant ...fused_forward:TC_SPLIT="'rna'"; "mask" is fused_forward.TC_SPLIT's default):
        # the high part by masking the low 13 mantissa bits (truncation), the low part the exact remainder, rounded to
        # TF32 (mask-rna) or passed raw (mask: the tensor core reads its top 19 bits).
        "tc-mask-rna": ("falsifier", lambda s: _sub(s, TC_SPLIT_RNA_HI, TC_SPLIT_MASK_HI), ()),
        "tc-mask": ("falsifier", lambda s: _sub(_sub(s, TC_SPLIT_RNA_HI, TC_SPLIT_MASK_HI), TC_SPLIT_RNA_LO,
                                                TC_SPLIT_RAW_LO), ()),
        "F-tc-no-product": ("falsifier", lambda s: _sub(s, "for (int k0 = 0; k0 < KP; k0 += 8) {",
                                                        "for (int k0 = 0; k0 < 0; k0 += 8) {"), ()),
        "tc-presplit": ("exact", _tc_presplit(kernel), ()),
        # Exponent-shifted 3xFP16 m16n8k16 (round 2, 5 November 2026): rounded 22-bit split, 2x the TF32 MMA rate.
        "tc-rerender": ("exact", _tc_rerender(kernel, True, "tc-rerender"), ()),
        "tc-tp": ("rounding", _tc_rerender(kernel, True, "tc-tp", tctp=True), ()),  # the re-render check: bitwise
        "simt": ("rounding", _tc_rerender(kernel, False, "simt"), ()),  # the FP32-SIMT product: a third opinion
        # Fewer warps per block: the launch bound allows more registers (the A100 kernel spills 40 bytes at 168).
        "tc-warps8": ("exact", _tc_rerender(kernel, True, "tc-warps8", warps=8), ()),
        # Launch shapes for other SM counts (H200, 5 November 2026): the persistent grid at 1, 3 or 4 blocks per SM
        # (BLOCKS_PER_SM, 2 by default) and work chunks of 1, 3 or 4 receivers (RECEIVERS_PER_CHUNK, 2). Bitwise.
        **{f"grid{n}": ("exact", _grid(kernel, n), ()) for n in (1, 3, 4)},
        **{f"rc{n}": ("exact", lambda s, n=n: _sub(s, "constexpr int RC = 2;", f"constexpr int RC = {n};"), ())
           for n in (1, 3, 4)},
        "tc-warps10": ("exact", _tc_rerender(kernel, True, "tc-warps10", warps=10), ()),
        # Round 4: the staging double-buffered with cp.async (fused_forward.PREFETCH): bitwise.
        "tc-prefetch": ("exact", _tc_rerender(kernel, True, "tc-prefetch", prefetch=True), ()),
        # The FP32-SIMT kernel (the plan's own, no TENSOR_CORES constant) re-rendered, without and with prefetch.
        "simt-rerender": ("exact", _tc_rerender(kernel, False, "simt-rerender"), ()),
        "simt-prefetch": ("exact", _tc_rerender(kernel, False, "simt-prefetch", prefetch=True), ()),
        "simt-prefetch-x": ("exact", _tc_rerender(kernel, False, "simt-prefetch-x", prefetch="x"), ()),
        "tc-prefetch-x": ("exact", _tc_rerender(kernel, True, "tc-prefetch-x", prefetch="x"), ()),
        "tc-prefetch-x-w10": ("exact", _tc_rerender(kernel, True, "tc-prefetch-x-w10", warps=10, prefetch="x"), ()),
        "tc-prefetch-w10": ("exact", _tc_rerender(kernel, True, "tc-prefetch-w10", warps=10, prefetch=True), ()),
        "tc-fp16": ("rounding", _tc_rerender(kernel, "fp16", "tc-fp16"), ()),
        "tc-fp16-1pass": ("falsifier", lambda s: _sub(_sub(_tc_rerender(kernel, "fp16", "tc-fp16-1pass")(s),
                                                            "mma_f16(cross, al, bh);", ""),
                                                       "mma_f16(cross, ah, bl);", ""), ()),
        "F-tc-no-redistribute": ("falsifier", lambda s: _sub(s, "w[g][b] = r_s[(g * B + b) * RS + lane];",
                                                             "w[g][b] = wacc[g][b & 1][b >> 1 & 3];"), ()),
        # Round 4 (5 November 2026): the staging's global loads, each replaced by a constant (timing bounds of the
        # load latency: a, the sender features x and the edge harmonics sh).
        "F-no-aload": ("falsifier", lambda s: _sub(s, "a[(e0 + b) * KA + k]", "0.5f"), ()),
        "F-no-xload": ("falsifier", lambda s: _sub(s, r"\(snd >= 0\) \? x\[[^\]]*\] : 0\.f",
                                                   "(snd >= 0) ? 0.5f : 0.f", regex=True), ()),
        "F-no-shload": ("falsifier", lambda s: _sub(s, "sh[e0 * ED + i]", "0.5f"), ()),
        "F-no-loads": ("falsifier", lambda s: _sub(_sub(_sub(s, "a[(e0 + b) * KA + k]", "0.5f"),
                                                        r"\(snd >= 0\) \? x\[[^\]]*\] : 0\.f",
                                                        "(snd >= 0) ? 0.5f : 0.f", regex=True),
                                                   "sh[e0 * ED + i]", "0.5f"), ()),
        # H200 ideas plan (5 November 2026): the same bounds for the float4 staging (VEC_A, VEC_SH; idea 7).
        "F-no-aload4": ("falsifier", lambda s: _sub(
            s, "__ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q)", "make_float4(0.5f, 0.5f, 0.5f, 0.5f)"),
            ()),
        "F-no-shload4": ("falsifier", lambda s: _sub(
            s, "__ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i)", "make_float4(0.5f, 0.5f, 0.5f, 0.5f)"), ()),
        "F-no-ash4": ("falsifier", lambda s: _sub(_sub(
            s, "__ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q)", "make_float4(0.5f, 0.5f, 0.5f, 0.5f)"),
            "__ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i)", "make_float4(0.5f, 0.5f, 0.5f, 0.5f)"), ()),
        "tc-a4": ("exact", _tc_a4, (), _a4_args),
        "simt-a4": ("exact", _simt_a4, (), _a4_args),
        "xs": ("exact", _xs(kernel), ()),
        "tc-a4+xs": ("exact", lambda s: _xs(kernel)(_tc_a4(s)), (), _a4_args),
        "simt-a4+sh4": ("exact", lambda s: _sh4(_simt_a4(s)), (), _a4_args),
        "sh4": ("exact", lambda s: _sh4(s), ()),
        "tc-wfrag": ("exact", _wfrag, ()),
        "tc-a4+wfrag": ("exact", lambda s: _wfrag(_tc_a4(s)), (), _a4_args),
        "x-ldg": ("exact", lambda s: _hints(s, "ldg"), ()),
        "x-el": ("exact", lambda s: _hints(s, "el"), ()),
        "tc-a4+x-el": ("exact", lambda s: _hints(_tc_a4(s), "el"), (), _a4_args),
        "tc-a4-na": ("exact", lambda s: _hints(_tc_a4(s), stream=True), (), _a4_args),
        "tc-a4-na+x-el": ("exact", lambda s: _hints(_tc_a4(s), "el", stream=True), (), _a4_args),
        # The TP's coefficient multiplies dropped (x e instead of x e c): how much its arithmetic weighs.
        "F-tp-no-coef": ("falsifier", lambda s: _sub(s, r" \* -?0x[0-9a-f.]+p[-+]\d+f", "", regex=True), ()),
        # H200 ideas plan, idea 4: the per-lane TP of an S tile (one product per output and input component).
        "F-s-tile": ("falsifier", _s_tile, ()),
        # Ideas session idea 10 (5 November 2026): de-phase the forward's warps. The chunk loop has no block barrier,
        # so warps that start out of phase stay so unless something locks them; odd warps (or warp % 4) start late by
        # a sleep. Bitwise: only the start time changes.
        # Ideas session third revision (6 November 2026): idea 15(a) (exact) and idea 14's bound.
        "x-lane-idx": ("exact", _lane_indices, ()),
        "F-tp-stored-w": ("falsifier", _stored_w, (), _stored_w_args(kernel)),
        # Ideas session, fifth revision (6 November 2026). 14(A), the owner-pass forward: its pass 1 is the base with
        # STORE_W on (w for every edge here, so twice pass 1's stores); estimate (A) = store-w / 2 + F-tp-stored-w / 2.
        "store-w": ("exact", lambda s: _sub(s, "constexpr bool STORE_W = false;", "constexpr bool STORE_W = true;", 1),
                    (), _stored_w_args(kernel)),
        # Idea 17 (warp groups share each W fragment): timing bounds with G = 2's and G = 4's W traffic. Both m-tiles
        # (F-w-half), and also paired path groups (F-w-quarter), read one fragment; outputs wrong, time only.
        "F-w-half": ("falsifier", lambda s: _sub(s, _WFRAG_INDEX, _WFRAG_INDEX.replace("* 2 + m)", "* 2 + 0)")), ()),
        "F-w-quarter": ("falsifier", lambda s: _sub(s, _WFRAG_INDEX, _WFRAG_INDEX.replace("* 2 + m)", "* 2 + 0)")
                                                     .replace("((g * (KP", "(((g & ~1) * (KP")), ()),
        # Idea 17 built at G = 2 (fused_forward.W_GROUP): warp pairs share each W fragment. Bitwise.
        "w-group": ("exact", _w_group(kernel), ()),
        "F-skew-100ns": ("exact", lambda s: _skew(s, "(warp & 1) * 100"), ()),
        "F-skew-300ns": ("exact", lambda s: _skew(s, "(warp & 1) * 300"), ()),
        "F-skew-1us": ("exact", lambda s: _skew(s, "(warp & 1) * 1000"), ()),
        "F-skew-4us": ("exact", lambda s: _skew(s, "(warp & 1) * 4000"), ()),
        "F-skew-16us": ("exact", lambda s: _skew(s, "(warp & 1) * 16000"), ()),
        "F-skew4-4us": ("exact", lambda s: _skew(s, "(warp & 3) * 4000"), ()),
        # Kernel review (5 November 2026): x gathered at hashed sender indices without loading senders (the same
        # random L2 gathers, no load-to-load dependency; wrong results), and the TP's sh_s reads as float4 (exact).
        "F-x-nodep": ("falsifier", lambda s: _sub(s, "senders[e0 + b]",
                                                  "(int)((unsigned)((e0 + b) * 2654435761u) % (unsigned)N)"), ()),
        "sh-lds4": ("exact", _sh_lds4, ()),
        "F-no-out": ("falsifier", lambda s: _sub(s, r"out(\d)\[cur \* (\d+) \+ ([^\]]*)\] = acc\[(\d+)\];",
                                                 r"if (acc[\4] == 12345.f) out\1[cur * \2 + \3] = acc[\4];",
                                                 regex=True), ()),
        # H200 ideas plan, revision 7 (6 November 2026).
        # Idea 20, F-bias: w (STORE_W on) and the messages against float64 for the 3xTF32 splits "mask" (truncating
        # high part, raw low part), "round", B1 (W rounded to 22 bits, a's high part rounded) and the FP32-SIMT
        # product; F-bias-gemm: one CUTLASS 3xTF32 GEMM at K = 128 (and cuBLAS IEEE), no forward launch.
        "F-bias-mask": ("bias", _then(_split_mode("mask"), _store_w_on), (), _shared_w_args(kernel)),
        "F-bias-round": ("bias", _then(_split_mode("round"), _store_w_on), (), _shared_w_args(kernel)),
        "F-bias-b1": ("bias", _then(_b1, _store_w_on), (), _shared_w_args(kernel)),
        "F-bias-simt": ("bias", _then(_tc_rerender(kernel, False, "F-bias-simt"), _store_w_on), (),
                        _shared_w_args(kernel)),
        "F-bias-gemm": ("bias-gemm", None, ()),
        # 14(A), pass 1's stores: store-w with evict-first stores, and with LDS.128 + STG.128 from r_s. Both bitwise
        # against store-w (FORWARD_REFERENCES).
        "store-w-cs": ("exact", _then(lambda s: s, _store_w_on, _stream_store), (), _shared_w_args(kernel)),
        "store-w-v4": ("exact", _then(lambda s: s, _store_w_on, _store_v4), (), _shared_w_args(kernel)),
        # Idea 17's a path on the W_GROUP render (timing bounds, wrong w).
        "F-wg-own-a": ("falsifier", _wg_own_a("0"), ()),
        "F-wg-own-a-ld": ("falsifier", _wg_own_a("1"), ()),
        "F-wg-no-bar": ("falsifier", _wg_no_bar, ()),
        # Idea 21 (A100-SXM4): the fragment-sized W region at 12 warps, without and with W_GROUP. Bitwise.
        "tc-warps12": ("exact", _wfrag_region(kernel, "tc-warps12"), ()),
        "w-group-12": ("exact", _wfrag_region(kernel, "w-group-12", w_group=True), ()),
        # The same region at as many warps as fit this GPU (the A100-SXM4: 12, i.e. tc-warps12; the RTX 5060 Ti: 5,
        # and 4 with W_GROUP): the render's bitwise check where 12 warps do not fit.
        "tc-region": ("exact", _wfrag_region(kernel, "tc-region", warps=None), ()),
        "w-group-region": ("exact", _wfrag_region(kernel, "w-group-region", warps=None, w_group=True), ()),
        # Round 9 (H200 ideas revision 9). Idea 22(ii): W_GROUP with the fragment-sized region at 12, 14 and 16 warps
        # (the launch bound caps the registers: 168, 144, 128), with and without the x loads (F-no-xload: x not live
        # across the k-loop, as in 22(i)). Idea 25: one part's blocks only. 22(ii)'s carve-out: F-l1-28.
        **{f"wg{w}": ("exact", _wfrag_region(kernel, f"wg{w}", warps=w, w_group=True), ()) for w in (12, 14, 16)},
        **{f"F-no-xload-wg{w}": ("falsifier", _then(_wfrag_region(kernel, f"F-no-xload-wg{w}", warps=w, w_group=True),
                                                     _no_xload), ()) for w in (12, 14, 16)},
        "F-part0-only": ("falsifier", _part_only(0), ()),
        "F-part1-only": ("falsifier", _part_only(1), ()),
        "F-l1-28": ("exact", _l1_pad(kernel, 224 * 1024), ()),
    }


def _lane_indices(source):
    """Ideas session idea 15(a): each batch's sender and receiver indices loaded by lanes 0..B-1 at once and
    broadcast with shuffles, so the x gathers no longer wait on a serial index chain. Bitwise."""
    old = ("rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;\n"
           "                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;")
    new = ("rv[b] = __shfl_sync(FULL, lane_rv, b);\n"
           "                    int snd = __shfl_sync(FULL, lane_snd, b);")
    source = _sub(source, old, new)
    head = r"(\n(\s*)float xv\[B\]\[\d+\];)"
    load = (r"\n\2const int lane_rv = (lane < nb && !tail) ? receivers[e0 + lane] : 0x7fffffff;"
            r"\n\2const int lane_snd = (lane < nb && !tail) ? senders[e0 + lane] : -1;\1")
    return _sub(source, head, load, regex=True)


_WFRAG_INDEX = "((g * (KP / 8) + (k0 >> 3)) * 2 + m) * 32 + lane]"


def _stored_w(source):
    """Ideas session idea 14 bound: the forward with w read from HBM (``wout``, one row per edge, as the pair-shared
    form must) instead of the product: no a staging and no MMAs (the W loads and splits then fall away too)."""
    source = _sub(source, "if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);", "")
    source = _sub(source, r"\n\s*mma_tf32\(wacc\[g\]\[m\], a[hl], b[hl]\);", "", regex=True)
    out, pos = [], 0
    pattern = re.compile(r"float w\[(\d+)\]\[B\];(.*?)w\[g\]\[b\] = r_s\[\(g \* B \+ b\) \* RS \+ lane\];"
                         r"(.*?)const int wst\[\1\] = (\{[^}]*\});", re.S)
    for m in pattern.finditer(source):
        out.append(source[pos:m.start()])
        out.append(f"const int wsl[{m.group(1)}] = {m.group(4)};\n                float w[{m.group(1)}][B];{m.group(2)}"
                   f"w[g][b] = (b < nb) ? __ldg(wout + (e0 + b) * Q + wsl[g] + ch) : 0.f;{m.group(3)}"
                   f"const int wst[{m.group(1)}] = {m.group(4)};")
        pos = m.end()
    if not out:
        raise ValueError("F-tp-stored-w: the w redistribution was not found")
    return "".join(out) + source[pos:]


def _stored_w_args(kernel):
    """``F-tp-stored-w``'s w buffer: [E, Q] (the plan's H200 forward stores no w, so its own buffer is empty)."""
    def transform(args):
        import torch
        args = list(args)
        edges = args[3].shape[0]
        args[8] = torch.zeros(edges, kernel.weight_dim, device=args[3].device, dtype=args[3].dtype)
        return args
    return transform


def _skew(source, nanoseconds):
    """The forward's warps start their chunk loop after ``__nanosleep(nanoseconds)`` (an expression of ``warp``)."""
    anchor = "const int gid = lane >> 2, tq = lane & 3;"
    return _sub(source, anchor, f"{anchor}\n        if ({nanoseconds}) __nanosleep({nanoseconds});")


def _sh_lds4(s):
    """The tensor product's sh_s reads as four float4 loads per edge (ED = 16) instead of one scalar LDS per
    component. Bitwise (kernel review, 5 November 2026)."""
    if "constexpr int ED = 16;" not in s:
        raise ValueError("sh-lds4: needs ED = 16")
    s = _sub(s, "if (b >= nb) break;", "if (b >= nb) break;\n"
             "                        const float4* shp = reinterpret_cast<const float4*>(sh_s + b * ED);\n"
             "                        const float4 shq[4] = {shp[0], shp[1], shp[2], shp[3]};")
    return _sub(s, r"sh_s\[b \* ED \+ (\d+)\]",
                lambda m: f"shq[{int(m.group(1)) // 4}].{'xyzw'[int(m.group(1)) % 4]}", regex=True)


def forward_launcher(kernel, source, extra):
    """A launch function for a variant of ``kernel`` (a FusedFoldedForward) compiled from ``source``."""
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.fused_forward import CHANNEL_BLOCK
    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    cubin = compile_cubin(source, torch.cuda.get_device_capability(), extra)
    module = _check(driver.cuModuleLoadData(cubin))
    function = _check(driver.cuModuleGetFunction(module, b"fused_folded_forward"))
    attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    rule = _RULES.get(source, kernel.rule)
    _check(driver.cuFuncSetAttribute(function, attribute, rule.shared_bytes))
    registers = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    spills = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                              function))
    counters = torch.zeros((kernel.channels // CHANNEL_BLOCK) * rule.parts, dtype=torch.int32, device="cuda")

    # VEC_A (the A100 default) reads a from rows padded to AS floats (fused_forward.a_columns); a variant rendered
    # without it reads rows of KA floats. Give such a variant a compacted copy of the captured a, else it reads
    # misaligned rows (5 November 2026: every A100 variant without VEC_A differed in all 5,000 rows).
    ka = int(re.search(r"constexpr int KA = (\d+);", source).group(1))
    vec_a = "reinterpret_cast<const float4*>(a + (e0 + b) * AS)" in source
    compacted = {}

    def launch(args):
        a, weight, x, sh, senders, receivers, row_ptr, out, w = args
        if not vec_a and a.shape[1] != ka:
            key = (a.data_ptr(), a.shape[0])
            if key not in compacted:
                compacted.clear()
                compacted[key] = a[:, :ka].contiguous()
            a = compacted[key]
        counters.zero_()
        p = ctypes.c_void_p
        outs = [o.data_ptr() for o in out] if isinstance(out, list) else [out.data_ptr()]  # BLOCK_MESSAGE groups
        values = (x.shape[0], sh.shape[0], a.data_ptr(), weight.data_ptr(), x.data_ptr(), sh.data_ptr(),
                  senders.data_ptr(), receivers.data_ptr(), row_ptr.data_ptr(), counters.data_ptr(), *outs,
                  w.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * (len(values) - 2)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(function, rule.blocks_per_slice, kernel.channels // CHANNEL_BLOCK,
                                     rule.parts, rule.warps * 32, 1, 1, rule.shared_bytes, stream,
                                     (values, types), 0))
    return launch, dict(registers=int(registers), local_bytes=int(spills))


def ptxas_report(source, capability, extra=()):
    """(registers, spill bytes) of ``source`` compiled with NVRTC for ``capability`` (no device needed)."""
    from cuda.bindings import nvrtc

    from fastmace.kernels.oeq_streaming.runtime import _check
    major, minor = capability
    # A unique symbol in the PTX: NVRTC caches compiled programs across processes (keyed past the comments), and a
    # cache hit prints no ptxas info (6 November 2026: "registers None" for sources compiled before).
    source = f"{source}\n__device__ int fm_ptxas_report_{time.time_ns()};\n"
    program = _check(nvrtc.nvrtcCreateProgram(source.encode(), b"lab.cu", 0, [], []))
    options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"-default-device",
               b"--ptxas-options=-v"] + [o.encode() for o in extra]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    size = _check(nvrtc.nvrtcGetProgramLogSize(program))
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(program, log)
    log = log.decode(errors="replace")
    if int(result[0]) != 0:
        raise RuntimeError(log[-2000:])
    registers = [int(v) for v in re.findall(r"Used (\d+) registers", log)]
    spills = [int(v) for v in re.findall(r"(\d+) bytes spill stores", log)]
    return dict(registers=max(registers) if registers else None, spill_store_bytes=max(spills) if spills else 0)


# ------------------------------------------------------------------------------------------------------- capture
def capture(args):
    """(kernel objects, argument lists) of one eager force call of the default plan; the engine is freed."""
    import torch

    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig, RewriteConfig
    from fastmace.kernels.oeq_streaming import fused_forward, wprime_backward
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    captured, seen, recording = {}, [], [True]
    edge_arg = {"forward": 3, "wprime": 1}  # sh [E, 16]

    def recorder(name, original):
        def call(self, *a):
            edges = a[edge_arg[name]].shape[0]
            capturing = torch.cuda.is_current_stream_capturing()
            seen.append((name, edges, capturing))
            largest = captured[name][1][edge_arg[name]].shape[0] if name in captured else -1
            if recording[0] and not capturing and edges > largest:
                captured[name] = (self, [t.clone() if isinstance(t, torch.Tensor)
                                         else [u.clone() for u in t] if isinstance(t, list) else t for t in a])
            return original(self, *a)
        return call

    def full():
        return all(name in captured and captured[name][1][edge_arg[name]].shape[0] >= args.min_edges
                   for name in edge_arg)

    originals = {fused_forward.FusedFoldedForward: fused_forward.FusedFoldedForward.__call__,
                 wprime_backward.WPrimeBackward: wprime_backward.WPrimeBackward.__call__}
    fused_forward.FusedFoldedForward.__call__ = recorder("forward", originals[fused_forward.FusedFoldedForward])
    wprime_backward.WPrimeBackward.__call__ = recorder("wprime", originals[wprime_backward.WPrimeBackward])
    try:
        planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=NeighbourConfig(skin=0.5))
        sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
        name, base = ab_step_time.read_structure(args.structure, None)
        rewrites = {}
        for item in args.rewrites:
            key, value = item.split("=", 1)
            rewrites[key] = value
        field = ForceField(loaded, base, precision="float32",
                           rewrites=RewriteConfig(**rewrites) if rewrites else "auto")
        integrator = IntegratorConfig(temperature_K=300.0, timestep_fs=1.0)
        plan = planner.plan(field, base, integrator, Workload(max_temperature_K=300.0), planner.hardware())
        engine = planner.build(plan, field, base.copy(), temperature_K=300.0)
        engine.prepare()
        if args.equilibrate_steps:  # thermal disorder first: the ideal lattice's edge lengths keep the tables in L2
            recording[0] = False
            for _ in range(args.equilibrate_steps):
                engine.step()
            engine.flush()
            torch.cuda.synchronize()
            recording[0] = True
            captured.clear()  # prepare()'s warm-up calls ran on the ideal lattice
            engine.invalidate_graphs("omat0_kernel_lab: eager call for the capture after equilibration")
        for attempt in range(3):
            if full():
                break
            if attempt:  # the full-size calls ran under graph capture: drop the graphs, step again
                engine.invalidate_graphs("omat0_kernel_lab: eager call for the capture")
            engine.step()
            engine.flush()
        torch.cuda.synchronize()
        print("calls seen (kernel, edges, capturing):", sorted(set(seen)), flush=True)
        lines = [line.strip() for line in plan.describe(full=True).splitlines()
                 if any(f"RewriteConfig.{f} " in line for f in ("radial_tables", "fused_folded_forward",
                                                                "wprime_backward", "chunked_conv_backward",
                                                                "tf32x3_gemms"))]
        meta = dict(structure=name, natoms=len(base), checkpoint_sha256=sha, plan_lines=lines)
    finally:
        fused_forward.FusedFoldedForward.__call__ = originals[fused_forward.FusedFoldedForward]
        wprime_backward.WPrimeBackward.__call__ = originals[wprime_backward.WPrimeBackward]
    if not full():
        raise SystemExit(f"no eager call of both kernels with at least {args.min_edges} edges (is the plan's fused "
                         f"forward / wprime on?); calls seen: {sorted(set(seen))}")
    del engine, field, plan, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return captured, meta


def outputs_of(name, args):
    """Fresh output buffers of a captured argument list (the launch writes them)."""
    import torch
    args = list(args)
    if name == "forward":  # out: one tensor, or a list of groups (wprime_backward.BLOCK_MESSAGE)
        args[7] = [torch.zeros_like(o) for o in args[7]] if isinstance(args[7], list) else torch.zeros_like(args[7])
        args[8] = torch.zeros_like(args[8])  # zeros: the variants' cleared buffers (padding rows stay unwritten)
    else:
        args[6] = torch.zeros_like(args[6])  # dx: atomic adds
        args[7] = torch.empty_like(args[7])
        args[17] = torch.empty_like(args[17])
    return args


def poison(name, args):
    """Zero every output a variant writes, as the base's fresh buffers are (the variants share buffers: a variant
    that writes nothing then shows zeros, not the previous variant's values; the forward leaves rows without edges
    unwritten, as production's zero fill expects). wprime's dx is zeroed by ``reset`` before each launch."""
    if name == "forward":
        for o in (args[7] if isinstance(args[7], list) else [args[7]]):
            o.zero_()
        args[8].zero_()
    else:
        args[7].zero_()
        args[17].zero_()


def reset(name, args):
    if name == "wprime":
        args[6].zero_()


def compare(name, base, other):
    """Bitwise comparison of a variant's outputs with the base kernel's (dx with a relative tolerance)."""
    import torch
    if name == "forward":
        outs = base[7] if isinstance(base[7], list) else [base[7]]
        others = other[7] if isinstance(other[7], list) else [other[7]]
        pairs = {f"out{i}" if len(outs) > 1 else "out": (a, b) for i, (a, b) in enumerate(zip(outs, others))}
        pairs["w"] = (base[8], other[8])
    else:
        pairs = {"dsh": (base[7], other[7]), "dr": (base[17], other[17]), "dx": (base[6], other[6])}
    result = {}
    for key, (a, b) in pairs.items():
        if key == "dx":
            scale = float(a.abs().max()) or 1.0
            result[key] = f"max|d|/max|a| = {float((a - b).abs().max()) / scale:.2e}"
        else:
            same = bool(torch.equal(a, b))
            scale = float(a.abs().max()) or 1.0
            result[key] = "bitwise" if same else f"max|d|/max|a| = {float((a - b).abs().max()) / scale:.2e}"
            if name == "forward" and key != "w" and not same and a.dim() >= 1:  # where: which rows, how many edges
                bad = ((a - b).abs().reshape(a.shape[0], -1).amax(1) > 1e-4 * scale).nonzero().flatten()
                if bad.numel():
                    row_ptr = base[6]
                    degree = (row_ptr[1:] - row_ptr[:-1]).long()
                    rows = bad[bad < degree.numel()]
                    result[key] += (f" in {bad.numel()}/{a.shape[0]} rows ({int(bad.min())}..{int(bad.max())}; "
                                    f"{degree.numel()} CSR rows; degrees of the bad rows "
                                    f"{sorted(set(degree[rows].tolist()))[:8]})")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE",
                   help="RewriteConfig overrides (e.g. radial_tables=on fused_folded_forward=on wprime_backward=on "
                        "on a GPU outside decision 66)")
    p.add_argument("--constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE",
                   help="Module constants of the captured plan (A/B switches such as "
                        "fastmace.rewrites.radial:DEGREE=5)")
    p.add_argument("--kernels", nargs="*", default=["wprime", "forward"])
    p.add_argument("--variants", nargs="*", default=None, help="variant names (default: all)")
    p.add_argument("--min-edges", type=int, default=100000, help="the smallest edge count of a captured call")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--flush-l2-mb", type=int, default=256,
                   help="MB written before every timed launch (0: none, the warm L2 of lab runs before 5 November "
                        "2026 about 13:00 UTC)")
    p.add_argument("--equilibrate-steps", type=int, default=300,
                   help="MD steps (300 K, 1 fs) before the captured eager call; 0: the first steps from the ideal "
                        "lattice, as every lab run before 5 November 2026 about 12:40 UTC")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ptxas-archs", nargs="*", default=[], metavar="MAJOR.MINOR",
                   help="also report ptxas registers and spills of every variant for these architectures")
    p.add_argument("--ptxas-only", action="store_true", help="report ptxas only; no launches or timings")
    p.add_argument("--profile-variants", nargs="*", default=[], metavar="NAME",
                   help="after the timings, launch each of these variants once between cudaProfilerStart and "
                        "cudaProfilerStop (ncu --profile-from-start off; tools/modal/ncu_forward.sh)")
    args = p.parse_args(argv)
    ab_step_time.prepare_fastmace_hybrid()
    import torch
    from ab_rewrites import constant_overrides

    constants = constant_overrides(args.constant)
    for (module, name), value in constants.items():
        setattr(sys.modules[module], name, value)
    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    captured, meta = capture(args)
    meta.update(tool="omat0_kernel_lab", created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit(), model=args.model, rounds=args.rounds,
                capture_s=time.perf_counter() - started, kernels={},
                constants={f"{m}:{n}": v for (m, n), v in constants.items()})
    if "wprime" in captured and captured["wprime"]:
        _WPRIME_CAPTURE["args"] = captured["wprime"][1]
    for name in args.kernels:
        kernel, cargs = captured[name]
        if name == "wprime":
            gprime, starts = cargs[10], cargs[11]
            species = round((gprime.shape[0] // starts.shape[0]) ** 0.5)
            table, launcher = wprime_variants(species, kernel), wprime_launcher
            shapes = dict(edges=cargs[1].shape[0], gprime=list(gprime.shape),
                          gprime_MB=gprime.numel() * gprime.element_size() / 2**20, species=species,
                          pieces=starts.shape[0], threads=kernel.num_threads, cohort=kernel.cohort)
        else:
            table, launcher = forward_variants(kernel), forward_launcher
            shapes = dict(edges=cargs[3].shape[0], k=kernel.k, rule=str(kernel.rule),
                          weight_last_column_zero=bool((cargs[1][:, -1] == 0).all()))
        (args.output / f"{name}.cu").write_text(kernel.source)
        names = [v for v in (args.variants or table) if v in table]
        if "base" not in names:
            names.insert(0, "base")
        for variant, reference in (FORWARD_REFERENCES.items() if name == "forward" else ()):
            if variant in names and reference in table and (reference not in names
                                                             or names.index(reference) > names.index(variant)):
                if reference in names:  # the reference runs first
                    names.remove(reference)
                names.insert(names.index(variant), reference)
        pseudo = [v for v in names if table[v][0] == "bias-gemm"]
        names = [v for v in names if v not in pseudo]
        if pseudo and not args.ptxas_only:
            meta.setdefault("bias_gemm", {}).update(gemm_bias())
            for key, value in meta["bias_gemm"].items():
                print(f"{name} F-bias-gemm {key}: {_fmt_bias(dict(c=value))[3:]}", flush=True)
        # Two output sets (the forward's w is 1.9 GB): the base kernel's, and one all other variants share; each
        # exact variant is compared with the base right after its first launch, before the next one overwrites it.
        base_args, shared_args = outputs_of(name, cargs), outputs_of(name, cargs)
        captured[name] = None
        cargs = None
        built = {}
        if args.ptxas_archs:
            report = {}
            for variant in names:
                kind, transform, extra, *_ = table[variant]
                try:
                    source = transform(kernel.source)
                    report[variant] = {arch: ptxas_report(source, tuple(int(x) for x in arch.split(".")), extra)
                                       for arch in args.ptxas_archs}
                except Exception as error:
                    report[variant] = dict(error=str(error)[-400:])
                print(f"{name} ptxas {variant:28s} {report[variant]}", flush=True)
            meta.setdefault("ptxas", {})[name] = report
            if args.ptxas_only:
                continue
        for variant in names:
            kind, transform, extra, *arguments = table[variant]  # arguments: a transform of the argument list
            try:
                launch, info = launcher(kernel, transform(kernel.source), extra)
            except Exception as error:  # a pattern that does not apply here, or NVRTC
                built[variant] = dict(kind=kind, error=str(error)[-600:])
                print(f"{name} {variant}: {error}", flush=True)
                continue
            own = base_args if variant == "base" else (arguments[0](shared_args) if arguments else shared_args)
            built[variant] = dict(kind=kind, launch=launch, args=own, **info)
        live = [v for v in names if "launch" in built[v]]
        references = {r for v, r in FORWARD_REFERENCES.items() if v in live} if name == "forward" else set()
        snapshots, messages = {}, None
        for variant in live:  # base first: the reference of the comparisons
            b = built[variant]
            if b["kind"] == "bias" and messages is None:
                try:
                    messages = message_reference(kernel, b["args"])
                except Exception as error:
                    messages = False
                    print(f"{name} F-bias message reference: {error}", flush=True)
                    meta.setdefault("bias_message_reference_error", str(error)[-400:])
            if variant != "base":  # the variants share buffers: cleared first, so a stale output cannot pass as theirs
                poison(name, b["args"])
            for _ in range(args.warmup):
                reset(name, b["args"])
                b["launch"](b["args"])
            torch.cuda.synchronize()
            if variant in references:  # its outputs, for the variants that must match it
                own = b["args"]
                snapshots[variant] = list(own)
                snapshots[variant][7] = [o.clone() for o in own[7]] if isinstance(own[7], list) else own[7].clone()
                snapshots[variant][8] = own[8].clone()
            reference = FORWARD_REFERENCES.get(variant) if name == "forward" else None
            if b["kind"] == "bias":
                b["bias"] = bias_of(kernel, b["args"], messages or None)
            elif reference is not None:
                b["compare"] = (dict(against=reference, **compare(name, snapshots[reference], b["args"]))
                                if reference in snapshots else dict(error=f"the reference {reference} did not run"))
            elif variant != "base" and b["kind"] in ("exact", "rounding") and "launch" in built["base"]:
                b["compare"] = compare(name, base_args, shared_args)
        snapshots = None
        times = {v: [] for v in live}
        flush = (torch.empty(int(args.flush_l2_mb) << 18, dtype=torch.float32, device="cuda")
                 if args.flush_l2_mb else None)
        for _ in range(args.rounds):  # round-robin
            for variant in live:
                b = built[variant]
                reset(name, b["args"])
                if flush is not None:  # a cold L2, as in a force call (the kernels before it evict the inputs)
                    flush.fill_(0.0)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                b["launch"](b["args"])
                end.record()
                end.synchronize()
                times[variant].append(start.elapsed_time(end) * 1000.0)
        profiled = [v for v in live if v in args.profile_variants]
        for index, variant in enumerate(profiled):  # one launch each inside the profiler range (ncu)
            b = built[variant]
            reset(name, b["args"])
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            b["launch"](b["args"])
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print(f"{name} profiled launch {index}: {variant}", flush=True)
        meta.setdefault("profiled", {})[name] = profiled
        base_median = sorted(times["base"])[len(times["base"]) // 2] if "base" in times else None
        rows = {}
        for variant in names:
            b = built[variant]
            row = {k: v for k, v in b.items() if k not in ("launch", "args")}
            if variant in times:
                ordered = sorted(times[variant])
                row.update(median_us=ordered[len(ordered) // 2], min_us=ordered[0], max_us=ordered[-1])
                if base_median:
                    row["ratio"] = row["median_us"] / base_median
            rows[variant] = row
            print(f"{name:8s} {variant:16s} {row.get('median_us', float('nan')):9.1f} us  "
                  f"ratio {row.get('ratio', float('nan')):.3f}  regs {row.get('registers')} "
                  f"local {row.get('local_bytes')}  {row.get('compare', row.get('error', ''))}", flush=True)
            if "bias" in row:
                print(f"{name:8s} {variant:16s} bias {_fmt_bias(row['bias'])}", flush=True)
        meta["kernels"][name] = dict(shapes=shapes, variants=rows)
        built = base_args = shared_args = None
        _W_BUFFER.clear()
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "lab.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
