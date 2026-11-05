"""OMAT-0 kernel lab (owner, 3 November 2026: "come up with more ideas to speed up omat-0 on a100").

The A100 profile of Benchmark_loup (results/omat0-a100-profile-20261103) puts 65 % of the force call in two kernels:
the radial r-gradient backward ``oeq_projected_backward_wprime`` (6.9 ms) and the generic fused forward
``fused_folded_forward`` (5.1 ms). This tool captures one force call's arguments of both from an eager call of the
default plan (SrTiO3 10x10x10, 5,000 atoms), frees the engine, and times variants of the two kernels' rendered
sources on those arguments, round-robin in one process (CUDA events, median per variant):

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

    return {
        "B-piece-sorted": ("falsifier", lambda s: s, (), by_piece),
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
    }


def wprime_launcher(kernel, source, extra):
    """A launch function for a variant of ``kernel`` (a WPrimeBackward) compiled from ``source``."""
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    cubin = compile_cubin(source, torch.cuda.get_device_capability(), extra)
    module = _check(driver.cuModuleLoadData(cubin))
    function = _check(driver.cuModuleGetFunction(module, b"oeq_projected_backward_wprime"))
    registers = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    spills = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                              function))

    def launch(args):
        (x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins, r_lo, bin_scale,
         r_max, dr) = args
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (edges * kernel.cohort + kernel.num_threads - 1) // kernel.num_threads
        p = ctypes.c_void_p
        douts = list(dout) if isinstance(dout, (list, tuple)) else [dout]  # BLOCK_MESSAGE: one pointer per group
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), *[d.data_ptr() for d in douts], dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr())
        types = ((ctypes.c_longlong, ctypes.c_longlong) + (p,) * (13 + len(douts))
                 + (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, p))
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


def _tc_rerender(kernel, tc, name, warps=None, prefetch=False):
    """The tensor-core kernel re-rendered in mode ``tc`` (fused_forward.render) with its own launch rule;
    ``warps``: that many warps per block (the launch bound, so more registers per thread with fewer)."""
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
        source = ff.render(kernel.schedule, kernel.k, rule, kernel.store_w, kernel.used, tc=tc,
                           blocks=kernel.blocks, prefetch=prefetch)
        _RULES[source] = rule
        print(f"{name} rule {rule} (plan {kernel.rule})", flush=True)
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


def _ns_slab(s):
    """No store, and w from a 6-term slab with float2 loads instead of the a.W product (wprime_backward.TABLE_W in the
    forward): w[g][b] = sum_k a_k G_k, the a row's first 6 values standing in for the Chebyshev basis and W's own
    buffer for the table (three groups by edge index: an L2-resident timing bound, not the real values)."""
    s = _sub(s, "constexpr bool STORE_W = true;", "constexpr bool STORE_W = false;", 1)

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
    s = _sub(s, "constexpr int KA = ", f"constexpr int AS = {A4_STRIDE};\nconstexpr int KA = ", 1)
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
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);""")


def _a4_args(args):
    import torch

    a, *rest = args
    padded = torch.nn.functional.pad(a, (0, A4_STRIDE - a.shape[1])).contiguous()
    return (padded, *rest)


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
        "tc-rerender": ("exact", _tc_rerender(kernel, True, "tc-rerender"), ()),  # the re-render check: bitwise
        "simt": ("rounding", _tc_rerender(kernel, False, "simt"), ()),  # the FP32-SIMT product: a third opinion
        # Fewer warps per block: the launch bound allows more registers (the A100 kernel spills 40 bytes at 168).
        "tc-warps8": ("exact", _tc_rerender(kernel, True, "tc-warps8", warps=8), ()),
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
        "tc-a4": ("exact", _tc_a4, (), _a4_args),
        "sh4": ("exact", lambda s: _sub(s, "for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? "
                                              "sh[e0 * ED + i] : 0.f;", """for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);"""),
                   ()),
        # The TP's coefficient multiplies dropped (x e instead of x e c): how much its arithmetic weighs.
        "F-tp-no-coef": ("falsifier", lambda s: _sub(s, r" \* -?0x[0-9a-f.]+p[-+]\d+f", "", regex=True), ()),
        "F-no-out": ("falsifier", lambda s: _sub(s, r"out(\d)\[cur \* (\d+) \+ ([^\]]*)\] = acc\[(\d+)\];",
                                                 r"if (acc[\4] == 12345.f) out\1[cur * \2 + \3] = acc[\4];",
                                                 regex=True), ()),
    }


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

    def launch(args):
        a, weight, x, sh, senders, receivers, row_ptr, out, w = args
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

    captured, seen = {}, []
    edge_arg = {"forward": 3, "wprime": 1}  # sh [E, 16]

    def recorder(name, original):
        def call(self, *a):
            edges = a[edge_arg[name]].shape[0]
            capturing = torch.cuda.is_current_stream_capturing()
            seen.append((name, edges, capturing))
            if not capturing and edges > (captured[name][1][edge_arg[name]].shape[0] if name in captured else -1):
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
        args[8] = torch.empty_like(args[8])
    else:
        args[6] = torch.zeros_like(args[6])  # dx: atomic adds
        args[7] = torch.empty_like(args[7])
        args[17] = torch.empty_like(args[17])
    return args


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
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ptxas-archs", nargs="*", default=[], metavar="MAJOR.MINOR",
                   help="also report ptxas registers and spills of every variant for these architectures")
    p.add_argument("--ptxas-only", action="store_true", help="report ptxas only; no launches or timings")
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
        for variant in live:  # base first: the reference of the comparisons
            b = built[variant]
            for _ in range(args.warmup):
                reset(name, b["args"])
                b["launch"](b["args"])
            torch.cuda.synchronize()
            if variant != "base" and b["kind"] in ("exact", "rounding") and "launch" in built["base"]:
                b["compare"] = compare(name, base_args, shared_args)
        times = {v: [] for v in live}
        for _ in range(args.rounds):  # round-robin
            for variant in live:
                b = built[variant]
                reset(name, b["args"])
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                b["launch"](b["args"])
                end.record()
                end.synchronize()
                times[variant].append(start.elapsed_time(end) * 1000.0)
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
        meta["kernels"][name] = dict(shapes=shapes, variants=rows)
        built = base_args = shared_args = None
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "lab.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
