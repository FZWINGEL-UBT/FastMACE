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
def wprime_variants(species):
    """name -> (kind, source transform, extra NVRTC options); kind "exact" or "falsifier"."""
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

    return {
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
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), dout.data_ptr(), dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p, p, p, p, p, p, ctypes.c_float,
                 ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, p)
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
        "F-tc-no-redistribute": ("falsifier", lambda s: _sub(s, "w[g][b] = r_s[(g * B + b) * RS + lane];",
                                                             "w[g][b] = wacc[g][b & 1][b >> 1 & 3];"), ()),
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
        values = (x.shape[0], sh.shape[0], a.data_ptr(), weight.data_ptr(), x.data_ptr(), sh.data_ptr(),
                  senders.data_ptr(), receivers.data_ptr(), row_ptr.data_ptr(), counters.data_ptr(), out.data_ptr(),
                  w.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong, p, p, p, p, p, p, p, p, p, p)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(function, rule.blocks_per_slice, kernel.channels // CHANNEL_BLOCK,
                                     rule.parts, rule.warps * 32, 1, 1, rule.shared_bytes, stream,
                                     (values, types), 0))
    return launch, dict(registers=int(registers), local_bytes=int(spills))


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
                captured[name] = (self, [t.clone() if isinstance(t, torch.Tensor) else t for t in a])
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
    if name == "forward":
        args[7] = torch.empty_like(args[7])
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
        pairs = {"out": (base[7], other[7]), "w": (base[8], other[8])}
    else:
        pairs = {"dsh": (base[7], other[7]), "dr": (base[17], other[17]), "dx": (base[6], other[6])}
    result = {}
    for key, (a, b) in pairs.items():
        if key == "dx":
            scale = float(a.abs().max()) or 1.0
            result[key] = f"max|d|/max|a| = {float((a - b).abs().max()) / scale:.2e}"
        else:
            same = bool(torch.equal(a, b))
            result[key] = "bitwise" if same else f"max|d| = {float((a - b).abs().max()):.3e}"
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
            table, launcher = wprime_variants(species), wprime_launcher
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
            if variant != "base" and b["kind"] == "exact" and "launch" in built["base"]:
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
