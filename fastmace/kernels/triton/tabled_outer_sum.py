"""The first interaction's radial table and outer-product sums with one fused backward (round 2, 5 November 2026).

The reassociated first interaction (tabled generic family, OMAT-0) evaluates its folded input ``a = g(r) [E, K]`` from
the radial table (``radial_table``), then the per-slot sums ``S[key] += y a^T`` (``edge_outer_sum``). Its backward
ran three passes: ``edge_outer_sum_backward`` wrote ``grad_a [E, K]``, a compiled pass assembled the table's gradient
from it, and ``radial_table_backward`` read that back for ``dL/dr = sum_j grad_g_j g_j'(r)``. ``tabled_outer_sum``
keeps the forward and fuses the backward into one walk over the edges (``edge_outer_sum``'s ``BACKWARD_WALK``): per
edge, ``grad_a = G[key]^T y`` stays in registers and is contracted at once with the table's derivative row, so
neither ``grad_a`` nor the assembled gradient is written. The edge density's gradient joins the same sum.

Rounding: the sums are regrouped (tree sums over d and j). Any key order is correct; runs of equal keys (the segmented
first sum's receiver and sender-species order) load each slot's block once. Float32 CUDA tensors only; elsewhere the
reference composition (``edge_outer_sum_backward`` and ``radial_table_backward``) runs.
"""

from __future__ import annotations

from functools import cache

import torch

from fastmace.kernels.gpu_models import GPUModels

CHUNK = 16  # edges per program (the walk's A100 optimum, tools/eos_backward_lab.py)
WARPS = 1
# The recomputing backward (TABLED_FORWARD) evaluates the table's values inside the kernel and never reads ``a``; the
# op still needs its shape. True: a stride-0 view carries it; False: ``[E, K]`` floats, as before (an A/B side only).
# The floats were the force peak's largest block: 4.1 GiB, 18.9 kB/atom at SrTiO3 233,280 on the RTX 4000 Ada
# (results/memory-profile-20261109). Work removal, no rounding change.
SHAPE_CARRIER = True
# Phase 2: the forward evaluates a = g(r) inside the segmented sum's walk (no [E, K] table output written and read
# back) and the backward evaluates it again next to the derivative row (nothing of a saved). Rounding (its own
# Clenshaw sums, the same recurrence). None: on for TABLED_FORWARD_GPUS (decision 66, after the MD-step A/Bs); True /
# False: an A/B side's choice on every GPU. MD step b/a (ab / ba) with FUSED_FIRST_BACKWARD against the default,
# OMAT-0 medium, SrTiO3 10x10x10 and NbCl5: RTX 5060 Ti 0.945 / 0.945, 0.948 / 0.955; RTX 4000 Ada 0.949 / 0.944,
# 0.978 / 0.968 (FUSED_FIRST_BACKWARD alone 0.962-0.979); A100-SXM4-80GB 0.966 / 0.962, 0.970 / 0.982 (alone
# 0.966-0.984; every A100 model); NVIDIA H200 (5 November 2026, after the recapture fix, SrTiO3, TABLED_FORWARD
# alone against the H200 defaults) 0.984 / 0.987.
TABLED_FORWARD = None
TABLED_FORWARD_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation",
                                 "NVIDIA A100-SXM4-80GB", "NVIDIA H200"))
# Idea 40 (plans/h200-ideas-20261105.md, 6 November 2026). POWER_BACKWARD (with TABLED_FORWARD): the backward takes a
# and its slope (and the density column's slope) by one dual Horner pass from the value series in the power basis of
# x (the table's ``pcoef``, converted in float64 at build), six row loads instead of six value and five derivative
# rows. Rounding. None: on for POWER_BACKWARD_GPUS; True / False: an A/B side's choice.
POWER_BACKWARD = None
# NVIDIA H200 (6 November 2026, results/ideas-queue-20261106): MD step b/a 0.9954 / 0.9957 (OMAT-0 medium, SrTiO3
# 10x10x10, abab / abba); force/stress gate as the default's (worst max-error ratio 1.016, forces).
# NVIDIA A100-SXM4-80GB: 1.0020 / 0.9977, a tie, off.
# RTX 4000 Ada (ada-5060-screen, 10-11 November 2026, results/ada-5060-screen-20261110; OMAT-0 medium, Benchmark_loup,
# SrTiO3 10x10x10 / NbCl5, ab / ba): alone 0.9935 / 0.9929, 0.9935 / 0.9933; left out of the
# adopted set (after the SplineForward registry fix) 1.0125 / 1.0052, 1.0087 / 1.0074.
# RTX 5060 Ti (ada-5060-screen, 11 November 2026, results/ada-5060-screen-20261110; OMAT-0 medium, Benchmark_loup,
# SrTiO3 10x10x10 / NbCl5, ab / ba): alone 0.9917 / 0.9946, 0.9944 / 0.9941;
# left out of the candidate set 1.0126 / 1.0070, 1.0068 / 1.0091.
POWER_BACKWARD_GPUS = GPUModels(("NVIDIA H200", "NVIDIA RTX 4000 Ada Generation", "NVIDIA GeForce RTX 5060 Ti"))
FORWARD_CHUNK = 32  # edges per program of the tabled forward (edge_outer_sum.FORWARD_CHUNK)
FORWARD_WARPS = 1
# The tabled forward's register cap (Triton's maxnreg; None: Triton's choice). step_limiters (9 November 2026):
# Triton gives the 1-warp forward 128 registers on sm_80 (64 on sm_90), so the A100 runs 16 blocks per SM (25 %
# of warps) and the kernel waits on global loads (long scoreboard 10 per issue), 398 us against the H200's 112.
# A module constant for the A/B (lab, off).
FORWARD_MAXNREG = None
# Idea 3 (Codex H200 report, 7 November 2026). PACKED_COEF (with TABLED_FORWARD): the walks read the value series (the
# forward's and RECOMPUTE's coef, POWER_BACKWARD's pcoef) from a copy packed per adjacent channel pair,
# ``[P, pieces, pairs, TERMS, 2]`` (``packed_coef``, built with the table), and load two successive terms of a pair as
# one 16-byte vector: TERMS / 2 vector loads per lane instead of TERMS 8-byte loads from ``[P, pieces, TERMS, J]``.
# The density column stays in the unpacked table. The same Clenshaw / Horner recurrences in the same order: bitwise.
# Even TERMS and an even number of columns only (else the unpacked rows). None: on for PACKED_COEF_GPUS (none yet);
# True / False: an A/B side's choice. Read when the table is built (no packed copy where off).
PACKED_COEF = None
PACKED_COEF_GPUS = GPUModels(())


def packed_coef_default(device_or_name=None) -> bool:
    """``PACKED_COEF`` (None: the device's name, or the name given, in ``PACKED_COEF_GPUS``)."""
    on = PACKED_COEF
    if on is None:
        name = device_or_name
        try:
            device = torch.device(device_or_name)
        except (RuntimeError, TypeError):
            device = None
        if device is not None:
            name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        on = name in PACKED_COEF_GPUS
    return bool(on)


def packed_coef(coef):
    """``coef [P, pieces, TERMS, J]`` (J - 1 channels, then the density column) packed per adjacent channel pair as
    ``[P, pieces, ceil((J - 1) / 2), TERMS, 2]``: ``out[p, q, i, t, c] = coef[p, q, t, 2 i + c]`` (an odd last pair
    padded with zeros); None for odd TERMS. Contiguous, so two successive terms of a pair are 16 aligned bytes."""
    if coef is None or coef.dim() != 4 or coef.shape[2] % 2:
        return None
    count, pieces, terms, columns = coef.shape
    channels = columns - 1
    width = channels + channels % 2
    values = coef.new_zeros((count, pieces, terms, width))
    values[..., :channels] = coef[..., :channels]
    return values.reshape(count, pieces, terms, width // 2, 2).permute(0, 1, 3, 2, 4).contiguous()


def _packs(packed, coef, columns):
    """Whether ``packed`` (``packed_coef(coef)``) serves a walk over the leading ``columns`` channels of ``coef``."""
    return (packed is not None and coef is not None and columns % 2 == 0 and columns > 0
            and packed.shape[2] * 2 >= columns and packed.shape[3] == coef.shape[2] and packed.shape[4] == 2
            and packed.dtype == coef.dtype and packed.device == coef.device and packed.is_contiguous())


@torch.compiler.assume_constant_result
def _listed(index: int) -> bool:
    """Whether CUDA device ``index`` is one of ``TABLED_FORWARD_GPUS``. A compile-time constant for Dynamo: the
    device name cannot change within a process. Traced as a plain call, ``torch.cuda.get_device_name`` (it returns a
    str) broke the graph inside ``_TabledOuterSum.forward``; Dynamo then compiled that forward as a frame of its own,
    and its tracer state (``AutogradFunctionContextVariable``) kept the eager call's ctx, with the saved edge lengths
    and their autograd graph, in a reference cycle. That graph held the positions' AccumulateGrad of the eager force
    call (the engine's initial forces, default stream) until the next garbage collection, so a CUDA-graph capture
    before it failed ("stale reference to the default stream"; ab_rewrites --order ba with a None side built second,
    H200, 5 November 2026)."""
    return torch.cuda.get_device_name(index) in TABLED_FORWARD_GPUS


@torch.compiler.assume_constant_result
def _power_listed(index: int) -> bool:
    """Whether CUDA device ``index`` is one of ``POWER_BACKWARD_GPUS`` (a Dynamo constant, as ``_listed``)."""
    return torch.cuda.get_device_name(index) in POWER_BACKWARD_GPUS


def power_backward(device):
    """``POWER_BACKWARD`` on ``device``: the constant when set, else on for ``POWER_BACKWARD_GPUS``."""
    if POWER_BACKWARD is not None:
        return bool(POWER_BACKWARD)
    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _power_listed(device.index if device.index is not None else torch.cuda.current_device())


def tabled_forward(device):
    """``TABLED_FORWARD`` on ``device``: the constant when set, else on for ``TABLED_FORWARD_GPUS``."""
    if TABLED_FORWARD is not None:
        return bool(TABLED_FORWARD)
    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _listed(device.index if device.index is not None else torch.cuda.current_device())


@cache
def _packed_fns():
    """The packed walks' series (``PACKED_COEF``): (Clenshaw value, dual Horner value and slope)."""
    import triton
    import triton.language as tl

    @triton.jit
    def quads(pack_ptr, base, s, K: tl.constexpr, TERMS: tl.constexpr, HALF: tl.constexpr):
        """Terms 2 s (lo) and 2 s + 1 (hi) of the leading K channels, each [HALF, 2] (channel 2 i + c at [i, c]): per
        channel pair one 16-byte load of ``[pair, t, c]`` at t = 2 s, 2 s + 1."""
        h = tl.arange(0, HALF)
        q = tl.arange(0, 4)
        offsets = h[:, None] * (TERMS * 2) + q[None, :]
        mask = (h < K // 2)[:, None] & (q < 4)[None, :]
        quad = tl.load(pack_ptr + base + 4 * s + offsets, mask=mask, other=0.0)  # [i, (t, c)]
        lo, hi = tl.split(tl.permute(tl.reshape(quad, (HALF, 2, 2)), (0, 2, 1)))  # [i, c, t] -> t = 2 s, 2 s + 1
        return lo, hi

    @triton.jit
    def clenshaw(pack_ptr, base, x, K: tl.constexpr, TERMS: tl.constexpr, HALF: tl.constexpr):
        """sum_t c_t T_t(x) over the leading K channels, [2 HALF]: the unpacked walk's recurrence and order."""
        v1 = tl.zeros((HALF, 2), dtype=tl.float32)
        v2 = tl.zeros((HALF, 2), dtype=tl.float32)
        for s in tl.static_range(TERMS // 2 - 1, -1, -1):
            lo, hi = quads(pack_ptr, base, s, K, TERMS, HALF)
            v = hi + 2.0 * x * v1 - v2
            v2 = v1
            v1 = v
            if s > 0:  # term 2 s >= 2: a step; term 0 closes the sum
                v = lo + 2.0 * x * v1 - v2
                v2 = v1
                v1 = v
            else:
                last = lo
        return tl.reshape(last + x * v1 - v2, (2 * HALF,))

    @triton.jit
    def horner(pack_ptr, base, x, K: tl.constexpr, TERMS: tl.constexpr, HALF: tl.constexpr):
        """(p(x), p'(x)) over the leading K channels by dual Horner from the power-basis series, [2 HALF] each: the
        unpacked walk's recurrence and order."""
        pd = tl.zeros((HALF, 2), dtype=tl.float32)
        for s in tl.static_range(TERMS // 2 - 1, -1, -1):
            lo, hi = quads(pack_ptr, base, s, K, TERMS, HALF)
            if s == TERMS // 2 - 1:  # the leading term starts the sum
                pa = hi
            else:
                pd = pd * x + pa
                pa = pa * x + hi
            pd = pd * x + pa
            pa = pa * x + lo
        return tl.reshape(pa, (2 * HALF,)), tl.reshape(pd, (2 * HALF,))

    return clenshaw, horner


@cache
def _kernel():
    import triton
    import triton.language as tl

    from fastmace.kernels.triton.edge_outer_sum import _base_fn

    _base = _base_fn()
    _clenshaw, _horner = _packed_fns()

    @triton.jit
    def kernel(grad_ptr, a_ptr, y_ptr, key_ptr, r_ptr, pair_ptr, dcoef_ptr, starts_ptr, scales_ptr, bins_ptr,
               grad_density_ptr, dr_ptr, grad_y_ptr, num_edges, a_stride, E_DENSITY, n_pieces, n_bins, r_lo,
               bin_scale, r_max, K: tl.constexpr, D: tl.constexpr, INNER: tl.constexpr, TERMS: tl.constexpr,
               J: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
               RECOMPUTE: tl.constexpr, coef_ptr, VTERMS: tl.constexpr, POWER: tl.constexpr, pack_ptr,
               PACKED: tl.constexpr, PAIRS: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * (INNER * K) + k[None, :]
        current = tl.load(key_ptr + start).to(tl.int64)
        block = tl.load(grad_ptr + _base(current, D, K, INNER) + offsets, mask=mask, other=0.0)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:  # a new run: its slot's block, once
                block = tl.load(grad_ptr + _base(key, D, K, INNER) + offsets, mask=mask, other=0.0)
                current = key
            row = edge.to(tl.int64)
            # The table's lookup at r (radial_table's rule).
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            if POWER and PACKED:  # as below, the channels from the packed series (pack [P, pieces, PAIRS, VTERMS, 2])
                vbase = (pair * n_pieces + piece) * VTERMS * J
                pa, pd = _horner(pack_ptr, (pair * n_pieces + piece) * (PAIRS * VTERMS * 2), x, K, VTERMS,
                                 BLOCK_K // 2)
                qa = tl.load(coef_ptr + vbase + (VTERMS - 1) * J + (J - 1))
                qd = 0.0 * x
                for t in tl.static_range(VTERMS - 2, -1, -1):
                    cd = tl.load(coef_ptr + vbase + t * J + (J - 1))
                    qd = qd * x + qa
                    qa = qa * x + cd
                a = tl.where(r < r_max, pa, 0.0)
            elif POWER:  # a, da/dx and the density's d/dx by dual Horner (pcoef [P, pieces, VTERMS, J], power basis)
                vbase = (pair * n_pieces + piece) * VTERMS * J
                pa = tl.load(coef_ptr + vbase + (VTERMS - 1) * J + k, mask=k < K, other=0.0)
                pd = tl.zeros((BLOCK_K,), dtype=tl.float32)
                qa = tl.load(coef_ptr + vbase + (VTERMS - 1) * J + (J - 1))
                qd = 0.0 * x
                for t in tl.static_range(VTERMS - 2, -1, -1):
                    c = tl.load(coef_ptr + vbase + t * J + k, mask=k < K, other=0.0)
                    pd = pd * x + pa
                    pa = pa * x + c
                    cd = tl.load(coef_ptr + vbase + t * J + (J - 1))
                    qd = qd * x + qa
                    qa = qa * x + cd
                a = tl.where(r < r_max, pa, 0.0)
            elif RECOMPUTE and PACKED:  # as below, from the packed series
                a = _clenshaw(pack_ptr, (pair * n_pieces + piece) * (PAIRS * VTERMS * 2), x, K, VTERMS, BLOCK_K // 2)
                a = tl.where(r < r_max, a, 0.0)
            elif RECOMPUTE:  # a = g(r) again from the value series (coef [P, pieces, VTERMS, J])
                vbase = (pair * n_pieces + piece) * VTERMS * J
                v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                for t in tl.static_range(VTERMS - 1, 0, -1):
                    c = tl.load(coef_ptr + vbase + t * J + k, mask=k < K, other=0.0)
                    v = c + 2.0 * x * v1 - v2
                    v2 = v1
                    v1 = v
                a = tl.load(coef_ptr + vbase + k, mask=k < K, other=0.0) + x * v1 - v2
                a = tl.where(r < r_max, a, 0.0)
            else:
                a = tl.load(a_ptr + row * a_stride + k, mask=k < K, other=0.0)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            tl.store(grad_y_ptr + row * D + d, tl.sum(block * a[None, :], axis=1), mask=d < D)
            grad_a = tl.sum(block * y[:, None], axis=0)
            if POWER:  # d/dr = d/dx * the piece's scale
                sc = tl.load(scales_ptr + piece)
                slope = pd * sc
                slope_density = qd * sc
            else:
                # The derivative row (dcoef [P, pieces, TERMS, J]).
                base = (pair * n_pieces + piece) * TERMS * J
                b1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                b2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                c1 = 0.0 * x  # the density column J - 1 (a scalar Clenshaw sum)
                c2 = 0.0 * x
                for t in tl.static_range(TERMS - 1, 0, -1):
                    c = tl.load(dcoef_ptr + base + t * J + k, mask=k < K, other=0.0)
                    b = c + 2.0 * x * b1 - b2
                    b2 = b1
                    b1 = b
                    cd = tl.load(dcoef_ptr + base + t * J + (J - 1))
                    cn = cd + 2.0 * x * c1 - c2
                    c2 = c1
                    c1 = cn
                slope = tl.load(dcoef_ptr + base + k, mask=k < K, other=0.0) + x * b1 - b2
                slope_density = tl.load(dcoef_ptr + base + (J - 1)) + x * c1 - c2
            grad_density = tl.load(grad_density_ptr + row, mask=edge < E_DENSITY, other=0.0)
            total = tl.sum(grad_a * slope, axis=0) + grad_density * slope_density
            total = tl.where((r < r_max) & (r >= r_lo), total, 0.0)
            tl.store(dr_ptr + row, total)

    return kernel


@cache
def _forward_kernel():
    import triton
    import triton.language as tl

    from fastmace.kernels.triton.edge_outer_sum import _base_fn

    _base = _base_fn()
    _clenshaw, _ = _packed_fns()

    @triton.jit
    def kernel(y_ptr, key_ptr, r_ptr, pair_ptr, coef_ptr, starts_ptr, scales_ptr, bins_ptr, out_ptr, density_ptr,
               num_edges, n_pieces, n_bins, r_lo, bin_scale, r_max, K: tl.constexpr, D: tl.constexpr,
               INNER: tl.constexpr, TERMS: tl.constexpr, J: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr,
               BLOCK_D: tl.constexpr, pack_ptr, PACKED: tl.constexpr, PAIRS: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * (INNER * K) + k[None, :]
        acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
        current = tl.load(key_ptr + start).to(tl.int64)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:  # a run ends: add its sum to its slot
                tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")
                acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
                current = key
            row = edge.to(tl.int64)
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            base = (pair * n_pieces + piece) * TERMS * J
            inside = r < r_max
            if PACKED:  # the channels from the packed series (pack [P, pieces, PAIRS, TERMS, 2]), the density below
                a = _clenshaw(pack_ptr, (pair * n_pieces + piece) * (PAIRS * TERMS * 2), x, K, TERMS, BLOCK_K // 2)
                a = tl.where(inside, a, 0.0)
                c1 = 0.0 * x
                c2 = 0.0 * x
                for t in tl.static_range(TERMS - 1, 0, -1):
                    cd = tl.load(coef_ptr + base + t * J + (J - 1))
                    cn = cd + 2.0 * x * c1 - c2
                    c2 = c1
                    c1 = cn
            else:
                v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                c1 = 0.0 * x
                c2 = 0.0 * x
                for t in tl.static_range(TERMS - 1, 0, -1):
                    c = tl.load(coef_ptr + base + t * J + k, mask=k < K, other=0.0)
                    v = c + 2.0 * x * v1 - v2
                    v2 = v1
                    v1 = v
                    cd = tl.load(coef_ptr + base + t * J + (J - 1))
                    cn = cd + 2.0 * x * c1 - c2
                    c2 = c1
                    c1 = cn
                a = tl.where(inside, tl.load(coef_ptr + base + k, mask=k < K, other=0.0) + x * v1 - v2, 0.0)
            density = tl.where(inside, tl.load(coef_ptr + base + (J - 1)) + x * c1 - c2, 0.0)
            tl.store(density_ptr + row, density)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            acc += y[:, None] * a[None, :]
        tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")

    return kernel


@torch.library.custom_op("fastmace::tabled_outer_sum_forward", mutates_args=())
def tabled_outer_sum_forward(r: torch.Tensor, pair: torch.Tensor, coef: torch.Tensor, starts: torch.Tensor,
                             scales: torch.Tensor, bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                             columns: int, y: torch.Tensor, key: torch.Tensor, slots: int,
                             inner: int = 1, packed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(S [slots, D * K], edge density [E, 1]) with a = the table's leading ``columns`` values evaluated in the walk
    (``TABLED_FORWARD``); float32 CUDA (elsewhere: the table, then ``edge_outer_sum_forward``). ``packed``
    (``PACKED_COEF``): ``packed_coef(coef)``, the channels' series read from it where it serves (``_packs``)."""
    import triton

    from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum_forward
    from fastmace.kernels.triton.radial_table import radial_table_forward
    edges, d, k = r.numel(), y.shape[1], int(columns)
    if not (r.is_cuda and coef.dtype == torch.float32 and y.dtype == torch.float32):
        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, True, k)
        return edge_outer_sum_forward(a, y, key, slots, inner), density
    r, pair, y, key = r.reshape(-1).contiguous(), pair.contiguous(), y.contiguous(), key.contiguous()
    out = torch.zeros((slots, d * k), dtype=y.dtype, device=y.device)
    density = r.new_empty((edges, 1))
    packs = _packs(packed, coef, k)
    if edges:
        _forward_kernel()[(triton.cdiv(edges, FORWARD_CHUNK),)](
            y, key, r, pair, coef, starts, scales, bins, out, density, edges, coef.shape[1], bins.shape[0],
            float(r_lo), float(bin_scale), float(r_max), K=k, D=d, INNER=inner, TERMS=coef.shape[2],
            J=coef.shape[-1], CHUNK=FORWARD_CHUNK, BLOCK_K=triton.next_power_of_2(k),
            BLOCK_D=triton.next_power_of_2(d), pack_ptr=packed if packs else coef, PACKED=packs,
            PAIRS=packed.shape[2] if packs else 1, num_warps=FORWARD_WARPS,
            **({} if FORWARD_MAXNREG is None else {"maxnreg": FORWARD_MAXNREG}))
    return out, density


@tabled_outer_sum_forward.register_fake
def _(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y, key, slots, inner=1, packed=None):
    return y.new_empty((slots, y.shape[1] * int(columns))), r.new_empty((r.numel(), 1))


@torch.library.custom_op("fastmace::tabled_outer_sum_backward", mutates_args=())
def tabled_outer_sum_backward(grad: torch.Tensor, a: torch.Tensor, y: torch.Tensor, key: torch.Tensor,
                              r: torch.Tensor, pair: torch.Tensor, dcoef: torch.Tensor, starts: torch.Tensor,
                              scales: torch.Tensor, bins: torch.Tensor, grad_density: torch.Tensor, r_lo: float,
                              bin_scale: float, r_max: float, inner: int = 1,
                              coef: torch.Tensor | None = None, columns: int = 0, power: bool = False,
                              packed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(dL/dr [E], grad_y [E, D]) from the slots' gradient ``grad [slots, D * K]`` (``inner`` layout), the table rows
    ``a [E, K]`` (the table's leading K columns) and the edge density's gradient (``[E, 0]``: none). With ``coef``
    (``TABLED_FORWARD``): ``a`` is evaluated again from the value table's leading ``columns`` (``a`` unused);
    ``power`` (``POWER_BACKWARD``): ``coef`` is the power-basis table ``pcoef``, a and the slopes by dual Horner;
    ``packed`` (``PACKED_COEF``): ``packed_coef(coef)``, the channels' series read from it where it serves."""
    if coef is not None:
        if not (grad.is_cuda and coef.dtype == torch.float32):
            if power:
                raise ValueError("the power-basis backward runs only in the fused CUDA float32 kernel")
            from fastmace.kernels.triton.radial_table import radial_table_forward
            a, _ = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, False,
                                        int(columns))
            coef = None
        else:
            # shape only: the kernel evaluates a (SHAPE_CARRIER: a stride-0 view, not [E, K] floats)
            a = (r.new_empty((1, int(columns))).expand(r.numel(), int(columns)) if SHAPE_CARRIER
                 else r.new_empty((r.numel(), int(columns))))
    num_edges, k = a.shape
    d = y.shape[1]
    grad_density = grad_density.reshape(-1)
    fused = (grad.is_cuda and grad.dtype == torch.float32 and a.dtype == torch.float32 and dcoef.dtype == torch.float32
             and k <= dcoef.shape[-1] - 1)
    if not fused:
        from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum_backward
        from fastmace.kernels.triton.radial_table import radial_table_backward

        grad_a, grad_y = edge_outer_sum_backward(grad, a, y, key, inner)
        density = grad_density.reshape(-1, 1) if grad_density.numel() else grad_density.reshape(-1, 0)
        dr = radial_table_backward(r.reshape(-1), pair, dcoef, starts, scales, bins, grad_a, density, r_lo, bin_scale,
                                   r_max)
        return dr.reshape(-1), grad_y
    import triton

    from fastmace.kernels.triton.edge_outer_sum import _rows
    grad, y, key = grad.contiguous(), y.contiguous(), key.contiguous()
    a = a if coef is not None else _rows(a)  # with coef, a is the stride-0 shape carrier: never read
    r, pair, grad_density = r.reshape(-1).contiguous(), pair.contiguous(), grad_density.contiguous()
    dr = r.new_empty((num_edges,))
    grad_y = torch.empty_like(y)
    packs = coef is not None and _packs(packed, coef, k)
    if num_edges:
        _kernel()[(triton.cdiv(num_edges, CHUNK),)](
            grad, a, y, key, r, pair, dcoef, starts, scales, bins, grad_density if grad_density.numel() else r, dr,
            grad_y, num_edges, a.stride(0), grad_density.numel(), dcoef.shape[1], bins.shape[0], float(r_lo),
            float(bin_scale), float(r_max), K=k, D=d, INNER=inner, TERMS=dcoef.shape[2], J=dcoef.shape[-1],
            CHUNK=CHUNK, BLOCK_K=triton.next_power_of_2(k), BLOCK_D=triton.next_power_of_2(d),
            RECOMPUTE=coef is not None, coef_ptr=coef if coef is not None else dcoef,
            VTERMS=coef.shape[2] if coef is not None else 1, POWER=bool(power) and coef is not None,
            pack_ptr=packed if packs else dcoef, PACKED=packs, PAIRS=packed.shape[2] if packs else 1,
            num_warps=WARPS)
    return dr, grad_y


@tabled_outer_sum_backward.register_fake
def _(grad, a, y, key, r, pair, dcoef, starts, scales, bins, grad_density, r_lo, bin_scale, r_max, inner=1,
      coef=None, columns=0, power=False, packed=None):
    return r.new_empty((r.numel(),)), torch.empty_like(y)


class _TabledOuterSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y, key, slots,
                inner, pcoef, packed, ppacked):
        from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum_forward
        from fastmace.kernels.triton.radial_table import radial_table_forward

        ctx.constants = (r_lo, bin_scale, r_max)
        ctx.inner, ctx.columns = inner, int(columns)
        ctx.shape, ctx.slots = r.shape, int(slots)
        ctx.tabled = tabled_forward(r.device) and r.dtype == torch.float32
        ctx.power = ctx.tabled and pcoef is not None and r.is_cuda and power_backward(r.device)
        if ctx.tabled:  # nothing of a kept: the backward evaluates it again
            # PACKED_COEF: the packed copies exist where it was on at the table's build (saved last, if any)
            pack = ppacked if ctx.power else packed
            ctx.save_for_backward(pcoef if ctx.power else coef, y, key, r, pair, dcoef, starts, scales, bins,
                                  *(() if pack is None else (pack,)))
            return tabled_outer_sum_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y,
                                            key, slots, inner, packed)
        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, True, columns)
        ctx.save_for_backward(a, y, key, r, pair, dcoef, starts, scales, bins)
        return edge_outer_sum_forward(a, y, key, slots, inner), density

    @staticmethod
    def backward(ctx, grad, grad_density):
        first, y, key, r, pair, dcoef, starts, scales, bins, *packed = ctx.saved_tensors
        if grad is None:
            grad = y.new_zeros((ctx.slots, y.shape[1] * ctx.columns))
        if grad_density is None:
            grad_density = y.new_zeros((0,))
        if ctx.tabled:  # first is coef
            dr, grad_y = tabled_outer_sum_backward(grad, y.new_empty((0, 0)), y, key, r, pair, dcoef, starts, scales,
                                                   bins, grad_density, *ctx.constants, ctx.inner, first, ctx.columns,
                                                   ctx.power, packed[0] if packed else None)
        else:  # first is a
            dr, grad_y = tabled_outer_sum_backward(grad, first, y, key, r, pair, dcoef, starts, scales, bins,
                                                   grad_density, *ctx.constants, ctx.inner)
        return (dr.reshape(ctx.shape), None, None, None, None, None, None, None, None, None, None, grad_y, None,
                None, None, None, None, None)


def tabled_outer_sum(r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y, key, slots,
                     inner=1, pcoef=None, packed=None, ppacked=None):
    """``(S [slots, D * K], edge density [E, 1])``: the table's leading ``columns`` = K values ``a`` of each edge
    length ``r``, summed as ``S[key] += y a^T`` (``edge_outer_sum``'s segmented forward, d major), with the fused
    backward (module docstring). Differentiable in ``r`` and ``y``. ``packed`` / ``ppacked``: ``packed_coef`` of
    ``coef`` / ``pcoef`` (``PACKED_COEF``; None: the unpacked rows)."""
    return _TabledOuterSum.apply(r, pair, coef, dcoef, starts, scales, bins, float(r_lo), float(bin_scale),
                                 float(r_max), int(columns), y, key, int(slots), int(inner), pcoef, packed, ppacked)


__all__ = ["CHUNK", "FORWARD_CHUNK", "PACKED_COEF", "PACKED_COEF_GPUS", "POWER_BACKWARD", "POWER_BACKWARD_GPUS",
           "TABLED_FORWARD", "TABLED_FORWARD_GPUS", "WARPS", "packed_coef", "packed_coef_default", "power_backward",
           "tabled_forward", "tabled_outer_sum", "tabled_outer_sum_backward", "tabled_outer_sum_forward"]
