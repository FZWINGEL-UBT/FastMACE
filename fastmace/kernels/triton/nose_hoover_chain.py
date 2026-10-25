# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: ASE 3.27's Nose-Hoover chain and MTK cell step (ase/md/nose_hoover_chain.py) as Triton kernels, via
# fastmace/integrators/nose_hoover.py and mtk.py; LGPL-2.1-or-later; see NOTICE.md.
"""Nose-Hoover chain halves and the MTK cell's 3 x 3 exp and exprel in one Triton program each (4-T03b; exact).

The PyTorch stages of ``fastmace.integrators.nose_hoover.NoseHooverChain.integrate`` and
``fastmace.integrators.mtk.exp_and_exprel`` are hundreds of scalar kernels per MD step: a chain half is
``loops`` x 3 Suzuki-Yoshida sub-steps of ``2 length`` link updates, a matrix function 14 Horner steps and 4
squarings of 3 x 3 products. Each runs here as one single-program kernel with the same arithmetic in the
same order, so the results are bitwise those of the PyTorch stage on the same GPU (as OPT3-T03's
``fused_integrator``):

- float64 throughout, every coefficient loaded from a float64 tensor (a Python float argument would be
  float32), FMA contraction off (``enable_fp_fusion=False``);
- the chain's scalars live in a small vector; an element is read as the sum of the vector masked to it
  (exact: the others are zeros) and written with ``where``;
- a 3 x 3 product is ``times_3x3``'s left fold ``(r0 m0 + r1 m1) + r2 m2``, elementwise.

The caller computes the chain force (a PyTorch reduction, the same in both stages) and applies the
returned scale.
"""

from __future__ import annotations

import torch

try:  # Triton is optional: without it (the CPU environments) only ``available`` is used
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the environment
    triton = None

if triton is not None:

    @triton.jit
    def _get(vector, idx, j):
        return tl.sum(tl.where(idx == j, vector, 0.0), axis=0)

    @triton.jit
    def _link(p, idx, damp_c, inv, d2, F, nf, kT, j: tl.constexpr, M: tl.constexpr):
        """NoseHooverChain._link: p[j] damped by p[j + 1], kicked by its force, damped again."""
        pj = _get(p, idx, j)
        damp = pj  # unused unless j < M - 1
        if j < M - 1:
            damp = tl.exp(_get(p, idx, j + 1) * _get(damp_c, idx, j))
            pj = pj * damp
        if j == 0:
            g = F - nf * kT
        else:
            pm = _get(p, idx, j - 1)
            g = pm * pm * _get(inv, idx, j - 1) - kT
        pj = pj + g * d2
        if j < M - 1:
            pj = pj * damp
        return tl.where(idx == j, pj, p)

    @triton.jit
    def chain_kernel(p_ptr, eta_ptr, table_ptr, inv_ptr, nf_ptr, kt_ptr, f_ptr, s_ptr, M: tl.constexpr,
                     BM: tl.constexpr, LOOPS: tl.constexpr):
        """One chain half: LOOPS x the three Suzuki-Yoshida sub-steps; writes p_eta, eta and the total scale."""
        idx = tl.arange(0, BM)
        live = idx < M
        p = tl.load(p_ptr + idx, mask=live, other=0.0)
        eta = tl.load(eta_ptr + idx, mask=live, other=0.0)
        inv = tl.load(inv_ptr + idx, mask=live, other=0.0)
        nf = tl.load(nf_ptr)
        kT = tl.load(kt_ptr)
        F = tl.load(f_ptr)
        S = tl.full([], 1.0, tl.float64)
        for _ in tl.static_range(LOOPS):
            for k in tl.static_range(3):
                row = table_ptr + k * (2 + 2 * M)
                d2 = tl.load(row)
                scale_c = tl.load(row + 1)
                eta_c = tl.load(row + 2 + idx, mask=live, other=0.0)
                damp_c = tl.load(row + 2 + M + idx, mask=live, other=0.0)
                for jj in tl.static_range(M):
                    p = _link(p, idx, damp_c, inv, d2, F, nf, kT, M - 1 - jj, M)
                eta = eta + p * eta_c
                s = tl.exp(_get(p, idx, 0) * scale_c)
                F = F * s * s
                S = S * s
                for j in tl.static_range(M):
                    p = _link(p, idx, damp_c, inv, d2, F, nf, kT, j, M)
        tl.store(p_ptr + idx, p, mask=live)
        tl.store(eta_ptr + idx, eta, mask=live)
        tl.store(s_ptr, S)

    @triton.jit
    def _times(a, b, rows, cols):
        """times_3x3(a, b) on 4 x 4 blocks (the fourth row and column zero)."""
        out = tl.zeros([4, 4], tl.float64)
        for k in tl.static_range(3):
            column = tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)
            row = tl.sum(tl.where(rows[:, None] == k, b, 0.0), axis=0)
            if k == 0:
                out = column[:, None] * row[None, :]
            else:
                out = out + column[:, None] * row[None, :]
        return out

    @triton.jit
    def expm_kernel(a_ptr, consts_ptr, e_ptr, r_ptr, DEGREE: tl.constexpr, SCALING: tl.constexpr):
        """exp_and_exprel: Horner for exprel of a / 2^SCALING, then SCALING squarings of both."""
        rows = tl.arange(0, 4)
        cols = tl.arange(0, 4)
        live = (rows[:, None] < 3) & (cols[None, :] < 3)
        offsets = rows[:, None] * 3 + cols[None, :]
        b = tl.load(a_ptr + offsets, mask=live, other=0.0) * tl.load(consts_ptr)       # times 2^-SCALING
        eye = tl.where((rows[:, None] == cols[None, :]) & live, 1.0, 0.0).to(tl.float64)
        r = eye
        for i in tl.static_range(DEGREE - 1):
            r = eye + _times(b, r, rows, cols) * tl.load(consts_ptr + 1 + i)          # 1 / (k + 1), k descending
        e = eye + _times(b, r, rows, cols)
        for _ in tl.static_range(SCALING):
            r = _times(r, e + eye, rows, cols) * 0.5
            e = _times(e, e, rows, cols)
        tl.store(e_ptr + offsets, e, mask=live)
        tl.store(r_ptr + offsets, r, mask=live)


def available(device: torch.device) -> str | None:
    """``None`` if the fused kernels run on ``device``, else why not."""
    if device.type != "cuda":
        return f"the fused chain kernels need CUDA, the state is on {device}"
    if triton is None:
        return "Triton is not installed"
    return None


def chain_half(p_eta, eta, table, inv, nf, kT, force, scale, *, length: int, loops: int) -> None:
    """Launch one chain half (``NoseHooverChain.integrate``'s arithmetic): p_eta and eta in place, the total
    momentum scale into ``scale``."""
    block = 1 << max(length - 1, 0).bit_length()
    chain_kernel[(1,)](p_eta, eta, table, inv, nf, kT, force, scale, M=length, BM=max(block, 2), LOOPS=loops,
                       enable_fp_fusion=False, num_warps=1)


def exp_and_exprel(a, consts, e, r, *, degree: int, scaling: int) -> None:
    """Launch ``mtk.exp_and_exprel`` of the 3 x 3 ``a`` into ``e`` and ``r``; ``consts`` holds 2^-scaling and the
    Horner reciprocals."""
    expm_kernel[(1,)](a, consts, e, r, DEGREE=degree, SCALING=scaling, enable_fp_fusion=False, num_warps=1)
