"""POLAR k-space sums without the dense [k, atoms] cosine and sine matrices (off-polar-alex, 6 November 2026).

graph_longrange's reciprocal-space terms multiply ``cos(k . r)`` and ``sin(k . r)`` ([K, N], one structure; K
k-vectors, N atoms) by thin matrices: the structure factor ``C X``, ``S X`` ([K, m], the charge density's Fourier
coefficients) and the projection ``A^T C + B^T S`` ([m, N], the potential at the atoms). PyTorch builds C and S,
multiplies with SIMT GEMMs of 4 to 8 columns, and autograd materialises dense [K, N] gradients of C and S, adds
them and multiplies by the phases' derivative (A100-SXM4-80GB, POLAR-1 on NbCl5, K = 2989, N = 1944: about 2.65 ms
of a force call for every model size). Here each sum recomputes the phases in [BLOCK_K, BLOCK_N] tiles:

- ``structure_factor(k, r, mask, x)``: ``(Yc, Ys)``, ``Yc[k] = mask[k] sum_n cos(theta_kn) x[n]`` (and sin);
- ``fourier_project(k, r, mask, a, b)``: ``P[n] = sum_k mask[k] (a[k] cos(theta_kn) + b[k] sin(theta_kn))``;

with ``theta_kn = k_k . r_n`` and ``mask`` the k-vectors kept (the fixed k-set marks those outside the cutoff).
Each backward is the other sum plus one phase-gradient pass, ``dtheta_kn = mask_k sum_m z_nm (alpha_km cos +
beta_km sin)``, reduced to ``dL/dr_n = sum_k dtheta_kn k_k`` and ``dL/dk_k = sum_n dtheta_kn r_n`` (the cell's
gradient, for the stress). Every tile writes its own partial sums, added in a fixed order afterwards: deterministic,
no atomics. Rounding against the dense path (the summation order, and the phase as three fused products instead
of cuBLAS's). Sines and cosines are libdevice's (as ``torch.sin``/``torch.cos``), not the approximate
instructions. Float32 CUDA tensors take the kernels; everything else (the CPU, float64) the dense PyTorch sums, with
the same backward formulas. First derivatives only (MD forces and stress).
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK_K, BLOCK_N, WARPS = 64, 64, 4


def _uses_triton(*tensors) -> bool:
    return all(t.is_cuda and t.dtype == torch.float32 for t in tensors)


def _phases(k, r):
    return k @ r.t()  # [K, N]


# --------------------------------------------------------------------------- kernels


@cache
def _kernels():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _tile(k_ptr, r_ptr, K, N, pk, pn, BK: tl.constexpr, BN: tl.constexpr):
        rows = pk * BK + tl.arange(0, BK)
        cols = pn * BN + tl.arange(0, BN)
        in_k, in_n = rows < K, cols < N
        theta = tl.zeros((BK, BN), dtype=tl.float32)
        for d in tl.static_range(3):
            kd = tl.load(k_ptr + rows * 3 + d, mask=in_k, other=0.0)
            rd = tl.load(r_ptr + cols * 3 + d, mask=in_n, other=0.0)
            theta += kd[:, None] * rd[None, :]
        return rows, cols, in_k, in_n, libdevice.cos(theta), libdevice.sin(theta)

    @triton.jit
    def ksum(k_ptr, r_ptr, mask_ptr, x_ptr, yc_ptr, ys_ptr, K, N, M: tl.constexpr,
             BK: tl.constexpr, BN: tl.constexpr):
        """Partials [N blocks, K, M]: mask[k] sum_n over this tile of cos / sin(theta) x[n, m]."""
        pk, pn = tl.program_id(0), tl.program_id(1)
        rows, cols, in_k, in_n, c, s = _tile(k_ptr, r_ptr, K, N, pk, pn, BK, BN)
        keep = tl.load(mask_ptr + rows, mask=in_k, other=0.0)
        base = pn * K * M
        for m in tl.static_range(M):
            xm = tl.load(x_ptr + cols * M + m, mask=in_n, other=0.0)
            tl.store(yc_ptr + base + rows * M + m, tl.sum(c * xm[None, :], axis=1) * keep, mask=in_k)
            tl.store(ys_ptr + base + rows * M + m, tl.sum(s * xm[None, :], axis=1) * keep, mask=in_k)

    @triton.jit
    def nsum(k_ptr, r_ptr, mask_ptr, u_ptr, v_ptr, out_ptr, K, N, M: tl.constexpr,
             BK: tl.constexpr, BN: tl.constexpr):
        """Partials [K blocks, N, M]: sum_k over this tile of mask[k] (u[k, m] cos + v[k, m] sin)."""
        pk, pn = tl.program_id(0), tl.program_id(1)
        rows, cols, in_k, in_n, c, s = _tile(k_ptr, r_ptr, K, N, pk, pn, BK, BN)
        keep = tl.load(mask_ptr + rows, mask=in_k, other=0.0)
        base = pk * N * M
        for m in tl.static_range(M):
            um = tl.load(u_ptr + rows * M + m, mask=in_k, other=0.0) * keep
            vm = tl.load(v_ptr + rows * M + m, mask=in_k, other=0.0) * keep
            tl.store(out_ptr + base + cols * M + m, tl.sum(c * um[:, None] + s * vm[:, None], axis=0), mask=in_n)

    @triton.jit
    def phase_grad(k_ptr, r_ptr, mask_ptr, alpha_ptr, beta_ptr, z_ptr, gr_ptr, gk_ptr, K, N, M: tl.constexpr,
                   BK: tl.constexpr, BN: tl.constexpr):
        """dtheta = mask[k] sum_m z[n, m] (alpha[k, m] cos + beta[k, m] sin) on this tile; partials
        gr [K blocks, N, 3] = sum_k dtheta k and gk [N blocks, K, 3] = sum_n dtheta r."""
        pk, pn = tl.program_id(0), tl.program_id(1)
        rows, cols, in_k, in_n, c, s = _tile(k_ptr, r_ptr, K, N, pk, pn, BK, BN)
        keep = tl.load(mask_ptr + rows, mask=in_k, other=0.0)
        p = tl.zeros((BK, BN), dtype=tl.float32)
        q = tl.zeros((BK, BN), dtype=tl.float32)
        for m in tl.static_range(M):
            zm = tl.load(z_ptr + cols * M + m, mask=in_n, other=0.0)
            am = tl.load(alpha_ptr + rows * M + m, mask=in_k, other=0.0)
            bm = tl.load(beta_ptr + rows * M + m, mask=in_k, other=0.0)
            p += am[:, None] * zm[None, :]
            q += bm[:, None] * zm[None, :]
        dtheta = (c * p + s * q) * keep[:, None]
        for d in tl.static_range(3):
            kd = tl.load(k_ptr + rows * 3 + d, mask=in_k, other=0.0)
            rd = tl.load(r_ptr + cols * 3 + d, mask=in_n, other=0.0)
            tl.store(gr_ptr + pk * N * 3 + cols * 3 + d, tl.sum(dtheta * kd[:, None], axis=0), mask=in_n)
            tl.store(gk_ptr + pn * K * 3 + rows * 3 + d, tl.sum(dtheta * rd[None, :], axis=1), mask=in_k)

    return ksum, nsum, phase_grad


def _grid(K, N):
    return (-(-K // BLOCK_K), -(-N // BLOCK_N))


# --------------------------------------------------------------------------- custom ops (dense on the CPU)


@torch.library.custom_op("fastmace::fourier_ksum", mutates_args=())
def fourier_ksum(k: torch.Tensor, r: torch.Tensor, mask: torch.Tensor,
                 x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mask * cos(k r^T) @ x, mask * sin(k r^T) @ x)``, ``[K, M]`` each."""
    if not _uses_triton(k, r, mask, x):
        theta = _phases(k, r)
        return mask[:, None] * (torch.cos(theta) @ x), mask[:, None] * (torch.sin(theta) @ x)
    K, N, M = k.shape[0], r.shape[0], x.shape[1]
    grid = _grid(K, N)
    yc = k.new_empty((grid[1], K, M))
    ys = k.new_empty((grid[1], K, M))
    if K and N and M:
        _kernels()[0][grid](k.contiguous(), r.contiguous(), mask.contiguous(), x.contiguous(), yc, ys, K, N, M,
                            BK=BLOCK_K, BN=BLOCK_N, num_warps=WARPS)
    else:
        yc.zero_(), ys.zero_()
    return yc.sum(0), ys.sum(0)


@fourier_ksum.register_fake
def _(k, r, mask, x):
    return k.new_empty((k.shape[0], x.shape[1])), k.new_empty((k.shape[0], x.shape[1]))


@torch.library.custom_op("fastmace::fourier_nsum", mutates_args=())
def fourier_nsum(k: torch.Tensor, r: torch.Tensor, mask: torch.Tensor, u: torch.Tensor,
                 v: torch.Tensor) -> torch.Tensor:
    """``cos(k r^T)^T @ (mask u) + sin(k r^T)^T @ (mask v)``, ``[N, M]``."""
    if not _uses_triton(k, r, mask, u, v):
        theta = _phases(k, r)
        return torch.cos(theta).t() @ (mask[:, None] * u) + torch.sin(theta).t() @ (mask[:, None] * v)
    K, N, M = k.shape[0], r.shape[0], u.shape[1]
    grid = _grid(K, N)
    out = k.new_empty((grid[0], N, M))
    if K and N and M:
        _kernels()[1][grid](k.contiguous(), r.contiguous(), mask.contiguous(), u.contiguous(), v.contiguous(), out,
                            K, N, M, BK=BLOCK_K, BN=BLOCK_N, num_warps=WARPS)
    else:
        out.zero_()
    return out.sum(0)


@fourier_nsum.register_fake
def _(k, r, mask, u, v):
    return r.new_empty((r.shape[0], u.shape[1]))


@torch.library.custom_op("fastmace::fourier_phase_grad", mutates_args=())
def fourier_phase_grad(k: torch.Tensor, r: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor,
                       beta: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dL/dr [N, 3], dL/dk [K, 3])`` for ``dtheta = mask (alpha z^T * cos + beta z^T * sin)``."""
    if not _uses_triton(k, r, mask, alpha, beta, z):
        theta = _phases(k, r)
        dtheta = mask[:, None] * (torch.cos(theta) * (alpha @ z.t()) + torch.sin(theta) * (beta @ z.t()))
        return dtheta.t() @ k, dtheta @ r
    K, N, M = k.shape[0], r.shape[0], z.shape[1]
    grid = _grid(K, N)
    gr = k.new_empty((grid[0], N, 3))
    gk = k.new_empty((grid[1], K, 3))
    if K and N and M:
        _kernels()[2][grid](k.contiguous(), r.contiguous(), mask.contiguous(), alpha.contiguous(),
                            beta.contiguous(), z.contiguous(), gr, gk, K, N, M,
                            BK=BLOCK_K, BN=BLOCK_N, num_warps=WARPS)
    else:
        gr.zero_(), gk.zero_()
    return gr.sum(0), gk.sum(0)


@fourier_phase_grad.register_fake
def _(k, r, mask, alpha, beta, z):
    return r.new_empty((r.shape[0], 3)), k.new_empty((k.shape[0], 3))


# --------------------------------------------------------------------------- autograd


def _zero_if_none(grad, like):
    return torch.zeros_like(like) if grad is None else grad


class _StructureFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, r, mask, x):
        ctx.save_for_backward(k, r, mask, x)
        return fourier_ksum(k, r, mask, x)

    @staticmethod
    def backward(ctx, grad_cos, grad_sin):
        k, r, mask, x = ctx.saved_tensors
        grad_cos = _zero_if_none(grad_cos, k.new_empty((k.shape[0], x.shape[1])))
        grad_sin = _zero_if_none(grad_sin, k.new_empty((k.shape[0], x.shape[1])))
        grad_x = fourier_nsum(k, r, mask, grad_cos, grad_sin) if ctx.needs_input_grad[3] else None
        grad_k = grad_r = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # d/dtheta (cos, sin) = (-sin, cos): alpha (the cosine's factor) = grad_sin, beta = -grad_cos
            grad_r, grad_k = fourier_phase_grad(k, r, mask, grad_sin, -grad_cos, x)
        return grad_k, grad_r, None, grad_x


class _FourierProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, r, mask, a, b):
        ctx.save_for_backward(k, r, mask, a, b)
        return fourier_nsum(k, r, mask, a, b)

    @staticmethod
    def backward(ctx, grad):
        k, r, mask, a, b = ctx.saved_tensors
        grad_a = grad_b = grad_k = grad_r = None
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            grad_a, grad_b = fourier_ksum(k, r, mask, grad)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # d/dtheta (a cos + b sin) = b cos - a sin
            grad_r, grad_k = fourier_phase_grad(k, r, mask, b, -a, grad)
        return grad_k, grad_r, None, grad_a, grad_b


def structure_factor(k, r, mask, x):
    """``(Yc, Ys)``: ``mask[:, None] * cos(k r^T) @ x`` and the same with sin; ``k`` [K, 3], ``r`` [N, 3], ``mask``
    [K] (1 kept, 0 not), ``x`` [N, M]."""
    return _StructureFactor.apply(k, r, mask, x)


def fourier_project(k, r, mask, a, b):
    """``cos(k r^T)^T @ (mask a) + sin(k r^T)^T @ (mask b)``, ``[N, M]``; ``a``, ``b`` [K, M]."""
    return _FourierProject.apply(k, r, mask, a, b)


__all__ = ["fourier_ksum", "fourier_nsum", "fourier_phase_grad", "fourier_project", "structure_factor"]
