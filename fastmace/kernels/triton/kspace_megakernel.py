"""POLAR k-space field features in one persistent launch per direction (polar-megakernel, 7 November 2026).

One call of graph_longrange's ``GTOElectrostaticFeatures`` (full PBC, one structure) maps the source multipoles of
every atom, ``x`` [N, C, J] (C channels, e.g. the two spins; J = sigma * m density components), to field features
[N, C, F]:

    Yc[k, c, j] = mask_k sum_n cos(theta_kn) x[n, c, j]          (Ys with sin; theta_kn = k_k . r_n)
    ur[k, c]    = sum_j dr[k, j] Yc + di[k, j] Ys,   ui = sum_j di Yc - dr Ys
    a[k, c, f]  = w_k (ur pr[k, f] + ui pi[k, f]),   b = w_k (ur pi - ui pr)
    out[n, c, f] = s0 sum_k (a cos(theta_kn) + b sin(theta_kn)),   s0 = 2 / (2 pi)^3

with ``w = (2 pi)^3 / volume * k_factor_coulomb * FIELD_CONSTANT`` (density scale, Coulomb kernel) and ``pr, pi``
the projection basis times ``k_factor_proj``, all per k-vector and formed by the caller. graph_longrange runs this as
dense [K, N] cosines and sines, two thin GEMMs per side and about ten pointwise kernels; fastmace.kernels.triton.
fourier (FUSED_FOURIER) recomputes the phases in tiles but keeps the per-k glue and the partial sums as separate
launches. Here the whole call is one persistent kernel with a grid barrier:

- phase 1, per k-tile over all atoms: Yc, Ys with ``tl.dot`` (IEEE FP32), the per-k glue in registers, then a, b
  (and Yc, Ys, kept for the backward) to a small [K, ...] scratch;
- grid barrier (an atomic arrival counter; the grid is at most one CTA per SM, so every CTA is resident);
- phase 2, per atom-tile over all k-vectors: out.

The backward is the same shape: phase 1 sums the output gradient into dA, dB per k-tile and applies the per-k glue's
backward (the k-vector factors' gradients dr, di, w, pr, pi for the stress, and dYc, dYs); phase 2 per atom tile
gives dx and the phase gradient, reduced to dL/dr (complete per tile) and per-tile partials of dL/dk; phase 3 (only
when the k-vectors need a gradient) adds those partials in a fixed order. Deterministic: no atomic sums, only the
barrier counter. Rounding against the dense path (summation order; the per-k factors combined before use).
Float32 CUDA tensors take the kernels; everything else the dense PyTorch formulas (the same as the reference).
First derivatives only (MD forces and stress).
"""

from __future__ import annotations

import math
from functools import cache

import torch

S0 = 2.0 / (2.0 * math.pi) ** 3
BLOCK_K, BLOCK_N, WARPS = 64, 64, 4

_COUNTERS: dict = {}


def _uses_triton(*tensors) -> bool:
    return all(t.is_cuda and t.dtype == torch.float32 for t in tensors)


def _pad(n: int) -> int:
    return max(16, 1 << (int(n) - 1).bit_length())


def _counter(device):
    """A grid-barrier arrival counter per device, never reset: each launch adds its grid size once per barrier,
    so a barrier waits for the next multiple of the grid (one grid size per counter: see ``_grid``)."""
    key = (device.type, device.index)
    counter = _COUNTERS.get(key)
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int64, device=device)
        _COUNTERS[key] = counter
    return counter


def _grid(device) -> int:
    """One CTA per SM: every CTA is resident (4 warps, under 100 KB of shared memory), so the barrier cannot hang.
    The same grid size for every launch on the device keeps the counter at a multiple of it between launches."""
    return torch.cuda.get_device_properties(device).multi_processor_count


# --------------------------------------------------------------------------- dense reference


def _glue(yc, ys, dr, di, w, pr, pi):
    """(ur, ui, a, b) from Yc, Ys [K, C, J]; ur, ui [K, C]; a, b [K, C, F]."""
    ur = (dr[:, None, :] * yc + di[:, None, :] * ys).sum(-1)
    ui = (di[:, None, :] * yc - dr[:, None, :] * ys).sum(-1)
    rr, ri = w[:, None] * ur, w[:, None] * ui
    a = rr[..., None] * pr[:, None, :] + ri[..., None] * pi[:, None, :]
    b = rr[..., None] * pi[:, None, :] - ri[..., None] * pr[:, None, :]
    return ur, ui, a, b


def dense_forward(k, r, mask, x, dr, di, w, pr, pi):
    """``(out [N, C, F], yc, ys [K, C, J], a, b [K, C, F])`` with dense cosines and sines (any device, dtype)."""
    theta = k @ r.t()
    cos, sin = torch.cos(theta), torch.sin(theta)
    yc = mask[:, None, None] * torch.einsum("kn,ncj->kcj", cos, x)
    ys = mask[:, None, None] * torch.einsum("kn,ncj->kcj", sin, x)
    _, _, a, b = _glue(yc, ys, dr, di, w, pr, pi)
    out = S0 * (torch.einsum("kn,kcf->ncf", cos, a) + torch.einsum("kn,kcf->ncf", sin, b))
    return out, yc, ys, a, b


def dense_backward(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi):
    """``(dk, dr_pos, dx, d_dr, d_di, dw, dpr, dpi)`` for the output gradient ``g`` [N, C, F]."""
    theta = k @ r.t()
    cos, sin = torch.cos(theta), torch.sin(theta)
    da = S0 * torch.einsum("kn,ncf->kcf", cos, g)
    db = S0 * torch.einsum("kn,ncf->kcf", sin, g)
    ur, ui, _, _ = _glue(yc, ys, dr, di, w, pr, pi)
    rr, ri = w[:, None] * ur, w[:, None] * ui
    drr = (da * pr[:, None, :] + db * pi[:, None, :]).sum(-1)
    dri = (da * pi[:, None, :] - db * pr[:, None, :]).sum(-1)
    dpr = (da * rr[..., None] - db * ri[..., None]).sum(1)
    dpi = (da * ri[..., None] + db * rr[..., None]).sum(1)
    dw = (drr * ur + dri * ui).sum(1)
    wr, wi = w[:, None] * drr, w[:, None] * dri
    dyc = mask[:, None, None] * (wr[..., None] * dr[:, None, :] + wi[..., None] * di[:, None, :])
    dys = mask[:, None, None] * (wr[..., None] * di[:, None, :] - wi[..., None] * dr[:, None, :])
    d_dr = (wr[..., None] * yc - wi[..., None] * ys).sum(1)
    d_di = (wr[..., None] * ys + wi[..., None] * yc).sum(1)
    dx = torch.einsum("kn,kcj->ncj", cos, dyc) + torch.einsum("kn,kcj->ncj", sin, dys)
    dtheta = (cos * (torch.einsum("ncj,kcj->kn", x, dys) + S0 * torch.einsum("ncf,kcf->kn", g, b))
              - sin * (torch.einsum("ncj,kcj->kn", x, dyc) + S0 * torch.einsum("ncf,kcf->kn", g, a)))
    return dtheta @ r, dtheta.t() @ k, dx, d_dr, d_di, dw, dpr, dpi


# --------------------------------------------------------------------------- kernels


@cache
def _kernels():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _barrier(counter_ptr, G):
        tl.debug_barrier()
        old = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
        target = (old // G + 1) * G
        seen = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
        while seen < target:
            seen = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
        tl.debug_barrier()

    @triton.jit
    def _phase(k_ptr, r_ptr, rows, cols, in_k, in_n):
        """theta [len(rows), len(cols)] = k[rows] . r[cols]."""
        kx = tl.load(k_ptr + rows * 3, mask=in_k, other=0.0)
        ky = tl.load(k_ptr + rows * 3 + 1, mask=in_k, other=0.0)
        kz = tl.load(k_ptr + rows * 3 + 2, mask=in_k, other=0.0)
        rx = tl.load(r_ptr + cols * 3, mask=in_n, other=0.0)
        ry = tl.load(r_ptr + cols * 3 + 1, mask=in_n, other=0.0)
        rz = tl.load(r_ptr + cols * 3 + 2, mask=in_n, other=0.0)
        return kx[:, None] * rx[None, :] + ky[:, None] * ry[None, :] + kz[:, None] * rz[None, :]

    @triton.jit
    def forward(k_ptr, r_ptr, mask_ptr, x_ptr, dr_ptr, di_ptr, w_ptr, pr_ptr, pi_ptr,
                out_ptr, yc_ptr, ys_ptr, a_ptr, b_ptr, counter_ptr, K, N, G, s0,
                C: tl.constexpr, J: tl.constexpr, F: tl.constexpr, CJP: tl.constexpr, CFP: tl.constexpr,
                BK: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        cj = tl.arange(0, CJP)
        cf = tl.arange(0, CFP)
        in_cj = cj < C * J
        in_cf = cf < C * F
        # ---- phase 1: per k-tile, Yc/Ys over all atoms, the per-k glue, a and b
        for t in range(pid, tl.cdiv(K, BK), G):
            rows = t * BK + tl.arange(0, BK)
            in_k = rows < K
            yc = tl.zeros((BK, CJP), dtype=tl.float32)
            ys = tl.zeros((BK, CJP), dtype=tl.float32)
            for n0 in range(0, N, BN):
                cols = n0 + tl.arange(0, BN)
                in_n = cols < N
                theta = _phase(k_ptr, r_ptr, rows, cols, in_k, in_n)
                xs = tl.load(x_ptr + cols[:, None] * (C * J) + cj[None, :], mask=in_n[:, None] & in_cj[None, :],
                             other=0.0)
                yc = tl.dot(libdevice.cos(theta), xs, yc, input_precision="ieee")
                ys = tl.dot(libdevice.sin(theta), xs, ys, input_precision="ieee")
            keep = tl.load(mask_ptr + rows, mask=in_k, other=0.0)
            yc = yc * keep[:, None]
            ys = ys * keep[:, None]
            kj = in_k[:, None] & in_cj[None, :]
            tl.store(yc_ptr + rows[:, None] * (C * J) + cj[None, :], yc, mask=kj)
            tl.store(ys_ptr + rows[:, None] * (C * J) + cj[None, :], ys, mask=kj)
            jj = cj % J
            dr = tl.load(dr_ptr + rows[:, None] * J + jj[None, :], mask=kj, other=0.0)
            di = tl.load(di_ptr + rows[:, None] * J + jj[None, :], mask=kj, other=0.0)
            pre = dr * yc + di * ys
            pim = di * yc - dr * ys
            w = tl.load(w_ptr + rows, mask=in_k, other=0.0)
            kf = in_k[:, None] & in_cf[None, :]
            ff = cf % F
            pr = tl.load(pr_ptr + rows[:, None] * F + ff[None, :], mask=kf, other=0.0)
            pi = tl.load(pi_ptr + rows[:, None] * F + ff[None, :], mask=kf, other=0.0)
            a = tl.zeros((BK, CFP), dtype=tl.float32)
            b = tl.zeros((BK, CFP), dtype=tl.float32)
            for c in tl.static_range(C):
                ur = tl.sum(tl.where((cj // J == c)[None, :], pre, 0.0), axis=1) * w
                ui = tl.sum(tl.where((cj // J == c)[None, :], pim, 0.0), axis=1) * w
                on = (cf // F == c)[None, :]
                a += tl.where(on, ur[:, None] * pr + ui[:, None] * pi, 0.0)
                b += tl.where(on, ur[:, None] * pi - ui[:, None] * pr, 0.0)
            tl.store(a_ptr + rows[:, None] * (C * F) + cf[None, :], a, mask=kf)
            tl.store(b_ptr + rows[:, None] * (C * F) + cf[None, :], b, mask=kf)
        _barrier(counter_ptr, G)
        # ---- phase 2: per atom tile, out over all k-vectors
        for t in range(pid, tl.cdiv(N, BN), G):
            cols = t * BN + tl.arange(0, BN)
            in_n = cols < N
            acc = tl.zeros((BN, CFP), dtype=tl.float32)
            for k0 in range(0, K, BK):
                rows = k0 + tl.arange(0, BK)
                in_k = rows < K
                theta = _phase(r_ptr, k_ptr, cols, rows, in_n, in_k)  # [BN, BK] = r . k
                kf = in_k[:, None] & in_cf[None, :]
                a = tl.load(a_ptr + rows[:, None] * (C * F) + cf[None, :], mask=kf, other=0.0, cache_modifier=".cg")
                b = tl.load(b_ptr + rows[:, None] * (C * F) + cf[None, :], mask=kf, other=0.0, cache_modifier=".cg")
                acc = tl.dot(libdevice.cos(theta), a, acc, input_precision="ieee")
                acc = tl.dot(libdevice.sin(theta), b, acc, input_precision="ieee")
            tl.store(out_ptr + cols[:, None] * (C * F) + cf[None, :], acc * s0, mask=in_n[:, None] & in_cf[None, :])

    @triton.jit
    def backward(k_ptr, r_ptr, mask_ptr, x_ptr, g_ptr, yc_ptr, ys_ptr, a_ptr, b_ptr,
                 dr_ptr, di_ptr, w_ptr, pr_ptr, pi_ptr,
                 dyc_ptr, dys_ptr, gk_part_ptr,
                 gk_ptr, gr_ptr, dx_ptr, ddr_ptr, ddi_ptr, dw_ptr, dpr_ptr, dpi_ptr,
                 counter_ptr, K, N, G, s0,
                 C: tl.constexpr, J: tl.constexpr, F: tl.constexpr, CJP: tl.constexpr, CFP: tl.constexpr,
                 FACTORS: tl.constexpr, GK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        cj = tl.arange(0, CJP)
        cf = tl.arange(0, CFP)
        in_cj = cj < C * J
        in_cf = cf < C * F
        # ---- phase 1: per k-tile, dA/dB over all atoms, the glue's backward
        for t in range(pid, tl.cdiv(K, BK), G):
            rows = t * BK + tl.arange(0, BK)
            in_k = rows < K
            da = tl.zeros((BK, CFP), dtype=tl.float32)
            db = tl.zeros((BK, CFP), dtype=tl.float32)
            for n0 in range(0, N, BN):
                cols = n0 + tl.arange(0, BN)
                in_n = cols < N
                theta = _phase(k_ptr, r_ptr, rows, cols, in_k, in_n)
                gs = tl.load(g_ptr + cols[:, None] * (C * F) + cf[None, :], mask=in_n[:, None] & in_cf[None, :],
                             other=0.0)
                da = tl.dot(libdevice.cos(theta), gs, da, input_precision="ieee")
                db = tl.dot(libdevice.sin(theta), gs, db, input_precision="ieee")
            da = da * s0
            db = db * s0
            kf = in_k[:, None] & in_cf[None, :]
            kj = in_k[:, None] & in_cj[None, :]
            ff = cf % F
            jj = cj % J
            pr = tl.load(pr_ptr + rows[:, None] * F + ff[None, :], mask=kf, other=0.0)
            pi = tl.load(pi_ptr + rows[:, None] * F + ff[None, :], mask=kf, other=0.0)
            dr = tl.load(dr_ptr + rows[:, None] * J + jj[None, :], mask=kj, other=0.0)
            di = tl.load(di_ptr + rows[:, None] * J + jj[None, :], mask=kj, other=0.0)
            yc = tl.load(yc_ptr + rows[:, None] * (C * J) + cj[None, :], mask=kj, other=0.0)
            ys = tl.load(ys_ptr + rows[:, None] * (C * J) + cj[None, :], mask=kj, other=0.0)
            w = tl.load(w_ptr + rows, mask=in_k, other=0.0)
            keep = tl.load(mask_ptr + rows, mask=in_k, other=0.0)
            pre = dr * yc + di * ys
            pim = di * yc - dr * ys
            tpr = da * pr + db * pi
            tpi = da * pi - db * pr
            dyc = tl.zeros((BK, CJP), dtype=tl.float32)
            dys = tl.zeros((BK, CJP), dtype=tl.float32)
            dw = tl.zeros((BK,), dtype=tl.float32)
            # the per-k factors' gradients, accumulated over the channels as [BK, F] / [BK, J] in column c * F + f
            # of channel 0 (folded below)
            gpr = tl.zeros((BK, CFP), dtype=tl.float32)
            gpi = tl.zeros((BK, CFP), dtype=tl.float32)
            gdr = tl.zeros((BK, CJP), dtype=tl.float32)
            gdi = tl.zeros((BK, CJP), dtype=tl.float32)
            for c in tl.static_range(C):
                on_f = (cf // F == c)[None, :]
                on_j = (cj // J == c)[None, :]
                ur = tl.sum(tl.where(on_j, pre, 0.0), axis=1)
                ui = tl.sum(tl.where(on_j, pim, 0.0), axis=1)
                drr = tl.sum(tl.where(on_f, tpr, 0.0), axis=1)
                dri = tl.sum(tl.where(on_f, tpi, 0.0), axis=1)
                dw += drr * ur + dri * ui
                wr = w * drr
                wi = w * dri
                dyc += tl.where(on_j, wr[:, None] * dr + wi[:, None] * di, 0.0)
                dys += tl.where(on_j, wr[:, None] * di - wi[:, None] * dr, 0.0)
                if FACTORS:
                    rr = w * ur
                    ri = w * ui
                    gpr += tl.where(on_f, da * rr[:, None] - db * ri[:, None], 0.0)
                    gpi += tl.where(on_f, da * ri[:, None] + db * rr[:, None], 0.0)
                    gdr += tl.where(on_j, wr[:, None] * yc - wi[:, None] * ys, 0.0)
                    gdi += tl.where(on_j, wr[:, None] * ys + wi[:, None] * yc, 0.0)
            dyc = dyc * keep[:, None]
            dys = dys * keep[:, None]
            tl.store(dyc_ptr + rows[:, None] * (C * J) + cj[None, :], dyc, mask=kj)
            tl.store(dys_ptr + rows[:, None] * (C * J) + cj[None, :], dys, mask=kj)
            if FACTORS:
                tl.store(dw_ptr + rows, dw, mask=in_k)
                # fold the channels: column f of the output = sum over c of column c * F + f (fixed order)
                for c in tl.static_range(C):
                    tl.store(dpr_ptr + c * K * F + rows[:, None] * F + ff[None, :], gpr,
                             mask=kf & (cf // F == c)[None, :])
                    tl.store(dpi_ptr + c * K * F + rows[:, None] * F + ff[None, :], gpi,
                             mask=kf & (cf // F == c)[None, :])
                    tl.store(ddr_ptr + c * K * J + rows[:, None] * J + jj[None, :], gdr,
                             mask=kj & (cj // J == c)[None, :])
                    tl.store(ddi_ptr + c * K * J + rows[:, None] * J + jj[None, :], gdi,
                             mask=kj & (cj // J == c)[None, :])
        _barrier(counter_ptr, G)
        # ---- phase 2: per atom tile over all k: dx, the phase gradient -> dL/dr, partials of dL/dk
        for t in range(pid, tl.cdiv(N, BN), G):
            cols = t * BN + tl.arange(0, BN)
            in_n = cols < N
            nj = in_n[:, None] & in_cj[None, :]
            nf = in_n[:, None] & in_cf[None, :]
            xs = tl.load(x_ptr + cols[:, None] * (C * J) + cj[None, :], mask=nj, other=0.0)
            gs = tl.load(g_ptr + cols[:, None] * (C * F) + cf[None, :], mask=nf, other=0.0) * s0
            dx = tl.zeros((BN, CJP), dtype=tl.float32)
            grx = tl.zeros((BN,), dtype=tl.float32)
            gry = tl.zeros((BN,), dtype=tl.float32)
            grz = tl.zeros((BN,), dtype=tl.float32)
            rx = tl.load(r_ptr + cols * 3, mask=in_n, other=0.0)
            ry = tl.load(r_ptr + cols * 3 + 1, mask=in_n, other=0.0)
            rz = tl.load(r_ptr + cols * 3 + 2, mask=in_n, other=0.0)
            for k0 in range(0, K, BK):
                rows = k0 + tl.arange(0, BK)
                in_k = rows < K
                kj = in_k[:, None] & in_cj[None, :]
                kf = in_k[:, None] & in_cf[None, :]
                theta = _phase(r_ptr, k_ptr, cols, rows, in_n, in_k)  # [BN, BK]
                cos = libdevice.cos(theta)
                sin = libdevice.sin(theta)
                dyc = tl.load(dyc_ptr + rows[:, None] * (C * J) + cj[None, :], mask=kj, other=0.0,
                              cache_modifier=".cg")
                dys = tl.load(dys_ptr + rows[:, None] * (C * J) + cj[None, :], mask=kj, other=0.0,
                              cache_modifier=".cg")
                a = tl.load(a_ptr + rows[:, None] * (C * F) + cf[None, :], mask=kf, other=0.0)
                b = tl.load(b_ptr + rows[:, None] * (C * F) + cf[None, :], mask=kf, other=0.0)
                dx = tl.dot(cos, dyc, dx, input_precision="ieee")
                dx = tl.dot(sin, dys, dx, input_precision="ieee")
                p = tl.dot(xs, tl.trans(dys), input_precision="ieee")
                p = tl.dot(gs, tl.trans(b), p, input_precision="ieee")
                q = tl.dot(xs, tl.trans(dyc), input_precision="ieee")
                q = tl.dot(gs, tl.trans(a), q, input_precision="ieee")
                dtheta = cos * p - sin * q  # [BN, BK]
                kx = tl.load(k_ptr + rows * 3, mask=in_k, other=0.0)
                ky = tl.load(k_ptr + rows * 3 + 1, mask=in_k, other=0.0)
                kz = tl.load(k_ptr + rows * 3 + 2, mask=in_k, other=0.0)
                grx += tl.sum(dtheta * kx[None, :], axis=1)
                gry += tl.sum(dtheta * ky[None, :], axis=1)
                grz += tl.sum(dtheta * kz[None, :], axis=1)
                if GK:
                    base = gk_part_ptr + t * K * 3 + rows * 3
                    tl.store(base, tl.sum(dtheta * rx[:, None], axis=0), mask=in_k)
                    tl.store(base + 1, tl.sum(dtheta * ry[:, None], axis=0), mask=in_k)
                    tl.store(base + 2, tl.sum(dtheta * rz[:, None], axis=0), mask=in_k)
            tl.store(dx_ptr + cols[:, None] * (C * J) + cj[None, :], dx, mask=nj)
            tl.store(gr_ptr + cols * 3, grx, mask=in_n)
            tl.store(gr_ptr + cols * 3 + 1, gry, mask=in_n)
            tl.store(gr_ptr + cols * 3 + 2, grz, mask=in_n)
        if GK:
            _barrier(counter_ptr, G)
            # ---- phase 3: dL/dk = the atom tiles' partials, added in tile order
            tiles = tl.cdiv(N, BN)
            for t in range(pid, tl.cdiv(K * 3, BK), G):
                idx = t * BK + tl.arange(0, BK)
                ok = idx < K * 3
                acc = tl.zeros((BK,), dtype=tl.float32)
                for p in range(0, tiles):
                    acc += tl.load(gk_part_ptr + p * K * 3 + idx, mask=ok, other=0.0, cache_modifier=".cg")
                tl.store(gk_ptr + idx, acc, mask=ok)

    return forward, backward


# --------------------------------------------------------------------------- custom ops


@torch.library.custom_op("fastmace::kspace_field_forward", mutates_args=())
def kspace_field_forward(k: torch.Tensor, r: torch.Tensor, mask: torch.Tensor, x: torch.Tensor, dr: torch.Tensor,
                         di: torch.Tensor, w: torch.Tensor, pr: torch.Tensor,
                         pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                    torch.Tensor]:
    """``(out [N, C, F], yc, ys [K, C, J], a, b [K, C, F])`` (module docstring)."""
    if not _uses_triton(k, r, mask, x, dr, di, w, pr, pi):
        return tuple(t.contiguous() for t in dense_forward(k, r, mask, x, dr, di, w, pr, pi))
    K, N, C, J, F = k.shape[0], r.shape[0], x.shape[1], x.shape[2], pr.shape[1]
    out = r.new_empty((N, C, F))
    yc, ys = k.new_empty((K, C, J)), k.new_empty((K, C, J))
    a, b = k.new_empty((K, C, F)), k.new_empty((K, C, F))
    if K == 0 or N == 0:
        for t in (out, yc, ys, a, b):
            t.zero_()
        return out, yc, ys, a, b
    G = _grid(k.device)
    _kernels()[0][(G,)](k.contiguous(), r.contiguous(), mask.contiguous(), x.contiguous(), dr.contiguous(),
                        di.contiguous(), w.contiguous(), pr.contiguous(), pi.contiguous(), out, yc, ys, a, b,
                        _counter(k.device), K, N, G, S0, C=C, J=J, F=F, CJP=_pad(C * J), CFP=_pad(C * F),
                        BK=BLOCK_K, BN=BLOCK_N, num_warps=WARPS)
    return out, yc, ys, a, b


@kspace_field_forward.register_fake
def _(k, r, mask, x, dr, di, w, pr, pi):
    K, N, C, J, F = k.shape[0], r.shape[0], x.shape[1], x.shape[2], pr.shape[1]
    return (r.new_empty((N, C, F)), k.new_empty((K, C, J)), k.new_empty((K, C, J)), k.new_empty((K, C, F)),
            k.new_empty((K, C, F)))


@torch.library.custom_op("fastmace::kspace_field_backward", mutates_args=())
def kspace_field_backward(k: torch.Tensor, r: torch.Tensor, mask: torch.Tensor, x: torch.Tensor, g: torch.Tensor,
                          yc: torch.Tensor, ys: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dr: torch.Tensor,
                          di: torch.Tensor, w: torch.Tensor, pr: torch.Tensor, pi: torch.Tensor, factors: bool,
                          grad_k: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dk, dr_pos, dx, d_dr, d_di, dw, dpr, dpi)``; ``dk`` only with ``grad_k``, the k-vector factors' gradients
    only with ``factors`` (else empty)."""
    K, N, C, J, F = k.shape[0], r.shape[0], x.shape[1], x.shape[2], pr.shape[1]
    if not _uses_triton(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi):
        res = dense_backward(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi)
        empty = [k.new_empty((0,)) for _ in range(6)]
        dk = res[0] if grad_k else empty[0]
        rest = [t.contiguous() if factors else e for t, e in zip(res[3:], empty[1:])]
        return (dk, res[1].contiguous(), res[2].contiguous(), *rest)
    G = _grid(k.device)
    tiles = -(-N // BLOCK_N)
    dyc, dys = k.new_empty((K, C, J)), k.new_empty((K, C, J))
    gk_part = k.new_empty((tiles, K, 3) if grad_k else (1,))
    dk = k.new_empty((K, 3) if grad_k else (0,))
    gr, dx = r.new_empty((N, 3)), r.new_empty((N, C, J))
    if factors:
        ddr, ddi = k.new_empty((C, K, J)), k.new_empty((C, K, J))
        dpr, dpi = k.new_empty((C, K, F)), k.new_empty((C, K, F))
        dw = k.new_empty((K,))
    else:
        ddr, ddi, dpr, dpi, dw = (k.new_empty((1,)) for _ in range(5))
    if K == 0 or N == 0:
        for t in (dk, gr, dx, ddr, ddi, dpr, dpi, dw):
            t.zero_()
    else:
        _kernels()[1][(G,)](k.contiguous(), r.contiguous(), mask.contiguous(), x.contiguous(), g.contiguous(),
                            yc.contiguous(), ys.contiguous(), a.contiguous(), b.contiguous(), dr.contiguous(),
                            di.contiguous(), w.contiguous(), pr.contiguous(), pi.contiguous(), dyc, dys, gk_part,
                            dk, gr, dx, ddr, ddi, dw, dpr, dpi, _counter(k.device), K, N, G, S0,
                            C=C, J=J, F=F, CJP=_pad(C * J), CFP=_pad(C * F), FACTORS=factors, GK=grad_k,
                            BK=BLOCK_K, BN=BLOCK_N, num_warps=WARPS)
    if factors:  # the channels' terms, added in channel order (C is 1 or 2: one small kernel)
        ddr, ddi, dpr, dpi = (t.sum(0) for t in (ddr, ddi, dpr, dpi))
        return dk, gr, dx, ddr, ddi, dw, dpr, dpi
    return (dk, gr, dx, *(k.new_empty((0,)) for _ in range(5)))


@kspace_field_backward.register_fake
def _(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi, factors, grad_k):
    K, N, C, J, F = k.shape[0], r.shape[0], x.shape[1], x.shape[2], pr.shape[1]

    def shaped(wanted, *shape):
        return k.new_empty(shape if wanted else (0,))

    return (shaped(grad_k, K, 3), r.new_empty((N, 3)), r.new_empty((N, C, J)), shaped(factors, K, J),
            shaped(factors, K, J), shaped(factors, K), shaped(factors, K, F), shaped(factors, K, F))


# --------------------------------------------------------------------------- autograd


class _KspaceField(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, r, mask, x, dr, di, w, pr, pi):
        out, yc, ys, a, b = kspace_field_forward(k, r, mask, x, dr, di, w, pr, pi)
        ctx.save_for_backward(k, r, mask, x, yc, ys, a, b, dr, di, w, pr, pi)
        ctx.mark_non_differentiable(yc, ys, a, b)
        return out

    @staticmethod
    def backward(ctx, g):
        k, r, mask, x, yc, ys, a, b, dr, di, w, pr, pi = ctx.saved_tensors
        need = ctx.needs_input_grad
        factors = any(need[i] for i in (4, 5, 6, 7, 8))
        dk, d_r, dx, d_dr, d_di, dw, dpr, dpi = kspace_field_backward(
            k, r, mask, x, g.contiguous(), yc, ys, a, b, dr, di, w, pr, pi, factors, bool(need[0]))
        return (dk if need[0] else None, d_r if need[1] else None, None, dx if need[3] else None,
                d_dr if need[4] else None, d_di if need[5] else None, dw if need[6] else None,
                dpr if need[7] else None, dpi if need[8] else None)


def kspace_field(k, r, mask, x, dr, di, w, pr, pi):
    """Field features [N, C, F] of sources ``x`` [N, C, J] (module docstring); differentiable in every input but
    ``mask``."""
    return _KspaceField.apply(k, r, mask, x, dr, di, w, pr, pi)


__all__ = ["dense_backward", "dense_forward", "kspace_field", "kspace_field_backward", "kspace_field_forward"]
