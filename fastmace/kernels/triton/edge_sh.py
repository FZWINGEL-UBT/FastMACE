"""Edge spherical harmonics, l <= 3, in one Triton kernel each way (round 5, 5 November 2026).

MACE computes ``o3.SphericalHarmonics(lmax=3, normalize=True, normalization="component")`` of the edge vectors.
Compiled by Inductor, the forward saves about 16 per-edge intermediates for autograd (Benchmark_loup on the A100:
121 µs forward, about 107 µs backward for 373,760 edges). Here the forward reads the vector and writes the 16
values; the backward reads the vector and the 16 gradients and writes the vector's gradient from the analytic
derivative of e3nn's own polynomials (``e3nn/o3/_spherical_harmonics.py``, the same expressions in the same order),
through ``torch.nn.functional.normalize``'s ``v / max(|v|, eps)``. Rounding: the backward's sums are regrouped.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

EPS = 1e-12  # torch.nn.functional.normalize's default
BLOCK = 256

S3 = math.sqrt(3.0)
S5 = math.sqrt(5.0)
S56 = math.sqrt(5.0 / 6.0)
S38 = math.sqrt(3.0 / 8.0)
C1, C2, C3 = math.sqrt(3.0), math.sqrt(5.0), math.sqrt(7.0)  # component normalisation sqrt(2l + 1)


@triton.jit
def _unit(v_ptr, offs, mask, eps: tl.constexpr):
    vx = tl.load(v_ptr + offs * 3 + 0, mask=mask, other=0.0)
    vy = tl.load(v_ptr + offs * 3 + 1, mask=mask, other=0.0)
    vz = tl.load(v_ptr + offs * 3 + 2, mask=mask, other=0.0)
    n = tl.sqrt(vx * vx + vy * vy + vz * vz)
    d = tl.maximum(n, eps)
    return vx / d, vy / d, vz / d, n, d


@triton.jit
def _forward(v_ptr, out_ptr, E, BLOCK: tl.constexpr, eps: tl.constexpr, s3: tl.constexpr, s5: tl.constexpr,
             s56: tl.constexpr, s38: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < E
    x, y, z, n, d = _unit(v_ptr, offs, mask, eps)
    s20 = s3 * x * z
    s21 = s3 * x * y
    y2 = y * y
    x2z2 = x * x + z * z
    s22 = y2 - 0.5 * x2z2
    s23 = s3 * y * z
    s24 = s3 / 2.0 * (z * z - x * x)
    s30 = s56 * (s20 * z + s24 * x)
    s31 = s5 * s20 * y
    s32 = s38 * (4.0 * y2 - x2z2) * x
    s33 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    s34 = s38 * z * (4.0 * y2 - x2z2)
    s35 = s5 * s24 * y
    s36 = s56 * (s24 * z - s20 * x)
    # one coalesced [BLOCK, 16] store: column k picks its value
    k = tl.arange(0, 16)[None, :]
    one = tl.full([BLOCK], 1.0, tl.float32)
    vals = tl.where(k == 0, one[:, None], 0.0)
    vals = tl.where(k == 1, (x * c1)[:, None], vals)
    vals = tl.where(k == 2, (y * c1)[:, None], vals)
    vals = tl.where(k == 3, (z * c1)[:, None], vals)
    vals = tl.where(k == 4, (s20 * c2)[:, None], vals)
    vals = tl.where(k == 5, (s21 * c2)[:, None], vals)
    vals = tl.where(k == 6, (s22 * c2)[:, None], vals)
    vals = tl.where(k == 7, (s23 * c2)[:, None], vals)
    vals = tl.where(k == 8, (s24 * c2)[:, None], vals)
    vals = tl.where(k == 9, (s30 * c3)[:, None], vals)
    vals = tl.where(k == 10, (s31 * c3)[:, None], vals)
    vals = tl.where(k == 11, (s32 * c3)[:, None], vals)
    vals = tl.where(k == 12, (s33 * c3)[:, None], vals)
    vals = tl.where(k == 13, (s34 * c3)[:, None], vals)
    vals = tl.where(k == 14, (s35 * c3)[:, None], vals)
    vals = tl.where(k == 15, (s36 * c3)[:, None], vals)
    tl.store(out_ptr + offs[:, None] * 16 + k, vals, mask=mask[:, None])


@triton.jit
def _backward(v_ptr, g_ptr, out_ptr, E, BLOCK: tl.constexpr, eps: tl.constexpr, s3: tl.constexpr, s5: tl.constexpr,
              s56: tl.constexpr, s38: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < E
    x, y, z, n, d = _unit(v_ptr, offs, mask, eps)
    g = g_ptr + offs * 16
    g1x = tl.load(g + 1, mask=mask, other=0.0) * c1
    g1y = tl.load(g + 2, mask=mask, other=0.0) * c1
    g1z = tl.load(g + 3, mask=mask, other=0.0) * c1
    g20 = tl.load(g + 4, mask=mask, other=0.0) * c2
    g21 = tl.load(g + 5, mask=mask, other=0.0) * c2
    g22 = tl.load(g + 6, mask=mask, other=0.0) * c2
    g23 = tl.load(g + 7, mask=mask, other=0.0) * c2
    g24 = tl.load(g + 8, mask=mask, other=0.0) * c2
    g30 = tl.load(g + 9, mask=mask, other=0.0) * c3
    g31 = tl.load(g + 10, mask=mask, other=0.0) * c3
    g32 = tl.load(g + 11, mask=mask, other=0.0) * c3
    g33 = tl.load(g + 12, mask=mask, other=0.0) * c3
    g34 = tl.load(g + 13, mask=mask, other=0.0) * c3
    g35 = tl.load(g + 14, mask=mask, other=0.0) * c3
    g36 = tl.load(g + 15, mask=mask, other=0.0) * c3
    s20 = s3 * x * z
    s24 = s3 / 2.0 * (z * z - x * x)
    y2 = y * y
    x2z2 = x * x + z * z
    a = 4.0 * y2 - x2z2
    b = 2.0 * y2 - 3.0 * x2z2
    gs20 = g20 + s56 * (g30 * z - g36 * x) + s5 * g31 * y
    gs24 = g24 + s56 * (g30 * x + g36 * z) + s5 * g35 * y
    ga = s38 * (g32 * x + g34 * z)
    gb = 0.5 * y * g33
    gy2 = g22 + 4.0 * ga + 2.0 * gb
    gx2z2 = -0.5 * g22 - ga - 3.0 * gb
    gx = (g1x + s3 * z * gs20 + s3 * y * g21 - s3 * x * gs24 + 2.0 * x * gx2z2
          + s56 * (g30 * s24 - g36 * s20) + s38 * a * g32)
    gy = g1y + s3 * x * g21 + s3 * z * g23 + 2.0 * y * gy2 + s5 * (g31 * s20 + g35 * s24) + 0.5 * b * g33
    gz = (g1z + s3 * x * gs20 + s3 * y * g23 + s3 * z * gs24 + 2.0 * z * gx2z2
          + s56 * (g30 * s20 + g36 * s24) + s38 * a * g34)
    # normalize: u = v / max(|v|, eps); beyond eps the radial part of gu drops out
    inside = n > eps
    dot = tl.where(inside, x * gx + y * gy + z * gz, 0.0)
    o = out_ptr + offs * 3
    tl.store(o + 0, (gx - x * dot) / d, mask=mask)
    tl.store(o + 1, (gy - y * dot) / d, mask=mask)
    tl.store(o + 2, (gz - z * dot) / d, mask=mask)


def _grid(e):
    return (triton.cdiv(e, BLOCK),)


@torch.library.custom_op("fastmace::edge_sh3", mutates_args=())
def edge_sh3(vectors: torch.Tensor) -> torch.Tensor:
    """``[E, 3]`` float32 CUDA vectors -> ``[E, 16]`` spherical harmonics l <= 3 (e3nn, normalised, component)."""
    v = vectors.contiguous()
    out = v.new_empty((v.shape[0], 16))
    if v.shape[0]:
        _forward[_grid(v.shape[0])](v, out, v.shape[0], BLOCK=BLOCK, eps=EPS, s3=S3, s5=S5, s56=S56, s38=S38,
                                    c1=C1, c2=C2, c3=C3)
    return out


@edge_sh3.register_fake
def _(vectors):
    return vectors.new_empty((vectors.shape[0], 16))


@torch.library.custom_op("fastmace::edge_sh3_backward", mutates_args=())
def edge_sh3_backward(vectors: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    v, g = vectors.contiguous(), grad.contiguous()
    out = torch.empty_like(v)
    if v.shape[0]:
        _backward[_grid(v.shape[0])](v, g, out, v.shape[0], BLOCK=BLOCK, eps=EPS, s3=S3, s5=S5, s56=S56, s38=S38,
                                     c1=C1, c2=C2, c3=C3)
    return out


@edge_sh3_backward.register_fake
def _(vectors, grad):
    return torch.empty_like(vectors)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _backward_rule(ctx, grad):
    (vectors,) = ctx.saved_tensors
    return edge_sh3_backward(vectors, grad)


edge_sh3.register_autograd(_backward_rule, setup_context=_setup_context)

__all__ = ["edge_sh3", "edge_sh3_backward"]
