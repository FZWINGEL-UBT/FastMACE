"""Fused hidden block of MACE's radial networks: LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU.

fused_radial_hidden(h, ln1, linear, ln2) evaluates that block for rows of
width D (64 for MH-1) in one Triton kernel per tile of rows, with the D x D
product as an IEEE-FP32 dot (no TF32), so the intermediate tensors never go
to memory. The backward kernel recomputes the forward from the stored input
and returns only the input gradient (the parameters are constants in MD).
Same operations as the PyTorch modules; exact up to rounding. The CPU runs
the PyTorch formulas.
"""
from functools import cache

import torch

BLOCK_ROWS = 16  # rows per program (fixed for every crystal and GPU)


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def forward_kernel(h_ptr, stride, out_ptr, g1_ptr, b1_ptr, wt_ptr, bias_ptr, g2_ptr, b2_ptr,
                       rows, eps1, eps2, BE: tl.constexpr, D: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        x = tl.load(h_ptr + r64[:, None] * stride + c[None, :], mask=live[:, None], other=0.0)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=1) / D + eps1)
        y = xc * rstd[:, None] * tl.load(g1_ptr + c)[None, :] + tl.load(b1_ptr + c)[None, :]
        s = y * tl.sigmoid(y)
        wt = tl.load(wt_ptr + c[:, None] * D + c[None, :])
        z = tl.dot(s, wt, input_precision="ieee") + tl.load(bias_ptr + c)[None, :]
        mean2 = tl.sum(z, axis=1) / D
        zc = z - mean2[:, None]
        rstd2 = 1.0 / tl.sqrt(tl.sum(zc * zc, axis=1) / D + eps2)
        y2 = zc * rstd2[:, None] * tl.load(g2_ptr + c)[None, :] + tl.load(b2_ptr + c)[None, :]
        tl.store(out_ptr + r64[:, None] * D + c[None, :], y2 * tl.sigmoid(y2), mask=live[:, None])

    @triton.jit
    def backward_kernel(h_ptr, stride, go_ptr, dh_ptr, g1_ptr, b1_ptr, wt_ptr, w_ptr, bias_ptr, g2_ptr,
                        b2_ptr, rows, eps1, eps2, BE: tl.constexpr, D: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        g1 = tl.load(g1_ptr + c)[None, :]
        g2 = tl.load(g2_ptr + c)[None, :]
        x = tl.load(h_ptr + r64[:, None] * stride + c[None, :], mask=live[:, None], other=0.0)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=1) / D + eps1)
        xhat = xc * rstd[:, None]
        y = xhat * g1 + tl.load(b1_ptr + c)[None, :]
        sig = tl.sigmoid(y)
        s = y * sig
        wt = tl.load(wt_ptr + c[:, None] * D + c[None, :])
        z = tl.dot(s, wt, input_precision="ieee") + tl.load(bias_ptr + c)[None, :]
        mean2 = tl.sum(z, axis=1) / D
        zc = z - mean2[:, None]
        rstd2 = 1.0 / tl.sqrt(tl.sum(zc * zc, axis=1) / D + eps2)
        zhat = zc * rstd2[:, None]
        y2 = zhat * g2 + tl.load(b2_ptr + c)[None, :]
        sig2 = tl.sigmoid(y2)
        go = tl.load(go_ptr + r64[:, None] * D + c[None, :], mask=live[:, None], other=0.0)
        dy2 = go * sig2 * (1.0 + y2 * (1.0 - sig2))            # SiLU'
        dzhat = dy2 * g2                                        # LayerNorm 2
        dz = rstd2[:, None] * (dzhat - (tl.sum(dzhat, axis=1) / D)[:, None]
                               - zhat * (tl.sum(dzhat * zhat, axis=1) / D)[:, None])
        w = tl.load(w_ptr + c[:, None] * D + c[None, :])       # W [out, in]: ds = dz @ W
        ds = tl.dot(dz, w, input_precision="ieee")
        dy = ds * sig * (1.0 + y * (1.0 - sig))
        dxhat = dy * g1                                          # LayerNorm 1
        dx = rstd[:, None] * (dxhat - (tl.sum(dxhat, axis=1) / D)[:, None]
                              - xhat * (tl.sum(dxhat * xhat, axis=1) / D)[:, None])
        tl.store(dh_ptr + r64[:, None] * D + c[None, :], dx, mask=live[:, None])

    return {"forward": forward_kernel, "backward": backward_kernel}


def _reference(h, g1, b1, w, bias, g2, b2, eps1, eps2):
    d = h.shape[1]
    y = torch.nn.functional.silu(torch.nn.functional.layer_norm(h, (d,), g1, b1, eps1))
    z = torch.nn.functional.linear(y, w, bias)
    return torch.nn.functional.silu(torch.nn.functional.layer_norm(z, (d,), g2, b2, eps2))


def _reference_backward(h, grad, g1, b1, w, bias, g2, b2, eps1, eps2):
    """The backward kernel's formulas in PyTorch (custom ops run without autograd)."""
    def layer_norm(x, eps):
        xc = x - x.mean(dim=1, keepdim=True)
        rstd = torch.rsqrt((xc * xc).mean(dim=1, keepdim=True) + eps)
        return xc * rstd, rstd

    def layer_norm_backward(dxhat, xhat, rstd):
        return rstd * (dxhat - dxhat.mean(dim=1, keepdim=True) - xhat * (dxhat * xhat).mean(dim=1, keepdim=True))

    xhat, rstd = layer_norm(h, eps1)
    y = xhat * g1 + b1
    sig = torch.sigmoid(y)
    z = torch.nn.functional.linear(y * sig, w, bias)
    zhat, rstd2 = layer_norm(z, eps2)
    y2 = zhat * g2 + b2
    sig2 = torch.sigmoid(y2)
    dz = layer_norm_backward(grad * sig2 * (1 + y2 * (1 - sig2)) * g2, zhat, rstd2)
    dy = (dz @ w) * sig * (1 + y * (1 - sig))
    return layer_norm_backward(dy * g1, xhat, rstd)


def _use_triton(h):
    return h.is_cuda and h.dtype == torch.float32 and h.shape[1] in (16, 32, 64, 128) and h.stride(1) == 1


@torch.library.custom_op("fastmace::fused_radial_hidden_forward", mutates_args=())
def fused_radial_hidden_forward(h: torch.Tensor, g1: torch.Tensor, b1: torch.Tensor, w: torch.Tensor,
                                bias: torch.Tensor, g2: torch.Tensor, b2: torch.Tensor, eps1: float,
                                eps2: float) -> torch.Tensor:
    if not _use_triton(h):
        return _reference(h, g1, b1, w, bias, g2, b2, eps1, eps2)
    import triton
    rows, d = h.shape
    out = h.new_empty((rows, d))
    if rows:
        _kernels()["forward"][(triton.cdiv(rows, BLOCK_ROWS),)](
            h, h.stride(0), out, g1.contiguous(), b1.contiguous(), w.t().contiguous(), bias.contiguous(),
            g2.contiguous(), b2.contiguous(), rows, eps1, eps2, BE=BLOCK_ROWS, D=d, num_warps=4)
    return out


@fused_radial_hidden_forward.register_fake
def _(h, g1, b1, w, bias, g2, b2, eps1, eps2):
    return h.new_empty((h.shape[0], h.shape[1]))


@torch.library.custom_op("fastmace::fused_radial_hidden_backward", mutates_args=())
def fused_radial_hidden_backward(h: torch.Tensor, grad: torch.Tensor, g1: torch.Tensor, b1: torch.Tensor,
                                 w: torch.Tensor, bias: torch.Tensor, g2: torch.Tensor, b2: torch.Tensor,
                                 eps1: float, eps2: float) -> torch.Tensor:
    if not _use_triton(h):
        return _reference_backward(h, grad, g1, b1, w, bias, g2, b2, eps1, eps2)
    import triton
    rows, d = h.shape
    dh = h.new_empty((rows, d))
    if rows:
        _kernels()["backward"][(triton.cdiv(rows, BLOCK_ROWS),)](
            h, h.stride(0), grad.contiguous(), dh, g1.contiguous(), b1.contiguous(), w.t().contiguous(),
            w.contiguous(), bias.contiguous(), g2.contiguous(), b2.contiguous(), rows, eps1, eps2,
            BE=BLOCK_ROWS, D=d, num_warps=4)
    return dh


@fused_radial_hidden_backward.register_fake
def _(h, grad, g1, b1, w, bias, g2, b2, eps1, eps2):
    return h.new_empty((h.shape[0], h.shape[1]))


class _FusedRadialHidden(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, g1, b1, w, bias, g2, b2, eps1, eps2):
        ctx.save_for_backward(h, g1, b1, w, bias, g2, b2)
        ctx.eps = (eps1, eps2)
        return fused_radial_hidden_forward(h, g1, b1, w, bias, g2, b2, eps1, eps2)

    @staticmethod
    def backward(ctx, grad):
        h, g1, b1, w, bias, g2, b2 = ctx.saved_tensors
        dh = fused_radial_hidden_backward(h, grad, g1, b1, w, bias, g2, b2, *ctx.eps)
        return dh, None, None, None, None, None, None, None, None


def hidden_block(modules):
    """(ln1, linear, ln2) if modules are LayerNorm, SiLU, Linear(D, D), LayerNorm, SiLU; else None."""
    if len(modules) != 5:
        return None
    ln1, act1, linear, ln2, act2 = modules
    if not (isinstance(ln1, torch.nn.LayerNorm) and isinstance(act1, torch.nn.SiLU)
            and isinstance(linear, torch.nn.Linear) and isinstance(ln2, torch.nn.LayerNorm)
            and isinstance(act2, torch.nn.SiLU)):
        return None
    d = linear.in_features
    if (linear.out_features != d or linear.bias is None or ln1.normalized_shape != (d,)
            or ln2.normalized_shape != (d,) or not ln1.elementwise_affine or not ln2.elementwise_affine):
        return None
    return ln1, linear, ln2


def hidden_plan(modules):
    """Steps for a sequence of radial-network modules with every fusable run replaced.

    Scans left to right; each LayerNorm, SiLU, Linear(D, D), LayerNorm, SiLU
    run becomes one ("fused", (ln1, linear, ln2)) step, every other module a
    ("module", m) step. Returns None if nothing is fusable. MH-1's weight
    network (LN, SiLU, Linear, LN, SiLU, Linear, LN, SiLU before its last
    layer) gets one fused step followed by Linear, LN, SiLU.
    """
    modules = list(modules)
    plan, i = [], 0
    while i < len(modules):
        block = hidden_block(modules[i:i + 5])
        if block is not None:
            plan.append(("fused", block))
            i += 5
        else:
            plan.append(("module", modules[i]))
            i += 1
    return plan if any(kind == "fused" for kind, _ in plan) else None


def run_hidden_plan(h, plan):
    """Apply a hidden_plan to h (same result as the original modules, up to rounding)."""
    for kind, step in plan:
        h = fused_radial_hidden(h, *step) if kind == "fused" else step(h)
    return h


def fused_radial_hidden(h, ln1, linear, ln2):
    """SiLU(LN2(Linear(SiLU(LN1(h))))) with fused Triton forward and backward; parameters must be constants."""
    return _FusedRadialHidden.apply(h, ln1.weight, ln1.bias, linear.weight, linear.bias, ln2.weight, ln2.bias,
                                    float(ln1.eps), float(ln2.eps))
