"""The hidden stack of MACE's radial networks in one kernel per direction (opt-2 OPT2-T02).

``radial_mlp(x, modules)`` evaluates LayerNorm -> SiLU, then K times Linear(D, D) -> LayerNorm -> SiLU, the
hidden layers of a radial network before its last layer (MH-1: K = 2, D = 64), for rows of width D:

- **forward:** one Triton program per tile of rows reads x, runs every layer in registers and writes only the
  output;
- **backward:** one program per tile reads x and the output's gradient, recomputes the forward up to each
  layer as it walks back through the chain, and writes only the input's gradient. The parameters are
  constants in MD: no parameter gradients.

The products are IEEE FP32 dots (no TF32), the same operations as the PyTorch modules: exact up to rounding.
The kernels run for CUDA float32 rows only; anything else (the CPU, float64 reference plans) runs the
original modules themselves, so those results are bitwise the original's. Design and the reason the
15 October single-block fusion (``fused_radial``) gave no gain: ``docs/design/opt2-T02-radial-mlp-fusion.md``.
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK_ROWS = 32  # rows per program, for every crystal and GPU (register budget at D <= 128, K <= 4)
MAX_LAYERS = 4  # K, the hidden Linear layers one kernel takes
WIDTHS = (16, 32, 64, 128)


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _layer_norm_silu(a, g_ptr, b_ptr, layer, eps, c, D: tl.constexpr):
        mean = tl.sum(a, axis=1) / D
        ac = a - mean[:, None]
        rstd = 1.0 / tl.sqrt(tl.sum(ac * ac, axis=1) / D + eps)
        ahat = ac * rstd[:, None]
        y = ahat * tl.load(g_ptr + layer * D + c)[None, :] + tl.load(b_ptr + layer * D + c)[None, :]
        sig = tl.sigmoid(y)
        return ahat, rstd, y, sig

    @triton.jit
    def _linear(s, wt_ptr, bias_ptr, layer, c, D: tl.constexpr):
        wt = tl.load(wt_ptr + layer * D * D + c[:, None] * D + c[None, :])  # W^T [in, out]
        return tl.dot(s, wt, input_precision="ieee") + tl.load(bias_ptr + layer * D + c)[None, :]

    @triton.jit
    def forward_kernel(x_ptr, stride, out_ptr, g_ptr, b_ptr, wt_ptr, bias_ptr, rows, eps_ptr,
                       BE: tl.constexpr, D: tl.constexpr, K: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        a = tl.load(x_ptr + r64[:, None] * stride + c[None, :], mask=live[:, None], other=0.0)
        s = a
        for i in tl.static_range(K + 1):
            _, _, y, sig = _layer_norm_silu(a, g_ptr, b_ptr, i, tl.load(eps_ptr + i), c, D)
            s = y * sig
            if i < K:
                a = _linear(s, wt_ptr, bias_ptr, i, c, D)
        tl.store(out_ptr + r64[:, None] * D + c[None, :], s, mask=live[:, None])

    @triton.jit
    def backward_kernel(x_ptr, stride, grad_ptr, dx_ptr, g_ptr, b_ptr, wt_ptr, w_ptr, bias_ptr, rows, eps_ptr,
                        BE: tl.constexpr, D: tl.constexpr, K: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        x = tl.load(x_ptr + r64[:, None] * stride + c[None, :], mask=live[:, None], other=0.0)
        grad = tl.load(grad_ptr + r64[:, None] * D + c[None, :], mask=live[:, None], other=0.0)
        for j in tl.static_range(K, -1, -1):  # unrolled, last LayerNorm first
            # The input of LayerNorm j, recomputed from x (registers only; K <= 4).
            a = x
            for i in tl.static_range(j):
                _, _, y, sig = _layer_norm_silu(a, g_ptr, b_ptr, i, tl.load(eps_ptr + i), c, D)
                a = _linear(y * sig, wt_ptr, bias_ptr, i, c, D)
            ahat, rstd, y, sig = _layer_norm_silu(a, g_ptr, b_ptr, j, tl.load(eps_ptr + j), c, D)
            dy = grad * sig * (1.0 + y * (1.0 - sig))  # SiLU'
            dahat = dy * tl.load(g_ptr + j * D + c)[None, :]  # LayerNorm j
            da = rstd[:, None] * (dahat - (tl.sum(dahat, axis=1) / D)[:, None]
                                  - ahat * (tl.sum(dahat * ahat, axis=1) / D)[:, None])
            if j > 0:  # through Linear j - 1: d(s) = d(a) @ W, W [out, in]
                w = tl.load(w_ptr + (j - 1) * D * D + c[:, None] * D + c[None, :])
                grad = tl.dot(da, w, input_precision="ieee")
            else:
                tl.store(dx_ptr + r64[:, None] * D + c[None, :], da, mask=live[:, None])

    return {"forward": forward_kernel, "backward": backward_kernel}


# --------------------------------------------------------------------------- which networks qualify


def stack_layers(modules):
    """(layer_norms, linears) if ``modules`` are LN, SiLU, then K times (Linear(D, D), LN, SiLU); else a reason.

    Returns a str (why not) instead of the tuple when the run does not qualify: every LayerNorm affine over
    (D,), every Linear D x D with bias, D in WIDTHS, 1 <= K <= MAX_LAYERS.
    """
    modules = list(modules)
    if len(modules) < 5 or (len(modules) - 2) % 3:
        return f"{len(modules)} modules: not LayerNorm, SiLU, then (Linear, LayerNorm, SiLU) x K"
    norms, linears = [modules[0]], []
    pattern = [modules[1]]
    for i in range(2, len(modules), 3):
        linears.append(modules[i])
        norms.append(modules[i + 1])
        pattern.append(modules[i + 2])
    if not all(isinstance(m, torch.nn.LayerNorm) for m in norms):
        return "a LayerNorm is expected after each Linear and first"
    if not all(isinstance(m, torch.nn.SiLU) for m in pattern):
        return "a SiLU is expected after each LayerNorm"
    if not all(isinstance(m, torch.nn.Linear) for m in linears):
        return "a Linear is expected between the LayerNorm/SiLU pairs"
    d = norms[0].normalized_shape[-1] if norms[0].normalized_shape else 0
    if d not in WIDTHS:
        return f"width {d} is not one of {WIDTHS}"
    if len(linears) > MAX_LAYERS:
        return f"{len(linears)} hidden Linear layers; at most {MAX_LAYERS}"
    for norm in norms:
        if norm.normalized_shape != (d,) or not norm.elementwise_affine or norm.weight is None or norm.bias is None:
            return "every LayerNorm must be affine over the width"
    for linear in linears:
        if linear.in_features != d or linear.out_features != d or linear.bias is None:
            return f"every hidden Linear must be {d} x {d} with a bias"
    return tuple(norms), tuple(linears)


class RadialStack(torch.nn.Module):
    """The hidden stack as one module: the original modules (CPU, float64) or the fused kernels (CUDA float32).

    Holds the original modules, so ``state_dict`` and deep copies carry the same parameters; the stacked
    kernel operands are built from them at each call (a few small tensors).
    """

    def __init__(self, modules):
        super().__init__()
        layers = stack_layers(modules)
        if isinstance(layers, str):
            raise ValueError(f"radial stack not fusable: {layers}")
        self.modules_ = torch.nn.Sequential(*modules)
        self.norms, self.linears = layers
        self.register_buffer("eps", torch.tensor([float(norm.eps) for norm in self.norms], dtype=torch.float32,
                                                 device=self.norms[0].weight.device), persistent=False)

    def operands(self):
        gamma = torch.stack([norm.weight for norm in self.norms])
        beta = torch.stack([norm.bias for norm in self.norms])
        weight = torch.stack([linear.weight for linear in self.linears])
        bias = torch.stack([linear.bias for linear in self.linears])
        return gamma, beta, weight, bias, self.eps

    def forward(self, x):
        if not _use_triton(x):
            return self.modules_(x)
        return _RadialStack.apply(x, *self.operands())


def _use_triton(x):
    return x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.shape[1] in WIDTHS and x.stride(1) == 1


# --------------------------------------------------------------------------- custom ops


@torch.library.custom_op("fastmace::radial_mlp_forward", mutates_args=())
def radial_mlp_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    import triton
    rows, d = x.shape
    out = x.new_empty((rows, d))
    if rows:
        _kernels()["forward"][(triton.cdiv(rows, BLOCK_ROWS),)](
            x, x.stride(0), out, gamma.contiguous(), beta.contiguous(), weight.transpose(1, 2).contiguous(),
            bias.contiguous(), rows, eps.contiguous(), BE=BLOCK_ROWS, D=d, K=weight.shape[0], num_warps=4)
    return out


@radial_mlp_forward.register_fake
def _(x, gamma, beta, weight, bias, eps):
    return x.new_empty((x.shape[0], x.shape[1]))


@torch.library.custom_op("fastmace::radial_mlp_backward", mutates_args=())
def radial_mlp_backward(x: torch.Tensor, grad: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                        weight: torch.Tensor, bias: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    import triton
    rows, d = x.shape
    dx = x.new_empty((rows, d))
    if rows:
        _kernels()["backward"][(triton.cdiv(rows, BLOCK_ROWS),)](
            x, x.stride(0), grad.contiguous(), dx, gamma.contiguous(), beta.contiguous(),
            weight.transpose(1, 2).contiguous(), weight.contiguous(), bias.contiguous(), rows, eps.contiguous(),
            BE=BLOCK_ROWS, D=d, K=weight.shape[0], num_warps=4)
    return dx


@radial_mlp_backward.register_fake
def _(x, grad, gamma, beta, weight, bias, eps):
    return x.new_empty((x.shape[0], x.shape[1]))


class _RadialStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, weight, bias, eps):
        ctx.save_for_backward(x, gamma, beta, weight, bias, eps)
        return radial_mlp_forward(x, gamma, beta, weight, bias, eps)

    @staticmethod
    def backward(ctx, grad):
        x, gamma, beta, weight, bias, eps = ctx.saved_tensors
        return radial_mlp_backward(x, grad, gamma, beta, weight, bias, eps), None, None, None, None, None


__all__ = ["BLOCK_ROWS", "MAX_LAYERS", "RadialStack", "WIDTHS", "radial_mlp_backward", "radial_mlp_forward",
           "stack_layers"]
