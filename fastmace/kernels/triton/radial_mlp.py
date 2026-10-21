"""The hidden stack of MACE's radial networks in one kernel per direction (opt-2 OPT2-T02).

``radial_mlp(x, modules)`` evaluates LayerNorm -> SiLU, then K times Linear(D, D) -> LayerNorm -> SiLU, the
hidden layers of a radial network before its last layer (MH-1: K = 2, D = 64), for rows of width D:

- **forward:** one Triton program per tile of rows reads x, runs every layer in registers, writes the output
  and the inputs of LayerNorms 1..K (``saved``);
- **backward:** one program per tile reads x, ``saved`` and the output's gradient, walks back through the
  chain with only the K gradient products, and writes only the input's gradient. The parameters are
  constants in MD: no parameter gradients. (Recomputing the forward instead of storing it is kept as an
  option; it was slower and its backward spilled registers: OPT2-T02 screen, 21 October 2026.)

Tiles by a rule from the width alone (``kernel_config``): 8 elements of a row tile per thread, the register
budget at which neither kernel spills (Triton's spill counts, checked by a GPU test for every width).

The products are IEEE FP32 dots (no TF32), the same operations as the PyTorch modules: exact up to rounding.
The kernels run for CUDA float32 rows only; anything else (the CPU, float64 reference plans) runs the
original modules themselves, so those results are bitwise the original's. Design and the reason the
15 October single-block fusion (``fused_radial``) gave no gain: ``docs/design/opt2-T02-radial-mlp-fusion.md``.
"""

from __future__ import annotations

from functools import cache

import torch

MAX_LAYERS = 4  # K, the hidden Linear layers one kernel takes
# Widths up to 64: each program holds a D x D weight tile in registers, and at D = 128 both kernels spill at
# any tile tried (OPT2-T02's GPU test; 400-1,200 spilled registers), so 128 is refused, not slowed down.
WIDTHS = (16, 32, 64)
ELEMENTS_PER_THREAD = 8  # of a [rows, D] tile: the budget at which neither kernel spills
MAX_WARPS = 8


def kernel_config(d):
    """(rows per program, warps, store) for width ``d``: a rule from the shape, the same on every GPU.

    32 rows per program and as many warps as give each thread ELEMENTS_PER_THREAD elements of a row tile (at
    most MAX_WARPS); the forward stores the LayerNorm inputs. MH-1 (D = 64): 32 rows, 8 warps.
    """
    rows = 32
    warps = max(1, min(MAX_WARPS, rows * d // (32 * ELEMENTS_PER_THREAD)))
    return rows, warps, True


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
    def forward_kernel(x_ptr, stride, out_ptr, saved_ptr, g_ptr, b_ptr, wt_ptr, bias_ptr, rows, eps_ptr,
                       BE: tl.constexpr, D: tl.constexpr, K: tl.constexpr, STORE: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        a = tl.load(x_ptr + r64[:, None] * stride + c[None, :], mask=live[:, None], other=0.0)
        s = a
        for i in tl.static_range(K + 1):
            if STORE and i > 0:  # the input of LayerNorm i, for a backward without recomputation
                tl.store(saved_ptr + (i - 1) * rows * D + r64[:, None] * D + c[None, :], a, mask=live[:, None])
            _, _, y, sig = _layer_norm_silu(a, g_ptr, b_ptr, i, tl.load(eps_ptr + i), c, D)
            s = y * sig
            if i < K:
                a = _linear(s, wt_ptr, bias_ptr, i, c, D)
        tl.store(out_ptr + r64[:, None] * D + c[None, :], s, mask=live[:, None])

    @triton.jit
    def backward_kernel(x_ptr, stride, saved_ptr, grad_ptr, dx_ptr, g_ptr, b_ptr, wt_ptr, w_ptr, bias_ptr, rows,
                        eps_ptr, BE: tl.constexpr, D: tl.constexpr, K: tl.constexpr, STORE: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        x = tl.load(x_ptr + r64[:, None] * stride + c[None, :], mask=live[:, None], other=0.0)
        grad = tl.load(grad_ptr + r64[:, None] * D + c[None, :], mask=live[:, None], other=0.0)
        for j in tl.static_range(K, -1, -1):  # unrolled, last LayerNorm first
            # The input of LayerNorm j: stored by the forward, or recomputed from x (registers only; K <= 4).
            if STORE and j > 0:
                a = tl.load(saved_ptr + (j - 1) * rows * D + r64[:, None] * D + c[None, :], mask=live[:, None],
                            other=0.0)
            else:
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

    @triton.jit
    def _gathered_input(first_ptr, first_stride, src_ptr, src_stride, tgt_ptr, tgt_stride, ei_ptr, rows, r64, live,
                        c):
        # x = first[e] + source_table[sender[e]] + target_table[receiver[e]]: the first layer's output with the
        # gathered species terms, the add Inductor fuses into the first LayerNorm without this kernel.
        sender = tl.load(ei_ptr + r64, mask=live, other=0)
        receiver = tl.load(ei_ptr + rows + r64, mask=live, other=0)
        x = tl.load(first_ptr + r64[:, None] * first_stride + c[None, :], mask=live[:, None], other=0.0)
        x += tl.load(src_ptr + sender[:, None] * src_stride + c[None, :], mask=live[:, None], other=0.0)
        x += tl.load(tgt_ptr + receiver[:, None] * tgt_stride + c[None, :], mask=live[:, None], other=0.0)
        return x

    @triton.jit
    def entry_forward_kernel(first_ptr, first_stride, src_ptr, src_stride, tgt_ptr, tgt_stride, ei_ptr, cut_ptr,
                             out_ptr, saved_ptr, g_ptr, b_ptr, wt_ptr, bias_ptr, rows, eps_ptr, BE: tl.constexpr,
                             D: tl.constexpr, K: tl.constexpr, HAS_CUT: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        a = _gathered_input(first_ptr, first_stride, src_ptr, src_stride, tgt_ptr, tgt_stride, ei_ptr, rows, r64,
                            live, c)
        s = a
        for i in tl.static_range(K + 1):
            if i > 0:  # the input of LayerNorm i, for the backward
                tl.store(saved_ptr + (i - 1) * rows * D + r64[:, None] * D + c[None, :], a, mask=live[:, None])
            _, _, y, sig = _layer_norm_silu(a, g_ptr, b_ptr, i, tl.load(eps_ptr + i), c, D)
            s = y * sig
            if i < K:
                a = _linear(s, wt_ptr, bias_ptr, i, c, D)
        # The folded last layer's input [h c, c] (fold_cutoff), [rows, D + 1]; without a cutoff [h, 1].
        if HAS_CUT:
            cut = tl.load(cut_ptr + r64, mask=live, other=0.0)
        else:
            cut = tl.full((BE,), 1.0, tl.float32)
        tl.store(out_ptr + r64[:, None] * (D + 1) + c[None, :], s * cut[:, None], mask=live[:, None])
        tl.store(out_ptr + r64 * (D + 1) + D, cut, mask=live)

    @triton.jit
    def entry_backward_kernel(first_ptr, first_stride, src_ptr, src_stride, tgt_ptr, tgt_stride, ei_ptr, cut_ptr,
                              saved_ptr, grad_ptr, dx_ptr, dcut_ptr, g_ptr, b_ptr, w_ptr, rows, eps_ptr,
                              BE: tl.constexpr, D: tl.constexpr, K: tl.constexpr, HAS_CUT: tl.constexpr):
        r = tl.program_id(0) * BE + tl.arange(0, BE)
        live = r < rows
        r64 = r.to(tl.int64)
        c = tl.arange(0, D)
        grad_a = tl.load(grad_ptr + r64[:, None] * (D + 1) + c[None, :], mask=live[:, None], other=0.0)
        if HAS_CUT:
            cut = tl.load(cut_ptr + r64, mask=live, other=0.0)
        else:
            cut = tl.full((BE,), 1.0, tl.float32)
        grad = grad_a * cut[:, None]  # d/dh of h c
        for j in tl.static_range(K, -1, -1):  # unrolled, last LayerNorm first
            if j > 0:
                a = tl.load(saved_ptr + (j - 1) * rows * D + r64[:, None] * D + c[None, :], mask=live[:, None],
                            other=0.0)
            else:
                a = _gathered_input(first_ptr, first_stride, src_ptr, src_stride, tgt_ptr, tgt_stride, ei_ptr, rows,
                                    r64, live, c)
            ahat, rstd, y, sig = _layer_norm_silu(a, g_ptr, b_ptr, j, tl.load(eps_ptr + j), c, D)
            if HAS_CUT and j == K:  # d/dc of [h c, c] = sum(dA[:, :D] h) + dA[:, D]
                dcut = tl.sum(grad_a * (y * sig), axis=1) + tl.load(grad_ptr + r64 * (D + 1) + D, mask=live,
                                                                    other=0.0)
                tl.store(dcut_ptr + r64, dcut, mask=live)
            dy = grad * sig * (1.0 + y * (1.0 - sig))  # SiLU'
            dahat = dy * tl.load(g_ptr + j * D + c)[None, :]  # LayerNorm j
            da = rstd[:, None] * (dahat - (tl.sum(dahat, axis=1) / D)[:, None]
                                  - ahat * (tl.sum(dahat * ahat, axis=1) / D)[:, None])
            if j > 0:  # through Linear j - 1: d(s) = d(a) @ W, W [out, in]
                w = tl.load(w_ptr + (j - 1) * D * D + c[:, None] * D + c[None, :])
                grad = tl.dot(da, w, input_precision="ieee")
            else:
                tl.store(dx_ptr + r64[:, None] * D + c[None, :], da, mask=live[:, None])

    return {"forward": forward_kernel, "backward": backward_kernel, "entry_forward": entry_forward_kernel,
            "entry_backward": entry_backward_kernel}


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

    def __init__(self, modules, config=None):
        super().__init__()
        # (rows per program, warps, store the LayerNorm inputs); None takes the rule (kernel_config).
        self.config = tuple(config) if config is not None else None  # None: kernel_config(width)
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
        block_rows, num_warps, store = self.config or kernel_config(x.shape[1])
        return _RadialStack.apply(x, *self.operands(), int(block_rows), int(num_warps), bool(store))


def _use_triton(x):
    return x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.shape[1] in WIDTHS and x.stride(1) == 1


# --------------------------------------------------------------------------- custom ops


@torch.library.custom_op("fastmace::radial_mlp_forward", mutates_args=())
def radial_mlp_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor, eps: torch.Tensor, block_rows: int, num_warps: int,
                       store: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, saved): ``saved`` holds the inputs of LayerNorms 1..K, [K, rows, D], when ``store``; else it is empty."""
    import triton
    rows, d = x.shape
    k = weight.shape[0]
    out = x.new_empty((rows, d))
    saved = x.new_empty((k, rows, d) if store else (0,))
    if rows:
        _kernels()["forward"][(triton.cdiv(rows, block_rows),)](
            x, x.stride(0), out, saved, gamma.contiguous(), beta.contiguous(), weight.transpose(1, 2).contiguous(),
            bias.contiguous(), rows, eps.contiguous(), BE=block_rows, D=d, K=k, STORE=store, num_warps=num_warps)
    return out, saved


@radial_mlp_forward.register_fake
def _(x, gamma, beta, weight, bias, eps, block_rows, num_warps, store):
    rows, d = x.shape
    return x.new_empty((rows, d)), x.new_empty((weight.shape[0], rows, d) if store else (0,))


@torch.library.custom_op("fastmace::radial_mlp_backward", mutates_args=())
def radial_mlp_backward(x: torch.Tensor, saved: torch.Tensor, grad: torch.Tensor, gamma: torch.Tensor,
                        beta: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: torch.Tensor,
                        block_rows: int, num_warps: int, store: bool) -> torch.Tensor:
    import triton
    rows, d = x.shape
    dx = x.new_empty((rows, d))
    if rows:
        _kernels()["backward"][(triton.cdiv(rows, block_rows),)](
            x, x.stride(0), saved, grad.contiguous(), dx, gamma.contiguous(), beta.contiguous(),
            weight.transpose(1, 2).contiguous(), weight.contiguous(), bias.contiguous(), rows, eps.contiguous(),
            BE=block_rows, D=d, K=weight.shape[0], STORE=store, num_warps=num_warps)
    return dx


@radial_mlp_backward.register_fake
def _(x, saved, grad, gamma, beta, weight, bias, eps, block_rows, num_warps, store):
    return x.new_empty((x.shape[0], x.shape[1]))


class _RadialStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, weight, bias, eps, block_rows, num_warps, store):
        out, saved = radial_mlp_forward(x, gamma, beta, weight, bias, eps, block_rows, num_warps, store)
        ctx.save_for_backward(x, saved, gamma, beta, weight, bias, eps)
        ctx.config = (block_rows, num_warps, store)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, saved, gamma, beta, weight, bias, eps = ctx.saved_tensors
        dx = radial_mlp_backward(x, saved, grad, gamma, beta, weight, bias, eps, *ctx.config)
        return dx, None, None, None, None, None, None, None, None


# --------------------------------------------------------------------------- the entry: gathers in, [h c, c] out


@torch.library.custom_op("fastmace::radial_entry_forward", mutates_args=())
def radial_entry_forward(first: torch.Tensor, source: torch.Tensor, target: torch.Tensor, edge_index: torch.Tensor,
                         cutoff: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor, eps: torch.Tensor, has_cutoff: bool, block_rows: int,
                         num_warps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """([h c, c] of shape [rows, D + 1], saved LayerNorm inputs [K, rows, D])."""
    import triton
    rows, d = first.shape
    k = weight.shape[0]
    out = first.new_empty((rows, d + 1))
    saved = first.new_empty((k, rows, d))
    if rows:
        _kernels()["entry_forward"][(triton.cdiv(rows, block_rows),)](
            first, first.stride(0), source, source.stride(0), target, target.stride(0), edge_index.contiguous(),
            cutoff.contiguous(), out, saved, gamma.contiguous(), beta.contiguous(),
            weight.transpose(1, 2).contiguous(), bias.contiguous(), rows, eps.contiguous(), BE=block_rows, D=d, K=k,
            HAS_CUT=has_cutoff, num_warps=num_warps)
    return out, saved


@radial_entry_forward.register_fake
def _(first, source, target, edge_index, cutoff, gamma, beta, weight, bias, eps, has_cutoff, block_rows,
      num_warps):
    rows, d = first.shape
    return first.new_empty((rows, d + 1)), first.new_empty((weight.shape[0], rows, d))


@torch.library.custom_op("fastmace::radial_entry_backward", mutates_args=())
def radial_entry_backward(first: torch.Tensor, source: torch.Tensor, target: torch.Tensor, edge_index: torch.Tensor,
                          cutoff: torch.Tensor, saved: torch.Tensor, grad: torch.Tensor, gamma: torch.Tensor,
                          beta: torch.Tensor, weight: torch.Tensor, eps: torch.Tensor, has_cutoff: bool,
                          block_rows: int, num_warps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(gradient of x = first + gathered tables, [rows, D]; gradient of the cutoff, [rows, 1])."""
    import triton
    rows, d = first.shape
    dx = first.new_empty((rows, d))
    dcut = first.new_zeros((rows, 1))
    if rows:
        _kernels()["entry_backward"][(triton.cdiv(rows, block_rows),)](
            first, first.stride(0), source, source.stride(0), target, target.stride(0), edge_index.contiguous(),
            cutoff.contiguous(), saved, grad.contiguous(), dx, dcut, gamma.contiguous(), beta.contiguous(),
            weight.contiguous(), rows, eps.contiguous(), BE=block_rows, D=d, K=weight.shape[0], HAS_CUT=has_cutoff,
            num_warps=num_warps)
    return dx, dcut


@radial_entry_backward.register_fake
def _(first, source, target, edge_index, cutoff, saved, grad, gamma, beta, weight, eps, has_cutoff, block_rows,
      num_warps):
    rows, d = first.shape
    return first.new_empty((rows, d)), first.new_empty((rows, 1))


class _RadialEntry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, source, target, edge_index, cutoff, gamma, beta, weight, bias, eps, has_cutoff,
                block_rows, num_warps):
        out, saved = radial_entry_forward(first, source, target, edge_index, cutoff, gamma, beta, weight, bias, eps,
                                          has_cutoff, block_rows, num_warps)
        ctx.save_for_backward(first, source, target, edge_index, cutoff, saved, gamma, beta, weight, eps)
        ctx.config = (has_cutoff, block_rows, num_warps)
        ctx.table_rows = (source.shape[0], target.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad):
        first, source, target, edge_index, cutoff, saved, gamma, beta, weight, eps = ctx.saved_tensors
        dx, dcut = radial_entry_backward(first, source, target, edge_index, cutoff, saved, grad, gamma, beta, weight,
                                         eps, *ctx.config)
        need = ctx.needs_input_grad
        # The tables are per-node species terms; their gradients are the per-edge gradients summed per node.
        dsource = (dx.new_zeros((ctx.table_rows[0], dx.shape[1])).index_add_(0, edge_index[0], dx)
                   if need[1] else None)
        dtarget = (dx.new_zeros((ctx.table_rows[1], dx.shape[1])).index_add_(0, edge_index[1], dx)
                   if need[2] else None)
        dcut = dcut if need[4] and ctx.config[0] else None
        return (dx if need[0] else None, dsource, dtarget, None, dcut, None, None, None, None, None, None, None, None)


def entry_reference(stack, first, source, target, edge_index, cutoff):
    """[h c, c] (or [h, 1]) of a radial network from its first layer's output and gathered species tables, in
    PyTorch with the original modules: what the interaction computed before fused_radial_mlp."""
    hidden = first + source[edge_index[0]] + target[edge_index[1]]
    h = stack.modules_(hidden)
    if cutoff is None:
        return torch.cat([h, torch.ones_like(h[:, :1])], dim=1)
    return torch.cat([h * cutoff, cutoff], dim=1)


def _use_entry_triton(first, source, target, cutoff):
    tensors = [first, source, target] + ([cutoff] if cutoff is not None else [])
    return (_use_triton(first) and all(t.is_cuda and t.dtype == torch.float32 for t in tensors)
            and source.stride(1) == 1 and target.stride(1) == 1)


def radial_entry(stack, first, source, target, edge_index, cutoff):
    """The folded last layer's input of one radial network, from the joint first layer's output columns ``first``
    [E, D], the species tables ``source`` and ``target`` [N, D] (gathered by ``edge_index``) and the cutoff
    [E, 1] or None: the fused kernels on CUDA float32, else ``entry_reference`` (opt-2 OPT2-T02c)."""
    if not _use_entry_triton(first, source, target, cutoff):
        return entry_reference(stack, first, source, target, edge_index, cutoff)
    block_rows, num_warps, _ = stack.config or kernel_config(first.shape[1])
    has_cutoff = cutoff is not None
    cut = cutoff if has_cutoff else first.new_empty((0, 1))
    return _RadialEntry.apply(first, source, target, edge_index, cut, *stack.operands(), has_cutoff,
                              int(block_rows), int(num_warps))


__all__ = ["ELEMENTS_PER_THREAD", "MAX_LAYERS", "MAX_WARPS", "RadialStack", "WIDTHS", "entry_reference",
           "kernel_config", "radial_entry", "radial_entry_backward", "radial_entry_forward", "radial_mlp_backward",
           "radial_mlp_forward", "stack_layers"]
