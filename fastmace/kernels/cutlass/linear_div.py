"""A block linear's products divided by (density + 1) in the GEMM epilogue (H200 ideas plan, idea 31, first family;
6 November 2026; ``results/h200-ideas-20261105/idea31-glue.md``).

The density interactions end with ``self.linear(message) / (density + 1)``: one 3xTF32 product per output irrep
segment (``BlockLinear``, grouped), ``join_rows``'s ``torch.cat`` into [atoms, width] (Inductor's ``cat_view`` kernel)
and the division (``add_cat_div_slice_view``, which also writes ``reshape_irreps``' [atoms, irreps, mul] copy). On the
H200 these two kernels take about 54 us per OMAT-0 medium force call (Benchmark_loup, interaction 2's F13-F16).

``tf32x3_linear_div`` runs each product through ``tf32x3.remap_div_into``: the CUTLASS 64x64 kernel of ``mm_into``
whose epilogue stores row (a, m) of segment j straight into ``out[a, o_j + m * mul_j:...]`` and divides it there by
``density[a] + 1`` with Inductor's instructions (``add.f32``, then ``div.full.f32``; read from the generated PTX in
step 1). The mainloop and alpha/beta are ``mm_into``'s, so the result is bitwise the cat-then-divide path's, and the
caller reads it as ``reshape_irreps``' output through a view: no cat, no division kernel, no copy.

Backward: autograd's formulas for the division, ``grad / (d + 1)`` for the message and ``-grad * ((x / (d + 1)) /
(d + 1))`` summed over the columns for the density, with ``x / (d + 1)`` taken from the op's output (the same bits),
and each product's input gradient through ``tf32x3_mm(grad_j, weight_t_j)`` as ``Tf32x3Matmul`` does.

``LINEAR_DIV``: None: on for ``LINEAR_DIV_GPUS`` (decision 66); True / False: an A/B side's choice. Only where
``tf32x3_mm`` runs on CUTLASS (compute capability 8.0 and 9.0) and every product takes the kernel
(``tf32x3.takes``).
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

LINEAR_DIV = None
# MD-step A/Bs (6 November 2026, 7264f56f; OMAT-0 medium, SrTiO3 10x10x10, both build orders): H200 0.9912 /
# 0.9934, A100-SXM4-80GB 0.9896 / 0.9891. Bitwise at kernel level (output, input and density gradients).
LINEAR_DIV_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB"))


# h200-fastpath (bitwise): the backward's divided gradient written as contiguous segment blocks in one kernel
# (``kernels.triton.divided_blocks``) instead of Inductor's per-segment slice-divide-clone kernels.
LINEAR_DIV_SPLIT = None
LINEAR_DIV_SPLIT_GPUS = GPUModels(())


def linear_div_split_default(name=None) -> bool:
    """``LINEAR_DIV_SPLIT`` (None: ``name`` in ``LINEAR_DIV_SPLIT_GPUS``)."""
    on = LINEAR_DIV_SPLIT
    if on is None:
        on = name in LINEAR_DIV_SPLIT_GPUS
    return bool(on)


def linear_div_default(name=None) -> bool:
    """``LINEAR_DIV`` (None: ``name`` in ``LINEAR_DIV_GPUS``)."""
    on = LINEAR_DIV
    if on is None:
        on = name in LINEAR_DIV_GPUS
    return bool(on)


def applies(lefts, weights, density, offsets, width) -> bool:
    """Whether ``tf32x3_linear_div`` takes these operands (shapes, dtypes and devices only, so the compiled trace
    reads it without a graph break; the switch and the CUTLASS backend are the caller's, decided at build): float32
    CUDA, every product a tensor-core product the shape rule sends to CUTLASS (``tf32x3.takes``), widths and offsets
    aligned, one density per atom."""
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    device = density.device
    if device.type != "cuda" or density.dtype != torch.float32:
        return False
    n = density.shape[0]
    if density.numel() != n or width % tf32x3.ALIGNMENT:
        return False
    for left, weight, offset in zip(lefts, weights, offsets):
        m, k = left.shape
        if (left.dtype != torch.float32 or weight.dtype != torch.float32 or left.device != device
                or weight.shape[0] != k or offset % tf32x3.ALIGNMENT or m % n):
            return False
        if not (tf32x3_gemm.uses_tf32x3_tile(k, weight.shape[1]) and tf32x3.takes(m, k, weight.shape[1])):
            return False
    return True


@torch.library.custom_op("fastmace::tf32x3_linear_div", mutates_args=())
def tf32x3_linear_div(lefts: list[torch.Tensor], weights: list[torch.Tensor], weights_t: list[torch.Tensor],
                      offsets: list[int], width: int, density: torch.Tensor) -> torch.Tensor:
    """``cat_j(lefts[j] @ weights[j] as [atoms, dim_j * mul_j] at offsets[j]) / (density + 1)`` ([atoms, width]);
    ``density`` [atoms, 1]. The offsets' segments cover ``width``."""
    from fastmace.kernels.cutlass import tf32x3

    n = density.shape[0]
    out = lefts[0].new_empty((n, width))
    scale = density.reshape(n).contiguous()
    for left, weight, offset in zip(lefts, weights, offsets):
        a, b = tf32x3.fit(left, weight)
        tf32x3.remap_div_into(a, b, out[:, offset:], left.shape[0] // n, width, weight.shape[1], scale)
    return out


@tf32x3_linear_div.register_fake
def _(lefts, weights, weights_t, offsets, width, density):
    return lefts[0].new_empty((density.shape[0], width))


def _setup_context(ctx, inputs, output):
    lefts, weights, weights_t, offsets, width, density = inputs
    ctx.save_for_backward(density, output, *weights_t)
    ctx.offsets = list(offsets)
    ctx.shapes = [tuple(left.shape) for left in lefts]


def _backward(ctx, grad):
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    density, output, *weights_t = ctx.saved_tensors
    n = density.shape[0]
    # aten's derivatives of x / (d + 1): grad / other, and -grad * ((self / other) / other) summed to other's shape
    other = density + 1
    grad_density = (-grad * (output / other)).sum(dim=1, keepdim=True)
    widths = [(m // n) * weight_t.shape[0] for (m, _), weight_t in zip(ctx.shapes, weights_t)]
    split = (grad.is_cuda and len(widths) <= 4 and list(ctx.offsets) == [sum(widths[:k]) for k in range(len(widths))]
             and sum(widths) == grad.shape[1] and linear_div_split_default(torch.cuda.get_device_name(grad.device)))
    grad_lefts = []
    if split:  # LINEAR_DIV_SPLIT: the divided blocks contiguous from one kernel, then the same products
        from fastmace.kernels.triton.divided_blocks import divided_blocks

        for (m, _), weight_t, block in zip(ctx.shapes, weights_t, divided_blocks(grad, other, widths)):
            grad_lefts.append(tf32x3_mm(block.view(m, weight_t.shape[0]), weight_t))
    else:
        grad_x = grad / other
        for (m, _), weight_t, offset, width in zip(ctx.shapes, weights_t, ctx.offsets, widths):
            grad_lefts.append(tf32x3_mm(grad_x[:, offset:offset + width].reshape(m, weight_t.shape[0]), weight_t))
    nothing = [None] * len(weights_t)
    return grad_lefts, nothing, list(nothing), None, None, grad_density


tf32x3_linear_div.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["LINEAR_DIV", "LINEAR_DIV_GPUS", "applies", "linear_div_default", "tf32x3_linear_div"]
