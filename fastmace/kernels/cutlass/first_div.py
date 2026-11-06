"""The first interaction's reassociated products divided by (density + 1) in the GEMM epilogue, kept per block (H200
ideas plan, idea 31, F2-F5 with the B9-B11 side; 6 November 2026; ``results/h200-ideas-20261105/idea31-glue.md``).

A density first interaction with the folded linear (``generic_fold_product_linear``: its ``linear`` is the identity)
computes ``skip_tp(cat_l(reassociated_l) / (density + 1))``. Inductor then runs, per OMAT-0 medium force call:
``cat_view_7`` (the four [atoms * (2l+1), 128] products into [atoms, 2048], #28), four ``add_div_slice`` copies that
divide and cut the cat back into each block's [atoms * (2l+1), 128] rows for the species skip's GEMMs (#31-#38), and
in the backward the density reduction that also writes ``g / (d + 1)`` into [atoms, 2048] (#141) followed by three
copies back into rows (#143-#147) for the reassociation's input-gradient GEMMs.

``blocks_div`` runs each reassociated product through ``tf32x3.remap_div_into`` with the rows contiguous: the
epilogue divides by ``density[atom] + 1`` (``add.f32``, ``div.full.f32``, as Inductor) and stores the block's rows,
which the species skip reads in place (``SpeciesSkip`` takes the blocks as a tuple). No cat and no copies forward.
The blocks lie one after another in one buffer (the op's output; ``blocks_div`` returns views of it), so that the
backward's reduction reads them as one buffer and Inductor keeps it fused (at most 8 buffers per fused pointwise).
Backward: the density reduction over the same 2048 columns in the same order and with the same arithmetic (the
gradient and ``(x / (d + 1)) / (d + 1)`` joined by zero-padded additions as ``slice_backward`` joined them, the
product right before the sum, with ``x / (d + 1)`` the op's output), and each block's ``g / (d + 1)`` straight in
its row layout for ``tf32x3_mm(g_l / (d + 1), right_l.T)`` (no [atoms, 2048] write, no copies). The density is
summed before the products (the same ``scatter_sum`` as after them).

Bitwise against the cat-then-divide path (``tests/gpu/test_first_div_cuda.py``) except for the sign of exact zeros:
the old path's gradient went through ``slice_backward`` additions of +0.0, which turn -0.0 into +0.0 before the
division; a -0.0 there changes only the sign of products that are exactly zero.

``FIRST_DIV``: None: on for ``FIRST_DIV_GPUS`` (decision 66); True / False: an A/B side's choice. Only where
``tf32x3_mm`` runs on CUTLASS and every product takes the kernel (``tf32x3.takes``).
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

FIRST_DIV = None
FIRST_DIV_GPUS = GPUModels(())


def first_div_default(name=None) -> bool:
    """``FIRST_DIV`` (None: ``name`` in ``FIRST_DIV_GPUS``)."""
    on = FIRST_DIV
    if on is None:
        on = name in FIRST_DIV_GPUS
    return bool(on)


def applies(lefts, rights, density) -> bool:
    """Whether ``blocks_div`` takes these operands (shapes, dtypes and devices only: no graph break)."""
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    device = density.device
    if device.type != "cuda" or density.dtype != torch.float32:
        return False
    n = density.shape[0]
    if density.numel() != n:
        return False
    for left, right in zip(lefts, rights):
        m, k = left.shape
        if (left.dtype != torch.float32 or right.dtype != torch.float32 or right.shape[0] != k or m % n
                or right.requires_grad):
            return False
        if right.shape[1] % tf32x3.ALIGNMENT:
            return False
        if not (tf32x3_gemm.uses_tf32x3_tile(k, right.shape[1]) and tf32x3.takes(m, k, right.shape[1])):
            return False
    return True


def _spans(lefts, rights):
    """Each block's (offset, rows, columns) in the joined buffer."""
    spans, offset = [], 0
    for left, right in zip(lefts, rights):
        m, columns = left.shape[0], right.shape[1]
        spans.append((offset, m, columns))
        offset += m * columns
    return spans, offset


@torch.library.custom_op("fastmace::tf32x3_blocks_div", mutates_args=())
def tf32x3_blocks_div(lefts: list[torch.Tensor], rights: list[torch.Tensor], density: torch.Tensor) -> torch.Tensor:
    """``(lefts[l] @ rights[l]) / (density[row // (rows_l / atoms)] + 1)`` for each l, [rows_l, N_l] row-major one
    after another in one flat buffer; ``density`` [atoms, 1]."""
    from fastmace.kernels.cutlass import tf32x3

    n = density.shape[0]
    scale = density.reshape(n).contiguous()
    spans, total = _spans(lefts, rights)
    joined = lefts[0].new_empty((total,))
    for left, right, (offset, m, columns) in zip(lefts, rights, spans):
        a, b = tf32x3.fit(left, right)
        size = m // n
        tf32x3.remap_div_into(a, b, joined[offset:offset + m * columns].view(m, columns), size, size * columns,
                              columns, scale)
    return joined


@tf32x3_blocks_div.register_fake
def _(lefts, rights, density):
    return lefts[0].new_empty((_spans(lefts, rights)[1],))


def blocks_div(lefts, rights, density):
    """``tf32x3_blocks_div`` as each block's [rows_l, N_l] view."""
    joined = tf32x3_blocks_div(list(lefts), list(rights), density)
    return [joined[offset:offset + m * columns].view(m, columns) for offset, m, columns in _spans(lefts, rights)[0]]


def _setup_context(ctx, inputs, output):
    lefts, rights, density = inputs
    ctx.save_for_backward(density, output, *rights)
    ctx.spans = _spans(lefts, rights)[0]


def _joined(blocks, n):
    """``cat_l(block_l.reshape(n, -1))`` as a sum of zero-padded blocks, the last block first."""
    widths = [block.numel() // n for block in blocks]
    width = sum(widths)
    starts = [sum(widths[:i]) for i in range(len(widths))]
    joined = None
    for block, start, w in reversed(list(zip(blocks, starts, widths))):
        padded = torch.nn.functional.pad(block.reshape(n, w), (start, width - start - w))
        joined = padded if joined is None else joined + padded
    return joined


def _backward(ctx, grad):
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    density, output, *rights = ctx.saved_tensors
    n = density.shape[0]
    other = density + 1
    # the same (d + 1), but tied to the gradient (+ 0 * g[0], exact for a finite gradient): otherwise AOTAutograd's
    # partitioner computes the quotients below in the forward and saves them instead of the output (equal sizes), four
    # more forward kernels
    tied = other + grad[:1] * 0.0
    grads, quotients, grad_lefts = [], [], []
    for (offset, m, columns), right in zip(ctx.spans, rights):
        g = grad[offset:offset + m * columns].view(n, m // n, columns)
        grads.append(g)
        quotients.append(output[offset:offset + m * columns].view(n, m // n, columns) / tied[:, :, None])
        grad_lefts.append(tf32x3_mm((g / other[:, :, None]).reshape(m, columns), right.t()))
    # aten's derivative of x / (d + 1) for d, -g * ((x / (d + 1)) / (d + 1)), summed over the [atoms, width] columns
    # as the cat had them, the product right before the sum (where the compiler contracts it with the accumulation)
    grad_density = (-_joined(grads, n) * _joined(quotients, n)).sum(dim=1, keepdim=True)
    return grad_lefts, [None] * len(rights), grad_density


tf32x3_blocks_div.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["FIRST_DIV", "FIRST_DIV_GPUS", "applies", "blocks_div", "first_div_default", "tf32x3_blocks_div"]
