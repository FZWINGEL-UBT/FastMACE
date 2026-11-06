"""The species skip's own-species selection in the GEMM epilogue (H200 ideas plan, idea 31, second family: F6-F9 and
F12 with #40, #41 and #62; 6 November 2026; ``results/h200-ideas-20261105/idea31-glue.md``).

``SpeciesSkip`` (``generic_species_skip``) multiplies each input block by the present species' matrices side by side
(``[rows, mul_in] @ [mul_in, species * mul_out]``, 3xTF32), keeps each atom's own block (``sum_s y_s * onehot_s``) and
joins the blocks (``torch.cat``). Inductor runs the selection and the cat as one kernel (#40 ``cat_index_mul_sum``,
reading the three species' products) and, for the symmetric contraction, a copy of the cat (#41); 78 us per OMAT-0
medium force call on the H200 with F12's select (#62).

``tf32x3_skip_select`` runs each product through ``tf32x3.remap_select_into``: the CUTLASS 64x64 kernel of ``mm_into``
whose epilogue stores element (row (a, m), column j) only if j lies in atom a's species block, at column ``j % mul_out``
of row m of the atom's slice of the joined [atoms, width] output. The mainloop is ``mm_into``'s, so each stored value
is the own-species accumulator ``y_own``, and today's ``sum_s y_s * onehot_s`` with a one-hot row is ``y_own`` exactly
(``y * 1`` is ``y``; the other terms are zeros, ``y + 0`` is ``y``): bitwise, except that a zero ``y_own`` keeps its
own sign (the sum of signed zeros may flip -0.0 to +0.0) and that a non-finite product in another species' block no
longer turns the atom's value into NaN. ``onehot`` must be one-hot (the node attributes; ``species_gemm`` assumes the
same).

Backward: autograd's formulas for the selection, ``g[:, :, None, :] * onehot[:, None, :, None]`` as each block's
[rows, species * mul_out] gradient (Inductor's ``expand_index_mul`` kernel, as today), then ``tf32x3_mm(g_full,
w.T)`` as ``Tf32x3Matmul`` does.

``SKIP_SELECT``: None: on for ``SKIP_SELECT_GPUS`` (decision 66); True / False: an A/B side's choice. Only where
``tf32x3_mm`` runs on CUTLASS, the skip is in ir_mul with one product per output block (no sums of pairs) and every
product takes the kernel (``tf32x3.takes``).
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

SKIP_SELECT = None
SKIP_SELECT_GPUS = GPUModels(())


def skip_select_default(name=None) -> bool:
    """``SKIP_SELECT`` (None: ``name`` in ``SKIP_SELECT_GPUS``)."""
    on = SKIP_SELECT
    if on is None:
        on = name in SKIP_SELECT_GPUS
    return bool(on)


def applies(lefts, weights, onehot, offsets, width) -> bool:
    """Whether ``tf32x3_skip_select`` takes these operands (shapes, dtypes and devices only: no graph break)."""
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import tf32x3_gemm

    device = onehot.device
    if device.type != "cuda" or onehot.dtype != torch.float32 or width % tf32x3.ALIGNMENT:
        return False
    n, species = onehot.shape
    for left, weight, offset in zip(lefts, weights, offsets):
        m, k = left.shape
        if (left.dtype != torch.float32 or weight.dtype != torch.float32 or left.device != device
                or weight.shape[0] != k or weight.shape[1] % species or m % n or offset % tf32x3.ALIGNMENT
                or weight.requires_grad):
            return False
        if (weight.shape[1] // species) % tf32x3.ALIGNMENT:
            return False
        if not (tf32x3_gemm.uses_tf32x3_tile(k, weight.shape[1]) and tf32x3.takes(m, k, weight.shape[1])):
            return False
    return True


@torch.library.custom_op("fastmace::tf32x3_skip_select", mutates_args=())
def tf32x3_skip_select(lefts: list[torch.Tensor], weights: list[torch.Tensor], onehot: torch.Tensor, kind: torch.Tensor,
                       offsets: list[int], width: int) -> torch.Tensor:
    """``cat_j((lefts[j] @ weights[j]).view(n, size_j, species, mul_j) * onehot[:, None, :, None]).sum(2)`` as
    [n, width], block j at ``offsets[j]``; ``kind`` the atoms' species (int32, ``onehot.argmax(1)``). The blocks
    cover ``width``."""
    from fastmace.kernels.cutlass import tf32x3

    n, species = onehot.shape
    out = lefts[0].new_empty((n, width))
    kind = kind.contiguous()
    for left, weight, offset in zip(lefts, weights, offsets):
        a, b = tf32x3.fit(left, weight)
        columns = weight.shape[1] // species
        tf32x3.remap_select_into(a, b, out[:, offset:], left.shape[0] // n, width, columns, kind, columns)
    return out


@tf32x3_skip_select.register_fake
def _(lefts, weights, onehot, kind, offsets, width):
    return lefts[0].new_empty((onehot.shape[0], width))


def _setup_context(ctx, inputs, output):
    lefts, weights, onehot, kind, offsets, width = inputs
    ctx.save_for_backward(onehot, *weights)
    ctx.offsets = list(offsets)
    ctx.rows = [left.shape[0] for left in lefts]


def _backward(ctx, grad):
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    onehot, *weights = ctx.saved_tensors
    n, species = onehot.shape
    grad_lefts = []
    for m, weight, offset in zip(ctx.rows, weights, ctx.offsets):
        size, columns = m // n, weight.shape[1] // species
        g = grad[:, offset:offset + size * columns].reshape(n, size, columns)
        # autograd's: the sum over species expands g, the selection multiplies it by the one-hot row
        full = (g.unsqueeze(2) * onehot[:, None, :, None]).reshape(m, species * columns)
        grad_lefts.append(tf32x3_mm(full, weight.t()))
    return grad_lefts, [None] * len(weights), None, None, None, None


tf32x3_skip_select.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["SKIP_SELECT", "SKIP_SELECT_GPUS", "applies", "skip_select_default", "tf32x3_skip_select"]
