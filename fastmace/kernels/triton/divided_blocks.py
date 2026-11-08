"""``grad / other`` split into contiguous column blocks in one pass (``linear_div``'s backward, ``LINEAR_DIV_SPLIT``).

``tf32x3_linear_div``'s backward multiplies each segment's columns ``grad[:, o:o + w] / (density + 1)``, viewed as
``[atoms * (w / K), K]``, by its transposed weight. The column block of a row-major ``[atoms, width]`` tensor is not
one matrix with a single row stride, so the compiled backward copies each one (Inductor's slice-divide-clone kernels,
four for OMAT-0 medium's l = 0..3: about 59 us per force call on the H200, ``profile-spline``). ``divided_blocks``
writes all of them in one kernel into one buffer, path after path, each block ``[atoms, w]`` contiguous, and returns
the views. The division is Triton's ``/`` (``div.full.f32``), the instruction Inductor emits for the same expression
(``kernels.cutlass.linear_div``), so the blocks are bitwise the compiled backward's.
"""

from __future__ import annotations

import torch

BLOCK = 1024
_kernel_cache = []


def _kernel():
    if _kernel_cache:
        return _kernel_cache[0]
    import triton
    import triton.language as tl

    @triton.jit
    def divided_blocks_kernel(g_ptr, d_ptr, out_ptr, atoms, total, stride_g, o1, o2, o3, w0, w1, w2, w3,
                              BLOCK: tl.constexpr):
        a = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = c < total
        x = tl.load(g_ptr + a.to(tl.int64) * stride_g + c, mask=live, other=0.0)
        d = tl.load(d_ptr + a)
        off = tl.where(c >= o3, o3, tl.where(c >= o2, o2, tl.where(c >= o1, o1, 0)))
        w = tl.where(c >= o3, w3, tl.where(c >= o2, w2, tl.where(c >= o1, w1, w0)))
        dst = atoms.to(tl.int64) * off + a.to(tl.int64) * w + (c - off)
        tl.store(out_ptr + dst, x / d, mask=live)

    _kernel_cache.append(divided_blocks_kernel)
    return divided_blocks_kernel


def _reference(grad, other, widths):
    blocks, offset = [], 0
    for width in widths:
        blocks.append((grad[:, offset:offset + width] / other).reshape(-1))
        offset += width
    return torch.cat(blocks)


@torch.library.custom_op("fastmace::divided_blocks", mutates_args=())
def divided_blocks_flat(grad: torch.Tensor, other: torch.Tensor, widths: list[int]) -> torch.Tensor:
    """``torch.cat([(grad[:, o:o + w] / other).flatten() for each block])`` for ``grad [atoms, sum(widths)]`` and
    ``other [atoms, 1]``; one Triton kernel on CUDA float32 (at most four blocks)."""
    if grad.device.type != "cuda" or grad.dtype != torch.float32 or len(widths) > 4:
        return _reference(grad, other, widths)
    import triton

    atoms, total = grad.shape
    out = grad.new_empty(atoms * total)
    if not out.numel():
        return out
    if grad.stride(1) != 1:
        grad = grad.contiguous()
    other = other.reshape(atoms).contiguous()
    w = list(widths) + [0] * (4 - len(widths))
    o = [sum(w[:k]) for k in range(4)]
    o = [o[k] if k < len(widths) else total for k in range(4)]
    _kernel()[(atoms, triton.cdiv(total, BLOCK))](grad, other, out, atoms, total, grad.stride(0), o[1], o[2], o[3],
                                                  w[0], w[1], w[2], w[3], BLOCK=BLOCK)
    return out


@divided_blocks_flat.register_fake
def _(grad, other, widths):
    return grad.new_empty(grad.shape[0] * grad.shape[1])


def divided_blocks(grad, other, widths):
    """The blocks ``(grad[:, o:o + w] / other)`` as contiguous ``[atoms, w]`` views of one buffer."""
    flat = divided_blocks_flat(grad, other, list(widths))
    atoms, blocks, start = grad.shape[0], [], 0
    for width in widths:
        blocks.append(flat[start:start + atoms * width].view(atoms, width))
        start += atoms * width
    return blocks


__all__ = ["divided_blocks", "divided_blocks_flat"]
