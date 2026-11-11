"""Per-atom sums of per-edge scalars over the receiver index, as a segmented sum (idea 5(b), H200 ideas plan,
6 November 2026).

The generic interactions' densities are ``scatter_sum(edge_density, edge_index[1], dim=0, dim_size=N)``: an
``[E, 1]`` value per edge summed into ``[N, 1]``. Inductor compiles the step's scatters into one atomic kernel per
edge (Benchmark_loup on the A100-SXM4-80GB: 74.8 µs for three of them, 5,000 atoms with ≈ 75 edges each). With
receiver-sorted edges (``NeighbourConfig.sort_receivers``), a block of edges holds a few runs of equal receivers:
here each block sums its runs with a segmented scan and adds one atomic per run end, so ≈ 2 atomics per block
instead of ``BLOCK``. Unsorted edges stay correct (more, shorter runs). The backward is the same gather as
``scatter_sum``'s, ``grad[receivers]``, so only the forward's summation order changes: rounding, as the atomics'
order already is.
"""

from __future__ import annotations

from functools import cache

import torch

from fastmace.kernels.gpu_models import GPUModels

# Idea 5(b). RECEIVER_SUM: the generic interactions' density sums through ``receiver_sum``. Rounding. None: on for
# RECEIVER_SUM_GPUS; True / False: an A/B side's choice. A100-SXM4-80GB (6 November 2026): MD b/a 0.9998 / 0.9918 and
# 0.9990 / 0.9985, gate as the default; the A100 80GB PCIe follows it (the owner: relative results carry over).
# H200: a tie (0.9992 / 1.0012), off.
RECEIVER_SUM = None
# RTX 5060 Ti (ada-5060-screen, 11 November 2026, results/ada-5060-screen-20261110; OMAT-0 medium, Benchmark_loup,
# SrTiO3 10x10x10 / NbCl5, ab / ba): alone 0.9985 / 0.9994, 1.0073 / 0.9987;
# left out of the candidate set 1.0041 / 1.0006, 1.0034 / 1.0049.
RECEIVER_SUM_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA GeForce RTX 5060 Ti"))
BLOCK = 1024
WARPS = 4


@torch.compiler.assume_constant_result
def _listed(index: int) -> bool:
    """Whether CUDA device ``index`` is one of ``RECEIVER_SUM_GPUS`` (a Dynamo constant, as
    ``tabled_outer_sum._listed``)."""
    return torch.cuda.get_device_name(index) in RECEIVER_SUM_GPUS


def receiver_sum_default(device) -> bool:
    """``RECEIVER_SUM`` on ``device``: the constant when set, else on for ``RECEIVER_SUM_GPUS``."""
    if RECEIVER_SUM is not None:
        return bool(RECEIVER_SUM)
    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _listed(device.index if device.index is not None else torch.cuda.current_device())


def applies(values: torch.Tensor, index: torch.Tensor) -> bool:
    """Whether ``receiver_sum`` takes ``values`` ([E, 1] float32 CUDA) over ``index`` ([E] integer)."""
    return (values.is_cuda and values.dtype == torch.float32 and values.dim() == 2 and values.shape[1] == 1
            and index.dim() == 1 and index.shape[0] == values.shape[0] and not index.dtype.is_floating_point
            and receiver_sum_default(values.device))


@cache
def _forward_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def segmented_add(a_start, a_value, b_start, b_value):
        """Combine of a segmented sum: a run restarts where b starts one."""
        return a_start | b_start, tl.where(b_start, b_value, a_value + b_value)

    @triton.jit
    def kernel(values_ptr, index_ptr, out_ptr, E, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < E
        r = tl.load(index_ptr + offs, mask=mask, other=-1)
        v = tl.load(values_ptr + offs, mask=mask, other=0.0)
        r_prev = tl.load(index_ptr + offs - 1, mask=mask & (offs > 0), other=-1)
        r_next = tl.load(index_ptr + offs + 1, mask=offs + 1 < E, other=-1)
        first = tl.arange(0, BLOCK) == 0
        last = (tl.arange(0, BLOCK) == BLOCK - 1) | (offs + 1 >= E)
        start = first | (r != r_prev)
        end = mask & (last | (r != r_next))
        _, run = tl.associative_scan((start.to(tl.int32), v), 0, segmented_add)
        tl.atomic_add(out_ptr + r, run, mask=end, sem="relaxed")

    return kernel


@torch.library.custom_op("fastmace::receiver_sum", mutates_args=())
def receiver_sum(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """``[E, 1]`` float32 CUDA ``values`` summed over ``index`` into ``[num_nodes, 1]`` (``scatter_sum(values,
    index, dim=0, dim_size=num_nodes)``), one atomic per run of equal indices in a block."""
    v = values.contiguous()
    out = v.new_zeros((num_nodes, 1))
    e = v.shape[0]
    if e:
        _forward_kernel()[((e + BLOCK - 1) // BLOCK,)](v, index.contiguous(), out, e, BLOCK=BLOCK, num_warps=WARPS)
    return out


@receiver_sum.register_fake
def _(values, index, num_nodes):
    return values.new_empty((num_nodes, 1))


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[1])


def _backward_rule(ctx, grad):
    (index,) = ctx.saved_tensors
    return grad.index_select(0, index), None, None


receiver_sum.register_autograd(_backward_rule, setup_context=_setup_context)

__all__ = ["BLOCK", "RECEIVER_SUM", "RECEIVER_SUM_GPUS", "WARPS", "applies", "receiver_sum", "receiver_sum_default"]
