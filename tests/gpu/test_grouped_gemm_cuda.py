"""fastmace.kernels.triton.grouped_gemm on CUDA (plan opt-1, OPT-T03 candidate 2): small products in one launch.

Forward and input gradients against cuBLAS at the SPEC §6.4 float32 threshold, for separately allocated
operands (the kernel addresses them by offset); two launches agree bitwise (no atomics); and a CUDA graph
that captured the launch replays it correctly on new values in the same buffers.
"""

import pytest
import torch

from fastmace.kernels import triton_available
from fastmace.kernels.triton import grouped_gemm as gg

pytestmark = pytest.mark.skipif(not triton_available(), reason=triton_available().reason)

SHAPES = [(216, 512, 512), (648, 512, 512), (1080, 512, 512), (1512, 512, 512), (216, 512, 6)]  # 5 > MAX_GROUPS


def _case(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lefts = [torch.randn((m, k), generator=g, device="cuda").requires_grad_() for m, k, _ in SHAPES]
    weights = [torch.randn((k, n), generator=g, device="cuda") / k ** 0.5 for _, k, n in SHAPES]
    return lefts, weights, [w.t().contiguous() for w in weights]


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_grouped_products_match_cublas_and_repeat_bitwise():
    lefts, weights, weights_t = _case()
    grads = [torch.randn((m, n), device="cuda") for m, _, n in SHAPES]
    got = gg.grouped_products(lefts, weights, weights_t)
    want = [x @ w for x, w in zip(lefts, weights)]
    for a, b in zip(got, want):
        assert _rel(a, b) <= 1e-4
    total = sum((o * g).sum() for o, g in zip(got, grads))
    reference = sum((o * g).sum() for o, g in zip(want, grads))
    for a, b in zip(torch.autograd.grad(total, lefts), torch.autograd.grad(reference, lefts)):
        assert _rel(a, b) <= 1e-4
    again = gg.grouped_products(lefts, weights, weights_t)
    assert all(torch.equal(a, b) for a, b in zip(again, got))


def test_strided_rows_and_graph_replay():
    lefts, weights, weights_t = _case(seed=1)
    wide = torch.randn((216, 3 * 512), device="cuda")
    lefts = [wide[:, :512], *[x.detach() for x in lefts[1:]]]  # a strided view, as BlockLinear's l = 0 rows
    static = [x.clone() if i else x for i, x in enumerate(lefts)]
    gg.grouped_products(static, weights, weights_t)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gg.grouped_products(static, weights, weights_t)
    for x in static[1:]:
        x.normal_()
    wide.normal_()
    graph.replay()
    for out, x, w in zip(captured, static, weights):
        assert _rel(out, x @ w) <= 1e-4
