"""fastmace.kernels.cutlass.segment_rows on CUDA (round 5, 5 November 2026): a BlockLinear's per-irrep products read
from the [n, width] input and written into the joined output through CUTLASS's gathered entry. Values and input
gradients against float64 within the 3xTF32 bound, zero columns where no block writes, zero gradient where no block
reads, the shape-rule fallback (few rows), replay under a captured CUDA graph, and BlockLinear with a plan against
BlockLinear without one. SEGMENT_GROUPED (H200 ideas plan, idea 6): every block in one launch, bitwise against the
per-block launches, forward and input gradient, and under a captured graph."""

import pytest
import torch

from fastmace.kernels.cutlass import segment_rows as sr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

LIMIT = 2e-5  # relative to the row scale |x| |w|, the 3xTF32 bound of test_species_gemm_cuda


def _available():
    from fastmace.kernels.cutlass import tf32x3
    if not tf32x3.available("cuda", family=True):
        pytest.skip("the CUTLASS cubin needs compute capability 8.x")


# (o_in, mul_in, dim) segments in, (o_out, mul_out, dim) out, blocks (i, j): OMAT-0's 128x(0e+1o+2e+3o) with
# every segment used, and a case with an unread input segment, an unwritten output segment and narrower outputs.
CASES = {
    "omat0": ([(0, 128, 1), (128, 128, 3), (512, 128, 5), (1152, 128, 7)],
              [(0, 128, 1), (128, 128, 3), (512, 128, 5), (1152, 128, 7)], [(0, 0), (1, 1), (2, 2), (3, 3)]),
    "partial": ([(0, 64, 1), (64, 32, 3), (160, 64, 3)],
                [(0, 32, 1), (32, 64, 3), (224, 16, 5)], [(0, 0), (2, 1)]),
}


def _blocks(case, seed=0):
    segments_in, segments_out, pairs = CASES[case]
    g = torch.Generator(device="cuda").manual_seed(seed)
    weights = [torch.randn(segments_in[i][1], segments_out[j][1], device="cuda", generator=g) / segments_in[i][1] ** 0.5
               for i, j in pairs]
    width_in = sum(m * d for _, m, d in segments_in)
    width_out = sum(m * d for _, m, d in segments_out)
    plan = sr.plan_of(segments_in, segments_out, pairs, weights)
    assert plan is not None
    return segments_in, segments_out, pairs, weights, width_in, width_out, plan


def _reference(x, weights, plan, width_out):
    out = x.new_zeros((x.shape[0], width_out))
    abs_out = x.new_zeros((x.shape[0], width_out))
    for b, w in enumerate(weights):
        o_in, d, o_out = plan[3 * b:3 * b + 3]
        k, m = w.shape
        rows = x[:, o_in:o_in + d * k].reshape(-1, k)
        out[:, o_out:o_out + d * m] = (rows @ w).reshape(-1, d * m)
        abs_out[:, o_out:o_out + d * m] = (rows.abs() @ w.abs()).reshape(-1, d * m)
    return out, abs_out


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("n", [1000, 7])  # 7 atoms: too few tiles, the copy path's fallback
def test_values_and_gradient(case, n):
    _available()
    _, _, _, weights, width_in, width_out, plan = _blocks(case)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(n, width_in, device="cuda", generator=g).requires_grad_()
    weights_t = [w.t().contiguous() for w in weights]
    y = sr.segment_rows(x, weights, weights_t, plan, width_out)
    exact, scale = _reference(x.detach().double(), [w.double() for w in weights], plan, width_out)
    written = scale > 0
    assert torch.equal(y.detach()[~written], torch.zeros_like(y.detach()[~written]))
    assert float(((y.detach().double() - exact).abs()[written] / scale[written]).max()) < LIMIT
    grad = torch.randn(y.shape, device="cuda", generator=g)
    (gx,) = torch.autograd.grad(y, x, grad)
    exact, scale = _reference(grad.double(), [w.double() for w in weights_t], sr.reversed_plan(plan), width_in)
    read = scale > 0
    assert torch.equal(gx[~read], torch.zeros_like(gx[~read]))
    assert float(((gx.double() - exact).abs()[read] / scale[read]).max()) < LIMIT


def test_cuda_graph_replay():
    _available()
    _, _, _, weights, width_in, width_out, plan = _blocks("omat0")
    weights_t = [w.t().contiguous() for w in weights]
    x = torch.randn(1000, width_in, device="cuda")
    expected = sr.segment_rows(x, weights, weights_t, plan, width_out)  # warm-up builds the cached indices
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = sr.segment_rows(x, weights, weights_t, plan, width_out)
    x.copy_(torch.randn_like(x))
    graph.replay()
    assert torch.equal(y, sr.segment_rows(x, weights, weights_t, plan, width_out))
    assert not torch.equal(y, expected)


def test_plan_refuses_shared_segments():
    w = torch.zeros(4, 4)
    segments = [(0, 4, 1), (4, 4, 3)]
    assert sr.plan_of(segments, segments, [(0, 0), (1, 1)], [w, w]) == [0, 1, 0, 4, 3, 4]
    assert sr.plan_of(segments, segments, [(0, 0), (0, 1)], [w, w]) is None  # an input read twice
    assert sr.plan_of(segments, segments, [(0, 1), (1, 1)], [w, w]) is None  # an output written twice (beta)
    assert sr.plan_of(segments, segments, [((0, 1), 0)], [torch.zeros(8, 4)]) is None  # grouped over two inputs
    assert sr.plan_of(segments, segments, [((1,), 1)], [w]) == [4, 3, 4]


def test_block_linear_with_plan():
    _available()
    from fastmace.rewrites.linear.modules import BlockLinear

    segments_in, segments_out, pairs, weights, width_in, _, _ = _blocks("omat0")
    blocks = [(i, j, w) for (i, j), w in zip(pairs, weights)]
    plain = BlockLinear(torch.nn.Identity(), torch.nn.Identity(), segments_in, segments_out, blocks, grouped=True)
    gathered = BlockLinear(torch.nn.Identity(), torch.nn.Identity(), segments_in, segments_out, blocks, grouped=True)
    for module in (plain, gathered):
        module.tf32x3 = True
    count = len(gathered.block_ends)
    gathered.segment_plan = sr.plan_of(segments_in, segments_out, gathered.block_ends,
                                       [getattr(gathered, f"weight_{k}") for k in range(count)])
    assert gathered.segment_plan is not None
    x = torch.randn(1000, width_in, device="cuda", requires_grad=True)
    a, b = plain(x), gathered(x)
    grad = torch.randn_like(a)
    (ga,) = torch.autograd.grad(a, x, grad)
    (gb,) = torch.autograd.grad(b, x, grad)
    for left, right in ((a.detach(), b.detach()), (ga, gb)):
        assert float((left - right).abs().max() / left.abs().max()) < 1e-5


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("n", [1000, 7])
def test_grouped_bitwise(case, n, monkeypatch):
    _available()
    _, _, _, weights, width_in, width_out, plan = _blocks(case)
    weights_t = [w.t().contiguous() for w in weights]
    x = torch.randn(n, width_in, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    grad = torch.randn(n, width_out, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    results = {}
    for grouped in (False, True):
        monkeypatch.setattr(sr, "SEGMENT_GROUPED", grouped)
        xg = x.clone().requires_grad_()
        y = sr.segment_rows(xg, weights, weights_t, plan, width_out)
        (gx,) = torch.autograd.grad(y, xg, grad)
        results[grouped] = (y.detach(), gx)
    assert torch.equal(results[True][0], results[False][0])
    assert torch.equal(results[True][1], results[False][1])
    out = torch.empty(n, width_out, device="cuda")
    if case == "partial":  # different block shapes: never one launch
        assert not sr._grouped(x, weights, plan, out)
    elif n == 1000:
        assert sr._grouped(x, weights, plan, out)


def test_grouped_graph_replay(monkeypatch):
    _available()
    monkeypatch.setattr(sr, "SEGMENT_GROUPED", True)
    _, _, _, weights, width_in, width_out, plan = _blocks("omat0")
    weights_t = [w.t().contiguous() for w in weights]
    x = torch.randn(1000, width_in, device="cuda")
    sr.segment_rows(x, weights, weights_t, plan, width_out)  # warm-up builds the cached indices and stacked weights
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = sr.segment_rows(x, weights, weights_t, plan, width_out)
    x.copy_(torch.randn_like(x))
    graph.replay()
    monkeypatch.setattr(sr, "SEGMENT_GROUPED", False)
    assert torch.equal(y, sr.segment_rows(x, weights, weights_t, plan, width_out))
