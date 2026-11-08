"""``inductor.SMALL_MATMULS`` (A/B constant, off): small matrix products as Inductor code instead of GEMM calls.

The A100 profiles of POLAR-1 (results/off-polar-alex-20261106) show hundreds of launches under 5 µs per MD step,
among them matrix products with an inner dimension of 1 to 8 or an output of one to four rows or columns
([216, 2] x [2, 512], [216, 512] x [512, 1], ...). Each is a cuBLAS launch. With the constant on, a post-grad pass
writes these products as a broadcast multiply and sum, which Inductor fuses with their neighbours (an inner
dimension up to its unroll threshold becomes pointwise code). Rounding: the sums run in another order.
"""

import pickle

import pytest
import torch

from fastmace.rewrites import inductor


def test_the_rule_takes_small_products_only():
    small = inductor.small_matmul_kind
    assert small((216, 2), (2, 512)) == "inner"
    assert small((216, 8), (8, 512)) == "inner"
    assert small((216, 512), (512, 1)) == "outer"
    assert small((4, 216), (216, 512)) == "outer"
    assert small((216, 9), (9, 512)) is None
    assert small((216, 64), (64, 64)) is None
    assert small((5, 512), (512, 5)) is None
    assert small((1, 1 << 20), (1 << 20, 1)) is None  # a long reduction stays with the library


def test_the_pass_pickles_by_name():
    graph_pass = inductor.small_matmul_pass()
    copy = pickle.loads(pickle.dumps(graph_pass))
    assert type(copy) is type(graph_pass) is inductor.SmallMatmuls
    assert copy.uuid() == inductor.SMALL_MATMULS_UUID


def test_compile_options_follow_the_constant(monkeypatch):
    assert inductor.compile_options({}) == {}
    monkeypatch.setattr(inductor, "SMALL_MATMULS", True)
    options = inductor.compile_options({"combo_kernels": True})
    assert options["combo_kernels"] is True
    assert options["post_grad_custom_post_pass"].uuid() == inductor.SMALL_MATMULS_UUID


def test_compiled_products_match_and_lose_their_gemms(monkeypatch):
    """Forward and backward through the pass: the small products leave the graph, the values stay."""
    torch.manual_seed(2)

    def energy(x, w_in, w_out, bias):
        h = torch.nn.functional.silu(torch.addmm(bias, x, w_in))  # [216, 2] x [2, 64]: inner
        e = h @ w_out  # [216, 64] x [64, 1]: outer
        return (e * e).sum()

    x = torch.randn(216, 2, dtype=torch.float64, requires_grad=True)
    w_in = torch.randn(2, 64, dtype=torch.float64)
    w_out = torch.randn(64, 1, dtype=torch.float64)
    bias = torch.randn(64, dtype=torch.float64)
    eager_e = energy(x, w_in, w_out, bias)
    eager_g = torch.autograd.grad(eager_e, x)[0]
    seen = []
    real = inductor.small_matmul_kind
    monkeypatch.setattr(inductor, "small_matmul_kind",
                        lambda a, b: seen.append(real(a, b)) or real(a, b))
    torch._dynamo.reset()
    try:
        compiled = torch.compile(energy, options={"post_grad_custom_post_pass": inductor.small_matmul_pass(),
                                                  "fx_graph_cache": False})
        e = compiled(x, w_in, w_out, bias)
        g = torch.autograd.grad(e, x)[0]
    finally:
        torch._dynamo.reset()
    assert "inner" in seen and "outer" in seen
    torch.testing.assert_close(e, eager_e, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(g, eager_g, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shapes", [((7, 3), (3, 5)), ((3, 40), (40, 6)), ((9, 40), (40, 2))])
def test_replacements_equal_the_product(shapes):
    a = torch.randn(*shapes[0], dtype=torch.float64)
    b = torch.randn(*shapes[1], dtype=torch.float64)
    kind = inductor.small_matmul_kind(a.shape, b.shape)
    got = inductor.small_matmul(a, b, kind)
    torch.testing.assert_close(got, a @ b, rtol=1e-13, atol=1e-13)
