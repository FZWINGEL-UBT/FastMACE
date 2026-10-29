"""wprime_backward (opt-9 OPT9-T03) without a GPU: the w'-dot against the chain it replaces, the edge density's
gradient, the generated kernel source and its NVRTC compilation, the record and the planner's handling.

The chain: the streaming backward's weight gradient ``dw``, ``da = dw [W, b]``, then the radial table's backward
``dr = <da, a'(r)> + dd density'(r)``. The record: ``dr = <dw, w'(r)>`` with ``w' = sum_t T_t(x) G'_t``, ``G' =
C' [W, b]^T`` per (pair, piece), plus the table's backward on the density column alone.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastmace.kernels.oeq_streaming import wprime_backward as wp
from fastmace.kernels.triton.radial_table import evaluate, radial_table, radial_table_backward
from fastmace.kernels.triton.w_table import fold_tables

R_LO, R_MAX = 0.3, 6.0


def _tables(dtype=torch.float64, pairs=4, pieces=6, degree=10, width=66, outputs=40, seed=1029):
    """Random tables whose derivative series is the value series' derivative (as float32 tables are), so the
    chain's backward is the exact r-derivative of its forward."""
    gen = torch.Generator().manual_seed(seed)
    coef = torch.randn(pairs, pieces, degree + 1, width, generator=gen, dtype=torch.float64)
    starts = torch.linspace(R_LO, R_MAX, pieces + 1, dtype=torch.float64)[:-1]
    scale = 2 * pieces / (R_MAX - R_LO)
    dcoef = (torch.from_numpy(np.polynomial.chebyshev.chebder(coef.numpy(), axis=2)) * scale).contiguous()
    weight = torch.randn(outputs, width - 1, generator=gen, dtype=torch.float64)
    table = SimpleNamespace(coef=coef.to(dtype), dcoef=dcoef.to(dtype), starts=starts.to(dtype),
                            scales=torch.full((pieces,), scale, dtype=dtype),
                            bins=torch.arange(pieces, dtype=torch.int32), r_lo=R_LO,
                            bin_scale=pieces / (R_MAX - R_LO), r_max=R_MAX)
    return table, weight.to(dtype), gen


def _lengths(gen, dtype, inside=300):
    """Lengths inside the domain, at and beyond the cutoff and below its lower end."""
    r = torch.cat([0.35 + 5.6 * torch.rand(inside, generator=gen, dtype=torch.float64),
                   torch.tensor([6.0, 7.5, 0.1], dtype=torch.float64)])
    return r.to(dtype)


def _args(table):
    return table.starts, table.scales, table.bins, table.r_lo, table.bin_scale, table.r_max


def test_derivative_table_is_fold_tables_g_prime_without_padding():
    table, weight, _ = _tables()
    gprime = wp.derivative_table(table.dcoef, weight)
    pairs, pieces, terms, _ = table.dcoef.shape
    assert gprime.shape == (pairs * pieces, terms, weight.shape[0]) and gprime.dtype == table.dcoef.dtype
    _, dg = fold_tables(table.coef, table.dcoef, weight)
    assert torch.equal(gprime, dg[:-1, :terms])  # fold_tables pads one term and adds a zero group
    assert bool((dg[:-1, terms:] == 0).all())


@pytest.mark.parametrize("dtype, tolerance", [(torch.float64, 1e-12), (torch.float32, 2e-5)])
def test_the_w_prime_dot_equals_the_chain(dtype, tolerance):
    """``<dw, w'(r)>`` against ``radial_table_backward(dw W)`` with a zero density gradient; edges outside the
    table's domain (at and beyond the cutoff, below ``r_lo``) give exactly zero."""
    table, weight, gen = _tables(dtype)
    r = _lengths(gen, dtype)
    pair = torch.randint(0, table.coef.shape[0], (r.numel(),), generator=gen, dtype=torch.int32)
    dw = torch.randn(r.numel(), weight.shape[0], generator=gen, dtype=torch.float64).to(dtype)
    chain = radial_table_backward(r, pair, table.dcoef, table.starts, table.scales, table.bins, dw @ weight,
                                  torch.zeros(r.numel(), 1, dtype=dtype), table.r_lo, table.bin_scale, table.r_max)
    gprime = wp.derivative_table(table.dcoef, weight)
    got = wp.wprime_reference(dw, r, pair, gprime, *_args(table))
    assert got.shape == (r.numel(),)
    assert float((got - chain).abs().max()) <= tolerance * float(chain.abs().max())
    assert bool((got[-3:] == 0).all())  # r = r_max, beyond it, below r_lo


def test_arguments_follow_the_tables_locate_rule():
    table, _, gen = _tables(torch.float32)
    r = _lengths(gen, torch.float32)
    pair = torch.randint(0, table.coef.shape[0], (r.numel(),), generator=gen, dtype=torch.int32)
    x, group = wp.wprime_arguments(r, pair, *_args(table))
    pieces = table.starts.shape[0]
    assert x.dtype == torch.float32 and group.dtype == torch.int32
    assert bool((group[:-3] >= 0).all()) and group[-3:].tolist() == [-1, -1, -1]
    assert bool((group[:-3] // pieces == pair[:-3]).all())
    assert float(x[:-3].abs().max()) <= 1.0 + 1e-6


def _radial_loss(r, pair, table, weight, dw, dd):
    """``<dw, a(r) W^T> + <dd, density(r)>`` through the table's autograd function (CPU: ``evaluate``)."""
    a, density = radial_table(r, pair, table.coef, table.dcoef, *_args(table))
    return (dw * (a @ weight.T)).sum() + (dd * density).sum()


def test_gradcheck_and_the_density_column():
    """The chain's backward is the derivative of its forward (``gradcheck``, float64), and the record's two parts,
    the w'-dot and the table's backward on the density column alone (an ``[E, 0]`` folded gradient), add up to the
    chain's gradient, the density's share included."""
    table, weight, gen = _tables(torch.float64, pairs=2, pieces=3, outputs=12, width=8)
    r = 0.4 + 5.5 * torch.rand(40, generator=gen, dtype=torch.float64)
    pair = torch.randint(0, 2, (r.numel(),), generator=gen, dtype=torch.int32)
    dw = torch.randn(r.numel(), weight.shape[0], generator=gen, dtype=torch.float64)
    dd = torch.randn(r.numel(), 1, generator=gen, dtype=torch.float64)
    leaf = r.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x: _radial_loss(x, pair, table, weight, dw, dd), (leaf,), eps=1e-7,
                                    atol=1e-6, rtol=1e-6)
    (want,) = torch.autograd.grad(_radial_loss(leaf, pair, table, weight, dw, dd), leaf)
    gprime = wp.derivative_table(table.dcoef, weight)
    density = wp.density_column(table.dcoef)
    assert density.shape == (*table.dcoef.shape[:3], 1)
    from_density = wp.density_backward(r, pair, density, table.starts, table.scales, table.bins, dd, table.r_lo,
                                       table.bin_scale, table.r_max)
    got = wp.wprime_reference(dw, r, pair, gprime, *_args(table)) + from_density
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11 * float(want.abs().max()))
    # The density's share is not negligible here: dropping it would fail.
    assert float(from_density.abs().max()) > 1e-3 * float(want.abs().max())
    slopes = evaluate(r, pair, table.dcoef, *_args(table), derivative=True)[:, -1:]
    torch.testing.assert_close(from_density, (dd * slopes).reshape(-1), rtol=1e-13, atol=0.0)


def _schedule(layout="ir_mul", channels=64, edge_mul=1):
    """A two-path uvu schedule (0e x 0e -> 0e, 0e x 1o -> 1o) with what the vendored backward reads."""
    slot = lambda name, start, component, dim: SimpleNamespace(name=name, start=start, component=component,  # noqa: E731
                                                              irrep_dim=dim)
    grad = slot("gx0", 0, 0, 1)
    term = lambda i, j, c: SimpleNamespace(input_component=i, edge_component=j, coefficient=c,  # noqa: E731
                                           input_accumulator=grad)
    out0 = SimpleNamespace(accumulator=slot("o0", 0, 0, 1), terms=(term(0, 0, 1.0),))
    out1 = [SimpleNamespace(accumulator=slot(f"o1{m}", channels, m, 3), terms=(term(0, m, 0.5),)) for m in range(3)]
    paths = (SimpleNamespace(input_start=0, input_irrep_dim=1, edge_start=0, edge_irrep_dim=1, weight_start=0,
                             edge_mul=edge_mul, outputs=(out0,)),
             SimpleNamespace(input_start=0, input_irrep_dim=1, edge_start=1, edge_irrep_dim=3,
                             weight_start=channels * edge_mul, edge_mul=edge_mul, outputs=tuple(out1)))
    slots = (out0.accumulator, *(o.accumulator for o in out1))
    return SimpleNamespace(channels=channels, layout=layout, paths=paths, output_slots=slots,
                           input_gradient_slots=(grad,), edge_dim=4, input_dim=channels, output_dim=4 * channels,
                           weight_numel=2 * channels * edge_mul,
                           launch_config=SimpleNamespace(logical_cohort_width=32))


def test_render_both_layouts():
    source = wp.render(_schedule(), "float", 10)
    assert 'extern "C" __global__ void oeq_projected_backward_wprime(' in source
    assert "constexpr int TERMS = 10;" in source and "constexpr int WEIGHT_DIM = 128;" in source
    assert "dweights[" not in source  # the weight gradient stays in registers
    assert "dr[e] = scalar_t(radial_gradient);" in source
    assert "x[input_index]" in source and "const int64_t input_index = input_base\n" in source
    assert "input_base + 0 + channel" in source  # ir_mul: channel-minor
    assert "output_base + 66 + channel * 3" in wp.render(_schedule("mul_ir"), "float", 10)
    assert "using scalar_t = double;" in wp.render(_schedule(), "double", 17)


def test_generated_source_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    for layout in ("ir_mul", "mul_ir"):
        for capability in ((12, 0), (8, 9), (8, 0)):
            assert compile_cubin(wp.render(_schedule(layout), "float", 10), capability)


def test_record_and_requirements():
    pytest.importorskip("mace")
    from fastmace.rewrites import radial

    record = radial.RECORDS["wprime_backward"]
    assert record.requires == ("radial_tables", "chunked_conv_backward")
    assert record.exactness == "rounding" and record.default == "off" and not record.deterministic
    order = tuple(radial.RECORDS)
    assert order.index("radial_tables") < order.index("wprime_backward")


def test_planner_rules_and_resolution():
    from test_planner import plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.planner import fixed
    from fastmace.planner.fixed import PlanRefused

    assert fixed.AUTO_RULES["RewriteConfig.wprime_backward"] == "never_automatic"
    assert "wprime_backward" in fixed.REWRITE_FIELDS
    plan = plan_for("mh1-r2scan")
    assert plan.rewrites.wprime_backward == "off" and plan.rewrites.radial_tables == "on"
    on = plan_for("mh1-r2scan", rewrites=RewriteConfig(wprime_backward="on"))
    assert on.rewrites.wprime_backward == "on" and on.rewrites.chunked_conv_backward == "on"
    with pytest.raises(PlanRefused, match="wprime_backward='on' needs float32"):
        plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"), rewrites=RewriteConfig(wprime_backward="on"))
    with pytest.raises(PlanRefused, match="wprime_backward='on' requires radial_tables"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(wprime_backward="on", radial_tables="off"))
    with pytest.raises(PlanRefused, match="wprime_backward='on' requires radial_tables"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(wprime_backward="on", chunked_conv_backward="off"))
