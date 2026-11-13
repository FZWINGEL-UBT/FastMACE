"""wprime_backward (opt-9 OPT9-T03) without a GPU: the w'-dot against the chain it replaces, the edge density's
gradient, the generated kernel source and its NVRTC compilation, the record and the planner's handling.

The chain: the streaming backward's weight gradient ``dw``, ``da = dw [W, b]``, then the radial table's backward
``dr = <da, a'(r)> + dd density'(r)``. The record: ``dr = <dw, w'(r)>`` with ``w' = sum_t T_t(x) G'_t``, ``G' =
C' [W, b]^T`` per (pair, piece), plus the table's backward on the density column alone.
"""

import weakref
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


@pytest.mark.parametrize("degree, vector", [(10, 2), (10, 4), (7, 2), (7, 4)])
def test_q_major_derivative_table_holds_the_same_values(degree, vector):
    """``SLOPE_VECTOR``'s layout: each (group, weight)'s terms contiguous, zero terms up to a multiple of the load
    width, the term-major table's values bit for bit."""
    table, weight, _ = _tables(torch.float32, degree=degree)
    term_major = wp.derivative_table(table.dcoef, weight)
    q_major = wp.derivative_table(table.dcoef, weight, vector)
    terms = degree
    padded = wp.padded_terms(terms, vector)
    assert padded % vector == 0 and terms <= padded < terms + vector
    assert q_major.shape == (term_major.shape[0], weight.shape[0], padded) and q_major.is_contiguous()
    assert torch.equal(q_major[..., :terms], term_major.transpose(1, 2))
    assert bool((q_major[..., terms:] == 0).all())


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
    assert "__ldg(g + k * WEIGHT_DIM + weight_index)" in source and "float2" not in source  # term-major
    vector = wp.render(_schedule(), "float", 7, vector=4)  # SLOPE_VECTOR: q-major, one zero term
    assert "constexpr int TERMS = 7;" in vector and "constexpr int PADDED_TERMS = 8;" in vector
    assert "const float4 v = __ldg(gq + k);" in vector and "* v.w;" in vector
    with pytest.raises(ValueError, match="slope vector"):
        wp.render(_schedule(), "double", 17, vector=2)


def test_render_factored():
    """FACTORED: per path the u/v sums, then one weight-gradient, input- and edge-gradient update per component."""
    for layout in ("ir_mul", "mul_ir"):
        for table_w in (False, True):
            plain = wp.render(_schedule(layout, edge_mul=2), "float", 6, vector=2, table_w=table_w)
            source = wp.render(_schedule(layout, edge_mul=2), "float", 6, vector=2, table_w=table_w, factored=True)
            assert "FACTORED" not in plain and "u_0 += scaled * edge_value;" not in plain
            assert "u_0 += scaled * edge_value;" in source and "v_0 += scaled * input_value;" in source
            assert "weight_gradient += output_gradient * input_value" not in source
            assert "+= scaled_weight * u_0;" in source
            flat = " ".join(source.split())
            assert "edge_gradient[0] += scaled_weight * v_0;" in flat  # edge channel 0
            assert "edge_gradient[1] += scaled_weight * v_0;" in flat  # edge channel 1
    paths = wp.factored_paths(_schedule())
    assert len(paths) == 2 and all(p["inputs"] and p["edges"] for p in paths)
    assert all(p["alpha"] == 1.0 for p in paths)
    # FACTORED_SCALE: a path's common coefficient magnitude applied once per path.
    scaled = wp.factored_paths(_schedule(), scale=True)
    assert any(p["alpha"] != 1.0 for p in scaled)
    source = wp.render(_schedule(), "float", 6, vector=2, table_w=True, factored=True, scale=True)
    assert "const scalar_t scaled_weight = radial_weight * alpha;" in source and "weight_gradient *= alpha;" in source


def test_generated_source_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    for layout in ("ir_mul", "mul_ir"):
        for capability in ((12, 0), (8, 9), (8, 0)):
            assert compile_cubin(wp.render(_schedule(layout), "float", 10), capability)
            assert compile_cubin(wp.render(_schedule(layout), "float", 7, vector=2), capability)
            assert compile_cubin(wp.render(_schedule(layout), "float", 7, vector=2, table_w=True, factored=True),
                                 capability)


def test_record_and_requirements():
    pytest.importorskip("mace")
    from fastmace.rewrites import radial

    record = radial.RECORDS["wprime_backward"]
    assert record.requires == ("radial_tables", "chunked_conv_backward")
    assert record.exactness == "rounding" and record.default == "auto" and not record.deterministic
    order = tuple(radial.RECORDS)
    assert order.index("radial_tables") < order.index("wprime_backward")


def test_planner_rules_and_resolution():
    from test_planner import plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.planner import fixed
    from fastmace.planner.fixed import PlanRefused

    assert (fixed.AUTO_RULES["RewriteConfig.wprime_backward"]
            == "on_for_mh1_and_omat0_float32_unless_reproducible_on_decision_66_gpus")
    assert "wprime_backward" in fixed.REWRITE_FIELDS
    plan = plan_for("mh1-r2scan")  # the RTX 5060 Ti: decision 66 (test_planner.py has the per-GPU cases)
    assert plan.rewrites.wprime_backward == "on" and plan.rewrites.radial_tables == "on"
    assert plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64")).rewrites.wprime_backward == "off"
    on = plan_for("mh1-r2scan", rewrites=RewriteConfig(wprime_backward="on"))
    assert on.rewrites.wprime_backward == "on" and on.rewrites.chunked_conv_backward == "on"
    with pytest.raises(PlanRefused, match="wprime_backward='on' needs float32"):
        plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"), rewrites=RewriteConfig(wprime_backward="on"))
    with pytest.raises(PlanRefused, match="wprime_backward='on' requires radial_tables"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(wprime_backward="on", radial_tables="off"))
    with pytest.raises(PlanRefused, match="wprime_backward='on' requires radial_tables"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(wprime_backward="on", chunked_conv_backward="off"))


def _edge16(layout="ir_mul"):
    """``_schedule`` with edge_sh3's 16 edge features (the paths read components 0-3 of them)."""
    schedule = _schedule(layout)
    schedule.edge_dim = 16
    return schedule


@pytest.mark.parametrize("reduction", ["plain", "butterfly", "redux"])
def test_edge_force_source(reduction):
    """EDGE_FORCE (idea 13): the vectors after dr, dL/dv [E, 3] written through dsh by one lane, no dsh row."""
    options = dict(butterfly=reduction == "butterfly", redux=reduction == "redux")
    plain = wp.render(_edge16(), "float", 10, **options)
    source = wp.render(_edge16(), "float", 10, edge_force=True, **options)
    assert "const scalar_t* __restrict__ vec" not in plain and "dsh[3 * e" not in plain
    assert "scalar_t* dr,\n    const scalar_t* __restrict__ vec) {" in source
    assert source.count("dsh[3 * e + 0] = scalar_t((gx - x * dot) / d);") == 1
    assert "if (lane < 3) dsh[e * 3 + lane] = scalar_t(0);" in source  # a skipped edge's row
    assert "dsh[edge_base" not in source and "dsh[e * EDGE_DIM + c]" not in source
    assert "constexpr float C3 = 2.6457513110645907f;" in source


def test_edge_force_needs_float32_and_sixteen_features():
    with pytest.raises(ValueError, match="edge_force"):
        wp.render(_schedule(), "float", 10, edge_force=True)  # 4 edge features
    with pytest.raises(ValueError, match="edge_force"):
        wp.render(_edge16(), "double", 10, edge_force=True)
    assert wp.edge_force_supported(_edge16(), "float") and not wp.edge_force_supported(_schedule(), "float")


def test_edge_force_default(monkeypatch):
    monkeypatch.setattr(wp, "EDGE_FORCE", None)
    assert not wp.edge_force_default(torch.device("cpu"))  # EDGE_FORCE_GPUS is empty
    monkeypatch.setattr(wp, "EDGE_FORCE", True)
    assert wp.edge_force_default(torch.device("cpu"))


def test_edge_force_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    for capability in ((12, 0), (8, 9), (8, 0)):
        for reduction in ("plain", "butterfly", "redux"):
            options = dict(butterfly=reduction == "butterfly", redux=reduction == "redux")
            assert compile_cubin(wp.render(_edge16(), "float", 10, edge_force=True, **options), capability)


def test_skip_outside_source():
    """SKIP_OUTSIDE (12 November 2026, results/wprime-h200-20261111): an edge at or beyond the cutoff writes its zeros
    and leaves before the channel loop, whose table loads are then unconditional; off, the zero fill stays."""
    options = dict(vector=2, table_w=True)
    plain = wp.render(_schedule(), "float", 4, **options)
    source = wp.render(_schedule(), "float", 4, skip_outside=True, **options)
    assert "SKIP_OUTSIDE" not in plain and "if (inside) {" in plain
    assert "if (inside) {" not in source and "if (true) {" in source and "if (!inside) {" in source
    skip = source[source.index("if (!inside) {"):source.index("acc_t edge_gradient[EDGE_DIM]")]
    assert "dsh[e * EDGE_DIM + c] = scalar_t(0);" in skip and "dr[e] = scalar_t(0);" in skip and "return;" in skip
    force = wp.render(_edge16(), "float", 4, skip_outside=True, edge_force=True, **options)
    skip = force[force.index("if (!inside) {"):force.index("acc_t edge_gradient[EDGE_DIM]")]
    assert "if (lane < 3) dsh[e * 3 + lane] = scalar_t(0);" in skip
    with pytest.raises(ValueError, match="skip_outside"):
        wp.render(_schedule(), "float", 4, skip_outside=True)  # no TABLE_W: no inside flag


def test_skip_outside_lab_form_is_the_template():
    """The kernel lab's h2-skip transform (wprime_tuning.skip) is SKIP_OUTSIDE's template code."""
    import re

    from fastmace.kernels.oeq_streaming import wprime_tuning as wt

    def code(text):
        lines = (re.sub(r"//.*", "", line).strip() for line in text.splitlines())
        return [line.replace("dsh[edge_base + c]", "dsh[e * EDGE_DIM + c]") for line in lines if line]
    options = dict(vector=2, table_w=True, factored=True, scale=True, horner=True, blocks=[(1, [0]), (3, [64])],
                   stage_dout=True, stage_sh=4)
    assert code(wt.skip(wp.render(_edge16(), "float", 4, **options))) == code(
        wp.render(_edge16(), "float", 4, skip_outside=True, **options).replace("return;", "continue;"))
    assert wt.variant(wp.render(_edge16(), "float", 4, **options), "skip+vec2").count("if (!inside) {") == 1


A100_H200_FAMILIES = (("NVIDIA H200", True), ("NVIDIA H200 NVL", True), ("NVIDIA A100-SXM4-80GB", True),
                      ("NVIDIA A100-SXM4-40GB", True), ("NVIDIA A100-PCIE-40GB", True), ("NVIDIA A100 80GB PCIe", True),
                      ("NVIDIA GeForce RTX 5060 Ti", False), ("NVIDIA RTX 4000 Ada Generation", False))


def test_skip_outside_default(monkeypatch):
    monkeypatch.setattr(wp, "SKIP_OUTSIDE", None)
    assert not wp.skip_outside_default(torch.device("cpu"))
    # H200 MD step 0.9889 / 0.9926, 0.9823 / 0.9811; the A100 plan 0.9849 / 0.9828, 0.9751 / 0.9751 (12 Nov)
    assert set(wp.SKIP_OUTSIDE_GPUS) == {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"}
    for name, on in A100_H200_FAMILIES:
        monkeypatch.setattr(wp, "_device_name", lambda device, name=name: name)
        assert wp.skip_outside_default(None) is on and wp.hoist_dx_default(None) is on, name
    monkeypatch.setattr(wp, "_device_name", lambda device: "NVIDIA H200")
    assert wp.skip_outside_default(None)
    monkeypatch.setattr(wp, "SKIP_OUTSIDE", False)  # an A/B side's choice wins over the set
    assert not wp.skip_outside_default(None)
    monkeypatch.setattr(wp, "SKIP_OUTSIDE", True)
    assert wp.skip_outside_default(torch.device("cpu"))


def test_skip_outside_and_hoist_dx_together_only():
    """SKIP_OUTSIDE and HOIST_DX win on the H200 together only (lab 0.973; alone 1.021 and 0.999): one GPU set."""
    assert wp.HOIST_DX_GPUS is wp.SKIP_OUTSIDE_GPUS


def test_skip_outside_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    for capability in ((9, 0), (8, 0)):
        for edge_force in (False, True):
            assert compile_cubin(wp.render(_edge16(), "float", 4, vector=2, table_w=True, skip_outside=True,
                                           edge_force=edge_force), capability)


def _tuned_base():
    return wp.render(_edge16(), "float", 4, vector=2, table_w=True, factored=True, scale=True, horner=True,
                     blocks=[(1, [0]), (3, [64])], stage_dout=True, stage_sh=4)


def test_hoist_dx_and_channel_vector_sources():
    """HOIST_DX / CHANNEL_VECTOR (wprime_tuning, 12 November 2026): the dx address as one pointer per lane; N channels
    per lane with vector staged-row, x and dx accesses (vector atomics from sm_90 only)."""
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt

    base = _tuned_base()
    hoisted = wt.apply(base, hoist_dx=True)
    assert "scalar_t* dx_lane = dx + input_base + lane;" in hoisted and "dx_lane += LOGICAL_GROUP_SIZE;" in hoisted
    assert "dx + input_base + 0 + channel" in base and "dx + input_base + 0 + channel" not in hoisted
    for n in (2, 4):
        source = wt.apply(base, hoist_dx=True, channel_vector=n)  # the vector form needs no hoist
        assert f"for (int cbase = lane * {n}; cbase < CHANNEL_DIM; cbase += LOGICAL_GROUP_SIZE * {n})" in source
        assert f"reinterpret_cast<const float{n}*>(stage_row + " in source and "#if __CUDA_ARCH__ >= 900" in source
        assert f"atomicAdd(reinterpret_cast<float{n}*>(dx + input_base + 0 + cbase)" in source
        assert "for (int channel = lane;" not in source and "dx_lane" not in source
    assert wt.apply(base) == base
    with pytest.raises(ValueError):
        wt.vec(base, 3)
    with pytest.raises(ValueError):
        wt.hoist("no channel loop here")


def _quadrupole(channels=64):
    """``_edge16`` plus a 0e x 2e -> 2e path: five edge sums v_j (WEIGHTED_VECTOR's J > 3)."""
    schedule = _edge16()
    slot = lambda name, start, component, dim: SimpleNamespace(name=name, start=start, component=component,  # noqa: E731
                                                              irrep_dim=dim)
    grad = slot("gx0", 0, 0, 1)
    outs = [SimpleNamespace(accumulator=slot(f"o2{m}", 4 * channels, m, 5),
                            terms=(SimpleNamespace(input_component=0, edge_component=m, coefficient=-0.25,
                                                   input_accumulator=grad),)) for m in range(5)]
    path = SimpleNamespace(input_start=0, input_irrep_dim=1, edge_start=4, edge_irrep_dim=5,
                           weight_start=2 * channels, edge_mul=1, outputs=tuple(outs))
    schedule.paths = (*schedule.paths, path)
    schedule.output_slots = (*schedule.output_slots, *(o.accumulator for o in outs))
    schedule.output_dim, schedule.weight_numel = 9 * channels, 3 * channels
    return schedule


def test_weighted_vector_source():
    """WEIGHTED_VECTOR (12 November 2026): a path with more than 3 edge sums adds w x_i c g_o per term, without v_j;
    paths with 3 or fewer, and SCALAR_PATHS' paths, keep their form."""
    options = dict(vector=2, table_w=True, factored=True, horner=True)
    plain = wp.render(_quadrupole(), "float", 4, **options)
    source = wp.render(_quadrupole(), "float", 4, weighted_vector=True, **options)
    flat, flat_plain = " ".join(source.split()), " ".join(plain.split())
    assert "WEIGHTED_VECTOR" not in plain and flat.count("WEIGHTED_VECTOR:") == 1
    for j in range(4, 9):  # the 2e path: per term, from radial_weight, no v_j
        assert f"edge_gradient[{j}] += scaled * (radial_weight * input_value);" in flat
        assert f"edge_gradient[{j}] += scaled_weight * v_" in flat_plain
        assert f"edge_gradient[{j}] += scaled_weight * v_" not in flat
    assert "edge_gradient[2] += scaled_weight * v_1;" in flat  # the 1o path (J = 3) keeps v_j
    # with FACTORED_SCALE and without FOLD_ALPHA the path's alpha goes into the weight once
    scaled = " ".join(wp.render(_quadrupole(), "float", 4, scale=True, weighted_vector=True, **options).split())
    assert "const scalar_t vector_weight = radial_weight * static_cast<scalar_t>(0x1.0000000000000p-2f);" in scaled
    assert "edge_gradient[4] -= scaled * (vector_weight * input_value);" in scaled
    folded = " ".join(wp.render(_quadrupole(), "float", 4, scale=True, fold_alpha=True, weighted_vector=True,
                                **options).split())
    assert "vector_weight" not in folded and "edge_gradient[4] -= scaled * (radial_weight * input_value);" in folded
    # SCALAR_PATHS takes the scalar-input 2e path first
    both = wp.render(_quadrupole(), "float", 4, scalar_paths=True, weighted_vector=True, **options)
    assert both == wp.render(_quadrupole(), "float", 4, scalar_paths=True, **options)
    with pytest.raises(ValueError, match="weighted_vector"):
        wp.render(_quadrupole(), "float", 4, vector=2, table_w=True, weighted_vector=True)  # no FACTORED


def test_weighted_vector_default(monkeypatch):
    monkeypatch.setattr(wp, "WEIGHTED_VECTOR", None)
    assert not wp.weighted_vector_default(torch.device("cpu"))
    # the A100 family (decision 66, 12 Nov: with SKIP_OUTSIDE + HOIST_DX 0.9742 / 0.9730, 0.9710 / 0.9701); not the H200
    assert set(wp.WEIGHTED_VECTOR_GPUS) == {"NVIDIA A100-SXM4-80GB"}
    for name, on in A100_H200_FAMILIES:
        monkeypatch.setattr(wp, "_device_name", lambda device, name=name: name)
        assert wp.weighted_vector_default(None) is (on and "A100" in name), name
    monkeypatch.setattr(wp, "WEIGHTED_VECTOR", True)
    assert wp.weighted_vector_default(torch.device("cpu"))


def test_weighted_vector_yields_to_late_pullback(monkeypatch):
    """WEIGHTED_VECTOR is off under LATE_PULLBACK (its block parser is not checked against weighted sources) and on
    FACTORED kernels only; the A100 is not in LATE_PULLBACK_GPUS, so the guard changes nothing there today."""
    monkeypatch.setattr(wp, "WEIGHTED_VECTOR", None)
    monkeypatch.setattr(wp, "_device_name", lambda device: "NVIDIA A100-SXM4-80GB")
    assert wp.weighted_vector_on(None, True, False)
    assert not wp.weighted_vector_on(None, True, True) and not wp.weighted_vector_on(None, False, False)
    assert "NVIDIA A100-SXM4-80GB" not in wp.LATE_PULLBACK_GPUS
    monkeypatch.setattr(wp, "WEIGHTED_VECTOR", True)  # an A/B side's True still yields to LATE_PULLBACK
    assert not wp.weighted_vector_on(None, True, True)
    import inspect  # WPrimeBackward decides LATE_PULLBACK before the render and passes it to the guard
    init = inspect.getsource(wp.WPrimeBackward.__init__)
    assert init.index("self.late_pullback = (") < init.index("weighted_vector_on(device, self.factored, "
                                                              "self.late_pullback, model_label)") < init.index("render(")


def test_weighted_vector_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    options = dict(vector=2, table_w=True, factored=True, scale=True, horner=True, weighted_vector=True,
                   blocks=[(1, [0]), (3, [64]), (5, [256])], stage_dout=True, stage_sh=4)
    for capability in ((12, 0), (9, 0), (8, 0)):
        source = wp.render(_quadrupole(), "float", 4, skip_outside=True, **options)
        assert compile_cubin(source, capability) and compile_cubin(wt.hoist(source), capability)


def test_hoist_dx_and_channel_vector_defaults(monkeypatch):
    cpu = torch.device("cpu")
    monkeypatch.setattr(wp, "HOIST_DX", None)
    monkeypatch.setattr(wp, "CHANNEL_VECTOR", None)
    assert not wp.hoist_dx_default(cpu) and wp.channel_vector_default(cpu) == 0
    assert set(wp.HOIST_DX_GPUS) == {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"} and len(wp.CHANNEL_VECTOR_GPUS) == 0
    monkeypatch.setattr(wp, "HOIST_DX", True)
    monkeypatch.setattr(wp, "CHANNEL_VECTOR", 2)
    assert wp.hoist_dx_default(cpu) and wp.channel_vector_default(cpu) == 2


def test_tuned_sources_compile_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming import wprime_tuning as wt
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    base = _tuned_base()
    for capability in ((9, 0), (8, 0)):
        for options in (dict(hoist_dx=True), dict(channel_vector=2), dict(channel_vector=4)):
            assert compile_cubin(wt.apply(base, **options), capability)


def test_edge_vectors_only_for_edge_sh_output():
    """EDGE_FORCE's hand-over (rewrites.generic.modules): an interaction gets EdgeSH's vectors only for the very
    tensor EdgeSH returned last; anything else (another tensor, no EdgeSH reference) keeps the dsh path."""
    from fastmace.rewrites.generic import modules as gm

    source = gm.EdgeSH(torch.nn.Identity())
    interaction = torch.nn.Module()
    assert gm.edge_vectors(interaction, torch.zeros(2, 16)) is None  # no reference
    interaction.__dict__["edge_sh_source"] = source
    vectors, attrs = torch.zeros(2, 3), torch.zeros(2, 16)
    source.last = (weakref.ref(vectors), weakref.ref(attrs))
    assert gm.edge_vectors(interaction, attrs) is vectors
    assert gm.edge_vectors(interaction, attrs.clone()) is None
    del vectors, attrs  # weak: a finished forward's tensors (and their autograd graph) are not kept alive
    assert gm.edge_vectors(interaction, torch.zeros(2, 16)) is None and source.last[0]() is None
    assert "edge_sh_source" not in dict(interaction.named_modules())  # a reference, not a submodule
