"""fused_folded_forward (opt-8 OPT8-T02) without a GPU: the launch rule, the path parts, the generated source,
its NVRTC compilation, the record and the planner's handling."""

from types import SimpleNamespace

import pytest

from fastmace.kernels.oeq_streaming import fused_forward as ff

SM120 = dict(shared_optin=101376, sms=36)  # RTX 5060 Ti
SM89 = dict(shared_optin=101376, sms=48)  # RTX 4000 Ada
SM80 = dict(shared_optin=166912, sms=108)  # A100


def test_launch_rule_mh1_shapes():
    rule = ff.launch_rule(10, 65, 16, 128, **SM120)
    # paths per part <= 40 / BATCH = 5, so two parts; 12 warps fit beside the 43 KB slice
    assert (rule.parts, rule.warps, rule.blocks_per_slice) == (2, 12, 9)
    assert rule.shared_bytes <= SM120["shared_optin"]
    assert rule.shared_bytes == 4 * ff.slice_floats(5, 65) + 12 * 4 * ff.staging_floats(65, 16)
    assert ff.launch_rule(10, 65, 16, 128, **SM89).blocks_per_slice == 12
    # More shared memory does not change the parts (the register rule binds), only fits the same warps.
    rule80 = ff.launch_rule(10, 65, 16, 128, **SM80)
    assert (rule80.parts, rule80.warps) == (2, 12)


def test_launch_rule_small_shared_memory_splits_more():
    rule = ff.launch_rule(10, 65, 16, 128, shared_optin=49152, sms=20)
    assert rule.parts > 2 and ff.MIN_WARPS <= rule.warps <= ff.MAX_WARPS
    assert rule.shared_bytes <= 49152
    with pytest.raises(ValueError):
        ff.launch_rule(1, 4096, 16, 128, shared_optin=49152, sms=20)


def test_balanced_parts_equal_counts_and_terms():
    costs = [1, 3, 3, 3, 11, 5, 11, 21, 7, 21]  # MH-1's second interaction (terms per path)
    parts = ff.balanced_parts(costs, 2)
    assert sorted(i for p in parts for i in p) == list(range(10))
    assert [len(p) for p in parts] == [5, 5]
    assert abs(sum(costs[i] for i in parts[0]) - sum(costs[i] for i in parts[1])) <= 1


def _schedule(layout="ir_mul", channels=64, edge_mul=1):
    """A two-path uvu schedule (0e x 0e -> 0e, 0e x 1o -> 1o) with the attributes the generator reads."""
    slot = lambda name, start, component, dim: SimpleNamespace(name=name, start=start, component=component,  # noqa: E731
                                                              irrep_dim=dim)
    term = lambda i, j, c: SimpleNamespace(input_component=i, edge_component=j, coefficient=c)  # noqa: E731
    out0 = SimpleNamespace(accumulator=slot("o0", 0, 0, 1), terms=(term(0, 0, 1.0),))
    out1 = [SimpleNamespace(accumulator=slot(f"o1{m}", channels, m, 3), terms=(term(0, m, 0.5),)) for m in range(3)]
    paths = (SimpleNamespace(input_start=0, input_irrep_dim=1, edge_start=0, edge_irrep_dim=1, weight_start=0,
                             edge_mul=edge_mul, outputs=(out0,)),
             SimpleNamespace(input_start=0, input_irrep_dim=1, edge_start=1, edge_irrep_dim=3, weight_start=channels,
                             edge_mul=edge_mul, outputs=tuple(out1)))
    slots = (out0.accumulator, *(o.accumulator for o in out1))
    return SimpleNamespace(channels=channels, layout=layout, paths=paths, output_slots=slots, edge_dim=4,
                           input_dim=channels, output_dim=4 * channels)


def test_render_ir_mul_and_mul_ir():
    rule = ff.launch_rule(2, 9, 4, 64, **SM120)
    source = ff.render(_schedule(), 9, rule)
    assert 'extern "C" __global__ void __launch_bounds__(' in source and "fused_folded_forward(" in source
    assert "constexpr int64_t Q = 128;" in source and "wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];" in source
    assert "x[(int64_t)snd * IN_DIM + 0 + ch]" in source  # ir_mul: channel-minor
    assert "out[cur * OUT_DIM + 64 + ch]" in source and "out[cur * OUT_DIM + 192 + ch]" in source
    mul_ir = ff.render(_schedule("mul_ir"), 9, rule)
    assert "out[cur * OUT_DIM + 66 + ch * 3]" in mul_ir  # start 64, component 2, dim 3


def test_render_refuses_unsupported_problems():
    rule = ff.launch_rule(2, 9, 4, 64, **SM120)
    with pytest.raises(ValueError, match="divisible"):
        ff.render(_schedule(channels=48), 9, rule)
    with pytest.raises(ValueError, match="one edge channel"):
        ff.render(_schedule(edge_mul=2), 9, rule)


def test_generated_source_compiles_with_nvrtc():
    from fastmace.kernels import nvrtc_available
    if not nvrtc_available():
        pytest.skip("NVRTC is not available")
    from fastmace.kernels.oeq_streaming.runtime import compile_cubin

    for layout in ("ir_mul", "mul_ir"):
        rule = ff.launch_rule(2, 9, 4, 64, **SM120)
        assert compile_cubin(ff.render(_schedule(layout), 9, rule), (12, 0))
        assert compile_cubin(ff.render(_schedule(layout), 65, ff.launch_rule(2, 65, 4, 64, **SM89)), (8, 9))


def test_record_and_requirements():
    pytest.importorskip("mace")
    from fastmace.rewrites import mh1

    record = mh1.RECORDS["fused_folded_forward"]
    assert record.requires == ("chunked_conv_backward",)
    assert record.exactness == "exact" and record.default == "off" and not record.deterministic
    order = tuple(mh1.RECORDS)
    assert order.index("chunked_conv_backward") < order.index("fused_folded_forward")


def test_planner_rules():
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.fused_folded_forward"] == "never_automatic"
    assert fixed.REWRITE_FIELDS.index("fused_folded_forward") == fixed.REWRITE_FIELDS.index(
        "chunked_conv_backward") + 1


def test_planner_resolution():
    from test_planner import plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.planner.fixed import PlanRefused

    assert plan_for("mh1-r2scan").rewrites.fused_folded_forward == "off"  # opt-in
    plan = plan_for("mh1-r2scan", rewrites=RewriteConfig(fused_folded_forward="on"))
    assert plan.rewrites.fused_folded_forward == "on" and plan.rewrites.chunked_conv_backward == "on"
    with pytest.raises(PlanRefused, match="fused_folded_forward='on' needs float32"):
        plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"),
                 rewrites=RewriteConfig(fused_folded_forward="on"))
    with pytest.raises(PlanRefused, match="fused_folded_forward='on' requires chunked_conv_backward"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(fused_folded_forward="on", chunked_conv_backward="off"))
