"""fused_folded_forward (opt-8 OPT8-T02) without a GPU: the launch rule, the path parts, the generated source,
its NVRTC compilation, the record and the planner's handling."""

from types import SimpleNamespace

import pytest
import torch

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
    assert "constexpr int K = 9;" in source and "constexpr int KA = 9;" in source


def test_skip_zero_column_reads_one_input_fewer(monkeypatch):
    """``SKIP_ZERO_COLUMN``: the k-loop stops before an all-zero last weight column; a and W keep their stride."""
    rule = ff.launch_rule(2, 9, 4, 64, **SM120)
    source = ff.render(_schedule(), 9, rule, used=8)
    assert "constexpr int K = 8;" in source and "constexpr int KA = 9;" in source
    assert "a[(e0 + b) * KA + k]" in source and "W[(int64_t)(wst[p] + cb * CB + l) * KA + k]" in source
    with pytest.raises(ValueError, match="used columns"):
        ff.render(_schedule(), 9, rule, used=10)
    weight = torch.ones(128, 9)
    weight[:, -1] = 0.0
    assert ff.used_columns(weight) == 9  # by default only on SKIP_ZERO_COLUMN_GPUS (a CPU weight: none)
    monkeypatch.setattr(ff, "SKIP_ZERO_COLUMN", False)
    assert ff.used_columns(weight) == 9
    monkeypatch.setattr(ff, "SKIP_ZERO_COLUMN", True)
    assert ff.used_columns(weight) == 8
    weight[3, -1] = 1e-30
    assert ff.used_columns(weight) == 9  # one nonzero entry keeps the column


def test_stream_store_w_only_changes_the_store():
    rule = ff.launch_rule(2, 9, 4, 64, **SM120)
    plain, streamed = ff.render(_schedule(), 9, rule), ff.render(_schedule(), 9, rule, stream_store=True)
    assert "__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);" in streamed and "__stcs" not in plain
    assert streamed.replace("__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);",
                            "wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];") == plain


def test_tensor_cores_render_and_launch_rule(monkeypatch):
    """``TENSOR_CORES``: the a.W product in 3xTF32 mma.sync tiles over a k padded to a multiple of 8, the W slice
    with row stride ``TC_STRIDE``, a per-warp redistribution tile; sm_80 and later only."""
    assert ff.TENSOR_CORES is None  # per GPU (decision 66): every A100 model (owner, 4 November 2026)
    assert ff.tensor_cores((8, 0), "NVIDIA A100-SXM4-80GB") and ff.tensor_cores((8, 0), "NVIDIA A100 80GB PCIe")
    assert ff.tensor_cores((8, 0), "NVIDIA A100-SXM4-40GB")
    assert not ff.tensor_cores((8, 0)) and not ff.tensor_cores((8, 6), "NVIDIA RTX A4000")
    assert not ff.tensor_cores((12, 0), "NVIDIA GeForce RTX 5060 Ti")
    monkeypatch.setattr(ff, "TENSOR_CORES", False)
    assert not ff.tensor_cores((8, 0), "NVIDIA A100-SXM4-80GB")
    monkeypatch.setattr(ff, "TENSOR_CORES", True)
    assert ff.tensor_cores((8, 0)) and ff.tensor_cores((12, 0)) and not ff.tensor_cores((7, 5))
    for device in (SM80, SM89, SM120):
        rule = ff.launch_rule(10, 64, 16, 128, tc=True, **device)
        assert rule.shared_bytes <= device["shared_optin"] and rule.warps >= ff.MIN_WARPS
        assert rule.shared_bytes == 4 * ff.slice_floats(-(-10 // rule.parts), 64, True) + rule.warps * 4 * (
            ff.staging_floats(64, 16, True, -(-10 // rule.parts)))
    rule = ff.launch_rule(2, 65, 4, 64, tc=True, **SM80)
    source = ff.render(_schedule(), 65, rule, tc=True)
    assert "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32" in source
    assert ff.TC_SPLIT == "mask" and "hi = __float_as_uint(x) & 0xffffe000u;" in source
    for split, line in (("rna", "cvt.rna.tf32.f32"),
                        ("round", "lo = (__float_as_uint(x - __uint_as_float(hi)) + 0x1000u) & 0xffffe000u;")):
        monkeypatch.setattr(ff, "TC_SPLIT", split)
        assert line in ff.render(_schedule(), 65, rule, tc=True)
    monkeypatch.setattr(ff, "TC_SPLIT", "mask")
    assert "constexpr int KP = 72;" in source and f"constexpr int WS = {ff.TC_STRIDE};" in source
    assert "a_s[k * B + b] = (b < nb && k < K) ? a[(e0 + b) * KA + k] : 0.f;" in source
    skipped = ff.render(_schedule(), 65, ff.launch_rule(2, 64, 4, 64, tc=True, **SM80), used=64, tc=True)
    assert "constexpr int K = 64;" in skipped and "constexpr int KP = 64;" in skipped
    plain = ff.render(_schedule(), 65, ff.launch_rule(2, 65, 4, 64, **SM80))
    assert "mma.sync" not in plain and "constexpr int KP = 65;" in plain


SM90 = dict(shared_optin=232448, sms=132)  # H200


def test_wgmma_switch_launch_rule_and_source(monkeypatch):
    """``WGMMA`` (round 5): sm_90 only, off everywhere by default; parts of at most 8 paths, warps a multiple of 4,
    the two W planes beside the warps' staging; the product as wgmma m64nNk8 tf32 with the warpgroup vote; NVRTC
    compiles it for sm_90a."""
    assert ff.WGMMA is None and "NVIDIA H200" not in ff.WGMMA_GPUS
    assert not ff.wgmma_default((9, 0), "NVIDIA H200") and not ff.wgmma_default((9, 0))
    monkeypatch.setattr(ff, "WGMMA", True)
    assert ff.wgmma_default((9, 0)) and not ff.wgmma_default((8, 0)) and not ff.wgmma_default((12, 0))
    rule = ff.launch_rule_wgmma(10, 64, 16, 128, **SM90)
    assert (rule.parts, rule.warps, rule.blocks_per_slice) == (2, 8, 17)
    assert rule.shared_bytes == 8 * 5 * 32 * 64 + 8 * 4 * ff.wgmma_staging_floats(64, 16, 5) <= SM90["shared_optin"]
    small = ff.launch_rule_wgmma(2, 9, 4, 64, **SM90)
    assert small.parts == 1 and small.warps % 4 == 0
    source = ff.render(_schedule(), 9, small, tc=True, wgmma=True)
    assert "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32" in source and "bar.red.or.pred" in source
    assert "constexpr int AW = KP + 4;" in source
    assert "wgmma_n64(d, al[ks], gmma_desc(bh_addr + 256 * ks), ks > 0);" in source
    kernel = source.split('extern "C"')[-1]
    assert "mma_tf32(" not in kernel
    monkeypatch.setattr(ff, "WGMMA_PIPELINED", True)
    piped = ff.render(_schedule(), 9, small, tc=True, wgmma=True)
    assert piped != source and "if (pend) {" in piped and "const int64_t e0 = pew + 0;" in piped
    assert "mma_tf32(" not in piped.split('extern "C"')[-1]
    monkeypatch.setattr(ff, "WGMMA_PIPELINED", False)
    with pytest.raises(ValueError, match="wgmma"):
        ff.render(_schedule(), 9, small, tc=False, wgmma=True)
    from fastmace.kernels import nvrtc_available
    if nvrtc_available():
        from fastmace.kernels.oeq_streaming.runtime import compile_cubin
        for layout in ("ir_mul", "mul_ir"):
            assert compile_cubin(ff.render(_schedule(layout), 9, small, tc=True, wgmma=True), (9, 0), suffix="a")
            assert compile_cubin(ff.render(_schedule(layout), 65, ff.launch_rule_wgmma(2, 64, 4, 64, **SM90), used=64,
                                           tc=True, vec_a=True, vec_sh=True, wgmma=True, store_w=False), (9, 0),
                                 suffix="a")
        monkeypatch.setattr(ff, "WGMMA_PIPELINED", True)
        assert compile_cubin(ff.render(_schedule(), 9, small, tc=True, wgmma=True), (9, 0), suffix="a")


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
        assert compile_cubin(ff.render(_schedule(layout), 65, ff.launch_rule(2, 65, 4, 64, **SM80), used=64,
                                       stream_store=True), (8, 0))
        for capability, device in (((8, 0), SM80), ((8, 9), SM89), ((12, 0), SM120)):  # TENSOR_CORES
            assert compile_cubin(ff.render(_schedule(layout), 65, ff.launch_rule(2, 64, 4, 64, tc=True, **device),
                                           used=64, tc=True), capability)
            assert compile_cubin(ff.render(_schedule(layout), 9, ff.launch_rule(2, 9, 4, 64, tc=True, **device),
                                           tc=True), capability)


def test_record_and_requirements():
    pytest.importorskip("mace")
    from fastmace.rewrites import mh1

    record = mh1.RECORDS["fused_folded_forward"]
    assert record.requires == ("chunked_conv_backward",)
    assert record.exactness == "exact" and record.default == "auto" and not record.deterministic
    order = tuple(mh1.RECORDS)
    assert order.index("chunked_conv_backward") < order.index("fused_folded_forward")


def test_planner_rules():
    from fastmace.planner import fixed

    assert fixed.AUTO_RULES["RewriteConfig.fused_folded_forward"] == (
        "on_for_mh1_float32_unless_reproducible_and_for_omat0_with_wprime_on_decision_66_gpus")
    assert fixed.REWRITE_FIELDS.index("fused_folded_forward") == fixed.REWRITE_FIELDS.index(
        "chunked_conv_backward") + 1


def test_planner_resolution():
    from test_planner import plan_for

    from fastmace.config import PrecisionConfig, RewriteConfig
    from fastmace.planner.fixed import PlanRefused

    # Owner decision 64: on for MH-1 in float32 plans that are not reproducible; off for every other model.
    assert plan_for("mh1-r2scan").rewrites.fused_folded_forward == "on"
    assert plan_for("mh1-omat").rewrites.fused_folded_forward == "on"
    for label in ("polar1-m", "omol0-xl"):  # POLAR-1-M: with wprime_backward on the RTX 4000 Ada only (polar-T01)
        plan = plan_for(label)
        assert plan.rewrites.fused_folded_forward == "off" and plan.rewrites.chunked_conv_backward == "on", label
        assert "not MH-1, decision 64" in dict(plan.auto_options_off)["fused_folded_forward"]
    # POLAR-1-S: one, reassociated interaction, nothing for the fused forward to act on.
    assert "no interaction after the reassociated first one" in dict(
        plan_for("polar1-s").auto_options_off)["fused_folded_forward"]
    assert plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64")).rewrites.fused_folded_forward == "off"
    assert plan_for("omol0-xl", rewrites=RewriteConfig(fused_folded_forward="on")).rewrites.fused_folded_forward == "on"
    plan = plan_for("mh1-r2scan", rewrites=RewriteConfig(fused_folded_forward="on"))
    assert plan.rewrites.fused_folded_forward == "on" and plan.rewrites.chunked_conv_backward == "on"
    with pytest.raises(PlanRefused, match="fused_folded_forward='on' needs float32"):
        plan_for("mh1-r2scan", precision=PrecisionConfig(dtype="float64"),
                 rewrites=RewriteConfig(fused_folded_forward="on"))
    with pytest.raises(PlanRefused, match="fused_folded_forward='on' requires chunked_conv_backward"):
        plan_for("mh1-r2scan", rewrites=RewriteConfig(fused_folded_forward="on", chunked_conv_backward="off"))
