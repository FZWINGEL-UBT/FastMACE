"""tools/ab_rewrites.py without a GPU: its two rewrite sets."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("_tool_ab_rewrites", TOOLS / "ab_rewrites.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_sides_parse_and_must_differ():
    args = tool.parse_args(["--output", "o", "--checkpoint", "c", "--a", "gated_rows=on", "--b", "gated_rows=on",
                            "segment_gemm=on"])
    assert args.a == {"gated_rows": "on"} and args.b == {"gated_rows": "on", "segment_gemm": "on"}
    with pytest.raises(SystemExit):
        tool.parse_args(["--output", "o", "--checkpoint", "c", "--a", "gated_rows=on", "--b", "gated_rows=on"])
    with pytest.raises(SystemExit):
        tool.parse_args(["--output", "o", "--checkpoint", "c", "--b", "nonsense=on"])


def test_engine_sides_take_booleans_floats_and_choices():
    assert tool.engine_overrides(["step_graph=true", "force_graphs=keep"]) == {"step_graph": True,
                                                                               "force_graphs": "keep"}
    for bad in (["step_graph=on"], ["no_such_field=1"], ["force_graphs="]):
        with pytest.raises(ValueError):
            tool.engine_overrides(bad)


def test_each_step_sets_its_sides_constants():
    from fastmace.kernels.triton import tile_rule

    seen = []

    class Side:
        engine = "engine"

        def step(self):
            seen.append(tile_rule.TRITON_NUM_WARPS)

    key = ("fastmace.kernels.triton.tile_rule", "TRITON_NUM_WARPS")
    before = tile_rule.TRITON_NUM_WARPS
    try:
        a, b = tool.ConstantsSide(Side(), {key: 8}), tool.ConstantsSide(Side(), {key: 4})
        a.step(), b.step(), a.step()
        assert seen == [8, 4, 8] and a.engine == "engine"
    finally:
        tile_rule.TRITON_NUM_WARPS = before


def test_a_sides_plan_reads_its_constants_not_the_last_steps():
    """OMAT-0 small on the H200 (8 November 2026): the plan ran outside the side's constants, so the label-keyed
    decisions read whatever the previous side's last step had left (2.2 or 5.4 ms per step by build order)."""
    from fastmace.planner import fixed

    key = ("fastmace.planner.fixed", "DECISION_66_TF32X3_LABELS")
    before = fixed.DECISION_66_TF32X3_LABELS
    seen = []
    try:
        tool.ConstantsSide(type("Side", (), {"step": lambda self: None})(), {key: ("omat0-medium", None)}).step()
        assert fixed.DECISION_66_TF32X3_LABELS == ("omat0-medium", None)  # a step leaves its side's value set
        for values in ({key: before}, {key: ("omat0-medium", None)}):
            tool.built_under(values, lambda: seen.append(("field", fixed.DECISION_66_TF32X3_LABELS)) or "field",
                             lambda field: seen.append(("plan", fixed.DECISION_66_TF32X3_LABELS)) or "plan",
                             lambda plan, field: seen.append(("build", fixed.DECISION_66_TF32X3_LABELS)) or "engine")
            assert fixed.DECISION_66_TF32X3_LABELS == ("omat0-medium", None)  # restored to what it found
        assert seen == [(stage, before) for stage in ("field", "plan", "build")] + [
            (stage, ("omat0-medium", None)) for stage in ("field", "plan", "build")]
    finally:
        fixed.DECISION_66_TF32X3_LABELS = before


def test_pack_entries_split_constants_from_rewrite_fields():
    entries = tool.pack_entries("t4|fastmace.kernels.triton.tile_rule:TRITON_NUM_WARPS=4 ;"
                                " seg|segment_gemm=on fastmace.kernels.triton.tile_rule:TRITON_NUM_WARPS=8;")
    key = ("fastmace.kernels.triton.tile_rule", "TRITON_NUM_WARPS")
    assert entries == [("t4", {}, {key: 4}), ("seg", {"segment_gemm": "on"}, {key: 8})]
    for bad in ("", "noitems|", "|segment_gemm=on", "a b|segment_gemm=on", "x|nonsense=on",
                "x|segment_gemm=on;x|gated_rows=on", "x|fastmace.kernels.triton.tile_rule:TRITON_NUM_WARPS='4'"):
        with pytest.raises(ValueError):
            tool.pack_entries(bad)


def test_pack_entries_must_differ_from_side_a():
    args = tool.parse_args(["--output", "o", "--checkpoint", "c", "--pack", "s|segment_gemm=on;g|gated_rows=on"])
    assert [name for name, _, _ in args.pack] == ["s", "g"] and args.b == {}
    with pytest.raises(SystemExit):  # entry "same" equals side a
        tool.parse_args(["--output", "o", "--checkpoint", "c", "--a", "segment_gemm=on",
                         "--pack", "s|gated_rows=on;same|segment_gemm=on"])


def test_a_dict_literal_sets_a_per_model_gpu_table():
    """A per-model table (MappingProxyType, e.g. DECISION_66_TF32X3_MODEL_GPUS) takes a dict literal, read-only as the
    default; a space in a GPU name is written as the escape backslash-x20, since --pack splits items at whitespace."""
    from types import MappingProxyType

    out = tool.constant_overrides(
        [r"fastmace.planner.fixed:DECISION_66_TF32X3_MODEL_GPUS={'omat0-small':('NVIDIA\x20A100-SXM4-80GB',)}"])
    value = out[("fastmace.planner.fixed", "DECISION_66_TF32X3_MODEL_GPUS")]
    assert isinstance(value, MappingProxyType)
    assert "NVIDIA A100-SXM4-80GB" in value["omat0-small"]
