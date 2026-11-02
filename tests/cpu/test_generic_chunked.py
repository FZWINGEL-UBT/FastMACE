"""The generic chunked streaming backward (``generic_chunked_conv_backward``; docs/validation/omat0-capacity.md).

- The record: the generic family's, on ``RewriteConfig.chunked_conv_backward``, after ``generic_streaming`` and
  ``generic_fused_radial_backward``; rounding class (``(h c) W`` against ``(h W) c``); not deterministic (the
  streaming convolution's atomic sender sums); its memory term removes the widest weight gradient per edge.
- Support: MH-1's folded class as before ("ok"), and every model with a generic interaction ("ok (generic folded
  path)").
- Stage 2 (decision 66): "auto" keeps it on for a generic model only on ``DECISION_66_GENERIC_CHUNKED_GPUS``,
  matched by exact name; an explicit "on" is kept; MH-1 keeps decision 60's default.
The CUDA forward and backward against the unchunked path: tests/gpu/test_rewrites_generic_gpu.py.
"""

import json
from pathlib import Path

import pytest
import test_planner as tp

from fastmace.config import RewriteConfig
from fastmace.model.capabilities import ModelDims
from fastmace.planner import fixed
from fastmace.rewrites import generic
from fastmace.rewrites import memory as terms

FIT = json.loads((Path(__file__).resolve().parents[1] / "_support" / "data" / "rule12_fit.json").read_text())


def test_the_record():
    record = generic.RECORDS["generic_chunked_conv_backward"]
    assert record["config_field"] == "chunked_conv_backward" and record["exactness"] == "rounding"
    assert record["requires"] == ("generic_streaming", "generic_fused_radial_backward")
    assert record["deterministic"] is False and record["default"] == "auto"
    assert record["workload_constraints"] == ("sort_receivers", "per_engine_edge_counter")
    assert record["memory"] is terms.generic_chunked_conv_backward
    dims = ModelDims.from_record(FIT["dims"]["omat0-medium"])
    assert terms.generic_chunked_conv_backward(dims, 4) == (-1280 * 4, 0)  # the second interaction's [E, 1280]


@pytest.mark.parametrize(("label", "reason"), [("mh1-r2scan", "ok"), ("omat0-medium", "ok (generic folded path)"),
                                               ("mp0-medium", "ok (generic folded path)"),
                                               ("off23-small", "ok (generic folded path)")])
def test_support(label, reason):
    assert tuple(tp.traits(label).rewrite_support["chunked_conv_backward"]) == (True, reason)


def test_stage_2_keeps_it_to_measured_gpus(monkeypatch):
    assert fixed.DECISION_66_GENERIC_CHUNKED_GPUS == ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_CHUNKED_GPUS", ("NVIDIA A100-SXM4-40GB",))
    plan = tp.plan_for("omat0-medium")
    _, choice, why = plan.decision("RewriteConfig.chunked_conv_backward")
    assert choice == "off" and "decision 66" in str(why) and tp.GPU.name in str(why)
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_CHUNKED_GPUS", (tp.GPU.name,))
    measured = tp.plan_for("omat0-medium")
    assert measured.rewrites.chunked_conv_backward == "on"
    assert "generic model on a measured GPU" in str(measured.decision("RewriteConfig.chunked_conv_backward")[2])
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_CHUNKED_GPUS", ())
    explicit = tp.plan_for("omat0-medium", rewrites=RewriteConfig(chunked_conv_backward="on"))
    assert explicit.rewrites.chunked_conv_backward == "on"
    assert tp.plan_for("mh1-r2scan").rewrites.chunked_conv_backward == "on"  # decision 60, unchanged


# --------------------------------------------------------------------------- the rescue rung


def test_the_rescue_rung():
    """generic_recompute_conv_weights: on MH-1's field, off by default, after the chunked path; rounding; float32
    only; in the memory rescue after MH-1's own rung, its own stage of the allocator reserve."""
    from fastmace.planner import memory

    record = generic.RECORDS["generic_recompute_conv_weights"]
    assert record["config_field"] == "recompute_conv_weights" and record["default"] == "off"
    assert record["requires"] == ("generic_chunked_conv_backward",) and record["exactness"] == "rounding"
    dims = ModelDims.from_record(FIT["dims"]["omat0-medium"])
    assert terms.generic_recompute_conv_weights(dims, 4) == (-(1280 - 65) * 4, 0)  # keeps [h c, c], 65 floats
    rescue = memory.MEMORY_RESCUE
    assert rescue.index("generic_recompute_conv_weights") == rescue.index("recompute_conv_weights") + 1
    assert memory.RESCUE_FLOAT32_ONLY == ("generic_recompute_conv_weights",)
    assert memory.rescue_stage(["generic_chunked_conv_backward", "generic_recompute_conv_weights"]) == (
        "generic_recompute_conv_weights")


def test_an_explicit_rung_needs_the_chunked_path_and_float32():
    from fastmace.config import PrecisionConfig
    from fastmace.errors import PlanRefused

    both = RewriteConfig(chunked_conv_backward="on", recompute_conv_weights="on")
    assert tp.plan_for("omat0-medium", rewrites=both).rewrites.recompute_conv_weights == "on"
    with pytest.raises(PlanRefused, match="requires chunked_conv_backward"):
        tp.plan_for("omat0-medium", rewrites=RewriteConfig(chunked_conv_backward="off", recompute_conv_weights="on"))
    with pytest.raises(PlanRefused, match="needs float32"):
        tp.plan_for("omat0-medium", rewrites=both, precision=PrecisionConfig(dtype="float64"))
    assert tp.plan_for("omat0-medium").rewrites.recompute_conv_weights == "off"  # auto: the rescue's


@pytest.mark.rule(12)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_rule_12_takes_the_generic_rung_in_float32_only(monkeypatch, precision):
    """A tiny generic model with MACE-OMAT-0 medium's widths (the tiny model's weights are narrower than the hidden
    layer the rung keeps, so it would save nothing) on a cell that cannot fit (the overstatement at 1, as
    test_memory_estimate's rung tests): rule 12 turns recompute_conv_weights on with the chunked path on, never in
    float64."""
    import dataclasses
    import types

    import test_memory_estimate as tme
    import test_minimal_planner as tmp

    from _support.structures import nacl
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.config import IntegratorConfig
    from fastmace.model import capabilities
    from fastmace.planner import memory

    monkeypatch.setattr(memory, "RESCUE_OVERSTATEMENT", 1.0)
    traits = dataclasses.replace(capabilities.model_traits(tiny_mace_model(list(tme.Z), **FAMILIES["density"])),
                                 dims=ModelDims.from_record(FIT["dims"]["omat0-medium"]))
    rewrites = RewriteConfig(chunked_conv_backward="on")

    def plan(memory_bytes):
        planner = tmp.minimal()
        planner._free_memory_bytes = lambda hardware: hardware.memory_bytes
        hardware = dataclasses.replace(tmp.GPU, memory_bytes=int(memory_bytes))
        loaded = types.SimpleNamespace(label=None, head=None, sha256=None, backend=None)
        return planner.plan_traits(traits, nacl(repeat=16), IntegratorConfig(), tme.WORKLOAD, hardware,
                                   precision=precision, rewrites=rewrites, loaded=loaded)

    full = plan(tmp.GPU.memory_bytes)
    assert full.rewrites.recompute_conv_weights == "off"
    tight = plan(full.predicted_memory_bytes / memory.MARGIN * 0.8)  # below what it needs (the weights resident)
    taken = tight.rewrites.recompute_conv_weights == "on"
    assert taken is (precision == "float32"), (tight.decision("RewriteConfig.recompute_conv_weights"),
                                               tight.decision("memory"), tight.rewrites.chunked_conv_backward)
    if taken:
        assert tight.predicted_memory_bytes < full.predicted_memory_bytes
        assert tight.memory_ladder.chosen.stage == "generic_recompute_conv_weights"
