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
