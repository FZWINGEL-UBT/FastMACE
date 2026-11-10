"""The generic chunked streaming backward (``generic_chunked_conv_backward``; docs/validation/omat0/omat0-capacity.md).

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
import types
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
    # the second interaction's [E, 1280] weight gradient, less the saved folded input [h c, c] (65 floats)
    assert terms.generic_chunked_conv_backward(dims, 4) == (-(1280 - 65) * 4, 0)


@pytest.mark.parametrize(("label", "reason"), [("mh1-r2scan", "ok"), ("omat0-medium", "ok (generic folded path)"),
                                               ("mp0-medium", "ok (generic folded path)"),
                                               ("off23-small", "ok (generic folded path)")])
def test_support(label, reason):
    assert tuple(tp.traits(label).rewrite_support["chunked_conv_backward"]) == (True, reason)


def test_stage_2_keeps_it_to_measured_gpus(monkeypatch):
    assert fixed.DECISION_66_GENERIC_CHUNKED_GPUS == ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_CHUNKED_GPUS", ("NVIDIA A100-SXM4-40GB",))
    monkeypatch.setattr(fixed, "DECISION_66_GENERIC_WPRIME_GPUS", ())  # else on as wprime_backward's carrier
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
    assert terms.generic_recompute_conv_weights(dims, 4) == (-1280 * 4, 0)  # the stored [E, 1280] weights
    rescue = memory.MEMORY_RESCUE
    assert rescue.index("generic_recompute_conv_weights") == rescue.index("recompute_conv_weights") + 1
    assert memory.RESCUE_FLOAT32_ONLY == ("generic_recompute_conv_weights", "generic_recompute_first_table")
    assert memory.rescue_stage(["generic_chunked_conv_backward", "generic_recompute_conv_weights"]) == (
        "generic_recompute_conv_weights")


def test_the_rung_saves_nothing_beside_generic_wprime():
    """Beside generic_wprime_backward (which stores no per-edge weights either) the rung saves nothing per edge and
    its path keeps the message gradient dense, ``messages`` floats per node more (OMAT-0 medium: 5,120; SrTiO3 39^3
    ran out of memory on it, results/memory-slack-20261109): its context term says so, and RESCUE_NOT_BESIDE keeps
    rule 12 and the fallback from taking it there."""
    from fastmace.planner import memory

    dims = ModelDims.from_record(FIT["dims"]["omat0-medium"])
    assert terms.generic_recompute_conv_weights_beside_wprime(dims, 4) == (0, 5120 * 4)
    assert ("generic_recompute_conv_weights", "generic_wprime_backward",
            "generic_recompute_conv_weights_beside_wprime") in memory.CONTEXT_TERMS
    assert memory.RESCUE_NOT_BESIDE == (("generic_recompute_conv_weights", "generic_wprime_backward"),)
    terms_ = dict(memory.rewrite_memory_terms(
        {"generic_recompute_conv_weights": types.SimpleNamespace(memory=terms.generic_recompute_conv_weights)},
        dims, ["generic_wprime_backward", "generic_recompute_conv_weights"], "float32"))
    assert terms_["generic_recompute_conv_weights"] == memory.MemoryTerm(0, 5120 * 4)


def test_the_first_layer_rung():
    """generic_recompute_first_layer (results/rung-b-20261109): its own field, off by default, after the first
    table's rung; rounding; float32 only; product 0's input (the first interaction's ``target``, 2,048 floats per node
    at OMAT-0 medium) is not kept, while the table's rung keeps its per-edge term only."""
    from fastmace.planner import fixed as fixed_planner
    from fastmace.planner import memory
    from fastmace.rewrites import radial

    record = radial.RECORDS["generic_recompute_first_layer"]
    assert record["config_field"] == "recompute_first_layer" and record["default"] == "off"
    assert record["requires"] == ("generic_recompute_first_table",) and record["exactness"] == "rounding"
    assert record["memory"] is terms.generic_recompute_first_layer
    dims = ModelDims.from_record(FIT["dims"]["omat0-medium"])
    assert terms.generic_recompute_first_layer(dims, 4) == (0, -2048 * 4)
    assert terms.generic_recompute_first_table(dims, 4) == (-65 * 4, 0)
    assert terms.generic_recompute_first_table_with_weights(dims, 4) == (-2 * 65 * 4, 0)
    rescue = memory.MEMORY_RESCUE
    assert rescue.index("generic_recompute_first_layer") == rescue.index("generic_recompute_first_table") + 1
    assert "generic_recompute_first_layer" in memory.RESCUE_FLOAT32_ONLY
    assert memory.rescue_stage(["generic_recompute_first_table", "generic_recompute_first_layer"]) == (
        "generic_recompute_first_layer")
    assert fixed_planner.AUTO_RULES["RewriteConfig.recompute_first_layer"] == (
        "on_by_rule_12_when_the_plan_does_not_fit_with_recompute_first_table")
    assert tp.plan_for("omat0-medium").rewrites.recompute_first_layer == "off"  # auto: the rescue's


def test_an_explicit_first_layer_needs_the_first_table_on_a_generic_model():
    from fastmace.errors import PlanRefused

    with pytest.raises(PlanRefused, match="recompute_first_layer='on' requires recompute_first_table"):
        tp.plan_for("omat0-medium", rewrites=RewriteConfig(recompute_first_layer="on", recompute_first_table="off"))
    with pytest.raises(PlanRefused, match="recompute_first_layer='on' requires recompute_first_table"):
        tp.plan_for("mh1-r2scan", rewrites=RewriteConfig(recompute_first_layer="on", recompute_first_table="on"))


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


def test_rule_12_counts_each_family_s_records():
    """minimal._family_applies: MH-1's radial records (recompute_conv_weights) for MH-1's class only, the generic
    records for generic models, generic_radial_tables (in the radial module) included."""
    from fastmace.planner import minimal

    registry = minimal._rewrite_registry()
    generic_traits, mh1_traits = tp.traits("omat0-medium"), tp.traits("mh1-r2scan")
    for name, generic_model in (("generic_radial_tables", True), ("generic_recompute_conv_weights", True),
                                ("recompute_conv_weights", False), ("radial_tables", False)):
        record = registry[name]
        assert minimal._family_applies(record, generic_traits) is generic_model, name
        assert minimal._family_applies(record, mh1_traits) is (not generic_model), name


def test_the_generic_tables_term_matches_the_estimate_s_count():
    """generic_radial_tables removes what rule 12's stock count holds for the tabulated networks: each network's
    input and its saved hidden units (``planner.memory.interaction_floats``), not its output (the folded path's
    ``[h c, c]``, counted with generic_chunked_conv_backward) and not more: the real saved tensors (two per hidden
    unit) were never in the count (docs/validation/omat0-capacity-2.md)."""
    from fastmace.planner import memory

    dims = ModelDims.from_record(FIT["dims"]["omat0-medium"])
    counted = sum(memory._network_floats(block.radial, block.radial_saved) - block.radial[-1]
                  + sum(memory._network_floats(widths, saved) - widths[-1] for widths, saved in block.other_radials)
                  for block in dims.interactions)
    assert counted == 2 * ((10 + 64 * 3) + 10)  # OMAT-0 medium: two interactions, the radial net and the density's
    assert terms.generic_radial_tables(dims, 4) == (-counted * 4, 0)


def test_generic_wprime_stands_aside_on_the_rescue_rung(monkeypatch):
    """generic_wprime_backward is not on the recompute path (its docstring): where rule 12 took
    generic_recompute_conv_weights for every candidate interaction, it applies to none and the build goes on.
    Raising there sent every cell that needs the rung to the fallback's rewrites-off plan, which ran out of memory
    (5060 Ti SrTiO3 40,000 atoms, results/omat0-ada-capacity-20261108). Without any candidate it still raises."""
    import torch

    from fastmace.rewrites import radial
    from fastmace.rewrites.registry import RewriteError

    class Interaction:
        radial_table = object()
        generic_folded = object()
        first_reassociation = None
        chunked_backward_edges = 256
        generic_recompute = object()

    monkeypatch.setattr(radial, "_generic_interactions", lambda model: [(1, Interaction()), (2, Interaction())])
    assert radial.attach_generic_wprime(torch.nn.Module()) == []

    class Unchunked(Interaction):
        chunked_backward_edges = 0
        generic_recompute = None

    monkeypatch.setattr(radial, "_generic_interactions", lambda model: [(1, Unchunked())])
    with pytest.raises(RewriteError, match="no generic interaction"):
        radial.attach_generic_wprime(torch.nn.Module())


def test_probe_runs_the_composed_skip_without_a_table_gradient():
    """``generic._run`` (the generic self-checks, ``generic_recompute_first_table``'s after ``tf32x3_gemms``): with
    ``compose_skip`` on the first interaction (inference only, refuses a species table with a gradient) the probe's
    table is bound and fed without one and its gradient is zero, the other leaves' gradients unchanged. Before, the
    rung's self-check failed on the RTX 4000 Ada and the plan fell to the terminal sequence
    (results/omat0-ada-capacity-20261108, SrTiO3 32^3)."""
    import torch

    class Reassociation:
        columns = [0, 1]

        def __init__(self, composed):
            self.compose_skip = object() if composed else None
            self.species_feats = None

        def bind(self, feats):
            if self.compose_skip is not None and feats is not None and feats.requires_grad:
                raise RuntimeError("compose_skip has no gradient for the weights: inference only (frozen parameters)")
            self.species_feats = feats

    class Interaction:
        def __init__(self, composed):
            self.first_reassociation = Reassociation(composed)

        def linear_up(self, table):
            return 2.0 * table

        def __call__(self, node_attrs, feats, edge_attrs, edge_feats, edge_index, cutoff=None):
            x = self.first_reassociation.species_feats
            if self.first_reassociation.compose_skip is not None and (x.requires_grad or feats.requires_grad):
                raise RuntimeError("compose_skip has no gradient for the weights: inference only (frozen parameters)")
            return (feats.sum() * edge_feats.square().sum() + x.sum() * edge_attrs.sum(),)

    generator = torch.Generator().manual_seed(3)
    probe = dict(node_attrs=torch.eye(2, dtype=torch.float64),
                 table=torch.randn(2, 3, generator=generator, dtype=torch.float64).requires_grad_(),
                 edge_attrs=torch.randn(4, 2, generator=generator, dtype=torch.float64).requires_grad_(),
                 edge_feats=torch.randn(4, 1, generator=generator, dtype=torch.float64).requires_grad_(),
                 edge_index=torch.zeros(2, 4, dtype=torch.long), node_feats=None, cutoff=None)
    plain = generic._run(Interaction(False), probe, False, [], generator)
    composed = generic._run(Interaction(True), probe, False, list([torch.ones(())]), generator)
    assert torch.equal(composed[1], torch.zeros_like(probe["table"]))  # no table gradient with the composed skip
    assert plain[1].abs().sum() > 0
    assert probe["table"].requires_grad  # the caller's probe is left as it was
