"""Rule 12 for the MACE model (plan task 5-T03; SPEC §9.3 rule 12, decision 51).

- ``capabilities.model_dims`` reads each family's widths, radial networks and their saved floats per
  hidden unit;
- the estimate's activations against the peak of a force call on the tiny models (CPU, float64 and
  float32, two sizes): never below it, at most 1.5x above;
- each declared rewrite term (``rewrites/memory.py``) against the measured change of the peak per edge;
- the planner: ``Plan.predicted_memory_bytes``, the ``memory`` decision with rule 12's code, decision 51's
  action (rewrites with an edge-sized term off, largest first, with their dependants; a rewrite the
  caller set kept and warned; rule 11's sizes, the tier and the precision never changed);
- an out-of-memory error carries the estimate.
"""

import copy
import dataclasses
import math
import types

import pytest
import test_minimal_planner as tmp
import torch

from _support.memory_peak import PeakTracker
from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_calculator, tiny_mace_model
from fastmace.config import IntegratorConfig, RewriteConfig
from fastmace.model import capabilities
from fastmace.planner import memory, rules
from fastmace.planner.workload import Workload
from fastmace.rewrites import fusion, generic, linear, mh1, radial
from fastmace.rewrites import memory as terms
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")

Z = (11, 17)
WORKLOAD = Workload(max_temperature_K=300.0)


def tiny(family, dtype="float64", repeat=2, z_values=Z):
    """(model, batch dict) of a tiny ``family`` model on rattled NaCl, weights frozen as in the engine."""
    calc = tiny_mace_calculator(list(z_values), **FAMILIES[family])
    model = calc.models[0]
    atoms = nacl(repeat=repeat, rattle=0.1, seed=8)
    if 8 in z_values:
        atoms.numbers[3] = 8  # a third species, once (as tests/cpu/test_rewrites_mh1.py)
    batch = calc._atoms_to_batch(atoms).to_dict()
    if dtype == "float32":
        model = model.float()
        batch = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                 for key, value in batch.items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, batch


def peak(model, batch):
    inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    with PeakTracker() as tracker:
        model(inputs, training=False, compute_force=True)
    return tracker.peak


def edges_nodes(batch):
    return int(batch["edge_index"].shape[1]), int(batch["positions"].shape[0])


# --------------------------------------------------------------------------- dims


@pytest.mark.rule(12)
def test_model_dims_of_the_families():
    mh1_dims = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["mh1"]))
    residual = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["residual"]))
    density = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["density"]))
    assert len(mh1_dims.interactions) == len(residual.interactions) == 2
    # MACE's RadialMLP (Linear, LayerNorm, SiLU) keeps two floats per hidden unit, e3nn's network one.
    assert {block.radial_saved for block in mh1_dims.interactions} == {2}
    assert {block.radial_saved for block in residual.interactions} == {1}
    # Widths: the radial basis and two embedded species blocks in, the tensor product's weights out.
    first = mh1_dims.interactions[0]
    assert first.radial[1:-1] == (64, 64, 64) and first.radial[0] > first.edge_feats == 6
    assert first.edge_attrs == 9 and first.messages > first.sender > 0 and first.correlation == 2
    # MH-1's density normalisation and the density family's are per-edge networks too; the residual has none.
    assert [widths for widths, _ in first.other_radials] == [(first.radial[0], 64, 1)]
    assert density.interactions[1].other_radials and not residual.interactions[0].other_radials
    # The last interaction's hidden irreps is a string in MACE.
    assert all(block.hidden > 0 for block in mh1_dims.interactions)
    assert mh1_dims.parameters == sum(p.numel() for p in tiny_mace_model(list(Z), **FAMILIES["mh1"]).parameters())
    assert capabilities.model_traits(tiny_mace_model(list(Z), **FAMILIES["mh1"])).dims == mh1_dims


def test_model_dims_record_round_trip():
    """A capture of the released checkpoints (5-T11's plan corpus) stores ``record()``; ``from_record`` reads it."""
    import json

    for family in ("mh1", "density"):
        dims = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES[family]))
        assert capabilities.ModelDims.from_record(json.loads(json.dumps(dims.record()))) == dims


def test_model_dims_of_a_stand_in_is_none():
    assert capabilities.model_dims(torch.nn.Linear(2, 2)) is None


# --------------------------------------------------------------------------- the estimate on the tiny models


@pytest.mark.rule(12)
@pytest.mark.parametrize("family", ["mh1", "residual", "density", "off"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_estimate_structure_matches_the_tiny_models_peak(family, dtype):
    """The run term's structure, unscaled: the activations it counts for stock MACE are within [0.8, 1.2] of a
    stock force call's peak (0.84-0.91 measured). ``SAFETY`` is fitted on the GPU peaks, where the fused paths keep
    less than stock; tests/cpu/test_memory_fit.py checks it."""
    for repeat in (2, 3):
        model, batch = tiny(family, dtype, repeat)
        measured = peak(model, batch)
        edges, nodes = edges_nodes(batch)
        found = memory.estimate(capabilities.model_dims(model), edges=edges, nodes=nodes, dtype=dtype,
                                workspace=False)
        unscaled = found.term("activations") / memory.SAFETY
        assert 0.8 * measured <= unscaled <= 1.2 * measured, (family, dtype, repeat, measured, unscaled)


def test_estimate_terms_and_text():
    dims = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["mh1"]))
    found = memory.estimate(dims, edges=10_000, nodes=500, dtype="float32", capacity=20_000,
                            rewrite_terms=(("x", memory.MemoryTerm(edge_bytes=8.0)),), resident_weights=True,
                            resident_terms=(("block_linear", 1000),), selfcheck_terms=(("a", 5), ("b", 7)))
    assert found.term("weights (pristine and execution copy)") == 2 * dims.parameters * 4
    assert found.resident == dims.parameters * 4 and found.term("rewrites' buffers") == 1000
    assert found.term("rewrites") == pytest.approx(memory.SAFETY * 80_000, abs=1)
    assert found.term("neighbour list") == 20_000 * (32 + 12 * 4 + 24)
    assert found.term("workspace") == memory.WORKSPACE_BYTES
    assert found.total == found.run == sum(value for _, value in found.terms) > found.build
    assert found.needed == found.total - found.resident
    # The build holds both weight copies, the rewrites' buffers, the workspace and the largest self-check.
    assert dict(found.build_terms) == {"weights (pristine and execution copy)": 2 * dims.parameters * 4,
                                       "rewrites' buffers": 1000, "workspace": memory.WORKSPACE_BYTES,
                                       "self-check b": 7}
    assert "10000 edges" in found.text(2**30) and "GiB free" in found.text(2**30)
    # float64 doubles the activations; a fused convolution keeps neither senders nor messages.
    wide = memory.estimate(dims, edges=10_000, nodes=500, dtype="float64", workspace=False)
    narrow = memory.estimate(dims, edges=10_000, nodes=500, dtype="float32", workspace=False)
    fused = memory.estimate(dims, edges=10_000, nodes=500, dtype="float32", workspace=False, fused_conv=True)
    assert wide.term("activations") == pytest.approx(2 * narrow.term("activations"), rel=1e-6)
    assert fused.term("activations") < narrow.term("activations")


def test_edges_and_capacity_follow_density_and_cutoff():
    atoms = nacl(repeat=3)
    volume = atoms.get_volume()
    edges = memory.model_edges(len(atoms), volume, 5.0, 0.05)
    assert memory.model_edges(len(atoms), volume, 5.0, 0.0) < edges < memory.model_edges(len(atoms), volume, 6.0, 0.05)
    capacity = memory.list_capacity(len(atoms), volume, 5.3, 1.15, 256)
    assert capacity % 256 == 0 and capacity > edges


def test_rule_12_counts_the_edges_in_the_structure():
    """docs/validation/estimate-refit.md: the uniform-density count is 1.2-1.4x a crystal's; rule 12 counts the pairs
    within r_max (both directions, periodic) and adds the edge bucket."""
    from ase.neighborlist import neighbor_list

    atoms = nacl(repeat=3)
    pairs = len(neighbor_list("i", atoms, 5.0))
    assert memory.counted_edges(atoms, 5.0, 0.05) == math.ceil(pairs * 1.05)
    assert memory.counted_edges(atoms, 5.0, 0.0) == pairs
    expected = memory.counted_edges(nacl(repeat=3), mh1_traits().r_max, 0.05)
    assert f"at {expected} edges" in str(plan().decision("memory")[2])


def test_rescue_terms_keep_a_fraction_of_the_remaining_activations():
    """docs/validation/estimate-refit.md: a memory-rescue record's term is the fraction of the remaining activations
    (SAFETY times the activations and the additive terms) it keeps, so the estimate's overstatement shrinks with the
    peak; its reported bytes are what it removes from the running remainder."""
    dims = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["mh1"]))
    base = memory.estimate(dims, edges=10_000, nodes=500, dtype="float32", workspace=False,
                           rewrite_terms=(("a", memory.MemoryTerm(edge_bytes=-8.0)),))
    half = memory.estimate(dims, edges=10_000, nodes=500, dtype="float32", workspace=False,
                           rewrite_terms=(("a", memory.MemoryTerm(edge_bytes=-8.0)),
                                          ("k", memory.MemoryTerm(keep=0.5)), ("q", memory.MemoryTerm(keep=0.5))))
    remaining = base.term("activations") + base.term("rewrites")
    kept = half.term("activations") + half.term("rewrites")
    assert kept == pytest.approx(remaining / 4, rel=1e-6)
    saved = dict(half.rewrite_terms)
    assert saved["k"] == pytest.approx(-remaining / 2, rel=1e-6) and saved["q"] == pytest.approx(-remaining / 4,
                                                                                                    rel=1e-6)
    assert half.total < base.total


# --------------------------------------------------------------------------- the rewrites' terms


def registry(*families):
    out = reg.Registry()
    for family in families:
        for record in family.RECORDS.values():
            out.register(record)
    return out


def rewritten(model, batch, names, families):
    regs = registry(*families)
    copied = copy.deepcopy(model)
    requested = {name: "on" if name in names else "off" for name in regs.names()}
    resolution = regs.resolve(copied, requested, strict=False)
    applied = regs.apply(copied, resolution, reg.Context(inputs=reg.structure_inputs(batch)), original=model,
                         check=False)
    assert set(names) <= {entry.name for entry in applied}, (names, applied)
    return copied


def per_edge(family, dtype, names, families):
    """The peak's change per edge (two sizes), bytes."""
    points = []
    for repeat in (3, 4):
        model, batch = tiny(family, dtype, repeat, z_values=(8, 11, 17))
        points.append((edges_nodes(batch)[0], peak(rewritten(model, batch, names, families), batch)))
    (e3, p3), (e4, p4) = points
    return (p4 - p3) / (e4 - e3)


MH1_STEPS = [  # (record, the records before it): each step's term against the measured change
    ("factor_species", ()),
    ("fold_cutoff", ("factor_species",)),
    ("joint_radial_input", ("factor_species", "fold_cutoff")),
    ("reassociate_first", ("factor_species", "fold_cutoff", "joint_radial_input")),
    ("fused_radial_hidden", ("factor_species", "fold_cutoff", "joint_radial_input")),
]


@pytest.mark.rule(12)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("name, before", MH1_STEPS, ids=[step[0] for step in MH1_STEPS])
def test_mh1_terms_match_the_measured_change(name, before, dtype):
    width = memory.DTYPE_BYTES[dtype]
    measured = (per_edge("mh1", dtype, (*before, name), (mh1, linear))
                - per_edge("mh1", dtype, before, (mh1, linear)))
    dims = capabilities.model_dims(tiny_mace_model([8, 11, 17], **FAMILIES["mh1"]))
    declared = mh1.RECORDS[name].memory(dims, width)[0]
    assert abs(declared - measured) <= max(0.25 * abs(measured), 16 * width), (name, declared, measured)


@pytest.mark.rule(12)
def test_generic_reassociation_term_matches_the_measured_change():
    """The reassociation with its segmented sum (the CUDA float32 path; on the CPU the pair runs as one)."""
    pair = ("generic_reassociate", "generic_segmented_first_sum")
    measured = per_edge("residual", "float64", pair, (generic,)) - per_edge("residual", "float64", (), (generic,))
    dims = capabilities.model_dims(tiny_mace_model([8, 11, 17], **FAMILIES["residual"]))
    declared = generic.RECORDS["generic_reassociate"]["memory"](dims, 8)[0]
    assert abs(declared - measured) <= max(0.25 * abs(measured), 16 * 8), (declared, measured)


def test_declared_terms():
    """Which records declare a term: the edge-sized changes; every term is (bytes per edge, bytes per node)."""
    declared = {name for family in (mh1, generic) for name, record in family.RECORDS.items()
                if (record["memory"] if isinstance(record, dict) else record.memory) is not None}
    assert fusion.RECORDS["segment_gemm"].selfcheck_memory is not None
    assert declared == {"factor_species", "fold_cutoff", "joint_radial_input", "fused_radial_hidden",
                        "reassociate_first", "streaming_conv", "chunked_conv_backward", "generic_reassociate",
                        "generic_streaming"}
    assert radial.RECORDS["radial_tables"].memory is terms.radial_tables
    assert radial.RECORDS["recompute_conv_weights"].memory is terms.recompute_conv_weights
    dims = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["mh1"]))
    assert terms.streaming_conv(dims, 4) == (8, 0)
    # Decision 67: MH-1's fused path keeps less per edge than the stock structure, so its terms are negative and
    # rule 12 never turns it off for memory (decision 51 drops only positive per-edge terms).
    for width in (4, 8):
        for term in (terms.mh1_streaming_conv, terms.chunked_conv_backward, terms.radial_tables):
            edge, node = term(dims, width)
            assert edge < 0 and node == 0, (term.__name__, width)
    assert mh1.RECORDS["streaming_conv"].memory is terms.mh1_streaming_conv
    with pytest.raises(ValueError, match="memory is not callable"):
        dataclasses.replace(mh1.RECORDS["fold_cutoff"], memory=3)
    assert {name for name, record in linear.RECORDS.items() if record.resident_memory is not None} == {
        "block_linear", "merged_residual"}


@pytest.mark.rule(12)
def test_block_linear_resident_term_matches_its_buffers():
    """block_linear keeps each converted linear's weight and its transpose: the buffers of the rewritten tiny
    model against the term from the pristine model's ``linear_weights``."""
    pytest.importorskip("cuequivariance_torch")
    from test_rewrites_mh1 import apply as apply_rewrites
    from test_rewrites_mh1 import only
    from test_rewrites_mh1 import tiny as tiny_cueq

    pristine = tiny_mace_model([8, 11, 17, 26], **FAMILIES["mh1"])  # test_rewrites_mh1.Z_VALUES
    model, batch = tiny_cueq("mh1", backend="cueq")
    rewritten, _, _ = apply_rewrites(model, batch, only("block_linear"), families=(linear,))
    held = sum(buffer.numel() * buffer.element_size() for name, buffer in rewritten.named_buffers()
               if name.rsplit(".", 1)[-1].startswith("weight_"))
    dims = capabilities.model_dims(pristine)
    assert dims.linear_weights > 0 and held == terms.block_linear_resident(dims, 8)


@pytest.mark.rule(12)
@pytest.mark.parametrize("sms", [10, 40])
def test_segment_selfcheck_term_bounds_its_transient(monkeypatch, sms):
    """segment_gemm's self-check probes ``2 x sms x 64`` atoms (the device's SM count, patched here); it checks
    one interaction at a time, so its peak is one interaction's tensors and its declared term bounds it."""
    pytest.importorskip("cuequivariance_torch")
    import test_rewrites_fusion as trf
    from test_rewrites_mh1 import context

    from fastmace.kernels.triton import tile_rule

    original, model, batch, _, _ = trf._rewritten_segment("on")
    monkeypatch.setattr(tile_rule, "device_sms", lambda device: sms)
    with PeakTracker() as tracker:
        assert fusion._segment_selfcheck(original, model, context(batch)).passed
    dims = capabilities.model_dims(original)
    width = memory.DTYPE_BYTES[str(next(model.parameters()).dtype).removeprefix("torch.")]
    declared = fusion.RECORDS["segment_gemm"].selfcheck_memory(dims, width, sms)
    assert len(dims.interactions) == 2  # two interactions held at once would be about twice the term
    assert tracker.peak <= declared <= 1.25 * tracker.peak, (tracker.peak, declared)


@pytest.mark.rule(12)
def test_the_build_estimate_wins_when_a_selfcheck_dominates():
    """max(build, run): with many SMs segment_gemm's self-check transient exceeds the run, and the plan predicts
    the build; turning the rewrite off (explicitly here) takes its transient with it."""
    many = dataclasses.replace(tmp.GPU, sms=2000)
    planner = tmp.minimal()
    planned = planner.plan_traits(mh1_traits(), nacl(repeat=3), IntegratorConfig(), WORKLOAD, many,
                                  precision="float32")
    assert planned.rewrites.segment_gemm == "on"
    reason = str(planned.decision("memory")[2])
    assert "(build;" in reason and "self-check segment_gemm" in reason
    without = planner.plan_traits(mh1_traits(), nacl(repeat=3), IntegratorConfig(), WORKLOAD, many,
                                  precision="float32", rewrites=RewriteConfig(segment_gemm="off"))
    assert "(run;" in str(without.decision("memory")[2])
    assert without.predicted_memory_bytes < planned.predicted_memory_bytes


@pytest.mark.rule(5)
@pytest.mark.rule(12)
@pytest.mark.precedence(5, 12)
def test_rule_5_wins_over_the_models_rule_12(monkeypatch):
    """A reproducible plan that does not fit: rule 12 may turn rewrites off and warns, but rule 5's backend,
    convolution and qualification decisions stay as they are (tests/cpu/test_rules_precedence.py's pair for
    the MACE model's estimate)."""
    import test_rules_precedence as trp

    from fastmace.planner import minimal as minimal_module

    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: False)
    traits = dataclasses.replace(tmp.tp.traits("mh1-r2scan"), dims=mh1_traits().dims, r_max=mh1_traits().r_max)
    workload = Workload(max_temperature_K=300.0, reproducible=True)

    def planned(memory_bytes):
        planner = tmp.minimal(versions=trp.PINNED)
        planner._free_memory_bytes = lambda hardware: hardware.memory_bytes
        return planner.plan_traits(traits, nacl(), IntegratorConfig(), workload,
                                   dataclasses.replace(tmp.GPU, memory_bytes=int(memory_bytes)), precision="float64")

    base = planned(tmp.GPU.memory_bytes)
    assert base.decision("memory")[1] == "fits"
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = planned(2**20)
    assert tight.decision("memory")[1] == "warn"
    assert trp.rule_5_decisions(tight) == trp.rule_5_decisions(base)
    assert tight.engine.backend == base.engine.backend and tight.precision == base.precision


@pytest.mark.rule(12)
def test_plan_text_memory_line():
    """str(plan)'s rule 12 line: the prediction, build or run, the decision and the rewrites it turned off."""
    full = plan()
    line = next(line for line in full.describe().splitlines() if "memory (rule 12)" in line)
    assert f"{full.predicted_memory_bytes / 2**30:.2f} GiB predicted (run; run" in line
    assert ", fits (" in line and "GiB free)" in line and len(line) <= 120
    without = plan(rewrites=RewriteConfig(reassociate_first="off"))
    free = (full.predicted_memory_bytes + without.predicted_memory_bytes) / 2 / memory.MARGIN
    tight = next(line for line in plan(memory_bytes=free).describe().splitlines() if "memory (rule 12)" in line)
    assert "turned off reassociate_first" in tight and len(tight) <= 120


@pytest.mark.rule(12)
@pytest.mark.rule(13)
def test_a_rule_13_revision_re_estimates_memory():
    """Rule 13 turns a memory-saving rewrite off without a re-plan; rule 12 runs again for the revision
    (``fallback.memory_revised``, as a rewrite candidate and the rule 13 path call it), and the prediction rises
    by what the rewrite saved instead of keeping the old, too low one."""
    from types import SimpleNamespace

    from fastmace.planner import fallback, reactions

    planned = plan()
    assert planned.rewrites.joint_radial_input == "on"
    failure = SimpleNamespace(rewrite="joint_radial_input", disabled=("joint_radial_input",),
                              fields=("joint_radial_input",), reason="relative error 2e-4 > 1e-4")
    revised = reactions.selfcheck_revision(planned, failure)
    assert revised.rewrites.joint_radial_input == "off"
    assert revised.predicted_memory_bytes == planned.predicted_memory_bytes  # the revision alone keeps the old one
    planner = tmp.minimal()
    planner._free_memory_bytes = lambda hardware: hardware.memory_bytes
    planner.hardware = lambda: tmp.GPU
    again = fallback.memory_revised(revised, SimpleNamespace(traits=mh1_traits()), nacl(repeat=3), planner)
    assert again.predicted_memory_bytes > planned.predicted_memory_bytes
    assert again.decision("memory")[2].code == "rule-12-memory" and again.decision("memory")[1] == "fits"
    assert [entry[0] for entry in again.decisions].count("memory") == 1
    assert again.decision("RewriteConfig.joint_radial_input") == revised.decision("RewriteConfig.joint_radial_input")
    # Without an estimate (stand-in traits) nothing changes.
    stand_in = tmp.plan_label("mh1-r2scan")
    assert fallback.memory_revised(stand_in, SimpleNamespace(traits=None), nacl()) is stand_in


def test_case_errors_equal_check_tensors():
    expected, got = torch.ones(3), torch.tensor([1.0, 1.0, 1.0 + 1e-3])
    cases = [("a", expected, expected), ("b", expected, got)]
    errors = reg.CaseErrors()
    for case in cases:
        errors.add(*case)
    assert errors.result() == reg.check_tensors(cases)
    assert reg.CaseErrors().result("float32") == reg.check_tensors([], "float32")


# --------------------------------------------------------------------------- decision 51's order


def test_drop_order_largest_first_never_explicit():
    found = (("a", 100), ("b", 300), ("c", 200), ("d", -50), ("e", 400))
    assert memory.drop_order(found, {"a", "b", "c", "d"}, set()) == ["b", "c", "a"]
    assert memory.drop_order(found, {"a", "b", "c", "d"}, {"b"}) == ["c", "a"]
    assert memory.fits(memory.MemoryEstimate(terms=(("x", 90),), edges=0, nodes=0, capacity=0, dtype="float32"), 100)
    assert not memory.fits(memory.MemoryEstimate(terms=(("x", 91),), edges=0, nodes=0, capacity=0,
                                                 dtype="float32"), 100)
    assert memory.fits(memory.MemoryEstimate(terms=(("x", 10**12),), edges=0, nodes=0, capacity=0,
                                             dtype="float32"), None)


# --------------------------------------------------------------------------- the planner


def mh1_traits():
    return capabilities.model_traits(tiny_mace_model(list(Z), **FAMILIES["mh1"]))


MH1 = "mh1-r2scan"  # decision 66's defaults apply to MH-1 on this file's GPU


def plan(memory_bytes=tmp.GPU.memory_bytes, rewrites="auto", atoms=None, label=None):
    """``label``: a label-table label, so that decision 66's per-model defaults apply (a stand-in loaded model)."""
    planner = tmp.minimal()
    planner._free_memory_bytes = lambda hardware: hardware.memory_bytes  # as on CUDA before initialisation
    hardware = dataclasses.replace(tmp.GPU, memory_bytes=int(memory_bytes))
    loaded = None if label is None else types.SimpleNamespace(label=label, head=None, sha256=None, backend=None)
    return planner.plan_traits(mh1_traits(), atoms if atoms is not None else nacl(repeat=3), IntegratorConfig(),
                               WORKLOAD, hardware, precision="float32", rewrites=rewrites, loaded=loaded)


def check(reason):
    assert rules.check_reason(reason) == (), reason


@pytest.mark.rule(12)
def test_the_plan_predicts_its_memory():
    planned = plan()
    assert planned.predicted_memory_bytes > memory.WORKSPACE_BYTES
    key, choice, reason = planned.decision("memory")
    assert choice == "fits" and (reason.rule, reason.code) == (12, "rule-12-memory")
    check(reason)
    assert "rule 12" in str(reason) and "GiB free" in str(reason)
    assert planned.rewrites.factor_species == "on"


@pytest.mark.rule(12)
def test_stand_in_traits_are_not_estimated():
    planned = tmp.plan_label("mh1-r2scan")
    assert planned.predicted_memory_bytes is None
    assert planned.decision("memory")[1] == "not estimated"
    check(planned.decision("memory")[2])


@pytest.mark.rule(12)
def test_rule_12_turns_off_the_largest_edge_sized_rewrite_and_its_dependants():
    """Decision 51: above 90 % of the free memory the rewrite with the largest edge-sized term goes, with the
    records that require it, until the estimate fits; rule 11's sizes, the tier and the precision stay. Decision
    67: only a group whose removal lowers the estimate goes. factor_species takes MH-1's fused path with it, which
    keeps less per edge, so it stays; reassociate_first (with segmented_first_sum) goes."""
    full = plan()
    without = plan(rewrites=RewriteConfig(reassociate_first="off"))
    assert without.predicted_memory_bytes < full.predicted_memory_bytes
    assert plan(rewrites=RewriteConfig(factor_species="off")).predicted_memory_bytes > full.predicted_memory_bytes
    free = (full.predicted_memory_bytes + without.predicted_memory_bytes) / 2 / memory.MARGIN
    tight = plan(memory_bytes=free)
    assert tight.predicted_memory_bytes <= memory.MARGIN * free < full.predicted_memory_bytes
    assert tight.rewrites.reassociate_first == "off" and tight.rewrites.factor_species == "on"
    for field in ("reassociate_first", "segmented_first_sum"):
        _, choice, reason = tight.decision(f"RewriteConfig.{field}")
        assert choice == "off" and (reason.rule, reason.code) == (12, "rule-12-memory-drop"), field
        check(reason)
    assert "reassociate_first" in str(tight.decision("RewriteConfig.segmented_first_sum")[2])
    assert tight.decision("memory")[1] == "fits"
    for part in ("neighbours", "tier", "precision", "engine"):
        assert getattr(tight, part) == getattr(full, part), part


@pytest.mark.rule(12)
def test_rule_12_warns_when_nothing_more_can_go():
    with pytest.warns(ResourceWarning, match="rule 12"):
        planned = plan(memory_bytes=2**20)
    _, choice, reason = planned.decision("memory")
    assert choice == "warn" and (reason.rule, reason.code) == (12, "rule-12-memory-warn")
    check(reason)
    assert "reassociate_first" in str(reason)  # turned off on the way
    assert planned.rewrites.factor_species == "on"  # decision 67: turning it off would raise the estimate
    assert planned.tier == "graph" and planned.precision == plan().precision


@pytest.mark.rule(12)
def test_rule_12_never_drops_a_group_that_saves_nothing(monkeypatch):
    """Decision 67: a rewrite whose own per-edge term is positive stays on when turning it off with its dependants
    would raise the estimate (here the streaming convolution with its index bytes alone, beside the chunked
    backward's measured saving). factor_species and reassociate_first are set explicitly, so their groups, which
    would save memory here, stay too."""
    monkeypatch.setattr(terms, "_MH1_STREAMING_SHARE", {4: 0.0, 8: 0.0})  # the index bytes alone: +8 per edge
    assert terms.mh1_streaming_conv(capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["mh1"])), 4)[0] == 8
    full = plan()
    if full.rewrites.chunked_conv_backward != "on":
        pytest.skip("the plan does not enable the chunked backward here")
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = plan(memory_bytes=2**20, rewrites=RewriteConfig(factor_species="on", reassociate_first="on"))
    assert tight.rewrites.streaming_conv == "on" and tight.rewrites.chunked_conv_backward == "on"
    assert getattr(tight.decision("RewriteConfig.streaming_conv")[2], "code", None) != "rule-12-memory-drop"


@pytest.mark.rule(12)
def test_rule_12_turns_on_a_memory_rescue_only_when_the_plan_does_not_fit():
    """CAP-T07: recompute_conv_weights ("auto") stays off while the plan fits; when it does not, rule 12 turns it on
    if wprime_backward is on and it lowers the estimate, with its own reason; an explicit "off" is kept."""
    from fastmace.planner.fixed import AUTO_RULES

    assert "recompute_conv_weights" in memory.MEMORY_RESCUE
    assert AUTO_RULES["RewriteConfig.recompute_conv_weights"] == "on_by_rule_12_when_the_plan_does_not_fit_without_it"
    full = plan(label=MH1)
    assert full.rewrites.recompute_conv_weights == "off"
    assert full.rewrites.wprime_backward == "on"
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = plan(label=MH1, memory_bytes=2**20)
    assert tight.rewrites.recompute_conv_weights == "on"
    _, choice, reason = tight.decision("RewriteConfig.recompute_conv_weights")
    assert choice == "on" and (reason.rule, reason.code) == (12, "rule-12-memory-rescue")
    check(reason)
    with pytest.warns(ResourceWarning, match="rule 12"):
        kept = plan(label=MH1, memory_bytes=2**20, rewrites=RewriteConfig(recompute_conv_weights="off"))
    assert kept.rewrites.recompute_conv_weights == "off"


@pytest.mark.rule(12)
def test_rule_12_turns_on_recompute_node_segments_only_after_recompute_conv_weights():
    """CAP-T08: recompute_node_segments ("auto") stays off while the plan fits, and while it fits with
    recompute_conv_weights on; otherwise rule 12 turns it on after recompute_conv_weights, with its own reason."""
    from fastmace.planner.fixed import AUTO_RULES

    assert memory.MEMORY_RESCUE == ("recompute_conv_weights", "recompute_first_table", "recompute_node_segments",
                                    "gate_epilogue", "recompute_node_region")
    assert AUTO_RULES["RewriteConfig.recompute_node_segments"] == (
        "on_by_rule_12_when_the_plan_does_not_fit_with_recompute_conv_weights")
    full = plan(label=MH1)
    assert full.rewrites.recompute_node_segments == "off"
    assert full.rewrites.wprime_backward == full.rewrites.gated_rows == "on"
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = plan(label=MH1, memory_bytes=2**20)  # nothing fits: both on
    assert tight.rewrites.recompute_conv_weights == "on" and tight.rewrites.recompute_node_segments == "on"
    _, choice, reason = tight.decision("RewriteConfig.recompute_node_segments")
    assert choice == "on" and (reason.rule, reason.code) == (12, "rule-12-memory-rescue")
    check(reason)
    assert "recompute_conv_weights" in str(reason)
    recompute = plan(label=MH1, rewrites=RewriteConfig(recompute_conv_weights="on"))
    free = recompute.predicted_memory_bytes / memory.MARGIN * 1.0001  # fits with recompute_conv_weights on
    fits = plan(label=MH1, memory_bytes=free, rewrites=RewriteConfig(recompute_conv_weights="on"))
    assert fits.rewrites.recompute_node_segments == "off"


def test_gate_epilogue_s_term_depends_on_the_node_segments():
    """docs/validation/region-norm.md: gate_epilogue keeps 12.3 kB per atom more without the node segments, 36.3 with
    the separate segments and 0.5 inside recompute_node_region's chunks (memory.CONTEXT_TERMS)."""
    from fastmace.planner.minimal import _rewrite_registry

    registry = _rewrite_registry()
    dims = capabilities.model_dims(tiny_mace_model(list(Z), **FAMILIES["mh1"]))
    assert registry["gate_epilogue"].memory is terms.gate_epilogue
    edge, node, keep = terms.gate_epilogue(dims, 4)
    assert edge == node == 0 and keep > 1
    alone = dict(memory.rewrite_memory_terms(registry, dims, ["gate_epilogue"], "float32"))
    assert alone["gate_epilogue"].keep == keep
    segments = dict(memory.rewrite_memory_terms(registry, dims, ["gate_epilogue", "recompute_node_segments"],
                                                "float32"))
    assert segments["gate_epilogue"].keep == terms.gate_epilogue_in_segments(dims, 4)[2] > keep
    region = dict(memory.rewrite_memory_terms(
        registry, dims, ["gate_epilogue", "recompute_node_segments", "recompute_node_region"], "float32"))
    assert "gate_epilogue" not in region and "recompute_node_region" in region


@pytest.mark.rule(12)
def test_rule_12_turns_gate_epilogue_off_only_after_recompute_node_segments():
    """docs/validation/region-norm.md: gate_epilogue (on by decision 66 on this file's GPU) stays on while the plan
    fits, and while it fits with recompute_node_segments on; otherwise rule 12 turns it off, the memory rescue's
    fourth rung, with its own reason. The caller's explicit "on" is kept."""
    assert memory.RESCUE_OFF == ("gate_epilogue",) and memory.rescue_value("gate_epilogue") == "off"
    full = plan(label=MH1)
    assert full.rewrites.gate_epilogue == full.rewrites.wprime_backward == "on"
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = plan(memory_bytes=2**20, label=MH1)  # nothing fits: every rung taken
    assert tight.rewrites.recompute_node_segments == "on" and tight.rewrites.gate_epilogue == "off"
    _, choice, reason = tight.decision("RewriteConfig.gate_epilogue")
    assert choice == "off" and (reason.rule, reason.code) == (12, "rule-12-memory-rescue")
    check(reason)
    assert "recompute_node_segments on" in str(reason)
    nodes = RewriteConfig(recompute_conv_weights="on", recompute_node_segments="on")
    free = plan(rewrites=nodes, label=MH1).predicted_memory_bytes / memory.MARGIN * 1.0001  # fits with the segments on
    assert plan(memory_bytes=free, rewrites=nodes, label=MH1).rewrites.gate_epilogue == "on"
    with pytest.warns(ResourceWarning, match="rule 12"):
        kept = plan(memory_bytes=2**20, rewrites=RewriteConfig(gate_epilogue="on"), label=MH1)
    assert kept.rewrites.gate_epilogue == "on"


@pytest.mark.rule(12)
def test_rule_12_gives_gate_epilogue_back_with_the_chunked_region(monkeypatch):
    """docs/validation/region-norm.md: when the rescue ends with recompute_node_region on, gate_epilogue keeps 0.5 kB
    per atom inside its chunks, so rule 12 turns it on again where the estimate still fits with it. The tiny model
    saves too little for rule 12 to take the rungs itself (``short_by_far``), so the overstatement is set to 1, and
    reassociate_first is set on, so that decision 51 drops nothing first."""
    monkeypatch.setattr(memory, "RESCUE_OVERSTATEMENT", 1.0)
    atoms = nacl(repeat=16)
    base = RewriteConfig(reassociate_first="on")
    full = plan(rewrites=base, atoms=atoms, label=MH1)
    assert full.rewrites.gate_epilogue == full.rewrites.wprime_backward == "on"
    every = dataclasses.replace(base, recompute_conv_weights="on", recompute_first_table="on",
                                recompute_node_segments="on", recompute_node_region="on")
    region = plan(rewrites=dataclasses.replace(every, gate_epilogue="on"), atoms=atoms, label=MH1)
    segments = plan(rewrites=dataclasses.replace(every, recompute_node_region="off", gate_epilogue="off"), atoms=atoms,
                    label=MH1)
    assert region.predicted_memory_bytes < segments.predicted_memory_bytes
    assert region.predicted_memory_bytes == plan(rewrites=dataclasses.replace(every, gate_epilogue="off"), atoms=atoms,
                                                 label=MH1).predicted_memory_bytes  # absorbed: no term
    free = region.predicted_memory_bytes / memory.MARGIN * 1.0001  # fits only with the region on
    auto = plan(memory_bytes=free, rewrites=base, atoms=atoms, label=MH1)
    assert auto.rewrites.recompute_node_region == "on", auto.decision("memory")
    assert auto.rewrites.gate_epilogue == "on"
    _, choice, reason = auto.decision("RewriteConfig.gate_epilogue")
    assert choice == "on" and (reason.rule, reason.code) == (12, "rule-12-memory-rescue")
    check(reason)
    assert "recompute_node_region" in str(reason)
    assert auto.predicted_memory_bytes <= memory.MARGIN * free
    assert auto.decision("memory")[1] == "fits"


@pytest.mark.rule(12)
def test_rule_12_turns_on_recompute_first_table_only_after_recompute_conv_weights():
    """Rebuild-a: recompute_first_table ("auto") stays off while the plan fits, and while it fits with
    recompute_conv_weights on; otherwise rule 12 turns it on after recompute_conv_weights and before
    recompute_node_segments, with its own reason."""
    from fastmace.planner.fixed import AUTO_RULES

    assert AUTO_RULES["RewriteConfig.recompute_first_table"] == (
        "on_by_rule_12_when_the_plan_does_not_fit_with_recompute_conv_weights")
    full = plan(label=MH1)
    assert full.rewrites.recompute_first_table == "off"
    assert full.rewrites.wprime_backward == full.rewrites.radial_tables == full.rewrites.reassociate_first == "on"
    with pytest.warns(ResourceWarning, match="rule 12"):
        # nothing fits: every rescue on; reassociate_first set on, so that decision 51 does not drop the table's
        # requirement first
        tight = plan(label=MH1, memory_bytes=2**20, rewrites=RewriteConfig(reassociate_first="on"))
    assert tight.rewrites.recompute_conv_weights == "on" and tight.rewrites.recompute_first_table == "on"
    _, choice, reason = tight.decision("RewriteConfig.recompute_first_table")
    assert choice == "on" and (reason.rule, reason.code) == (12, "rule-12-memory-rescue")
    check(reason)
    assert "recompute_conv_weights" in str(reason)
    with pytest.warns(ResourceWarning, match="rule 12"):
        kept = plan(label=MH1, memory_bytes=2**20, rewrites=RewriteConfig(recompute_first_table="off"))
    assert kept.rewrites.recompute_first_table == "off"
    recompute = plan(label=MH1, rewrites=RewriteConfig(recompute_conv_weights="on"))
    free = recompute.predicted_memory_bytes / memory.MARGIN * 1.0001  # fits with recompute_conv_weights on
    fits = plan(label=MH1, memory_bytes=free, rewrites=RewriteConfig(recompute_conv_weights="on"))
    assert fits.rewrites.recompute_first_table == "off"


@pytest.mark.rule(12)
def test_rule_12_leaves_the_rescue_to_the_run_when_the_cell_may_fit():
    """CAP-T07 (owner: only when memory would not be enough): the estimate overstates the peak by up to
    RESCUE_OVERSTATEMENT, so a plan that does not fit 90 % of the free memory, but whose estimate divided by it
    does, keeps recompute_conv_weights off; the run rebuilds with it after a real out-of-memory error."""
    full = plan(label=MH1)
    assert full.rewrites.wprime_backward == "on"
    pinned = RewriteConfig(**{f.name: "on" for f in dataclasses.fields(RewriteConfig)
                              if getattr(full.rewrites, f.name) == "on"})  # explicit: decision 51 drops none
    free = plan(label=MH1, rewrites=pinned).predicted_memory_bytes  # above the estimate / 1.34, below / 0.9
    with pytest.warns(ResourceWarning, match="rule 12"):
        tight = plan(label=MH1, memory_bytes=free, rewrites=pinned)
    assert tight.decision("memory")[1] == "warn"
    assert tight.rewrites.recompute_conv_weights == "off"
    _, choice, reason = tight.decision("RewriteConfig.recompute_conv_weights")
    assert choice == "off" and (reason.rule, reason.code) == (12, "rule-12-memory-rescue")
    assert "rebuilds with recompute_conv_weights on, one memory-rescue record per rebuild" in reason
    check(reason)
    bound = type("Estimate", (), {})
    bound.needed = free
    assert not memory.short_by_far(bound, free)
    bound.needed = 1.35 * free
    assert memory.short_by_far(bound, free)


@pytest.mark.rule(12)
def test_rule_12_keeps_an_explicit_on_and_warns():
    with pytest.warns(ResourceWarning, match="kept as set: factor_species"):
        planned = plan(memory_bytes=2**20, rewrites=RewriteConfig(factor_species="on"))
    assert planned.rewrites.factor_species == "on"
    assert getattr(planned.decision("RewriteConfig.factor_species")[2], "code", None) != "rule-12-memory-drop"
    assert planned.decision("memory")[1] == "warn"


@pytest.mark.rule(12)
def test_out_of_memory_carries_the_estimate():
    planned = plan()

    class OutOfMemoryError(RuntimeError):
        pass

    error = memory.annotate_out_of_memory(OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"), planned)
    assert str(error).startswith("CUDA out of memory") and "rule 12" in str(error) and "predicted" in str(error)
    assert str(memory.annotate_out_of_memory(error, planned)).count("rule 12): memory") == 1
    other = ValueError("x")
    assert memory.annotate_out_of_memory(other, planned) is other and str(other) == "x"
