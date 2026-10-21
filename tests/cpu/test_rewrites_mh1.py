"""The MH-1 family (SPEC §6.2; plan task 1a-T07): records, self-checks and the stacked package.

The tiny-model tests follow the research engine's ``FactorSpeciesRadialTests``
(``tests/core/test_gpu_langevin.py`` at ResearchModule tag research-engine-baseline-20261017):
NaCl 2x2x2 with one atom made oxygen, on models with an absent fourth species, eager, float64.
They run on the e3nn models and on their cuEquivariance conversions (built with cuEq's CPU
``naive`` method, as ``tests/cpu/test_rewrites_linear.py`` does), each rewrite through the
registry with its self-check at the SPEC §6.4 threshold, and then the whole MH-1 package
followed by the linear family. The baseline tests run the research engine's own rewrites from
the export named by ``FASTMACE_BASELINE`` (skipped when it is unset) and compare bitwise.
"""

import contextlib
import copy
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from packaging.specifiers import SpecifierSet

from fastmace.rewrites import linear, mh1
from fastmace.rewrites import registry as reg
from fastmace.rewrites.mh1 import modules as m

pytest.importorskip("mace")

Z_VALUES = [8, 11, 17, 26]  # Fe absent from the structure
MH1_ORDER = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
             "folded_backward_tiles", "fused_radial_hidden", "direct_node_feats", "folded_constants",
             "streaming_conv", "reassociate_first", "segmented_first_sum")
# The research engine's default MH-1 package on the CPU (no OEq: no streaming; fused radial hidden is opt-in).
PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
           "reassociate_first", "segmented_first_sum")


def registry(*families):
    """A registry of the given families in the research engine's order (MH-1 before the linear family)."""
    out = reg.Registry()
    for family in families or (mh1, linear):
        for record in family.RECORDS.values():
            out.register(record)
    return out


# --------------------------------------------------------------------------- tiny models


def structure():
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=0.1, seed=8)
    atoms.numbers[3] = 8  # three species, one of them once
    return atoms


def tiny(family="mh1", backend="e3nn", z_values=Z_VALUES, **options):
    """(model, batch dict) of a tiny model of ``family`` on the CPU; cuEq conversions use the naive kernel."""
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    calc = tiny_mace_calculator(z_values, **{**FAMILIES[family], **options})
    batch = calc._atoms_to_batch(structure()).to_dict()
    model = calc.models[0]
    if backend == "cueq":
        from cuequivariance_torch.primitives import segmented_polynomial
        from mace.cli.convert_e3nn_cueq import run

        with mock.patch.object(segmented_polynomial, "HAS_CUE_OPS", False):
            model = run(model, device="cpu", return_model=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, batch


def energy_and_forces(model, batch):
    out = model({key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()},
                training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()


def relative(got, expected):
    return float((got - expected).abs().max()) / float(expected.abs().max())


def context(batch):
    return reg.Context(inputs={"node_attrs": batch["node_attrs"]})


def apply(model, batch, requested, families=None):
    """Resolve and apply ``requested`` through the registry on a copy of ``model``; (copy, applied, resolution)."""
    regs = registry(*(families or (mh1, linear)))
    rewritten = copy.deepcopy(model)
    requested = {name: value for name, value in requested.items() if name in regs}
    resolution = regs.resolve(rewritten, requested, strict=False)
    applied = regs.apply(rewritten, resolution, context(batch), original=model)
    return rewritten, applied, resolution


def only(*names):
    """Requests: ``names`` on, every other MH-1 and linear record off."""
    everything = (*MH1_ORDER, *linear.RECORDS)
    return {name: "on" if name in names else "off" for name in everything}


BACKENDS = ["e3nn", pytest.param("cueq", marks=pytest.mark.skipif(
    importlib.util.find_spec("cuequivariance_torch") is None, reason="cuequivariance_torch is not installed"))]


# --------------------------------------------------------------------------- records and graph


def test_records_have_the_spec_fields():
    assert tuple(mh1.RECORDS) == MH1_ORDER
    for name, rewrite in mh1.RECORDS.items():
        assert isinstance(rewrite, reg.Rewrite) and rewrite.name == name and rewrite.conflicts == ()
        assert rewrite.mace_versions == SpecifierSet(">=0.3.16,<0.3.17")
    # Summation order only (SPEC §6.2); folded_backward_tiles keeps the order: bitwise (opt-2 OPT2-T03).
    # direct_node_feats and folded_constants (opt-3 OPT3-T05) move no value: bitwise.
    exact = {"streaming_conv", "segmented_first_sum", "folded_backward_tiles", "direct_node_feats", "folded_constants"}
    assert {name for name, r in mh1.RECORDS.items() if r.exactness == "exact"} == exact
    assert all(r.exactness == "rounding" for name, r in mh1.RECORDS.items() if name not in exact)
    assert {name for name, r in mh1.RECORDS.items() if r.default != "auto"} == {
        "fused_radial_hidden", "folded_backward_tiles", "direct_node_feats", "folded_constants"}
    assert mh1.RECORDS["fused_radial_hidden"].default == "off"  # opt-in, no gain (SPEC §6.2)
    pinned = {name for name, r in mh1.RECORDS.items() if r.source_hashes}
    assert pinned == {"factor_species", "fold_cutoff"}
    assert mh1.RECORDS["fold_cutoff"].source_hashes == {m_forward(): mh1.NONLINEAR_FORWARD_SHA256}
    assert mh1.NONLINEAR_FORWARD_SHA256 == "727c37b796de7d3b33793ae6317a6d09c1f4c8677b28c04963ffbb34899b0b44"
    assert set(mh1.RECORDS["reassociate_first"].workload_constraints) == {"single_graph", "fixed_species_superset"}
    assert set(mh1.RECORDS["streaming_conv"].workload_constraints) == {"oeq", "sort_receivers",
                                                                       "per_engine_edge_counter"}
    assert "sort_receivers" in mh1.RECORDS["segmented_first_sum"].workload_constraints


def m_forward():
    return "mace.modules.blocks:RealAgnosticResidualNonLinearInteractionBlock.forward"


def test_requires_graph_matches_the_spec():
    requires = {name: r.requires for name, r in mh1.RECORDS.items()}
    assert requires == {
        "factor_species": (), "fold_cutoff": ("factor_species",), "density_first": ("fold_cutoff",),
        "joint_radial_input": ("fold_cutoff",), "fused_folded_backward": ("fold_cutoff",),
        "folded_backward_tiles": ("fused_folded_backward",), "fused_radial_hidden": ("fold_cutoff",),
        "direct_node_feats": ("fold_cutoff",), "folded_constants": ("fold_cutoff",),
        "streaming_conv": ("fold_cutoff",),
        "reassociate_first": ("fold_cutoff",), "segmented_first_sum": ("reassociate_first",)}
    regs = registry()
    assert regs.order() == (*MH1_ORDER, "block_linear", "grouped_linear", "merged_residual")
    for name in MH1_ORDER:
        assert regs.config_field(name) == name


def test_installed_sources_qualify():
    for name in ("factor_species", "fold_cutoff"):
        assert registry(mh1).qualification(name).supported


def test_resolution_follows_the_requirements():
    model, _ = tiny("mh1")
    regs = registry()
    resolution = regs.resolve(model, {"fold_cutoff": "off"}, strict=False)
    for name in ("density_first", "joint_radial_input", "fused_folded_backward", "reassociate_first"):
        assert resolution.decision(name).reason == "requires fold_cutoff, which is off"
    assert resolution.decision("segmented_first_sum").reason == "requires reassociate_first, which is off"
    if importlib.util.find_spec("cuequivariance_torch"):  # cuEq: merged_residual needs density_first
        resolution = regs.resolve(tiny("mh1", "cueq")[0], {"density_first": "off"}, strict=False)
        assert resolution.decision("merged_residual").reason == "requires density_first, which is off"
        assert resolution.decision("block_linear").enabled
    with pytest.raises(reg.RewriteRefusedError, match="reassociate_first: requires fold_cutoff"):
        regs.resolve(model, {"fold_cutoff": "off", "reassociate_first": "on"})
    assert not regs.resolve(model).decision("fused_radial_hidden").enabled  # default off
    assert regs.resolve(model, {"fused_radial_hidden": "on"}).decision("fused_radial_hidden").enabled


@pytest.mark.parametrize("family, options, factor, fold, reassociate", [
    ("mh1", {}, True, True, True),
    ("omol", {}, True, True, True),  # per-graph joint embeddings keep the first inputs species-only
    ("mh1", {"apply_cutoff": True}, True, False, False),
    ("residual", {}, False, False, False),
    ("density", {}, False, False, False),
])
def test_applies_to_follows_the_model(family, options, factor, fold, reassociate):
    model, _ = tiny(family, **options)
    verdicts = {name: r.applies_to(model) for name, r in mh1.RECORDS.items()}
    assert verdicts["factor_species"].supported is factor
    for name in ("fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
                 "fused_radial_hidden"):
        assert verdicts[name].supported is fold, (name, verdicts[name])
    assert verdicts["reassociate_first"].supported is reassociate
    assert verdicts["segmented_first_sum"].supported is reassociate
    assert not verdicts["streaming_conv"].supported  # no OEq convolution on the CPU
    if not fold:
        assert "apply_cutoff" in verdicts["fold_cutoff"].reason or "are not" in verdicts["fold_cutoff"].reason


def test_density_first_refuses_a_biased_linear_1():
    model, batch = tiny("mh1")
    interaction = model.interactions[0]
    del interaction.linear_1.bias  # e3nn's empty placeholder
    interaction.linear_1.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))
    supported, reason = mh1.RECORDS["density_first"].applies_to(model)
    assert not supported and "bias-free" in reason
    mh1.factor_species_radial(model)
    mh1.fold_radial_cutoff(model)
    with pytest.raises(reg.RewriteError, match="bias-free"):
        mh1.density_before_linear(model)


def test_fold_refuses_a_changed_interaction_forward():
    model, _ = tiny("mh1")
    mh1.factor_species_radial(model)
    with mock.patch.object(mh1, "NONLINEAR_FORWARD_SHA256", "0" * 64), pytest.raises(reg.RewriteError,
                                                                                       match="0.3.16"):
        mh1.fold_radial_cutoff(model)
    with mock.patch.object(reg, "source_sha256", return_value="0" * 64):
        qualified = registry(mh1).qualification("fold_cutoff")
    assert not qualified.supported and "differs" in qualified.reason


def test_fold_needs_factor_species_and_reassociation_needs_the_fold():
    model, batch = tiny("mh1")
    with pytest.raises(reg.RewriteError, match="factor_species first"):
        mh1.fold_radial_cutoff(model)
    mh1.factor_species_radial(model)
    with pytest.raises(reg.RewriteError, match="folded"):
        mh1.reassociate_first_interaction(model, batch["node_attrs"])
    mh1.fold_radial_cutoff(model)
    with pytest.raises(reg.RewriteError, match="node_attrs"):
        mh1.RECORDS["reassociate_first"].apply(model, reg.Context())


def test_factor_species_replaces_both_networks_of_every_interaction():
    model, batch = tiny("mh1")
    rewritten, applied, _ = apply(model, batch, only("factor_species"))
    assert applied[0].detail == ["interactions.0", "interactions.1"]
    for interaction in rewritten.interactions:
        for net in (interaction.conv_tp_weights, interaction.density_fn):
            assert isinstance(net, m.FactoredRadialMLP)
            assert net.weight.shape[1] == interaction.edge_feats_irreps.num_irreps
        assert isinstance(interaction.source_embedding, m.ProjectedEmbedding)
    for interaction in model.interactions:  # the original is untouched
        assert type(interaction.conv_tp_weights).__name__ == "RadialMLP"


def test_reassociation_slots_are_the_structure_species():
    model, batch = tiny("mh1")
    rewritten, applied, _ = apply(model, batch, only("factor_species", "fold_cutoff", "reassociate_first"))
    reassociation = rewritten.interactions[0].first_reassociation
    assert applied[-1].detail == (0, 1, 2) and reassociation.columns.tolist() == [0, 1, 2]
    assert reassociation.ls == (0, 1, 2) and not reassociation.segmented
    assert not hasattr(rewritten.interactions[1], "first_reassociation")


def test_joint_fused_and_segmented_attributes():
    model, batch = tiny("mh1")
    rewritten, _, _ = apply(model, batch, only(*PACKAGE, "fused_radial_hidden"))
    for interaction in rewritten.interactions:
        assert interaction.forward.__func__ is m.cutoff_folded_forward and interaction.density_before_linear
        assert interaction.joint_radial_widths == [interaction.conv_tp_weights.weight.shape[0],
                                                   interaction.density_fn.weight.shape[0]]
        assert interaction.conv_tp_weights.fused_backward and not interaction.density_fn.fused_backward
        # MACE's default 64-64-64 radial network (as in MH-1): one fused block, then Linear, LN, SiLU.
        plan = interaction.conv_tp_weights.fused_hidden
        assert [kind for kind, _ in plan] == ["fused", "module", "module", "module"]
    assert rewritten.interactions[0].first_reassociation.segmented


# --------------------------------------------------------------------------- self-checks


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("family", ["mh1", "omol"])
@pytest.mark.parametrize("name", [*PACKAGE, "fused_radial_hidden"])
def test_each_rewrite_passes_its_selfcheck_and_keeps_the_model(family, backend, name):
    """Each record with what it requires, through the registry, at the SPEC §6.4 float64 threshold."""
    model, batch = tiny(family, backend)
    wanted = {name}
    while True:
        more = wanted | {r for n in wanted for r in mh1.RECORDS[n].requires}
        if more == wanted:
            break
        wanted = more
    rewritten, applied, _ = apply(model, batch, only(*wanted))
    assert [entry.name for entry in applied] == [n for n in MH1_ORDER if n in wanted]
    for entry in applied:
        assert entry.result.passed and entry.result.modules == 2, (entry.name, entry.result)
        assert entry.result.threshold == 1e-10 and entry.result.max_relative_error <= 1e-10
    energy_and_forces(model, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= 1e-10 and relative(new_forces, forces) <= 1e-10
    # the research engine's tolerances (FactorSpeciesRadialTests, eager)
    torch.testing.assert_close(new_forces, forces, rtol=0, atol=1e-12)
    assert abs(float(new_energy - energy)) < 1e-10


@pytest.mark.parametrize("name, perturb", [
    ("factor_species", lambda i: i.conv_tp_weights.weight),
    ("fold_cutoff", lambda i: i.conv_tp_weights.rest[-1].bias),
    ("joint_radial_input", lambda i: i.joint_radial_weight),
    ("reassociate_first", None),
])
def test_selfcheck_catches_a_wrong_rewrite(name, perturb):
    model, batch = tiny("mh1")
    requested = {"factor_species", "fold_cutoff", name}
    rewritten, _, _ = apply(model, batch, only(*requested))
    interaction = rewritten.interactions[0]
    if perturb is None:
        interaction.first_reassociation.gammas = tuple(g * (1 + 1e-8) for g in interaction.first_reassociation.gammas)
    else:
        with torch.no_grad():
            perturb(interaction).mul_(1 + 1e-8)
    result = registry(mh1).selfcheck(name, model, rewritten, context(batch))
    assert not result.passed and "interactions.0" in result.detail


# --------------------------------------------------------------------------- the stacked package


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("family", ["mh1", "omol"])
def test_stacked_package_agrees_with_the_original(family, backend):
    """The research engine's MH-1 package and then the linear family, all checked, as one registry resolves it."""
    model, batch = tiny(family, backend)
    rewritten, applied, resolution = apply(model, batch, {"block_linear": "auto"})
    linears = ("block_linear", "grouped_linear", "merged_residual") if backend == "cueq" else ()
    assert resolution.enabled == (*PACKAGE, *linears)
    assert resolution.decision("streaming_conv").reason.startswith("no factored interaction with an OEq")
    assert [entry.name for entry in applied] == [*PACKAGE, *linears]
    for entry in applied:
        assert entry.result.passed and entry.result.max_relative_error <= 1e-10, (entry.name, entry.result)
    if backend == "cueq":
        assert all(isinstance(i.residual_linear, linear.modules.ResidualBlockLinear) for i in rewritten.interactions)
    energy_and_forces(model, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= 1e-10 and relative(new_forces, forces) <= 1e-10
    torch.testing.assert_close(new_forces, forces, rtol=0, atol=1e-12)


def test_stacked_model_survives_a_deep_copy():
    model, batch = tiny("mh1")
    rewritten, _, _ = apply(model, batch, only(*PACKAGE, "fused_radial_hidden"))
    duplicate = copy.deepcopy(rewritten)
    assert duplicate.interactions[0].forward.__self__ is duplicate.interactions[0]
    plan = duplicate.interactions[0].conv_tp_weights.fused_hidden
    assert plan[0][1][1] is duplicate.interactions[0].conv_tp_weights.rest[2]
    energy_and_forces(rewritten, batch)
    for a, b in zip(energy_and_forces(rewritten, batch), energy_and_forces(duplicate, batch)):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- baseline (bitwise)


def _baseline(name):
    root = os.environ.get("FASTMACE_BASELINE")
    if not root:
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    directory = Path(root).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin"
    assert directory.is_dir(), f"FASTMACE_BASELINE={root} has no Experiments/Langevin_Thermostat/langevin"
    if "langevin" not in sys.modules:
        package = types.ModuleType("langevin")
        package.__path__ = [str(directory)]
        sys.modules["langevin"] = package
    return importlib.import_module(f"langevin.{name}")


BASELINE_FLAGS = {"factor_species": None, "fold_cutoff": "FOLD_RADIAL_CUTOFF", "density_first": "DENSITY_BEFORE_LINEAR",
                  "joint_radial_input": "JOINT_RADIAL_INPUT", "fused_folded_backward": "FUSED_FOLDED_BACKWARD",
                  "fused_radial_hidden": "FUSED_RADIAL_HIDDEN", "reassociate_first": "REASSOCIATE_FIRST_INTERACTION",
                  "segmented_first_sum": "SEGMENTED_FIRST_SUM"}


@pytest.mark.parametrize("names", [("factor_species",), ("factor_species", "fold_cutoff"), PACKAGE,
                                   (*PACKAGE, "fused_radial_hidden")])
@pytest.mark.parametrize("backend", BACKENDS)
def test_package_equals_the_baseline_bitwise(names, backend):
    gl = _baseline("gpu_langevin")
    model, batch = tiny("mh1", backend)
    if backend == "cueq" and "reassociate_first" in names:
        # cuEq's CPU (naive) convolution: path coefficients equal to 1e-15 only. The baseline's probe
        # demands them exactly equal and refuses; FastMACE's compares at the probe's tolerance.
        theirs = copy.deepcopy(model)
        with mock.patch.multiple(gl, FOLD_RADIAL_CUTOFF=True):
            gl.factor_species_radial(theirs)
        with pytest.raises(ValueError, match="differ within a path"):
            gl.reassociate_first_interaction(theirs, batch["node_attrs"])
        return
    ours, _, _ = apply(model, batch, only(*names), families=(mh1,))
    theirs = copy.deepcopy(model)
    flags = {flag: True for name, flag in BASELINE_FLAGS.items() if flag and name in names}
    with mock.patch.multiple(gl, **flags) if flags else contextlib.nullcontext():
        assert gl.factor_species_radial(theirs) == 2
        if "reassociate_first" in names:
            assert gl.reassociate_first_interaction(theirs, batch["node_attrs"]) == (0, 1, 2)
    old_params = dict(theirs.named_parameters())
    for name, tensor in ours.named_parameters():
        assert torch.equal(tensor, old_params[name]), name
    energy_and_forces(ours, batch)  # throwaway evaluation (docs/development.md)
    for a, b in zip(energy_and_forces(ours, batch), energy_and_forces(theirs, batch)):
        assert torch.equal(a, b)


@pytest.mark.skipif(importlib.util.find_spec("cuequivariance_torch") is None, reason="cuequivariance_torch is missing")
def test_block_linear_check_reaches_the_wrapped_species_embeddings():
    """block_linear after factor_species: the embeddings inside ProjectedEmbedding are checked against the originals."""
    model, batch = tiny("mh1", "cueq")
    rewritten, applied, _ = apply(model, batch, only("factor_species", "block_linear"))
    embedding = rewritten.interactions[0].source_embedding.embedding
    assert isinstance(embedding, linear.modules.BlockLinear)
    assert applied[-1].result.passed
    with torch.no_grad():
        embedding.weight_0.mul_(1 + 1e-8)
    result = registry().selfcheck("block_linear", model, rewritten, context(batch))
    assert not result.passed and "interactions.0.source_embedding.embedding" in result.detail
