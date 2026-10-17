"""The generic family: generic_species_skip, generic_reassociate, generic_fused_radial_backward, generic_streaming.

Ported from the research engine's ``tests/core/test_generic_rewrites.py`` (ResearchModule tag
research-engine-baseline-20261017) with its tolerances: the rewritten models reproduce the
original's energy, forces and stress (atol 1e-11, energy to 9 places), on the cell as built and
strained by 1%. The research engine compared through its ``MACEForces`` (eager and compiled);
here the rewritten tiny models are evaluated directly through MACE, eager only (compiled
evaluation belongs to the force field and engine, plan tasks 1a-T11 and 1a-T13).
``GENERIC_DENSITY_FIRST`` is not carried over, so its part of the ported test is dropped.

Further: the per-record self-checks on the tiny ``residual``, ``density`` and ``off`` families
(e3nn and cuEquivariance on the CPU, float64) at the SPEC §6.4 threshold; the MACE-OFF verdict
(the first block is not generic, so ``generic_reassociate`` never applies there); and bitwise
equality with the research engine's ``generic_rewrites`` when ``FASTMACE_BASELINE`` names the
baseline export. Streaming needs OpenEquivariance on a GPU (``tests/gpu``).
"""

import copy
import importlib
import os
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES as TINY
from _support.tiny_models import tiny_mace_calculator
from fastmace.rewrites import generic
from fastmace.rewrites import registry as reg
from fastmace.rewrites.generic import modules as gm

GENERIC_FIELDS = {"generic_species_skip": "species_skip", "generic_reassociate": "reassociate_first",
                  "generic_fused_radial_backward": "fused_folded_backward", "generic_streaming": "streaming_conv"}
FAMILIES = ("residual", "density", "off")


def mh1_available():
    try:
        importlib.import_module("fastmace.rewrites.mh1")
    except ImportError:
        return False
    return True


needs_mh1 = pytest.mark.skipif(not mh1_available(), reason="fastmace.rewrites.mh1 (plan task 1a-T07) is not merged")


def generic_registry():
    registry = reg.Registry()
    for record in generic.RECORDS.values():
        registry.register(record)
    return registry


def structure(shift=0.0, repeat=2):
    """The research test's rattled NaCl, optionally strained by ``shift`` with displaced atoms."""
    atoms = nacl(repeat=repeat)
    atoms.rattle(0.1, seed=11)
    atoms.set_cell(atoms.cell.array * (1 + shift), scale_atoms=True)
    atoms.positions += shift * np.random.default_rng(5).standard_normal(atoms.positions.shape)
    return atoms


def model_and_batch(family, z_values=(11, 17), cueq=False, atoms=None):
    calc = tiny_mace_calculator(list(z_values), **TINY[family])
    model = calc.models[0]
    if cueq:
        from cuequivariance_torch.primitives import segmented_polynomial
        from mace.cli.convert_e3nn_cueq import run

        with mock.patch.object(segmented_polynomial, "HAS_CUE_OPS", False):  # CPU: cuEq's naive kernel
            model = run(model, device="cpu", return_model=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, calc._atoms_to_batch(atoms if atoms is not None else structure())


def evaluate(model, batch):
    out = model(batch.clone().to_dict(), training=False, compute_force=True, compute_stress=True)
    return float(out["energy"].detach()), out["forces"].detach().numpy(), out["stress"].detach().numpy()


def context(batch, seed=0):
    return reg.Context(inputs={generic.NODE_ATTRS: batch["node_attrs"]}, seed=seed)


def rewrite(model, batch, names):
    """A rewritten copy of ``model`` with the generic records ``names`` (all must apply and pass)."""
    registry = generic_registry()
    rewritten = copy.deepcopy(model)
    resolution = registry.resolve(rewritten, {name: ("on" if name in names else "off") for name in registry.names()})
    applied = registry.apply(rewritten, resolution, context(batch), original=model)
    assert [entry.name for entry in applied] == [name for name in registry.names() if name in names]
    for entry in applied:
        assert entry.result.passed and entry.result.max_relative_error <= 1e-10, entry
    return rewritten, applied


def assert_same_physics(model, rewritten, z_values=(11, 17), atol=1e-11):
    calc_batch = tiny_mace_calculator(list(z_values), **TINY["residual"])._atoms_to_batch
    for shift in (0.0, 0.01):
        batch = calc_batch(structure(shift))
        evaluate(model, batch)  # throwaway evaluation (docs/development.md)
        e0, f0, s0 = evaluate(model, batch)
        e, f, s = evaluate(rewritten, batch)
        np.testing.assert_allclose(f, f0, atol=atol)
        np.testing.assert_allclose(s, s0, atol=atol)
        assert e == pytest.approx(e0, abs=1e-9)


# --------------------------------------------------------------------------- records


def test_records_have_the_spec_fields():
    registry = generic_registry()
    assert registry.names() == tuple(GENERIC_FIELDS)
    for name, field in GENERIC_FIELDS.items():
        rewrite_ = registry[name]
        assert registry.config_field(name) == field  # docs/design/switch-map.md
        assert rewrite_.default == "auto" and rewrite_.requires == () and rewrite_.conflicts == ()
        assert rewrite_.source_hashes == gm.source_hashes() and len(rewrite_.source_hashes) == 3
        assert registry.qualification(name).supported  # the pins match the installed MACE 0.3.16
    assert registry["generic_species_skip"].exactness == "exact"
    assert registry["generic_streaming"].exactness == "exact"
    assert registry["generic_reassociate"].exactness == "rounding"
    assert registry["generic_fused_radial_backward"].exactness == "rounding"
    assert "fixed_species_superset" in registry["generic_species_skip"].workload_constraints
    assert "single_graph" in registry["generic_reassociate"].workload_constraints


@pytest.mark.parametrize("family", ["residual", "density", "off", "mh1"])
def test_applies_to(family):
    model, _ = model_and_batch(family)
    support = {rewrite_.name: rewrite_.applies_to(model) for rewrite_ in generic_registry()}
    if family == "mh1":
        assert not any(verdict.supported for verdict in support.values())
        return
    assert support["generic_species_skip"].supported and support["generic_fused_radial_backward"].supported
    supported, reason = support["generic_streaming"]
    assert not supported and "OpenEquivariance" in reason  # e3nn and plain cuEq models
    supported, reason = support["generic_reassociate"]
    if family == "off":
        assert not supported and "RealAgnosticInteractionBlock" in reason
    else:
        assert supported, reason


def test_changed_mace_source_is_refused():
    model, batch = model_and_batch("residual")
    pins = dict(gm.GENERIC_INTERACTIONS)
    pins["RealAgnosticResidualInteractionBlock"] = ("residual", "0" * 64)
    with mock.patch.object(gm, "GENERIC_INTERACTIONS", types.MappingProxyType(pins)):
        record = reg.Rewrite.from_dict(generic.RECORDS["generic_species_skip"])[0]
        record = reg.Rewrite(**{**record.__dict__, "source_hashes": gm.source_hashes()})
        qualified, reason = reg.qualify(record)
        assert not qualified and "differs" in reason
        with pytest.raises(ValueError, match="source sha256"):
            generic.apply_species_skip(copy.deepcopy(model), context(batch))


def test_mh1_models_are_not_touched():
    model, batch = model_and_batch("mh1")
    registry = generic_registry()
    resolution = registry.resolve(model, strict=False)
    assert resolution.enabled == ()
    with pytest.raises(reg.RewriteRefusedError):
        registry.resolve(model, {"generic_species_skip": "on"})
    rewritten = copy.deepcopy(model)
    assert generic.apply_species_skip(rewritten, context(batch)) == []
    assert generic.apply_fused_radial_backward(rewritten, context(batch)) == []
    assert not any(hasattr(i, "generic_kind") for i in rewritten.interactions)


def test_structure_records_need_node_attrs():
    model, _ = model_and_batch("residual")
    with pytest.raises(ValueError, match="node attributes"):
        generic.apply_species_skip(copy.deepcopy(model), reg.Context())


# --------------------------------------------------------------------------- ported model tests


@pytest.mark.parametrize("family", ["residual", "density"])
def test_forward_alone_reproduces_each_class(family):
    """No option on: the replacement forward itself equals MACE's (all three classes)."""
    model, _ = model_and_batch(family)
    rewritten = copy.deepcopy(model)
    for _, interaction, kind in generic._generic_interactions(rewritten):
        generic._prepare(interaction, kind)
    assert all(i.forward.__func__ is gm.generic_interaction_forward for i in rewritten.interactions)
    assert_same_physics(model, rewritten, atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
def test_species_restricted_skip_matches_the_original(family):
    """generic_species_skip with one element absent from NaCl."""
    model, batch = model_and_batch(family, z_values=(8, 11, 17))
    rewritten, applied = rewrite(model, batch, ["generic_species_skip"])
    skips = [i.skip_tp for i in rewritten.interactions]
    generic_skips = [s for s in skips if isinstance(s, gm.SpeciesSkip)]
    assert len(generic_skips) == (1 if family == "off" else 2) == applied[0].result.modules
    assert all(len(s.columns) == 2 for s in generic_skips)  # Na and Cl only
    assert not isinstance(model.interactions[0].skip_tp, gm.SpeciesSkip)  # private copy, original untouched
    assert_same_physics(model, rewritten, z_values=(8, 11, 17))


@pytest.mark.parametrize("family", FAMILIES)
def test_fused_radial_backward_matches_the_original(family):
    """generic_fused_radial_backward: last radial layer through folded_linear (PyTorch fallback on the CPU)."""
    model, batch = model_and_batch(family)
    rewritten, applied = rewrite(model, batch, ["generic_fused_radial_backward"])
    flags = [getattr(i, "fused_radial_backward", False) for i in rewritten.interactions]
    assert flags == ([False, True] if family == "off" else [True, True])
    assert applied[0].detail == [f"interactions.{i}" for i, flag in enumerate(flags) if flag]
    assert_same_physics(model, rewritten)


@needs_mh1
@pytest.mark.parametrize("family", ["residual", "density"])
def test_reassociated_models_match_the_original(family):
    model, batch = model_and_batch(family)
    rewritten, _ = rewrite(model, batch, ["generic_reassociate"])
    first = rewritten.interactions[0]
    assert first.first_reassociation is not None
    assert first.forward.__func__ is gm.generic_interaction_forward
    assert not hasattr(model.interactions[0], "generic_kind")
    assert_same_physics(model, rewritten)


@needs_mh1
@pytest.mark.parametrize("family", ["residual", "density"])
def test_stacked_generic_rewrites_match_the_original(family):
    """Skip, reassociation and fused backward together; the reassociated first block is not fused."""
    model, batch = model_and_batch(family, z_values=(8, 11, 17))
    names = ["generic_species_skip", "generic_reassociate", "generic_fused_radial_backward"]
    rewritten, applied = rewrite(model, batch, names)
    assert applied[2].detail == ["interactions.1"]
    assert_same_physics(model, rewritten, z_values=(8, 11, 17))


# --------------------------------------------------------------------------- self-checks


@pytest.mark.parametrize("cueq", [False, True], ids=["e3nn", "cueq"])
@pytest.mark.parametrize("family", FAMILIES)
def test_selfchecks_on_tiny_models(family, cueq):
    if cueq:
        pytest.importorskip("cuequivariance_torch")
    model, batch = model_and_batch(family, z_values=(8, 11, 17), cueq=cueq)
    names = ["generic_species_skip", "generic_fused_radial_backward"]
    if family != "off" and mh1_available():
        names.insert(1, "generic_reassociate")
    rewritten, applied = rewrite(model, batch, names)
    expected_modules = 1 if family == "off" else 2
    by_name = {entry.name: entry.result for entry in applied}
    assert by_name["generic_species_skip"].modules == expected_modules
    assert by_name["generic_fused_radial_backward"].modules == (1 if len(names) == 3 else expected_modules)
    assert all(result.threshold == 1e-10 and result.dtype == "float64" for result in by_name.values())
    evaluate(model, batch)
    e0, f0, s0 = evaluate(model, batch)
    e, f, s = evaluate(rewritten, batch)
    np.testing.assert_allclose(f, f0, atol=1e-11)
    np.testing.assert_allclose(s, s0, atol=1e-11)


def test_species_skip_selfcheck_catches_a_wrong_matrix():
    model, batch = model_and_batch("density", z_values=(8, 11, 17))
    rewritten, _ = rewrite(model, batch, ["generic_species_skip"])
    with torch.no_grad():
        rewritten.interactions[1].skip_tp.w0.mul_(1 + 1e-8)
    result = generic.RECORDS["generic_species_skip"]["selfcheck"](model, rewritten, context(batch))
    assert not result.passed and "interactions.1" in result.detail


def test_species_skip_refuses_a_mismatch():
    model, _ = model_and_batch("residual")
    skip = model.interactions[0].skip_tp
    with mock.patch.object(gm.SpeciesSkip, "forward", lambda self, x1, x2: x1[:, :0] + 0.0):
        with pytest.raises(ValueError, match="does not reproduce"):
            gm.species_skip(skip, torch.tensor([0, 1]))


# --------------------------------------------------------------------------- MACE-OFF


def test_mace_off_first_block_is_not_rewritten():
    """The research engine reports reassociate_first for MACE-OFF, but no generic rewrite touches block 0."""
    model, batch = model_and_batch("off")
    assert [type(i).__name__ for i in model.interactions] == ["RealAgnosticInteractionBlock",
                                                              "RealAgnosticResidualInteractionBlock"]
    assert [index for index, _, _ in generic._generic_interactions(model)] == [1]
    rewritten = copy.deepcopy(model)
    if mh1_available():
        assert generic.apply_reassociate(rewritten, context(batch)) == []
    assert generic.apply_species_skip(rewritten, context(batch)) == ["interactions.1"]
    assert not hasattr(rewritten.interactions[0], "generic_kind")


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


BASELINE_CASES = {  # our records -> the research engine's switches
    ("generic_species_skip",): ("GENERIC_SPECIES_SKIP",),
    ("generic_fused_radial_backward",): ("GENERIC_FUSED_RADIAL_BACKWARD",),
    ("generic_species_skip", "generic_reassociate", "generic_fused_radial_backward"):
        ("GENERIC_SPECIES_SKIP", "GENERIC_REASSOCIATE", "GENERIC_FUSED_RADIAL_BACKWARD"),
}


@pytest.mark.parametrize("names", list(BASELINE_CASES), ids=lambda names: "+".join(names))
@pytest.mark.parametrize("family", FAMILIES)
def test_generic_rewrites_equal_the_baseline_bitwise(family, names):
    gl = _baseline("gpu_langevin")
    if "generic_reassociate" in names and (family == "off" or not mh1_available()):
        pytest.skip("no reassociation here")
    model, batch = model_and_batch(family, z_values=(8, 11, 17))
    ours, _ = rewrite(model, batch, list(names))
    theirs = copy.deepcopy(model)
    switches = {switch: True for switch in BASELINE_CASES[names]}
    with mock.patch.multiple(gl, GENERIC_STREAMING=False, **switches):
        count = gl.generic_rewrites(theirs, batch["node_attrs"])
    assert count == len(generic._generic_interactions(model))
    evaluate(ours, batch)  # throwaway evaluation (docs/development.md)
    for a, b in zip(evaluate(ours, batch), evaluate(theirs, batch)):
        assert np.array_equal(a, b)
