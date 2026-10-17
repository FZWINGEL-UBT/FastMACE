"""Model traits (SPEC §4.3, §9.2 "Model traits"): rewrite support, padding exactness, E0 classes.

Ported from the research engine's ``tests/core/test_gpu_langevin.py`` (``ModelSupportTests``) and
checked on one tiny model per §4.3 family. The expected verdicts are written out per family; each
comment names the research-engine function (``langevin/gpu_langevin.py``) whose logic gives it.
"""

import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from _support.tiny_models import FAMILIES, polar_available, tiny_family_calculator, tiny_mace_calculator
from fastmace.model import capabilities
from fastmace.model.capabilities import (
    ALL_ELECTRON,
    PSEUDOPOTENTIAL,
    SMALL,
    UNKNOWN,
    ModelTraits,
    RewriteSupport,
    classify_e0,
    first_inputs_species_only,
    model_traits,
    padding_is_exact,
    rewrite_support,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
Z_VALUES = [11, 17]
OPTIONS = ("factor_species", "fold_cutoff", "reassociate_first", "streaming_conv", "species_skip",
           "fused_folded_backward")
MH1 = "RealAgnosticResidualNonLinearInteractionBlock"
NOT_FACTORED = "interactions {} are not ('RealAgnosticResidualNonLinearInteractionBlock',)"
APPLIES_CUTOFF = "the model applies its cutoff inside the radial embedding (apply_cutoff=True)"
NOT_GENERIC = "no MACE-MP-0 or density-family interaction"


def _family(name):
    marks = []
    if FAMILIES[name].get("polar"):
        marks.append(pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange"))
    return pytest.param(name, marks=marks)


def nacl_cell():
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.1, seed=3)
    return atoms


def padded_energy_change(calc, atoms, padded_cutoff=6.0):
    """|E(edges within r_max) - E(plus every pair out to padded_cutoff)| of MACE's own model.

    The research engine measures this through MACEForces.use_edges; FastMACE's force field comes
    later (1a-T11), so the far pairs are added by building the graph with a larger cutoff. The
    model's radial embedding keeps its own r_max, so the extra edges are pure padding.
    """
    model = calc.models[0]
    r_max = calc.r_max
    energies = []
    try:
        for cutoff in (r_max, padded_cutoff):
            calc.r_max = cutoff
            batch = calc._atoms_to_batch(atoms)
            energies.append(float(model(batch.to_dict(), compute_force=False, training=False)["energy"].detach()))
    finally:
        calc.r_max = r_max
    return abs(energies[1] - energies[0])


# --------------------------------------------------------------------------- research-engine ports


def test_first_inputs_species_only():
    """Per-graph joint embeddings (MACE-OMOL-0: total charge and spin) keep the first inputs species-only."""
    LinearNodeEmbeddingBlock = type("LinearNodeEmbeddingBlock", (), {})
    model = types.SimpleNamespace(node_embedding=LinearNodeEmbeddingBlock())
    assert first_inputs_species_only(model)
    model.joint_embedding = object()
    model.embedding_specs = {"total_spin": {"per": "graph"}, "total_charge": {"per": "graph"}}
    assert first_inputs_species_only(model)
    model.embedding_specs["charges"] = {"per": "node"}
    assert not first_inputs_species_only(model)
    model.embedding_specs = {}
    assert not first_inputs_species_only(model)
    other = types.SimpleNamespace(node_embedding=object())
    assert not first_inputs_species_only(other)


# ModelSupportTests.CASES: (interaction, apply_cutoff, factor, fold/reassociate, padding exact)
SUPPORT_CASES = (
    ("RealAgnosticResidualInteractionBlock", True, False, False, True),  # MACE-MP style
    (MH1, True, True, False, False),
    (MH1, False, True, True, True),  # MH-1 style
)


@pytest.mark.parametrize("interaction, apply_cutoff, factor, fold, exact", SUPPORT_CASES)
def test_support_and_padding_match_the_model(interaction, apply_cutoff, factor, fold, exact):
    calc = tiny_mace_calculator(Z_VALUES, interaction=interaction, apply_cutoff=apply_cutoff)
    model = calc.models[0]
    support = rewrite_support(model)
    assert support["factor_species"][0] == factor
    assert support["fold_cutoff"][0] == fold
    generic = interaction in capabilities.GENERIC_INTERACTIONS  # GENERIC_* rewrites (MP-0 style)
    assert support["reassociate_first"][0] == (fold or generic)
    assert support["streaming_conv"][0] == (fold or generic)
    assert padding_is_exact(model) == exact
    change = padded_energy_change(calc, nacl_cell())
    if exact:  # the static rule agrees with the numbers
        assert change < 1e-10
    else:
        assert change > 1e-6


def test_padding_rule_on_stand_ins():
    """padding_is_exact's branches that the tiny models do not reach (research engine docstring)."""
    assert padding_is_exact(types.SimpleNamespace())  # no radial embedding: not a MACE model, nothing to check
    radial = types.SimpleNamespace(apply_cutoff=True)
    biased = types.SimpleNamespace(conv_tp_weights=torch.nn.Sequential(torch.nn.Linear(4, 4)))
    assert not padding_is_exact(types.SimpleNamespace(radial_embedding=radial, interactions=[biased]))
    bias_free = types.SimpleNamespace(conv_tp_weights=torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)),
                                      density_fn=torch.nn.Linear(4, 1, bias=False))
    assert padding_is_exact(types.SimpleNamespace(radial_embedding=radial, interactions=[bias_free]))
    biased_density = types.SimpleNamespace(density_fn=torch.nn.Linear(4, 1))
    assert not padding_is_exact(types.SimpleNamespace(radial_embedding=radial, interactions=[biased_density]))
    embedded = types.SimpleNamespace(source_embedding=object())
    assert not padding_is_exact(types.SimpleNamespace(radial_embedding=radial, interactions=[embedded]))
    passed = types.SimpleNamespace(apply_cutoff=False)
    assert padding_is_exact(types.SimpleNamespace(radial_embedding=passed, interactions=[biased, embedded]))


# --------------------------------------------------------------------------- one test per tiny family

# Expected traits per family of tests/_support/tiny_models.py. Sources (research engine,
# langevin/gpu_langevin.py at research-engine-baseline-20261017):
#   factor_species, fold_cutoff, reassociate_first, streaming_conv, species_skip,
#   fused_folded_backward: rewrite_support (and _radial_cutoff_passed, first_inputs_species_only);
#   padding exact: padding_is_exact; generic: membership in GENERIC_INTERACTIONS.
EXPECTED = {
    "mh1": dict(
        interactions=(MH1, MH1), apply_cutoff=False, joint=(), polar=False, exact=True, species_only=True,
        support={"factor_species": (True, "ok"), "fold_cutoff": (True, "ok"), "reassociate_first": (True, "ok"),
                 "streaming_conv": (True, "ok"), "species_skip": (False, NOT_GENERIC),
                 "fused_folded_backward": (True, "ok")}),
    # Per-graph total charge and spin: species-only first inputs, so reassociation stays (16 October 2026).
    "omol": dict(
        interactions=(MH1, MH1), apply_cutoff=False, joint=(("total_charge", "graph"), ("total_spin", "graph")),
        polar=False, exact=True, species_only=True,
        support={"factor_species": (True, "ok"), "fold_cutoff": (True, "ok"), "reassociate_first": (True, "ok"),
                 "streaming_conv": (True, "ok"), "species_skip": (False, NOT_GENERIC),
                 "fused_folded_backward": (True, "ok")}),
    # The tiny polar family passes the cutoff to the interactions (apply_cutoff=False), as the
    # released POLAR-1 models do, so it takes the MH-1 rewrites (SPEC §4.3) and its padding is exact.
    "polar": dict(
        interactions=(MH1, MH1), apply_cutoff=False, joint=(), polar=True, exact=True, species_only=True,
        support={"factor_species": (True, "ok"), "fold_cutoff": (True, "ok"), "reassociate_first": (True, "ok"),
                 "streaming_conv": (True, "ok"), "species_skip": (False, NOT_GENERIC),
                 "fused_folded_backward": (True, "ok")}),
    "residual": dict(
        interactions=("RealAgnosticResidualInteractionBlock",) * 2, apply_cutoff=True, joint=(), polar=False,
        exact=True, species_only=True,
        support={"factor_species": (False, NOT_FACTORED.format("['RealAgnosticResidualInteractionBlock']")),
                 "fold_cutoff": (False, NOT_FACTORED.format("['RealAgnosticResidualInteractionBlock']")),
                 "reassociate_first": (True, "ok (GENERIC_REASSOCIATE)"), "streaming_conv": (True, "ok"),
                 "species_skip": (True, "ok"), "fused_folded_backward": (True, "ok")}),
    "density": dict(
        interactions=("RealAgnosticDensityInteractionBlock", "RealAgnosticDensityResidualInteractionBlock"),
        apply_cutoff=True, joint=(), polar=False, exact=True, species_only=True,
        support={"factor_species": (False, NOT_FACTORED.format(
                     "['RealAgnosticDensityInteractionBlock', 'RealAgnosticDensityResidualInteractionBlock']")),
                 "fold_cutoff": (False, NOT_FACTORED.format(
                     "['RealAgnosticDensityInteractionBlock', 'RealAgnosticDensityResidualInteractionBlock']")),
                 "reassociate_first": (True, "ok (GENERIC_REASSOCIATE)"), "streaming_conv": (True, "ok"),
                 "species_skip": (True, "ok"), "fused_folded_backward": (True, "ok")}),
    # MACE-OFF: rewrite_support asks for *any* GENERIC_INTERACTIONS class, and the residual blocks
    # after the first one are, so the generic verdicts hold although the first block
    # (RealAgnosticInteractionBlock) is not in GENERIC_INTERACTIONS.
    "off": dict(
        interactions=("RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock"),
        apply_cutoff=True, joint=(), polar=False, exact=True, species_only=True,
        support={"factor_species": (False, NOT_FACTORED.format(
                     "['RealAgnosticInteractionBlock', 'RealAgnosticResidualInteractionBlock']")),
                 "fold_cutoff": (False, NOT_FACTORED.format(
                     "['RealAgnosticInteractionBlock', 'RealAgnosticResidualInteractionBlock']")),
                 "reassociate_first": (True, "ok (GENERIC_REASSOCIATE)"), "streaming_conv": (True, "ok"),
                 "species_skip": (True, "ok"), "fused_folded_backward": (True, "ok")}),
}


def test_every_family_has_expectations():
    assert set(EXPECTED) == set(FAMILIES)


@pytest.mark.parametrize("family", [_family(name) for name in FAMILIES])
def test_family_traits(family):
    calc = tiny_family_calculator(family, Z_VALUES)
    expected = EXPECTED[family]
    traits = model_traits(calc.models[0])
    assert isinstance(traits, ModelTraits)
    assert traits.model_class == ("PolarMACE" if expected["polar"] else "ScaleShiftMACE")
    assert traits.interaction_classes == expected["interactions"]
    assert traits.apply_cutoff is expected["apply_cutoff"]
    assert traits.radial_cutoff_passed is (not expected["apply_cutoff"])
    assert traits.joint_embedding is bool(expected["joint"])
    assert traits.joint_embeddings == expected["joint"]
    assert traits.polar is expected["polar"]
    assert traits.heads == ("Default",)
    assert traits.atomic_numbers == (11, 17)
    assert traits.r_max == 4.0
    assert traits.padding_is_exact is expected["exact"]
    assert traits.first_inputs_species_only is expected["species_only"]
    assert dict(traits.rewrite_support) == expected["support"]
    assert set(traits.rewrite_support) == set(OPTIONS)
    assert all(isinstance(v, RewriteSupport) for v in traits.rewrite_support.values())
    # The tiny models have zero E0: a small (relative) reference for every element.
    assert [(e.head, e.z, e.kind) for e in traits.e0] == [("Default", 11, SMALL), ("Default", 17, SMALL)]
    assert traits.e0_dtype == "float64"
    # The record agrees with the stand-alone functions the planner (T15) may call.
    assert dict(rewrite_support(calc.models[0])) == expected["support"]
    assert padding_is_exact(calc.models[0]) is expected["exact"]
    # A calculator is accepted too (its first model).
    assert model_traits(calc) == traits


def test_traits_are_frozen():
    traits = model_traits(tiny_family_calculator("mh1", Z_VALUES).models[0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        traits.polar = True
    with pytest.raises(TypeError):
        traits.rewrite_support["factor_species"] = RewriteSupport(False, "x")


# --------------------------------------------------------------------------- E0 classification

# SPEC §4.3 "Energy scale": all-electron Cl ≈ −12,522 eV, Na ≈ −4,412 eV, O ≈ −2,048 eV,
# Br ≈ −70,042 eV; MACE-OFF's table; pseudopotential (effective core) Nb, Ta ≈ −1,545 eV and
# I −8,102.52 eV; MH-1 matpes_r2scan Cl −4.8 eV, Ta −40.5 eV.
E0_CASES = (
    (17, -12522.0, ALL_ELECTRON),
    (11, -4412.0, ALL_ELECTRON),
    (8, -2048.0, ALL_ELECTRON),
    (35, -70042.0, ALL_ELECTRON),
    # MACE-OFF (all five checkpoints)
    (1, -13.57, ALL_ELECTRON),
    (6, -1030.57, ALL_ELECTRON),
    (7, -1486.38, ALL_ELECTRON),
    (8, -2043.93, ALL_ELECTRON),
    (9, -2715.32, ALL_ELECTRON),
    (15, -9287.41, ALL_ELECTRON),
    (16, -10834.48, ALL_ELECTRON),
    (17, -12522.65, ALL_ELECTRON),
    (35, -70045.28, ALL_ELECTRON),
    (53, -8102.52, PSEUDOPOTENTIAL),
    (41, -1545.0, PSEUDOPOTENTIAL),
    (73, -1545.0, PSEUDOPOTENTIAL),
    # a few eV
    (17, -4.8, SMALL),
    (73, -40.5, SMALL),
    (11, -1.3, SMALL),
    (8, -4.9, SMALL),
    (1, -1.1, SMALL),
    (41, 0.0, SMALL),
    (11, 0.5, SMALL),
)


@pytest.mark.parametrize("z, value, kind", E0_CASES)
def test_classify_e0(z, value, kind):
    result = classify_e0(z, value)
    assert result.kind == kind
    assert result.z == z and result.value_ev == value
    assert result.reason  # every class carries its reason


def test_classify_e0_unphysical_values_are_unknown():
    # Far below the all-electron energy of the atom, or a large positive value: no reference fits.
    assert classify_e0(17, -1.0e6).kind == UNKNOWN
    assert classify_e0(17, 5000.0).kind == UNKNOWN


def test_e0_reasons_name_the_numbers():
    assert "Cl" in classify_e0(17, -12522.0).reason
    assert "all-electron" in classify_e0(17, -12522.0).reason
    assert "effective core" in classify_e0(41, -1545.0).reason


def test_e0_of_a_tiny_model_with_an_all_electron_table():
    calc = tiny_mace_calculator(Z_VALUES, interaction=MH1, apply_cutoff=False,
                                atomic_energies=np.array([-4412.0, -12522.0]))
    traits = model_traits(calc.models[0])
    assert [(e.symbol, e.kind) for e in traits.e0] == [("Na", ALL_ELECTRON), ("Cl", ALL_ELECTRON)]
    assert traits.e0_kinds() == {11: ALL_ELECTRON, 17: ALL_ELECTRON}


def _multihead_stand_in():
    """A two-head model as MH-1 stores it: E0 rows per head, in the order of ``heads``."""
    table = torch.tensor([[-1.3, -4.8, -40.5], [-4412.0, -12522.0, -1545.0]], dtype=torch.float64)
    return types.SimpleNamespace(
        heads=["matpes_r2scan", "omol"],
        atomic_numbers=torch.tensor([11, 17, 73]),
        atomic_energies_fn=types.SimpleNamespace(atomic_energies=table),
        r_max=torch.tensor(6.0),
    )


def test_e0_per_head():
    traits = model_traits(_multihead_stand_in())
    assert traits.heads == ("matpes_r2scan", "omol")
    assert traits.e0_kinds("matpes_r2scan") == {11: SMALL, 17: SMALL, 73: SMALL}
    assert traits.e0_kinds("omol") == {11: ALL_ELECTRON, 17: ALL_ELECTRON, 73: PSEUDOPOTENTIAL}
    assert not traits.large_e0("matpes_r2scan")
    assert traits.large_e0("omol")
    with pytest.raises(ValueError, match="head"):
        traits.e0_kinds()  # two heads: the head must be named
    with pytest.raises(KeyError):
        traits.e0_kinds("pbe")


def test_stand_in_without_mace_parts():
    traits = model_traits(_multihead_stand_in())
    assert traits.interaction_classes == ()
    assert traits.apply_cutoff is None
    assert traits.padding_is_exact  # no radial embedding: nothing to check (research engine)
    assert not traits.polar and not traits.joint_embedding


# --------------------------------------------------------------------------- import hygiene


def test_import_does_not_import_mace():
    code = ("import sys, fastmace.model.capabilities; "
            "print(sorted(m for m in sys.modules if m == 'mace' or m.startswith('mace.')))")
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY_ROOT, capture_output=True, text=True,
                            check=False, timeout=300, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
