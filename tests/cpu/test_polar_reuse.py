"""PolarMACE with the MH-1 and OMAT-0 optimisations (docs/validation/polar-reuse.md, task polar-T01).

MACE-POLAR-1 runs MH-1's interaction class, so the MH-1 rewrite package is meant to carry over. These
tests check that on a tiny PolarMACE on the CPU, float64 (float32 for the radial tables), with the
long-range part included in the energy and forces:

- the MH-1 package, the block linears, ``gated_rows`` and ``polar_uvu`` pass their self-checks and
  keep the whole model's energy and forces at the SPEC §6.4 thresholds;
- ``radial_tables`` is certified on PolarMACE as on MH-1;
- the planner's POLAR-1 defaults are those the report records: the class-level MH-1 rewrites on,
  the label-gated decisions 64 and 66 off (no POLAR-1 measurement yet), and an explicit ``on`` of
  each of them accepted, so a same-process A/B can ask for them;
- the species-table family refuses PolarMACE (its pins cover ScaleShiftMACE's forward only).

The GPU-only kernels (streaming convolution, fused forward, wprime backward, EDGE_SH) are not
exercised here; their PolarMACE checks are GPU follow-ups in the report.
"""

import copy
import importlib.util

import pytest
import torch

from _support.tiny_models import polar_available

pytest.importorskip("mace")
pytestmark = pytest.mark.skipif(not polar_available(), reason="PolarMACE needs graph_longrange")

from test_rewrites_mh1 import PACKAGE, context, energy_and_forces, relative, tiny  # noqa: E402

from fastmace.config import RewriteConfig  # noqa: E402
from fastmace.planner.fixed import PlanRefused  # noqa: E402
from fastmace.rewrites import registry as reg  # noqa: E402

HAS_CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None
LINEARS = ("block_linear", "grouped_linear", "merged_residual")
POLAR_LABELS = ("polar1-s", "polar1-m", "polar1-l")
# The planner fields a POLAR-1 plan takes by default on every GPU (MH-1's class, not its label).
CLASS_DEFAULTS = ("factor_species", "fold_cutoff", "reassociate_first", "fused_folded_backward",
                  "segmented_first_sum", "streaming_conv", "chunked_conv_backward", "radial_tables",
                  "block_linear", "gated_rows", "segment_gemm", "fx_e3nn_linear", "polar_graph_sums",
                  "species_tables")
# Label-gated (decisions 64 and 66): on for MH-1 and, where measured, POLAR-1; explicit "on" accepted.
LABEL_GATED = ("fused_folded_forward", "wprime_backward", "gate_epilogue", "grouped_launch", "tf32x3_gemms")
SPECIES_FAMILY = ("species_tables", "fold_readout_tail", "fold_product_linear")
LAYER_FOLDS = ("fold_readout_tail", "fold_product_linear")  # refused: the long-range heads read every layer


def _float32(model, batch):
    return model.float(), {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                           for key, value in batch.items()}


def _apply(model, batch, names):
    """Every default-registry record in ``names`` on, every other off; (rewritten copy, applied, resolution)."""
    registry = reg.default_registry()
    asked = {name: "on" if name in names else "off" for name in registry.names()}
    rewritten = copy.deepcopy(model)
    resolution = registry.resolve(rewritten, asked, strict=False)
    applied = registry.apply(rewritten, resolution, context(batch), original=model)
    return rewritten, applied, resolution


def _agree(model, rewritten, batch, energy_tolerance, force_tolerance):
    energy_and_forces(model, batch)  # throwaway evaluation (docs/development.md)
    energy, forces = energy_and_forces(model, batch)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert relative(new_energy, energy) <= energy_tolerance
    assert relative(new_forces, forces) <= force_tolerance


# --------------------------------------------------------------------------- the rewrites on a tiny PolarMACE


@pytest.mark.parametrize("backend", ["e3nn", pytest.param("cueq", marks=pytest.mark.skipif(
    not HAS_CUEQ, reason="cuequivariance_torch is not installed"))])
def test_mh1_package_and_linears_keep_polarmace(backend):
    """The MH-1 package (and on cuEq the linear family, gated_rows) plus polar_uvu, each self-checked, and the
    whole PolarMACE (long range included) unchanged to float64 rounding."""
    model, batch = tiny("polar", backend)
    assert type(model).__name__ == "PolarMACE"
    extra = (*LINEARS, "gated_rows") if backend == "cueq" else ()
    rewritten, applied, resolution = _apply(model, batch, (*PACKAGE, *extra, "polar_uvu"))
    assert set(resolution.enabled) == {*PACKAGE, *extra, "polar_uvu"}
    for entry in applied:
        assert entry.result.passed and entry.result.max_relative_error <= 1e-10, (entry.name, entry.result)
    _agree(model, rewritten, batch, 1e-10, 1e-10)


def test_radial_tables_are_certified_on_polarmace():
    """Float32, as the planner enables them: certified by the record and within float32 noise of the network."""
    model, batch = _float32(*tiny("polar"))
    rewritten, applied, resolution = _apply(model, batch, (*PACKAGE, "radial_tables"))
    assert "radial_tables" in resolution.enabled
    (entry,) = [entry for entry in applied if entry.name == "radial_tables"]
    assert entry.result.passed, entry.result
    _agree(model, rewritten, batch, 1e-5, 1e-4)


@pytest.mark.parametrize("backend", ["e3nn", pytest.param("cueq", marks=pytest.mark.skipif(
    not HAS_CUEQ, reason="cuequivariance_torch is not installed"))])
def test_species_tables_keep_polarmace(backend):
    """PolarMACE.forward is pinned next to ScaleShiftMACE's: its embedding is read only by the first interaction, so
    the species tables apply; the whole model (long range included) stays at float64 rounding."""
    model, batch = tiny("polar", backend)
    extra = LINEARS if backend == "cueq" else ()
    rewritten, applied, resolution = _apply(model, batch, (*PACKAGE, *extra, "species_tables"))
    assert "species_tables" in resolution.enabled, resolution.decision("species_tables")
    for entry in applied:
        assert entry.result.passed and entry.result.max_relative_error <= 1e-10, (entry.name, entry.result)
    _agree(model, rewritten, batch, 1e-10, 1e-10)


def test_layer_folds_refuse_polarmace():
    """fold_readout_tail and fold_product_linear change what a layer's node features mean; PolarMACE's long-range
    heads (sources, Fukui, layer mixer) read every layer's features, so both stay refused."""
    model, batch = tiny("polar", "cueq" if HAS_CUEQ else "e3nn")
    _, _, resolution = _apply(model, batch, (*PACKAGE, *LINEARS, *SPECIES_FAMILY))
    for name in ("fold_readout_tail", "fold_product_linear"):
        decision = resolution.decision(name)
        assert not decision.enabled and "long-range heads read every layer's features" in decision.reason, decision


# --------------------------------------------------------------------------- the planner's POLAR-1 defaults


def _plan(label, hardware=None, **configs):
    import dataclasses

    import test_planner as tp

    from fastmace.config import IntegratorConfig

    hardware = tp.GPU if hardware is None else dataclasses.replace(tp.GPU, name=hardware)
    return tp.planner(**configs).plan(tp.traits(label), tp.nacl(), IntegratorConfig(), hardware, model_label=label)


DECISION_66_GPUS = ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB")
ADA = "NVIDIA RTX 4000 Ada Generation"
# Decision 66 for POLAR-1 (owner, 5 November 2026; docs/validation/polar-reuse.md section 4): measured GPUs only.
POLAR_WPRIME = ("fused_folded_forward", "wprime_backward")  # models with a second interaction (M, L)
POLAR_MEASURED = {"wprime": (ADA,), "gate_epilogue": (ADA,)}


def _expected_on(label, gpu):
    """The label-gated fields POLAR-1 takes by default on ``gpu``."""
    on = set()
    if gpu in POLAR_MEASURED["wprime"] and label != "polar1-s":
        on.update(POLAR_WPRIME)
    if gpu in POLAR_MEASURED["gate_epilogue"]:
        on.add("gate_epilogue")
    return on


@pytest.mark.parametrize("gpu", DECISION_66_GPUS)
@pytest.mark.parametrize("label", POLAR_LABELS)
def test_polar1_defaults(label, gpu):
    """The class-level MH-1 rewrites everywhere; the label-gated ones on the GPUs where POLAR-1 was measured."""
    rewrites = _plan(label, gpu).rewrites
    assert all(getattr(rewrites, name) == "on" for name in CLASS_DEFAULTS), rewrites
    on = _expected_on(label, gpu)
    assert {name for name in LABEL_GATED if getattr(rewrites, name) == "on"} == on, rewrites
    assert all(getattr(rewrites, name) == "off" for name in LAYER_FOLDS), rewrites
    assert rewrites.polar_uvu


@pytest.mark.parametrize("name", LABEL_GATED)
def test_polar1_accepts_each_label_gated_rewrite_when_asked(name):
    rewrites = _plan("polar1-m", rewrites=RewriteConfig(**{name: "on"})).rewrites
    assert getattr(rewrites, name) == "on"


@pytest.mark.parametrize("name", (*POLAR_WPRIME, "w_tables"))
def test_one_interaction_model_refuses_the_second_interaction_records(name):
    """POLAR-1-S has only the reassociated first interaction: wprime_backward, the fused forward and w_tables act on
    the interactions after it, so an explicit "on" is refused at plan time instead of failing the build."""
    from fastmace.config import PrecisionConfig

    precision = PrecisionConfig(dtype="float64" if name == "w_tables" else "float32")
    with pytest.raises(PlanRefused, match="no interaction after the reassociated first one"):
        _plan("polar1-s", ADA, precision=precision, rewrites=RewriteConfig(**{name: "on"}))


def test_one_interaction_model_keeps_auto_off_with_the_reason():
    plan = _plan("polar1-s", ADA)
    assert plan.rewrites.wprime_backward == "off" and plan.rewrites.fused_folded_forward == "off"
    off = {option.split(".")[-1]: reason for option, reason in plan.auto_options_off}
    for name in POLAR_WPRIME:
        assert "no interaction after the reassociated first one" in str(off[name]), off


def test_polar1_accepts_an_explicit_species_table_and_refuses_the_folds():
    assert _plan("polar1-s", rewrites=RewriteConfig(species_tables="on")).rewrites.species_tables == "on"
    with pytest.raises(PlanRefused, match="long-range heads read every layer's features"):
        _plan("polar1-m", rewrites=RewriteConfig(fold_product_linear="on"))


# --------------------------------------------------------------------------- EDGE_SH on MH-1's class


def test_edge_sh_for_mh1_class_is_off_until_measured(monkeypatch):
    """MH-1's streaming record applies EDGE_SH on EDGE_SH_MH1_GPUS (decision 66: none measured yet); the A/B constant
    overrides it as for the generic family."""
    from fastmace.rewrites.generic import modules as gm

    for name in (*DECISION_66_GPUS, "NVIDIA A100 80GB PCIe", None):
        assert not gm.edge_sh_default(name, gm.EDGE_SH_MH1_GPUS), name
    monkeypatch.setattr(gm, "EDGE_SH", True)
    assert gm.edge_sh_default(ADA, gm.EDGE_SH_MH1_GPUS)


def test_attach_edge_sh_wraps_l3_harmonics_and_falls_through_on_the_cpu(monkeypatch):
    """attach_edge_sh wraps e3nn's l <= 3 component harmonics once; off CUDA the wrapper is e3nn itself. The tiny
    PolarMACE (l <= 2) is left alone."""
    import torch
    from e3nn import o3

    from fastmace.rewrites.generic import modules as gm

    monkeypatch.setattr(gm, "EDGE_SH", True)
    model, _ = tiny("polar")
    assert not gm.attach_edge_sh(model, gm.EDGE_SH_MH1_GPUS)  # l <= 2: not EdgeSH's
    holder = torch.nn.Module()
    holder.spherical_harmonics = o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component")
    holder.weight = torch.nn.Parameter(torch.zeros(1))
    original = holder.spherical_harmonics
    assert gm.attach_edge_sh(holder, gm.EDGE_SH_MH1_GPUS)
    assert isinstance(holder.spherical_harmonics, gm.EdgeSH)
    assert not gm.attach_edge_sh(holder, gm.EDGE_SH_MH1_GPUS)  # idempotent
    vectors = torch.randn(64, 3)
    assert torch.equal(holder.spherical_harmonics(vectors), original(vectors))


# --------------------------------------------------------------------------- FUSED_FIRST_BACKWARD on MH-1's class


@pytest.mark.parametrize("family", ["mh1", "polar"])
def test_fused_first_message_on_mh1_class(monkeypatch, family):
    """The tabled outer sum (generic FUSED_FIRST_BACKWARD) in MH-1's reassociated first interaction: the folded
    weight [W, b] reads the table's leading h + 1 columns. On the CPU the op falls back to the table and
    edge_outer_sum, so with the switch on the model must agree with the plain path to float32 rounding, and the
    op must have been called. Off by default (FUSED_FIRST_MH1_GPUS empty: nothing measured)."""
    import fastmace.kernels.triton.tabled_outer_sum as tos
    from fastmace.rewrites.generic import modules as gm
    from fastmace.rewrites.mh1 import modules as mm

    assert not any(name in mm.FUSED_FIRST_MH1_GPUS for name in DECISION_66_GPUS)
    model, batch = _float32(*tiny(family))
    rewritten, applied, resolution = _apply(model, batch, (*PACKAGE, "radial_tables"))
    assert {"reassociate_first", "segmented_first_sum", "radial_tables"} <= set(resolution.enabled)
    calls = []
    original = tos.tabled_outer_sum

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tos, "tabled_outer_sum", spy)
    monkeypatch.setattr(gm, "FUSED_FIRST_BACKWARD", False)
    energy, forces = energy_and_forces(rewritten, batch)
    assert not calls
    monkeypatch.setattr(gm, "FUSED_FIRST_BACKWARD", True)
    new_energy, new_forces = energy_and_forces(rewritten, batch)
    assert calls
    assert relative(new_energy, energy) <= 1e-6 and relative(new_forces, forces) <= 1e-5
