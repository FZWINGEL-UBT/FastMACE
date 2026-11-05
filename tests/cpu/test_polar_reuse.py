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
                  "block_linear", "gated_rows", "segment_gemm", "fx_e3nn_linear", "polar_graph_sums")
# Label-gated to MH-1 (decisions 64 and 66): off for POLAR-1 until it is measured, explicit "on" accepted.
LABEL_GATED = ("fused_folded_forward", "wprime_backward", "gate_epilogue", "grouped_launch", "tf32x3_gemms")
SPECIES_FAMILY = ("species_tables", "fold_readout_tail", "fold_product_linear")


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


def test_species_family_refuses_polarmace():
    """Pinned to ScaleShiftMACE's forward: PolarMACE's forward is a follow-up (the report's adaptation list)."""
    model, batch = tiny("polar", "cueq" if HAS_CUEQ else "e3nn")
    _, _, resolution = _apply(model, batch, (*PACKAGE, *LINEARS, *SPECIES_FAMILY))
    for name in SPECIES_FAMILY:
        decision = resolution.decision(name)
        assert not decision.enabled and "only ScaleShiftMACE's forward is pinned" in decision.reason, decision


# --------------------------------------------------------------------------- the planner's POLAR-1 defaults


def _plan(label, hardware=None, **configs):
    import dataclasses

    import test_planner as tp

    from fastmace.config import IntegratorConfig

    hardware = tp.GPU if hardware is None else dataclasses.replace(tp.GPU, name=hardware)
    return tp.planner(**configs).plan(tp.traits(label), tp.nacl(), IntegratorConfig(), hardware, model_label=label)


DECISION_66_GPUS = ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB")


@pytest.mark.parametrize("gpu", DECISION_66_GPUS)
@pytest.mark.parametrize("label", POLAR_LABELS)
def test_polar1_defaults_take_the_class_rewrites_not_the_label_gated_ones(label, gpu):
    rewrites = _plan(label, gpu).rewrites
    assert all(getattr(rewrites, name) == "on" for name in CLASS_DEFAULTS), rewrites
    assert all(getattr(rewrites, name) == "off" for name in LABEL_GATED), rewrites
    assert all(getattr(rewrites, name) == "off" for name in SPECIES_FAMILY), rewrites
    assert rewrites.polar_uvu


@pytest.mark.parametrize("name", LABEL_GATED)
def test_polar1_accepts_each_label_gated_rewrite_when_asked(name):
    # POLAR-1-M: wprime_backward and the fused forward act on the interactions after the first, and POLAR-1-S has
    # only one (its plan takes them too, then the build fails: the report's follow-up on plan-time refusal).
    rewrites = _plan("polar1-m", rewrites=RewriteConfig(**{name: "on"})).rewrites
    assert getattr(rewrites, name) == "on"


def test_polar1_refuses_an_explicit_species_table():
    with pytest.raises(PlanRefused, match="only ScaleShiftMACE's forward is pinned"):
        _plan("polar1-s", rewrites=RewriteConfig(species_tables="on"))
