"""MACEForceField on the CPU (plan task 1a-T11; SPEC §3.2, §7.1, §8.1).

Ported from the research engine's ``MACEForcesTests``, ``FactorSpeciesRadialTests``,
``CompiledMACEForcesTests`` and ``AllocatorConfigTests`` (``tests/core/test_gpu_langevin.py`` at
ResearchModule tag research-engine-baseline-20261017), on the tiny float64 models of
``_support.tiny_models``:

- energy, forces and stress against MACE's own calculator (rtol 1e-12) and, with
  ``FASTMACE_BASELINE`` set, against the research engine's ``MACEForces`` bitwise after one
  throwaway evaluation (docs/development.md); rewrites off and on (the MH-1 package and the
  generic family through the registry, the research engine's globals on its side);
- deep copies re-apply MACE's OEq and fused-cuEq adapters (stand-ins with the packages'
  serialisation contracts; the real ones run in tests/gpu/test_forces_gpu.py);
- the streaming edge counter belongs to the force field: two force fields on one model each read
  their own count without rebinding;
- streaming refuses edges that are not receiver-sorted;
- a CPU cuEquivariance model must use cuEq's naive kernels, otherwise it is refused with a reason;
- the compiled-model cache is an object passed in, not a module dict.
"""

import contextlib
import copy
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from fastmace.config import DEFAULT_SKIN_A, EngineConfig, NeighbourConfig, RewriteConfig
from fastmace.model import compile as fc
from fastmace.model.forces import ForceField, ForceOutput, MACEForceField
from fastmace.neighbors.buckets import edge_bucket, round_up_edges
from fastmace.rewrites import registry as reg

pytest.importorskip("mace")

SKIN = 0.5
NEIGHBOURS = NeighbourConfig(skin=SKIN, sort_receivers="off", gpu_neighbour_list="off")
EAGER = EngineConfig(compile_mode="none")
Z_VALUES = [8, 11, 17, 26]  # Fe absent from the structure
MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")
GENERIC = ("generic_species_skip", "generic_reassociate", "generic_fused_radial_backward")
HAS_CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None


def structure(rattle=0.1, seed=8, oxygen=True):
    from _support.structures import nacl

    atoms = nacl(repeat=2, rattle=rattle, seed=seed)
    if oxygen:
        atoms.numbers[3] = 8  # three species, one of them once
    return atoms


def calculator(family="mh1", z_values=Z_VALUES):
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    return tiny_mace_calculator(z_values, **FAMILIES[family])


def only(*names):
    """Rewrite requests: ``names`` on, every other registered record off."""
    return {name: "on" if name in names else "off" for name in reg.default_registry().names()}


def moved(atoms, shift, seed=9):
    """Positions moved by ``shift`` x a standard normal (below skin / 2 for the shifts used here)."""
    return torch.tensor(atoms.positions + shift * np.random.default_rng(seed).standard_normal(atoms.positions.shape))


def evaluate(field, positions, padded=False):
    """Compact at ``positions``, read the exact or a padded edge count, return (energy, forces[, stress])."""
    field.compact(positions)
    count = int(field.edge_count)
    num_edges = round_up_edges(count, field.edge_capacity, edge_bucket(count, 0.25, 256)) if padded else count
    field.use_edges(num_edges)
    out = field(positions)
    values = [out.energy.detach().clone(), out.forces.detach().clone()]
    if out.stress is not None:
        values.append(out.stress.detach().clone())
    return values


def reference(calc, atoms, positions, stress=False):
    """MACE's calculator at ``positions`` (energy, forces[, stress as 3x3 in ASE's convention])."""
    current = atoms.copy()
    current.positions = positions.numpy()
    current.calc = calc
    values = [current.get_potential_energy(), current.get_forces()]
    if stress:
        values.append(current.get_stress(voigt=False))
    return values


def assert_matches_calculator(values, expected):
    np.testing.assert_allclose(float(values[0][0]), expected[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(values[1].numpy(), expected[1], rtol=1e-12, atol=1e-14)
    if len(expected) > 2:
        np.testing.assert_allclose(values[2].numpy(), expected[2], rtol=1e-10, atol=1e-14)


def relative(got, expected):
    return float((got - expected).abs().max()) / float(expected.abs().max())


# --------------------------------------------------------------------------- protocol and plumbing


def test_satisfies_the_protocol():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER)
    assert isinstance(field, ForceField)
    assert field.device == torch.device("cpu") and field.dtype == torch.float64
    assert field.compute_stress is False and field.padding_exact is True
    assert field.model is calc.models[0]  # no rewrites, not compiled: the calculator's own model
    out = field(torch.tensor(atoms.positions))
    assert isinstance(out, ForceOutput) and out.stress is None
    assert out.energy.shape == (1,) and out.forces.shape == (len(atoms), 3)
    assert field.edge_capacity >= int(field.edge_count) > 0 and field.edge_capacity % 256 == 0


def test_defaults_mirror_maceforces():
    """Without configs: uncompiled, no rewrites, matscipy order, the calculator's model."""
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms)
    assert field.info()["compile_mode"] == "none" and field.info()["rewrites"] == []
    assert field.resolution.enabled == () and field.model is calc.models[0]
    assert NeighbourConfig().skin == "auto" and field.neighbours.skin == DEFAULT_SKIN_A
    assert not field.neighbours.sort_receivers and not field.neighbours.gpu_build


def test_unresolved_neighbour_options_are_refused():
    with pytest.raises(ValueError, match="planner resolves it"):
        MACEForceField(calculator(), structure(), neighbours=NeighbourConfig(skin=SKIN))


@pytest.mark.parametrize("problem, match", [
    ("pbc", "Periodic boundaries"), ("compile", "uncompiled calculator"), ("units", "eV and Angstrom")])
def test_refuses_what_maceforces_refuses(problem, match):
    calc, atoms = calculator(), structure()
    if problem == "pbc":
        atoms.pbc = (True, True, False)
    elif problem == "compile":
        calc.use_compile = True
    else:
        calc.energy_units_to_eV = 2.0
    with pytest.raises(ValueError, match=match):
        MACEForceField(calc, atoms, neighbours=NEIGHBOURS)


def test_buffers_and_info():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True)
    buffers = field.buffers()
    for name in ("neighbours.edge_index", "neighbours.shifts", "neighbours.unit_shifts", "neighbours.count",
                 "neighbours.reference_positions", "neighbours.cell", "neighbours.build_cell", "cell"):
        assert torch.is_tensor(buffers[name]), name
    assert buffers["neighbours.count"] is field.edge_count
    info = field.info()
    assert info["neighbour_rebuilds"] == 1 and info["edge_capacity"] == field.edge_capacity
    assert info["backend"] == "e3nn" and info["streaming_interactions"] == 0 and info["polar"] is None


def test_rebuild_moves_buffers_only_when_capacity_grows():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS)
    edge_index = field.buffers()["neighbours.edge_index"]
    assert field.rebuild(torch.tensor(atoms.positions)) is False
    assert field.buffers()["neighbours.edge_index"] is edge_index
    assert bool(field.needs_rebuild(moved(atoms, 0.0))) is False
    far = torch.tensor(atoms.positions)
    far[0, 0] += SKIN
    assert bool(field.needs_rebuild(far)) is True
    assert bool(field.needs_rebuild(moved(atoms, 0.06), fraction=0.01)) is True


# --------------------------------------------------------------------------- agreement with MACE


@pytest.mark.parametrize("family", ["mh1", "residual", "density", "off"])
@pytest.mark.parametrize("padded", [False, True])
def test_matches_the_calculator_with_stale_candidates_and_padding(family, padded):
    """MACEForcesTests.test_matches_calculator_with_stale_candidates_and_padding, every family."""
    calc, atoms = calculator(family), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EAGER)
    positions = moved(atoms, 0.05, seed=4)  # < skin / 2: stale candidates, no rebuild
    current = atoms.copy()
    current.positions = positions.numpy()
    field.compact(positions)
    assert int(field.edge_count) == calc._atoms_to_batch(current)["edge_index"].shape[1]
    values = evaluate(field, positions, padded)
    assert_matches_calculator(values, reference(calc, atoms, positions))


@pytest.mark.parametrize("family", ["mh1", "density"])
def test_stress_matches_the_calculator_after_set_cell(family):
    calc, atoms = calculator(family), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True)
    strain = np.identity(3) + 0.01 * np.array([[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.3]])
    strained = atoms.copy()
    strained.set_cell(atoms.cell.array @ strain, scale_atoms=True)
    field.set_cell(torch.tensor(strained.cell.array))
    positions = torch.tensor(strained.positions)
    values = evaluate(field, positions, padded=True)
    assert_matches_calculator(values, reference(calc, strained, positions, stress=True))


def test_evaluate_is_the_one_off_api():
    calc, atoms = calculator(), structure()
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True)
    other = atoms.copy()
    other.positions += np.random.default_rng(3).uniform(-0.4, 0.4, atoms.positions.shape)  # beyond the skin
    energy, forces, stress = field.evaluate(other, stress=True)
    expected = reference(calc, other, torch.tensor(other.positions), stress=True)
    np.testing.assert_allclose(energy, expected[0], rtol=1e-12)
    np.testing.assert_allclose(forces, expected[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(stress, expected[2], rtol=1e-10, atol=1e-14)
    assert isinstance(energy, float) and forces.dtype == np.float64
    fixed = MACEForceField(calc, atoms, neighbours=NEIGHBOURS)
    assert fixed.evaluate(other)[2] is None
    with pytest.raises(ValueError, match="compute_stress"):
        fixed.evaluate(other, stress=True)
    swapped = other.copy()
    swapped.numbers[[0, 1]] = swapped.numbers[[1, 0]]
    with pytest.raises(ValueError, match="species"):
        fixed.evaluate(swapped)
    with pytest.raises(ValueError, match="cell"):
        strained = other.copy()
        strained.set_cell(other.cell.array * 1.01, scale_atoms=True)
        fixed.evaluate(strained)


@pytest.mark.parametrize("family, names", [("mh1", MH1_PACKAGE), ("density", GENERIC), ("residual", GENERIC)])
def test_rewrites_keep_energy_forces_and_stress(family, names):
    """FactorSpeciesRadialTests.test_eager_forces_match_original, through the registry."""
    calc, atoms = calculator(family), structure()
    plain = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True)
    rewritten = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True, rewrites=only(*names))
    assert [entry.name for entry in rewritten.applied] == list(names)
    assert all(entry.result.passed for entry in rewritten.applied)
    assert rewritten.model is not calc.models[0]  # private copy, calculator untouched
    assert all(type(i).__name__ != "FactoredRadialMLP" for i in calc.models[0].modules())
    for shift in (0.0, 0.05):
        positions = moved(atoms, shift)
        for a, b in zip(evaluate(rewritten, positions, padded=True), evaluate(plain, positions, padded=True)):
            assert relative(a, b) <= 1e-12


def test_reassociation_reads_the_structure_species():
    """``node_attrs`` reach the registry's Context: the reassociated slots are the structure's species."""
    field = MACEForceField(calculator(), structure(), neighbours=NEIGHBOURS,
                           rewrites=only("factor_species", "fold_cutoff", "reassociate_first"))
    assert field.species_columns == (0, 1, 2)  # O, Na, Cl present; Fe absent
    assert field.model.interactions[0].first_reassociation is not None


def test_rewrite_config_and_resolution_are_accepted():
    calc, atoms = calculator(), structure()
    off = RewriteConfig(**{name: "off" for name in ("factor_species", "fold_cutoff", "density_first",
                                                    "reassociate_first", "joint_radial_input",
                                                    "fused_folded_backward", "segmented_first_sum",
                                                    "streaming_conv", "species_skip", "fused_radial_hidden",
                                                    "block_linear", "grouped_linear", "merged_residual",
                                                    "single_graph_strain")}, polar_uvu=False)
    assert MACEForceField(calc, atoms, neighbours=NEIGHBOURS, rewrites=off).resolution.enabled == ()
    registry = reg.default_registry()
    resolution = registry.resolve(calc.models[0], only("factor_species"))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, rewrites=resolution, registry=registry)
    assert field.resolution is resolution and [e.name for e in field.applied] == ["factor_species"]
    with pytest.raises(reg.RewriteRefusedError):
        MACEForceField(calc, atoms, neighbours=NEIGHBOURS, rewrites=only("block_linear"))  # no cuEq Linear


# --------------------------------------------------------------------------- the research engine (bitwise)


def baseline(name):
    """langevin.<name> from the export named by FASTMACE_BASELINE (research-engine-baseline-20261017)."""
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


# Our records -> the research engine's switches (MACEForces reads them at construction).
BASELINE_SWITCHES = {
    "fold_cutoff": "FOLD_RADIAL_CUTOFF", "density_first": "DENSITY_BEFORE_LINEAR",
    "joint_radial_input": "JOINT_RADIAL_INPUT", "fused_folded_backward": "FUSED_FOLDED_BACKWARD",
    "reassociate_first": "REASSOCIATE_FIRST_INTERACTION", "segmented_first_sum": "SEGMENTED_FIRST_SUM",
    "generic_species_skip": "GENERIC_SPECIES_SKIP", "generic_reassociate": "GENERIC_REASSOCIATE",
    "generic_fused_radial_backward": "GENERIC_FUSED_RADIAL_BACKWARD",
}

BASELINE_CASES = [("mh1", ()), ("residual", ()), ("density", ()), ("off", ()), ("mh1", ("factor_species",)),
                  ("mh1", MH1_PACKAGE), ("density", GENERIC), ("residual", GENERIC)]


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("family, names", BASELINE_CASES, ids=lambda v: "+".join(v) if isinstance(v, tuple) else v)
def test_equals_maceforces_bitwise(family, names, stress):
    gl = baseline("gpu_langevin")
    calc, atoms = calculator(family), structure()
    switches = {BASELINE_SWITCHES[name]: True for name in names if name in BASELINE_SWITCHES}
    with mock.patch.multiple(gl, **switches) if switches else contextlib.nullcontext():
        theirs = gl.MACEForces(calc, atoms, skin=SKIN, factor_species="factor_species" in names,
                               compute_stress=stress)
    # 1a's energy sum: the research engine has no float64 accumulation (1b-T03).
    ours = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=stress, rewrites=only(*names),
                          float64_energy=False)
    assert (ours.model is calc.models[0]) is (theirs.model is calc.models[0])
    if names:
        theirs_params = dict(theirs.model.named_parameters())
        for name, tensor in ours.model.named_parameters():
            assert torch.equal(tensor, theirs_params[name]), name
    strain = torch.tensor(atoms.cell.array @ (np.identity(3) + 0.004 * np.ones((3, 3))))
    if stress:
        ours.set_cell(strain)
        theirs.set_cell(strain)
    for shift in (0.0, 0.05):
        positions = moved(atoms, shift)
        if stress:
            positions = positions @ torch.linalg.solve(torch.tensor(atoms.cell.array), strain)
        evaluate(ours, positions, padded=True)  # throwaway evaluation (docs/development.md)
        theirs.neighbours.compact(positions)
        count = int(theirs.neighbours.count)
        theirs.use_edges(round_up_edges(count, theirs.neighbours.capacity, edge_bucket(count, 0.25, 256)))
        expected = theirs(positions)
        got = evaluate(ours, positions, padded=True)
        assert int(ours.edge_count) == count
        for a, b in zip(got, expected):
            assert torch.equal(a, b.detach().reshape(a.shape)), f"max |difference| {(a - b).abs().max():.3e}"


# --------------------------------------------------------------------------- adapters on copies


def _fake_wrapper_ops():
    """mace.modules.wrapper_ops with MACE's OEq and cuEq adapters, as the research engine's test fakes them."""

    def with_oeq_conv_fusion(conv_tp, transpose_in=None, transpose_out=None):
        conv_tp.original_forward = conv_tp.forward
        conv_tp.layout_transpose_in = transpose_in
        conv_tp.layout_transpose_out = transpose_out

        def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
            sender, receiver = edge_index[0], edge_index[1]
            if self.layout_transpose_in is not None:
                node_feats = self.layout_transpose_in(node_feats)
            out = self.original_forward(node_feats, edge_attrs, tp_weights, receiver, sender)
            if self.layout_transpose_out is not None:
                out = self.layout_transpose_out(out)
            return out

        conv_tp.forward = MethodType(forward, conv_tp)
        return conv_tp

    def with_cueq_conv_fusion(conv_tp):
        conv_tp.original_forward = conv_tp.forward
        conv_tp.weight_numel = conv_tp.numel

        def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
            return self.original_forward(node_feats + edge_attrs + tp_weights, edge_index)

        conv_tp.forward = MethodType(forward, conv_tp)
        return conv_tp

    wrapper_ops = ModuleType("mace.modules.wrapper_ops")
    wrapper_ops.with_oeq_conv_fusion = with_oeq_conv_fusion
    wrapper_ops.with_oeq_scatter_sum = None
    wrapper_ops.with_cueq_conv_fusion = with_cueq_conv_fusion
    return wrapper_ops


class _Layout(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.register_buffer("offset", torch.tensor(float(offset)))

    def forward(self, values):
        return values + self.offset


class _FakeOeqConv(torch.nn.Module):
    """OEq's serialisation contract: only the C++ input description survives a copy."""

    def __init__(self):
        super().__init__()
        self.input_args = {"problem": SimpleNamespace(weight_numel=3)}

    def __getstate__(self):
        return self.input_args

    def __setstate__(self, state):
        torch.nn.Module.__init__(self)
        self.input_args = state

    def forward(self, node_feats, edge_attrs, tp_weights, rows, cols):
        return node_feats + edge_attrs + tp_weights + rows.to(node_feats) + cols.to(node_feats)


_FakeOeqConv.__module__ = "openequivariance._torch.TensorProductConv"


class _FakeCueqPolynomial(torch.nn.Module):
    """cuet.SegmentedPolynomial's contract: pickles through __reduce__ with its constructor arguments only."""

    def __init__(self, numel):
        super().__init__()
        self.numel = numel

    def __reduce__(self):
        return (type(self), (self.numel,))

    def forward(self, x, edge_index):
        return x * self.numel + edge_index.sum()


_FakeCueqPolynomial.__module__ = "cuequivariance_torch.primitives.segmented_polynomial"


class _AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.oeq = _FakeOeqConv()
        self.cueq = _FakeCueqPolynomial(5)


def test_private_copy_reapplies_oeq_and_cueq_adapters():
    wrapper_ops = _fake_wrapper_ops()
    model = _AdapterModel()
    wrapper_ops.with_oeq_conv_fusion(model.oeq, _Layout(10), _Layout(20))
    wrapper_ops.with_cueq_conv_fusion(model.cueq)
    bare = copy.deepcopy(model)  # what a plain copy gives: both adapters lost
    assert "forward" not in bare.oeq.__dict__ and "forward" not in bare.cueq.__dict__
    with mock.patch.dict(sys.modules, {"mace.modules.wrapper_ops": wrapper_ops}):
        private = fc.private_copy(model)
    one, edge_index = torch.ones(1, 1), torch.tensor([[0], [0]])
    assert float(private.oeq(one, one, one, edge_index)) == 33.0  # adapter and both layouts
    assert float(private.cueq(one, one, one, edge_index)) == 15.0 and private.cueq.weight_numel == 5
    assert private.oeq.layout_transpose_in is not model.oeq.layout_transpose_in  # independent copies
    model.oeq.layout_transpose_in.offset.fill_(100)
    assert float(private.oeq.layout_transpose_in.offset) == 10.0


def test_restore_refuses_an_unwrapped_oeq_module():
    model = _AdapterModel()
    with mock.patch.dict(sys.modules, {"mace.modules.wrapper_ops": _fake_wrapper_ops()}):
        with pytest.raises(RuntimeError, match="not wrapped by MACE"):
            fc.restore_oeq_wrappers(model, copy.deepcopy(model))
    plain = torch.nn.Linear(2, 2)
    assert fc.restore_oeq_wrappers(plain, copy.deepcopy(plain)) == 0
    assert fc.restore_cueq_conv_fusion(plain, copy.deepcopy(plain)) == 0


def test_force_field_copies_go_through_private_copy():
    """Every model copy of a force field re-applies the adapters (SPEC §3.2)."""
    calc, atoms = calculator(), structure()
    with mock.patch.object(fc, "private_copy", side_effect=fc.private_copy) as spy:
        MACEForceField(calc, atoms, neighbours=NEIGHBOURS, rewrites=only("factor_species"))
    spy.assert_called_once_with(calc.models[0])


# --------------------------------------------------------------------------- streaming counter and edge order


class _FakeStreaming(torch.nn.Module):
    """A module that reads ``streaming_count`` the way the streaming interactions do."""

    def __init__(self):
        super().__init__()
        self.streaming_kernels = object()
        self.register_buffer("streaming_count", torch.tensor(torch.iinfo(torch.int64).max), persistent=False)
        self.seen = []

    def forward(self):
        self.seen.append(self.streaming_count)


def test_each_force_field_reads_its_own_streaming_count():
    """Two force fields on one model: each call reads its own edge count, with no bind_streaming_count."""
    calc, atoms = calculator(), structure()
    streaming = _FakeStreaming()
    calc.models[0].add_module("fake_streaming", streaming)
    original_forward = type(calc.models[0]).forward

    def forward(self, *args, **kwargs):
        streaming()
        return original_forward(self, *args, **kwargs)

    sorted_neighbours = NeighbourConfig(skin=SKIN, sort_receivers="on", gpu_neighbour_list="off")
    # 1a's energy sum: the float64 forward is built from MACE's own source, not this patched one.
    with mock.patch.object(type(calc.models[0]), "forward", forward):
        first = MACEForceField(calc, atoms, neighbours=sorted_neighbours, float64_energy=False)
        second = MACEForceField(calc, structure(rattle=0.2, seed=3), neighbours=sorted_neighbours,
                                float64_energy=False)
        assert first.model is second.model  # one shared model
        assert first.info()["streaming_interactions"] == 1
        for field in (first, second, first, second):
            field(torch.tensor(atoms.positions))
            assert streaming.seen[-1] is field.edge_count
    assert first.edge_count is not second.edge_count


def test_streaming_refuses_unsorted_edges():
    calc, atoms = calculator(), structure()
    registry = reg.default_registry()
    enabled = ("factor_species", "fold_cutoff", "streaming_conv")
    resolution = reg.Resolution(tuple(reg.Decision(name, "on" if name in enabled else "off", name in enabled,
                                                   "on" if name in enabled else "off")
                                      for name in registry.order()))
    with pytest.raises(ValueError, match="receiver-sorted"):
        MACEForceField(calc, atoms, neighbours=NEIGHBOURS, rewrites=resolution, registry=registry)


def test_streaming_modules_without_sorted_edges_are_refused():
    """Also after the rewrites: any interaction with streaming kernels needs receiver-sorted edges."""
    calc, atoms = calculator(), structure()
    calc.models[0].add_module("fake_streaming", _FakeStreaming())
    with pytest.raises(ValueError, match="receiver-sorted"):
        MACEForceField(calc, atoms, neighbours=NEIGHBOURS)


# --------------------------------------------------------------------------- cuEquivariance on the CPU


def _cueq_on_cpu(naive):
    from cuequivariance_torch.primitives import segmented_polynomial
    from mace.cli.convert_e3nn_cueq import run

    calc = calculator()
    with fc.naive_cueq() if naive else contextlib.nullcontext():
        model = run(calc.models[0], device="cpu", return_model=True)
    calc.models = [model]
    return calc, segmented_polynomial.HAS_CUE_OPS


@pytest.mark.skipif(not HAS_CUEQ, reason="cuequivariance_torch is not installed")
def test_cpu_cueq_model_with_naive_kernels_runs():
    calc, atoms = calculator(), structure()
    expected = evaluate(MACEForceField(calc, atoms, neighbours=NEIGHBOURS), torch.tensor(atoms.positions))
    cueq, _ = _cueq_on_cpu(naive=True)
    assert fc.cpu_cueq_problems(cueq.models[0]) == []
    field = MACEForceField(cueq, atoms, neighbours=NEIGHBOURS)
    assert field.info()["backend"] == "cueq"
    for a, b in zip(evaluate(field, torch.tensor(atoms.positions)), expected):
        assert relative(a, b) <= 1e-12


@pytest.mark.skipif(not HAS_CUEQ, reason="cuequivariance_torch is not installed")
def test_cpu_cueq_model_with_cuda_kernels_is_refused():
    cueq, has_ops = _cueq_on_cpu(naive=False)
    if not has_ops:
        pytest.skip("cuequivariance_ops_torch is not importable: cuEq converts with its naive kernels anyway")
    problems = fc.cpu_cueq_problems(cueq.models[0])
    assert problems and all("naive" in problem for problem in problems)
    from fastmace.model.loading import BackendError

    with pytest.raises(BackendError, match="naive"):
        MACEForceField(cueq, structure(), neighbours=NEIGHBOURS)


# --------------------------------------------------------------------------- PolarMACE


def _polar_available():
    from _support.tiny_models import polar_available

    return polar_available()


needs_polar = pytest.mark.skipif(not (_polar_available() and HAS_CUEQ),
                                 reason="PolarMACE needs graph_longrange and cuequivariance_torch")


def polar_calculators():
    """(e3nn calculator, calculator holding its cuEq conversion with the naive kernels) of a tiny PolarMACE."""
    from mace.cli.convert_e3nn_cueq import run

    calc, cueq = calculator("polar", [11, 17]), calculator("polar", [11, 17])
    with fc.naive_cueq():
        cueq.models = [run(cueq.models[0], device="cpu", return_model=True)]
    return calc, cueq


@pytest.mark.skipif(not _polar_available(), reason="PolarMACE needs graph_longrange")
def test_polar_route_follows_longrange_polar():
    """PolarMACE runs only on the GPU engine's backends (fastmace.longrange.polar.route_support)."""
    from fastmace.longrange.polar import PolarUnavailableError

    with pytest.raises(PolarUnavailableError, match="backend 'e3nn'"):
        MACEForceField(calculator("polar", [11, 17]), structure(oxygen=False), neighbours=NEIGHBOURS)


@needs_polar
def test_polar_force_field_enables_its_own_copy():
    from fastmace.longrange import polar

    calc, cueq = polar_calculators()
    atoms = structure(oxygen=False)
    field = MACEForceField(cueq, atoms, neighbours=NEIGHBOURS, rewrites=only())
    assert field.model is not cueq.models[0] and polar.get_state(cueq.models[0]) is None
    assert polar.get_state(field.model) is not None and field.info()["polar"]["k_vectors"] > 0
    assert "polar.overflow" in field.buffers()
    positions = moved(atoms, 0.05)
    reference(calc, atoms, positions)  # throwaway: the first model of a process rounds differently
    energy, forces = evaluate(field, positions, padded=True)
    expected_energy, expected_forces = reference(calc, atoms, positions)
    # cuEq's naive kernels against e3nn, and graph_longrange's k-vectors summed with zeros
    # interleaved (fastmace.longrange.polar): equal to rounding.
    np.testing.assert_allclose(float(energy[0]), expected_energy, rtol=1e-12, atol=0)
    np.testing.assert_allclose(forces.numpy(), expected_forces, rtol=1e-12, atol=1e-12)


@needs_polar
def test_polar_charge_and_spin_are_validated():
    _, cueq = polar_calculators()
    atoms = structure(oxygen=False)
    with pytest.raises(ValueError, match="impossible"):
        MACEForceField(cueq, atoms, neighbours=NEIGHBOURS, spin_multiplicity=2)
    field = MACEForceField(cueq, atoms, neighbours=NEIGHBOURS, charge=0, spin_multiplicity=3)
    assert (field.charge, field.spin_multiplicity) == (0, 3)


def test_charge_and_spin_are_refused_without_an_embedding_or_polar():
    with pytest.raises(ValueError, match="charge"):
        MACEForceField(calculator(), structure(), neighbours=NEIGHBOURS, charge=1)


# --------------------------------------------------------------------------- compiled models and the cache


def test_cache_is_an_object_and_keeps_its_source():
    cache = fc.CompiledModelCache()
    source, compiled = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    assert cache.get(("k",), source) is None
    entry = cache.put(("k",), source, compiled, applied=("x",))
    assert cache.get(("k",), source) is entry and entry.model is compiled and entry.applied == ("x",)
    assert cache.get(("k",), torch.nn.Linear(1, 1)) is None  # another model under a reused id
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert "_COMPILED_MODELS" not in vars(fc)  # no module-level cache (SPEC §3.2)


def test_compiled_route_uses_the_cache_passed_in():
    """CompiledMACEForcesTests: compiled once per model and cache, then reused; another cache compiles again."""
    calc, atoms = calculator(), structure()
    engine = EngineConfig(compile_mode="default")
    calls = []

    def fake_compile(model, mode, **kwargs):
        calls.append((mode, kwargs["accelerated"], kwargs["oeq"]))
        private = fc.private_copy(model)
        if kwargs.get("prepare") is not None:
            kwargs["prepare"](private)
        return private

    cache = fc.CompiledModelCache()
    with mock.patch.object(fc, "compile_model", side_effect=fake_compile):
        first = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=engine, compile_cache=cache)
        again = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=engine, compile_cache=cache)
        other = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=engine, compile_cache=fc.CompiledModelCache())
        factored = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=engine, compile_cache=cache,
                                  rewrites=only("factor_species"))
    assert calls == [("default", False, False)] * 3
    assert again.model is first.model and other.model is not first.model and factored.model is not first.model
    assert first.model is not calc.models[0] and len(cache) == 2
    assert first.training is True and [e.name for e in factored.applied] == ["factor_species"]


def test_hybrid_compile_differentiates_outside_the_graph():
    """CompiledMACEForcesTests.test_hybrid_compile_uses_oeq_autograd_policy_and_eager_force_derivative."""
    calc, atoms = calculator(), structure()
    calc._enable_oeq = True
    calls = []

    def fake_compile(model, mode, **kwargs):
        calls.append((mode, kwargs["accelerated"], kwargs["oeq"]))
        return model

    with mock.patch.object(fc, "compile_model", side_effect=fake_compile):
        field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EngineConfig(compile_mode="default"))
    assert calls == [("default", True, True)] and field.training is False


def _fake_mace_compile():
    compile_module = ModuleType("mace.tools.compile")
    compile_module.configure_autograd_for_compile = mock.Mock()
    compile_module.disable_e3nn_codegen = mock.MagicMock()
    compile_module.simplify = mock.Mock(side_effect=lambda value: value)
    compile_module.prepare = mock.Mock()
    scripts_module = ModuleType("mace.tools.scripts_utils")
    scripts_module.extract_model = mock.Mock()
    return compile_module, scripts_module


def test_compile_model_selects_the_autograd_policy_and_dynamo_limit():
    """CompiledMACEForcesTests.test_compile_model_selects_distinct_autograd_policy_for_hybrid."""
    model = torch.nn.Linear(2, 1, bias=False)
    compile_module, scripts_module = _fake_mace_compile()
    fake_dynamo = SimpleNamespace(config=SimpleNamespace(cache_size_limit=8))
    prepared = []
    with mock.patch.object(fc, "install_engine_patches") as install, \
            mock.patch.object(torch, "_dynamo", fake_dynamo, create=True), \
            mock.patch.object(torch, "compile", side_effect=lambda value, **kwargs: value), \
            mock.patch.dict(sys.modules, {"mace.tools.compile": compile_module,
                                          "mace.tools.scripts_utils": scripts_module,
                                          "mace.modules.wrapper_ops": _fake_wrapper_ops()}):
        private = fc.compile_model(model, "default", accelerated=True, oeq=True, prepare=prepared.append)
        compile_module.configure_autograd_for_compile.assert_called_with(allow_autograd=False)
        fc.compile_model(model, "default", accelerated=True, oeq=False, engine=EngineConfig(dynamo_cache_size_limit=99))
        compile_module.configure_autograd_for_compile.assert_called_with(allow_autograd=True)
    assert prepared == [private] and private is not model
    assert not any(p.requires_grad for p in private.parameters())
    assert fake_dynamo.config.cache_size_limit == 99
    assert [c.kwargs["oeq"] for c in install.call_args_list] == [True, False]


def test_oeq_wrapper_and_layouts_survive_compile_model():
    """CompiledMACEForcesTests.test_oeq_wrapper_and_layouts_survive_private_copy_serialization."""
    wrapper_ops = _fake_wrapper_ops()
    model = _AdapterModel()
    wrapper_ops.with_oeq_conv_fusion(model.oeq, _Layout(10), _Layout(20))
    wrapper_ops.with_cueq_conv_fusion(model.cueq)
    compile_module, scripts_module = _fake_mace_compile()
    fake_dynamo = SimpleNamespace(config=SimpleNamespace(cache_size_limit=8))
    with mock.patch.object(fc, "install_engine_patches"), \
            mock.patch.object(torch, "_dynamo", fake_dynamo, create=True), \
            mock.patch.object(torch, "compile", side_effect=lambda value, **kwargs: value), \
            mock.patch.dict(sys.modules, {"mace.tools.compile": compile_module,
                                          "mace.tools.scripts_utils": scripts_module,
                                          "mace.modules.wrapper_ops": wrapper_ops}):
        private = fc.compile_model(model, "default", accelerated=True, oeq=True)
    one, edge_index = torch.ones(1, 1), torch.tensor([[0], [0]])
    assert float(private.oeq(one, one, one, edge_index)) == 33.0
    assert float(private.cueq(one, one, one, edge_index)) == 15.0


@pytest.mark.slow
def test_compiled_forces_match_uncompiled():
    """CompiledMACEForcesTests.test_compiled_forces_match_uncompiled (torch.compile on the CPU)."""
    pytest.importorskip("torch._inductor")
    calc, atoms = calculator(), structure()
    plain = MACEForceField(calc, atoms, neighbours=NEIGHBOURS)
    cache = fc.CompiledModelCache()
    compiled = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, engine=EngineConfig(compile_mode="default"),
                              compile_cache=cache)
    for shift in (0.0, 0.05):
        positions = moved(atoms, shift, seed=7)
        for a, b in zip(evaluate(compiled, positions), evaluate(plain, positions)):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------- allocator


def test_configure_allocator_sets_expandable_segments_only_when_unset_and_on_new_torch():
    """AllocatorConfigTests.test_sets_expandable_segments_only_when_unset_and_on_new_torch."""
    environ = {}
    assert fc.configure_allocator(environ=environ, torch_version="2.14.1+cu130") == fc.ALLOCATOR_SETTINGS
    assert environ == {"PYTORCH_CUDA_ALLOC_CONF": fc.ALLOCATOR_SETTINGS}
    environ = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}  # the user's setting wins
    assert fc.configure_allocator(environ=environ, torch_version="2.14.1+cu130") == "max_split_size_mb:128"
    assert fc.configure_allocator(environ={}, torch_version="2.5.1+cu124") is None  # untested with CUDA graphs
    assert fc.configure_allocator(enabled=False, environ={}, torch_version="2.14.1") is None


def test_construction_never_moves_the_batchs_edges(monkeypatch):
    """The batch's edge arrays (MACE's own neighbour list) are replaced by the skin list's before anything reads
    them, so the construction keeps them off the device: here they are meta tensors, which cannot be copied."""
    calc, atoms = calculator(), structure()
    batch_of = calc._atoms_to_batch

    class Meta:
        def __init__(self, batch):
            self.batch = batch

        def to_dict(self):
            out = self.batch.to_dict()
            for key in ("edge_index", "shifts", "unit_shifts"):
                out[key] = out[key].to("meta")
            return out

    reference = MACEForceField(calculator(), atoms, neighbours=NEIGHBOURS, compute_stress=True,
                               rewrites=only())
    reference.evaluate(atoms, stress=True)  # TorchScript's first call in a process rounds differently
    expected = reference.evaluate(atoms, stress=True)
    monkeypatch.setattr(calc, "_atoms_to_batch", lambda a: Meta(batch_of(a)))
    field = MACEForceField(calc, atoms, neighbours=NEIGHBOURS, compute_stress=True, rewrites=only())
    got = field.evaluate(atoms, stress=True)
    assert got[0] == expected[0]
    assert np.array_equal(got[1], expected[1]) and np.array_equal(got[2], expected[2])
