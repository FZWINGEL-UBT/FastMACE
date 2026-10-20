"""Instance isolation on the CPU (plan task 2-T08; SPEC §8.7, §12.2 "Isolation").

Several force fields and engines built from one model live in one process and are used in any
order, A -> B -> A, each giving what a fresh instance gives. Two crystals (NaCl and NbCl5) on the
tiny float64 models of ``_support.tiny_models``; the released models and the hybrid and polar
GPU paths are in tests/gpu/test_isolation_gpu.py. The audit of every shared object is
docs/design/isolation.md.

- each force field owns its batch, positions leaf, neighbour list and cell inputs (no tensor in
  common, even with one shared model);
- evaluations interleaved A -> B -> A equal fresh force fields bitwise: the eager source model
  shared as it is, private rewritten copies, and one compiled model shared through the cache;
- a module buffer swapped between calls (the streaming edge counter) is read as an input by
  dynamo: the right count every call, and no recompilation after the first A -> B round;
- the one shared state left open (finding 1, plan task 2-T08b): a failing ``single_graph_strain``
  self-check undoes the process-wide patch that earlier force fields were built with (pinned);
- PolarMACE: each force field enables its own copy, and A -> B -> A on two cells equals fresh;
- two engines stepped in turn equal each engine run alone, across neighbour rebuilds.
"""

import importlib.util
from unittest import mock

import numpy as np
import pytest
import torch

from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.langevin import Langevin
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField
from fastmace.neighbors.buckets import edge_bucket, round_up_edges
from fastmace.rewrites import registry as reg
from fastmace.state import SimulationState

pytest.importorskip("mace")

SKIN = 0.5
NEIGHBOURS = NeighbourConfig(skin=SKIN, sort_receivers="off", gpu_neighbour_list="off")
Z_VALUES = [11, 17, 41]  # Na, Cl, Nb: both crystals on one model
MH1_PACKAGE = ("factor_species", "fold_cutoff", "density_first", "joint_radial_input", "fused_folded_backward",
               "reassociate_first", "segmented_first_sum")
HAS_CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None


def crystals():
    """The two crystals: NaCl 2x2x2 (64 atoms) and NbCl5 (36 atoms), rattled."""
    from _support.structures import nacl, nbcl5

    return {"A": nacl(repeat=2, rattle=0.1, seed=8), "B": nbcl5(repeat=1, rattle=0.1, seed=4)}


def calculator(family="mh1", z_values=Z_VALUES):
    from _support.tiny_models import FAMILIES, tiny_mace_calculator

    return tiny_mace_calculator(z_values, **FAMILIES[family])


def only(*names):
    return {name: "on" if name in names else "off" for name in reg.default_registry().names()}


def moved(atoms, shift, seed):
    """Positions moved by ``shift`` x a standard normal (below skin / 2: no rebuild needed)."""
    return torch.tensor(atoms.positions + shift * np.random.default_rng(seed).standard_normal(atoms.positions.shape))


def graph_level(field, positions, padded=True):
    """The engine's calls between replays: compact, use (padded) edges, evaluate; copies of the outputs."""
    field.compact(positions)
    count = int(field.edge_count)
    num_edges = round_up_edges(count, field.edge_capacity, edge_bucket(count, 0.25, 256)) if padded else count
    field.use_edges(num_edges)
    out = field(positions)
    values = [out.energy.detach().clone(), out.forces.detach().clone()]
    if out.stress is not None:
        values.append(out.stress.detach().clone())
    return values


def schedule(atoms):
    """Two position sets per crystal, both within half the skin of the built list."""
    return {name: [moved(structure, 0.03, seed) for seed in (1, 2)] for name, structure in atoms.items()}


def fresh_values(build, atoms, positions):
    """Each crystal on its own fresh force field, every position set in turn: {name: [values, ...]}."""
    out = {}
    for name, structure in atoms.items():
        field = build(structure)
        out[name] = [graph_level(field, x) for x in positions[name]]
        del field
    return out


def interleaved_values(build, atoms, positions):
    """A and B built together, then evaluated A -> B -> A -> B: {name: [values, ...]} and the fields."""
    fields = {name: build(structure) for name, structure in atoms.items()}
    out = {name: [] for name in atoms}
    for index in range(2):
        for name in ("A", "B"):
            out[name].append(graph_level(fields[name], positions[name][index]))
    # And A once more after B, at A's first positions (A -> B -> A).
    out["A"].append(graph_level(fields["A"], positions["A"][0]))
    return out, fields


def assert_equal(got, expected, label):
    for index, (values, reference) in enumerate(zip(got, expected)):
        for part, (value, want) in enumerate(zip(values, reference, strict=True)):
            assert torch.equal(value, want), f"{label}: evaluation {index}, output {part}"


@pytest.fixture(scope="module", autouse=True)
def _throwaway():
    """The first model evaluation of a process can round differently (tests/cpu/test_forces.py)."""
    atoms = crystals()["A"]
    field = MACEForceField(calculator(), atoms, neighbours=NEIGHBOURS)
    graph_level(field, torch.tensor(atoms.positions))


# --------------------------------------------------------------------------- per-instance state


def test_force_fields_on_one_model_share_no_state_tensor():
    """Two force fields on one (shared) model: no buffer, batch tensor or positions leaf in common."""
    calc, atoms = calculator(), crystals()
    first, second = (MACEForceField(calc, atoms[name], neighbours=NEIGHBOURS, compute_stress=True)
                     for name in ("A", "B"))
    assert first.model is second.model is calc.models[0]  # eager, no rewrites: the source itself
    assert first.neighbours is not second.neighbours and first.data is not second.data

    def storages(field):
        tensors = list(field.buffers().values()) + [v for v in field.data.values() if torch.is_tensor(v)]
        return {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}

    assert storages(first).isdisjoint(storages(second))


# --------------------------------------------------------------------------- A -> B -> A evaluations


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("rewrites", [None, MH1_PACKAGE], ids=["source", "mh1-copy"])
def test_a_b_a_evaluations_equal_fresh_force_fields(rewrites, stress):
    """Eager: the source model shared as it is (no rewrites), or each field's own rewritten copy."""
    calc, atoms = calculator(), crystals()
    positions = schedule(atoms)
    request = None if rewrites is None else only(*rewrites)

    def build(structure):
        return MACEForceField(calc, structure, neighbours=NEIGHBOURS, rewrites=request, compute_stress=stress)

    expected = fresh_values(build, atoms, positions)
    got, fields = interleaved_values(build, atoms, positions)
    assert (fields["A"].model is fields["B"].model) is (rewrites is None)
    assert_equal(got["A"], expected["A"] + expected["A"][:1], "A")
    assert_equal(got["B"], expected["B"], "B")


def test_a_b_a_on_one_compiled_model_from_the_cache():
    """Two crystals share one compiled model through the cache (same key), A -> B -> A equals fresh."""
    calc, atoms = calculator(), crystals()
    positions = schedule(atoms)
    engine = EngineConfig(compile_mode="default")
    compiled = []

    def fake_compile(model, mode, **kwargs):
        private = fc.private_copy(model)
        if kwargs.get("prepare") is not None:
            kwargs["prepare"](private)
        compiled.append(private)
        return private

    with mock.patch.object(fc, "compile_model", side_effect=fake_compile):
        expected = fresh_values(lambda structure: MACEForceField(
            calc, structure, neighbours=NEIGHBOURS, engine=engine, compile_cache=fc.CompiledModelCache()),
            atoms, positions)
        cache = fc.CompiledModelCache()
        got, fields = interleaved_values(lambda structure: MACEForceField(
            calc, structure, neighbours=NEIGHBOURS, engine=engine, compile_cache=cache), atoms, positions)
    assert fields["A"].model is fields["B"].model and len(cache) == 1 and len(compiled) == 3
    assert_equal(got["A"], expected["A"] + expected["A"][:1], "A")
    assert_equal(got["B"], expected["B"], "B")


# --------------------------------------------------------------------------- a swapped buffer under dynamo


class _Counted(torch.nn.Module):
    """Sums the first ``streaming_count`` rows, reading the count as the streaming interactions do."""

    def __init__(self):
        super().__init__()
        self.register_buffer("streaming_count", torch.tensor(torch.iinfo(torch.int64).max), persistent=False)

    def forward(self, x):
        position = torch.arange(x.shape[0], device=x.device)
        return torch.where((position < self.streaming_count)[:, None], x, torch.zeros_like(x)).sum(dim=0)


def test_swapped_count_buffer_is_an_input_to_compiled_code():
    """MACEForceField._bind_streaming_count swaps a buffer of a shared compiled model before each call.

    Dynamo must read the buffer as an input: each field's own count every call (never the count
    the code was traced with), and no recompilation once A and B have each run once.
    """
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    module = _Counted()
    compiled = torch.compile(module, backend=counter, fullgraph=False)
    x = torch.arange(30, dtype=torch.float64).reshape(10, 3)
    counts = {"A": torch.tensor(7), "B": torch.tensor(4)}
    frames = []
    for name in ("A", "B", "A", "B", "A"):
        module.register_buffer("streaming_count", counts[name], persistent=False)
        assert torch.equal(compiled(x), x[: int(counts[name])].sum(dim=0)), name
        frames.append(counter.frame_count)
    counts["A"].fill_(9)  # in place, as compact() writes it inside a graph
    module.register_buffer("streaming_count", counts["A"], persistent=False)
    assert torch.equal(compiled(x), x[:9].sum(dim=0))
    assert frames[1:] == [frames[1]] * 4 and counter.frame_count == frames[1]
    torch._dynamo.reset()


# --------------------------------------------------------------------------- the one open shared state


def test_failing_strain_selfcheck_undoes_the_patch_of_earlier_force_fields():
    """Pins today's behaviour (docs/design/isolation.md, finding 1), which 2-T08b changes.

    ``single_graph_strain`` is a process-wide patch. B's failing self-check restores MACE's
    function (``Registry._selfcheck_failure``), and A, built with the record, then runs MACE's
    function: energy and forces unchanged, stress at rounding level, ``A.applied`` stale.
    2-T08b makes the strain path per model; then the patch is not undone and this test changes.
    """
    import dataclasses

    from fastmace import _patches
    from fastmace.errors import SelfCheckError

    def failing(original, rewritten, context):
        return reg.CheckResult(False, 1.0, 1e-10, 1, "injected failure (test)", "float64")

    registry = reg.default_registry()
    broken = reg.Registry()
    for name in registry.names():
        record = registry[name]
        if name == "single_graph_strain":
            record = dataclasses.replace(record, selfcheck=failing)
        broken.register(record, registry.config_field(name))
    calc, atoms = calculator(), crystals()
    request = only("single_graph_strain")
    installed_before = _patches.record_of("single_graph_strain") is not None
    try:
        first = MACEForceField(calc, atoms["A"], rewrites=request, neighbours=NEIGHBOURS, compute_stress=True)
        assert [entry.name for entry in first.applied] == ["single_graph_strain"]
        assert _patches.record_of("single_graph_strain") is not None
        before = first.evaluate(atoms["A"], stress=True)
        with pytest.raises(SelfCheckError, match="single_graph_strain"):
            MACEForceField(calc, atoms["B"], rewrites=request, neighbours=NEIGHBOURS, compute_stress=True,
                           registry=broken)
        assert _patches.record_of("single_graph_strain") is None  # undone for the whole process
        assert [entry.name for entry in first.applied] == ["single_graph_strain"]  # A's record is stale
        after = first.evaluate(atoms["A"], stress=True)
        assert after[0] == before[0] and np.array_equal(after[1], before[1])
        np.testing.assert_allclose(after[2], before[2], rtol=1e-12, atol=1e-16)
    finally:
        if installed_before:
            _patches.set_single_graph_strain(True)
        else:
            _patches.restore("single_graph_strain")


# --------------------------------------------------------------------------- PolarMACE


def _polar_available():
    from _support.tiny_models import polar_available

    return polar_available()


@pytest.mark.skipif(not (_polar_available() and HAS_CUEQ),
                    reason="PolarMACE needs graph_longrange and cuequivariance_torch")
def test_polar_a_b_a_equals_fresh_force_fields():
    """Each polar force field enables its own copy: A's k-vectors survive B's build and B's calls."""
    from mace.cli.convert_e3nn_cueq import run

    from _support.structures import nacl
    from fastmace.longrange import polar

    cueq = calculator("polar", [11, 17])
    with fc.naive_cueq():
        cueq.models = [run(cueq.models[0], device="cpu", return_model=True)]
    small = nacl(repeat=2, rattle=0.1, seed=8)
    large = small.copy()
    large.set_cell(small.cell.array * 1.04, scale_atoms=True)  # another cell, so another k-vector set
    atoms = {"A": small, "B": large}
    positions = schedule(atoms)

    def build(structure):
        return MACEForceField(cueq, structure, neighbours=NEIGHBOURS, rewrites=only())

    expected = fresh_values(build, atoms, positions)
    got, fields = interleaved_values(build, atoms, positions)
    states = [polar.get_state(fields[name].model) for name in ("A", "B")]
    assert fields["A"].model is not fields["B"].model and states[0] is not states[1]
    assert not np.array_equal(states[0].cell, states[1].cell)
    assert not any(bool(state.overflow) for state in states)
    assert_equal(got["A"], expected["A"] + expected["A"][:1], "A")
    assert_equal(got["B"], expected["B"], "B")


# --------------------------------------------------------------------------- engines in turn


ENGINE = EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu")
ENGINE_NEIGHBOURS = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off")


def build_engine(calc, structure, seed):
    field = MACEForceField(calc, structure, neighbours=ENGINE_NEIGHBOURS)
    state = SimulationState.from_atoms(structure, "cpu", seed)
    dyn = Langevin(state, IntegratorConfig(timestep_fs=2.0, friction=0.02, fixcm=True, noise_batch=7))
    dyn.set_temperature(1500.0)
    return GraphEngine(ENGINE, field, dyn, state, neighbours=ENGINE_NEIGHBOURS, capture=False,
                       constraints=constraints.ProcessRecord())


def rows(engine):
    return {key: value.copy() for key, value in engine.sync().items()}


def test_engines_stepped_in_turn_equal_engines_alone():
    """Two engines on one shared model, stepped A, B, A, B, ...: each equals its own run alone."""
    calc, atoms = calculator(), crystals()
    steps, seeds = 40, {"A": 7, "B": 11}
    alone = {}
    for name in ("A", "B"):
        engine = build_engine(calc, atoms[name], seeds[name])
        for _ in range(steps):
            engine.step()
        alone[name] = (rows(engine), engine.force_field.neighbours.rebuilds)
        del engine
    engines = {name: build_engine(calc, atoms[name], seeds[name]) for name in ("A", "B")}
    assert engines["A"].force_field.model is engines["B"].force_field.model
    for _ in range(steps):
        for name in ("A", "B"):
            engines[name].step()
    for name in ("A", "B"):
        got = rows(engines[name])
        assert got.keys() == alone[name][0].keys()
        for key in got:
            np.testing.assert_array_equal(got[key], alone[name][0][key], err_msg=f"{name}: {key}")
        rebuilds = engines[name].force_field.neighbours.rebuilds
        assert rebuilds == alone[name][1] and rebuilds > 1, name  # the steps crossed a rebuild
