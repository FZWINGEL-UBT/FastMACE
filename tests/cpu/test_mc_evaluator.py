"""The forward-only energy path and the MC evaluator on the CPU (plan task 3-T07; SPEC §8.1 "Energy-only
graph", §10.4 "Evaluation", "Baseline energy", "Energy precision", §7.3).

- ``MACEForceField.energy_only`` equals the force path's energy: bitwise in eager float64, within
  rounding compiled; for MH-1, generic models, OMOL-0's embedding readout and PolarMACE;
- its output carries no ``grad_fn`` and nothing requires a gradient;
- a force evaluation after it is unaffected (eager and compiled);
- a float64 copy that follows a float32 force field's edges equals a float64 force field's energy,
  and follows species changes through the derived-state refreshes;
- the MC evaluator's baseline after MD equals a fresh evaluation, never the MD energy, and its
  energy cache is tagged with evaluator, precision and state version.
"""

import copy
import dataclasses
import importlib.util

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig, PrecisionConfig
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField

pytest.importorskip("mace")
torch.serialization.add_safe_globals([slice])

NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="off")
EAGER = EngineConfig(compile_mode="none")
COMPILED = EngineConfig(compile_mode="default")
HAS_CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None
needs_polar = pytest.mark.skipif(not (polar_available() and HAS_CUEQ),
                                 reason="PolarMACE needs graph_longrange and cuequivariance_torch")
needs_cueq = pytest.mark.skipif(not HAS_CUEQ, reason="needs cuequivariance_torch")

# Family -> (Z values, extra constructor options): MH-1, a generic residual model, OMOL-0's embedding
# readout energy and PolarMACE with its local electron energy.
CASES = {
    "mh1": ([8, 11, 17, 26], {}),
    "residual": ([8, 11, 17, 26], {}),
    "omol": ([8, 11, 17, 26], {"use_embedding_readout": True}),
    "polar": ([11, 17], {"add_local_electron_energy": True}),
}
FAMILY_PARAMS = [pytest.param(name, marks=needs_polar) if name == "polar" else name for name in CASES]


def model64(family):
    z_values, extra = CASES[family]
    return tiny_mace_model(z_values, **{**FAMILIES[family], **extra})


def calculator(model, dtype="float64"):
    """MACE's calculator of a copy of ``model`` in ``dtype``; PolarMACE converted to cuEq (naive kernels)."""
    from mace.calculators import MACECalculator

    calc = MACECalculator(models=[copy.deepcopy(model)], device="cpu", default_dtype=dtype)
    if type(model).__name__ == "PolarMACE":
        from mace.cli.convert_e3nn_cueq import run

        with fc.naive_cueq():
            calc.models = [run(calc.models[0], device="cpu", return_model=True)]
    return calc


def structure(family, rattle=0.1, seed=8):
    atoms = nacl(repeat=2, rattle=rattle, seed=seed)
    if CASES[family][0][0] == 8:
        atoms.numbers[3] = 8
    return atoms


def field(calc, atoms, **options):
    options.setdefault("neighbours", NEIGHBOURS)
    options.setdefault("engine", EAGER)
    return MACEForceField(calc, atoms, **options)


def positions_of(atoms):
    return torch.tensor(atoms.positions, dtype=torch.float64)


# --------------------------------------------------------------------------- the energy-only path


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_energy_only_equals_the_force_path_energy_bitwise(family):
    atoms = structure(family)
    ff = field(calculator(model64(family)), atoms)
    positions = positions_of(atoms)
    expected = ff(positions).energy.detach().clone()
    got = ff.energy_only(positions)
    assert got.dtype == torch.float64 and got.shape == expected.shape
    assert torch.equal(got, expected)


@pytest.mark.parametrize("family", ["mh1", "residual"])
def test_energy_only_stress_force_field_equals_the_stress_path_energy(family):
    atoms = structure(family)
    ff = field(calculator(model64(family)), atoms, compute_stress=True)
    positions = positions_of(atoms)
    expected = ff(positions).energy.detach().clone()
    assert torch.equal(ff.energy_only(positions), expected)


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_energy_only_has_no_grad_fn(family):
    atoms = structure(family)
    ff = field(calculator(model64(family)), atoms, compute_stress=family != "polar")
    out = ff.energy_only(positions_of(atoms))
    assert out.grad_fn is None and not out.requires_grad


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_a_force_evaluation_after_energy_only_is_unaffected(family):
    atoms = structure(family)
    calc = calculator(model64(family))
    positions = positions_of(atoms)
    # e3nn's TorchScript-scripted modules (OMOL-0's tiny model reaches them) run the profiling graph in the
    # first call of a process and the optimised one afterwards; the two round one force differently in the
    # last bit (4e-19 eV/A here; identical with torch._C._set_graph_executor_optimize(False)). Whichever test
    # an xdist worker runs first made the comparison flaky, so both compared calls follow a warm-up call.
    # The GPU plans' cuEq and compiled paths have no TorchScript.
    field(calc, atoms)(positions)
    fresh = field(calc, atoms)(positions)
    ff = field(calc, atoms)
    ff.energy_only(positions)
    ff.energy_only(positions + 0.01)
    out = ff(positions)
    assert torch.equal(out.energy, fresh.energy) and torch.equal(out.forces, fresh.forces)


@pytest.mark.slow
@pytest.mark.parametrize("family", ["mh1", "residual"])
def test_compiled_energy_only_matches_and_leaves_the_force_path_alone(family):
    atoms = structure(family)
    calc = calculator(model64(family))
    positions = positions_of(atoms)
    ff = field(calc, atoms, engine=COMPILED)
    before = ff(positions)
    before = (before.energy.detach().clone(), before.forces.detach().clone())
    got = ff.energy_only(positions)
    assert got.grad_fn is None and not got.requires_grad
    np.testing.assert_allclose(got.numpy(), before[0].numpy(), rtol=1e-13, atol=0)
    after = ff(positions)
    # Compiled CPU forces vary between any two calls by about 1e-18 (energy-only or not); eager is bitwise above.
    assert torch.equal(after.energy, before[0])
    np.testing.assert_allclose(after.forces.numpy(), before[1].numpy(), rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- the float64 copy


def float32_field(family, atoms, **options):
    return field(calculator(model64(family), "float32"), atoms, **options)


@pytest.mark.parametrize("family", ["mh1", "residual", "omol"])
@pytest.mark.parametrize("stress", [False, True])
def test_float64_copy_follows_a_float32_force_field(family, stress):
    from fastmace.mc.evaluator import float64_copy

    atoms = structure(family)
    leader = float32_field(family, atoms, compute_stress=stress)
    copy64 = float64_copy(leader, calculator(model64(family)))
    assert copy64.dtype == torch.float64 and copy64.info()["follows"] == "float32"
    positions = positions_of(atoms)
    reference = field(calculator(model64(family)), atoms).energy_only(positions)
    np.testing.assert_allclose(copy64.energy_only(positions).numpy(), reference.numpy(), rtol=1e-13, atol=0)
    # The copy reads the leader's compacted edges, also padded and after the leader compacts again.
    moved = positions + 0.02 * torch.from_numpy(np.random.default_rng(3).standard_normal(positions.shape))
    leader.compact(moved)
    count = int(leader.edge_count)
    leader.use_edges(count + 7)
    copy64.use_edges(count + 7)
    at_moved = atoms.copy()
    at_moved.positions = moved.numpy()
    expected = field(calculator(model64(family)), at_moved).energy_only(moved)
    np.testing.assert_allclose(copy64.energy_only(moved).numpy(), expected.numpy(), rtol=1e-12, atol=0)


def test_float64_copy_follows_species_changes_through_the_refreshes():
    from fastmace.events import DerivedStateRegistry
    from fastmace.mc.evaluator import float64_copy

    family = "mh1"
    atoms = structure(family)
    leader = float32_field(family, atoms, species=[8, 11, 17, 26])
    copy64 = float64_copy(leader, calculator(model64(family)))
    registry = DerivedStateRegistry()
    leader.register_derived(registry)
    copy64.register_derived(registry, prefix="mc_float64")
    assert "mc_float64.node_attrs" in registry
    swapped = atoms.copy()
    swapped.numbers[[0, 1]] = swapped.numbers[[1, 0]]
    swapped.numbers[5] = 26
    leader.atomic_numbers.copy_(torch.as_tensor(swapped.numbers))
    for name, buffer, refresh, _ in leader._derived():  # what 3-T03's registry.refresh does
        refresh()
    for name, buffer, refresh, _ in copy64._derived():
        refresh()
    positions = positions_of(atoms)
    reference = field(calculator(model64(family)), swapped, species=[8, 11, 17, 26]).energy_only(positions)
    np.testing.assert_allclose(copy64.energy_only(positions).numpy(), reference.numpy(), rtol=1e-13, atol=0)


def test_float64_copy_refuses_another_structure():
    from fastmace.mc.evaluator import float64_copy

    atoms = structure("mh1")
    leader = float32_field("mh1", atoms)
    with pytest.raises(ValueError, match="float32"):
        float64_copy(field(calculator(model64("mh1")), atoms), calculator(model64("mh1")))
    other = calculator(tiny_mace_model([8, 11, 17], **FAMILIES["mh1"]))
    with pytest.raises(ValueError, match="species|E0 table"):  # another model
        float64_copy(leader, other)


# --------------------------------------------------------------------------- the MC energy cache


def test_energy_cache_is_tagged_with_evaluator_precision_and_version():
    from fastmace.mc.evaluator import EnergyCache

    cache = EnergyCache()
    assert cache.get(("e", "float32", 0)) is None
    cache.store(("e", "float32", 0), torch.tensor([-1.5], dtype=torch.float64))
    assert float(cache.get(("e", "float32", 0))) == -1.5
    assert cache.get(("e", "float64", 0)) is None and cache.get(("e", "float32", 1)) is None
    assert cache.get(("other", "float32", 0)) is None
    with pytest.raises(ValueError, match="float64"):
        cache.store(("e", "float32", 0), torch.tensor([-1.5], dtype=torch.float32))


# --------------------------------------------------------------------------- the MC evaluator on an engine


@pytest.fixture
def mace_engine(tmp_path, monkeypatch):
    """A fixed-planner graph engine (no capture) of a tiny MH-1 model on NaCl, on the CPU."""
    from fastmace.engine import constraints
    from fastmace.model import loading
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    pytest.importorskip("cuequivariance_torch")
    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    built = {}

    def make(dtype="float64"):
        planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"),
                               precision=PrecisionConfig(dtype=dtype), environ={})
        path = tmp_path / "tiny-mh1.model"
        if not path.exists():
            torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
        loaded = planner.load(str(path), trusted=True)
        atoms = nacl(rattle=0.05, seed=1)
        plan = planner.plan(loaded, atoms, IntegratorConfig(noise_batch=7), Hardware("cpu", (0, 0), 0))
        plan = dataclasses.replace(plan, neighbours=dataclasses.replace(plan.neighbours, skin=0.1))
        engine = planner.build(plan, loaded, atoms, temperature_K=600.0, constraints=constraints.ProcessRecord())
        built.update(path=path, plan=plan, loaded=loaded, atoms=atoms, planner=planner)
        return engine

    def load(dtype):
        planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"),
                               precision=PrecisionConfig(dtype=dtype), environ={})
        return planner.load(str(built["path"]), trusted=True)

    make.load = load
    make.built = built
    return make


def fresh_energy(make, engine):
    """The energy of a freshly built float64 force field at the engine's committed positions and species."""
    atoms = make.built["atoms"].copy()
    atoms.positions = engine.state.positions.detach().cpu().numpy()
    atoms.numbers = engine.state.species.detach().cpu().numpy()
    ff = MACEForceField(make.load("float64"), atoms, neighbours=make.built["plan"].neighbours, engine=EAGER)
    return ff.energy_only(positions_of(atoms))


def test_baseline_after_md_equals_a_fresh_evaluation(mace_engine):
    from fastmace.mc.evaluator import MCEvaluator

    engine = mace_engine()
    evaluator = MCEvaluator(engine, precision="float64")
    for _ in range(5):
        engine.step()
    baseline = evaluator.baseline()
    np.testing.assert_allclose(baseline.cpu().numpy(), fresh_energy(mace_engine, engine).numpy(), rtol=1e-12, atol=0)
    assert evaluator.cache.get(evaluator.tag()) is not None
    # Never the MD energy: the baseline is evaluated again after more MD, and the cache tag moves on.
    tag = evaluator.tag()
    engine.step()
    assert evaluator.tag() != tag or evaluator.cache.get(evaluator.tag()) is None
    np.testing.assert_allclose(evaluator.baseline().cpu().numpy(), fresh_energy(mace_engine, engine).numpy(),
                               rtol=1e-12, atol=0)


def test_accept_and_reject_update_the_cached_baseline(mace_engine):
    from fastmace.mc.evaluator import MCEvaluator

    engine = mace_engine()
    evaluator = MCEvaluator(engine, precision="float64")
    baseline = evaluator.baseline().clone()
    trial = baseline - 0.25
    evaluator.reject()
    assert torch.equal(evaluator.cache.get(evaluator.tag()), baseline)
    evaluator.accept(trial)
    assert torch.equal(evaluator.current(), trial)


def test_evaluator_precisions_and_refusals(mace_engine):
    from fastmace.mc.evaluator import MCEvaluator

    engine = mace_engine()
    assert MCEvaluator(engine).precision == "float64"  # None: the force precision
    with pytest.raises(ValueError, match="float32"):
        MCEvaluator(engine, precision="float32")  # below the force precision
    with pytest.raises(ValueError, match="precision"):
        MCEvaluator(engine, precision="auto")  # resolved by the planner (3-T11), never here
    info = MCEvaluator(engine, precision="float64").info()
    assert info["precision"] == "float64" and info["backend"] == "cueq" and info["captured"] is False


def test_float64_evaluator_of_a_float32_engine(mace_engine):
    from fastmace.mc.evaluator import MCEvaluator

    engine = mace_engine("float32")
    assert engine.force_field.dtype == torch.float32
    with pytest.raises(ValueError, match="float64 model"):
        MCEvaluator(engine, precision="float64")
    evaluator = MCEvaluator(engine, precision="float64", model=mace_engine.load("float64"))
    for _ in range(3):
        engine.step()
    np.testing.assert_allclose(evaluator.baseline().cpu().numpy(), fresh_energy(mace_engine, engine).numpy(),
                               rtol=1e-12, atol=0)
    info = evaluator.info()
    assert info["precision"] == "float64" and info["force_precision"] == "float32"
    assert info["backend"] == "cueq" and info["memory_bytes"] > 0
    # The float32 evaluator is the force field's own forward, and differs from float64 by float32 rounding only.
    single = MCEvaluator(engine, precision="float32")
    assert single.info()["memory_bytes"] == 0
    np.testing.assert_allclose(single.baseline().cpu().numpy(), evaluator.baseline().cpu().numpy(), rtol=1e-5)
    assert single.tag() != evaluator.tag()


# --------------------------------------------------------------------------- tools/bench/mc_cost.py


def _mc_cost():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools" / "bench" / "mc_cost.py"
    spec = importlib.util.spec_from_file_location("_mc_cost", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mc_cost_rotates_and_mirrors_the_block_order():
    tool = _mc_cost()
    names = ("md", "energy_float32", "energy_float64")
    orders = [tool.round_order(names, index) for index in range(6)]
    assert orders[0] == list(names) and orders[3] == list(names)[::-1]
    for position in range(3):  # every side is first, middle and last equally often
        assert sorted(order[position] for order in orders) == sorted(names * 2)
    summary = tool.summarise({"md": [10.0, 12.0], "energy_float32": [2.0, 3.0]}, ("md", "energy_float32"))
    assert summary["energy_float32"]["over_md_per_round"] == [0.2, 0.25]
    assert summary["energy_float32"]["over_md_median"] == pytest.approx(0.225)


def test_energy_graph_keys_and_the_copy_in_the_derived_registry(mace_engine):
    from fastmace.engine.graph import ENERGY_KEY
    from fastmace.mc.evaluator import MCEvaluator

    engine = mace_engine("float32")
    with pytest.raises(ValueError, match="float64 force field"):
        engine.energy(precision="float64")
    evaluator = MCEvaluator(engine, precision="float64", model=mace_engine.load("float64"))
    assert "energy.float64.node_attrs" in engine.derived
    assert set(engine.energy_fields) == {"float32", "float64"}
    assert MCEvaluator(engine, precision="float64").field is evaluator.field  # one copy per engine and precision
    with pytest.raises(ValueError, match="already"):
        engine.add_energy_field(evaluator.field)
    energy = engine.energy(precision="float64")
    assert energy is engine.energy(precision="float64") and energy.grad_fn is None
    stage, out = evaluator.stage()
    assert out is energy
    value = energy.clone()
    out.zero_()
    stage()
    assert torch.equal(out, value)
    assert engine.overflowed() == 0
    engine.flag_edge_overflow(engine.force_field.edge_count, 0)  # a host int limit (capturable: no copy)
    assert engine.overflowed() == 1
    engine.clear_overflow()
    info = engine.info()["energy_graphs"]
    assert info["float64"]["copy"] and info["float64"]["compile_mode"] == "none" and not info["float32"]["copy"]
    assert ENERGY_KEY == "E"


def test_float64_baseline_follows_a_species_transaction(mace_engine):
    from fastmace.events import Event, Transaction
    from fastmace.mc.evaluator import MCEvaluator

    engine = mace_engine("float32")
    evaluator = MCEvaluator(engine, precision="float64", model=mace_engine.load("float64"))
    single = MCEvaluator(engine)
    for _ in range(3):
        engine.step()
    before = evaluator.tag()
    evaluator.baseline()
    numbers = engine.state.species.detach().cpu().numpy().copy()
    numbers[[0, 1]] = numbers[[1, 0]]
    assert Transaction(engine, [Event("species", values=numbers)]).run() == "accepted"
    assert evaluator.tag() != before and evaluator.current() is None  # the cache does not carry over
    np.testing.assert_allclose(evaluator.baseline().cpu().numpy(), fresh_energy(mace_engine, engine).numpy(),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(single.baseline().cpu().numpy(), evaluator.current().cpu().numpy(), rtol=1e-5)


@pytest.mark.slow
def test_evaluator_on_the_compiled_tier_equals_eager():
    """The compiled tier (3-T06, no capture, exact edge counts): the same energies as an eager evaluation."""
    from fastmace.engine import constraints
    from fastmace.engine.compiled import CompiledEngine
    from fastmace.integrators.langevin import Langevin
    from fastmace.mc.evaluator import MCEvaluator
    from fastmace.state import SimulationState

    atoms = structure("mh1")
    calc = calculator(model64("mh1"))
    config = EngineConfig(device="cpu", compile_mode="default", dynamic_edges=True, pipelined="off")
    ff = field(calc, atoms, engine=config)
    state = SimulationState.from_atoms(atoms, "cpu", 5)
    engine = CompiledEngine(config, ff, Langevin(state, IntegratorConfig(noise_batch=7), temperature_K=600.0),
                            state, neighbours=NEIGHBOURS, constraints=constraints.ProcessRecord())
    evaluator = MCEvaluator(engine)
    for _ in range(4):
        engine.step()
    moved = atoms.copy()
    moved.positions = engine.state.positions.detach().numpy()
    expected = field(calculator(model64("mh1")), moved).energy_only(positions_of(moved))
    assert not evaluator.info()["captured"]
    np.testing.assert_allclose(evaluator.baseline().numpy(), expected.numpy(), rtol=1e-12, atol=0)


def test_float64_model_keeps_the_process_default_dtype():
    """MACE's calculator sets torch's default dtype to the loaded model's; the engine's float32 model reads it
    in its recompiles (found by the 3-T07 timing run: an Inductor recompile mixed double and float)."""
    from fastmace.config import EngineConfig, PrecisionConfig
    from fastmace.mc.evaluator import float64_model

    class Loader:
        def execution_model(self, plan):
            torch.set_default_dtype(torch.float64)  # as MACECalculator(default_dtype="float64")
            return plan

    @dataclasses.dataclass(frozen=True)
    class Plan:
        engine: EngineConfig
        precision: PrecisionConfig

    plan = Plan(EngineConfig(backend="cueq"), PrecisionConfig(dtype="float32"))
    saved = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        got = float64_model(Loader(), plan, backend="hybrid")
        assert torch.get_default_dtype() == torch.float32
        assert got.precision.dtype == "float64" and got.engine.backend == "hybrid"
    finally:
        torch.set_default_dtype(saved)


def test_float64_copy_evaluates_under_its_own_default_dtype():
    from fastmace.mc.evaluator import float64_copy

    atoms = structure("mh1")
    positions = positions_of(atoms)
    saved = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        reference = field(calculator(model64("mh1")), atoms).energy_only(positions)
        leader = float32_field("mh1", atoms)
        copy64 = float64_copy(leader, calculator(model64("mh1")))
        torch.set_default_dtype(torch.float32)  # the float32 engine's process default
        got = copy64.energy_only(positions)
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(saved)
    np.testing.assert_allclose(got.numpy(), reference.numpy(), rtol=1e-13, atol=0)
