"""Float64 energy accumulation on the CPU (plan task 1b-T03; SPEC §7.3, §6.2 ``float64_energy``, §4.3).

On the tiny models of ``_support.tiny_models``, with synthetic all-electron E0 tables:

- float64 models: the float64-accumulated energy equals MACE's stock float64 energy (1e-12);
- float32 models: the error against float64 is that of the forward pass, not of the E0 sum,
  and the exact (checkpoint) E0 table matters;
- forces and stress are bitwise those of 1a, with and without the float64 sum (float32 and
  float64, eager and compiled);
- the energy never comes from MACE's ``out["node_energy"]`` (OMOL's embedding readout term,
  PolarMACE's electron, electrostatic and field terms);
- per-atom energies sum to the total and name the terms they omit;
- ``float64_energy=False`` reproduces 1a's energy; an unqualified forward source is refused.
"""

import copy
import importlib.util

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.model import compile as fc
from fastmace.model import energy
from fastmace.model.forces import MACEForceField

pytest.importorskip("mace")
torch.serialization.add_safe_globals([slice])

NEIGHBOURS = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="off")
EAGER = EngineConfig(compile_mode="none")
# Synthetic all-electron E0 (eV), none of them a float32 number (SPEC §4.3 magnitudes).
E0 = {8: -2048.1234567891, 11: -4412.9876543219, 17: -12522.6512345678, 26: -34510.4321987654}
needs_polar = pytest.mark.skipif(not (polar_available() and importlib.util.find_spec("cuequivariance_torch")),
                                 reason="PolarMACE needs graph_longrange and cuequivariance_torch")

# Family -> (Z values, extra constructor options). OMOL with its embedding readout energy and
# PolarMACE with its local electron energy: the terms MACE's node_energy omits.
CASES = {
    "mh1": ([8, 11, 17, 26], {}),
    "residual": ([8, 11, 17, 26], {}),
    "density": ([8, 11, 17, 26], {}),
    "off": ([8, 11, 17, 26], {}),
    "omol": ([8, 11, 17, 26], {"use_embedding_readout": True}),
    "polar": ([11, 17], {"add_local_electron_energy": True}),
}
FAMILY_PARAMS = [pytest.param(name, marks=needs_polar) if name == "polar" else name for name in CASES]


def e0_table(z_values):
    return torch.tensor([E0[z] for z in sorted(z_values)], dtype=torch.float64)


def model64(family):
    """Tiny float64 model of ``family`` with the synthetic E0 table."""
    z_values, extra = CASES[family]
    model = tiny_mace_model(z_values, **{**FAMILIES[family], **extra})
    with torch.no_grad():
        model.atomic_energies_fn.atomic_energies.copy_(e0_table(z_values).reshape(
            model.atomic_energies_fn.atomic_energies.shape))
    return model


def calculator(model, dtype="float64"):
    """MACE's calculator of a copy of ``model`` in ``dtype``; PolarMACE converted to cuEq (naive kernels),
    the only backends its force field takes (``fastmace.longrange.polar.route_support``)."""
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


def stock(calc, atoms, stress=False):
    current = atoms.copy()
    current.calc = calc
    out = [current.get_potential_energy(), current.get_forces()]
    if stress:
        out.append(current.get_stress(voigt=False))
    return out


def raw_stock_output(model, atoms):
    """MACE's own forward output dict (float64 model) for ``atoms``."""
    calc = calculator(model)
    batch = calc._atoms_to_batch(atoms)
    return calc.models[0](batch.to_dict(), compute_stress=False, training=False)


# --------------------------------------------------------------------------- float64 models


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_float64_energy_equals_stock_float64(family):
    model, atoms = model64(family), structure(family)
    calc = calculator(model)
    expected = stock(calc, atoms)
    ff = field(calc, atoms)
    assert ff.float64_energy and ff.info()["float64_energy"] is True
    energy_value, forces, _ = ff.evaluate(atoms)
    np.testing.assert_allclose(energy_value, expected[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(forces, expected[1], rtol=1e-12, atol=1e-14)
    positions = torch.tensor(atoms.positions)
    assert ff(positions).energy.dtype == torch.float64


@pytest.mark.parametrize("family", [pytest.param("omol"), pytest.param("polar", marks=needs_polar)])
def test_energy_never_comes_from_node_energy(family):
    """The stock node_energy misses a term here; the float64 energy includes it (SPEC §7.3)."""
    model, atoms = model64(family), structure(family)
    out = raw_stock_output(model, atoms)
    missing = float(out["energy"].sum()) - float(out["node_energy"].sum())
    assert abs(missing) > 1e-6, "the regression test needs a term node_energy omits"
    ff = field(calculator(model), atoms)
    np.testing.assert_allclose(ff.evaluate(atoms)[0], float(out["energy"].sum()), rtol=1e-12, atol=0)


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_per_atom_energies_sum_to_the_total_and_name_omitted_terms(family):
    model, atoms = model64(family), structure(family)
    ff = field(calculator(model), atoms)
    per_atom = ff.per_atom_energies(atoms)
    assert per_atom.energies.shape == (len(atoms),) and per_atom.energies.dtype == np.float64
    total = ff.evaluate(atoms)[0]
    np.testing.assert_allclose(per_atom.energies.sum() + per_atom.unresolved, total, rtol=1e-13, atol=0)
    assert per_atom.terms == energy.terms(ff.source_model)
    if family == "polar":
        assert set(per_atom.terms["unresolved"]) == {"electrostatic", "field"}
        assert "local_electron" in per_atom.terms["resolved"] and per_atom.unresolved != 0.0
    else:
        assert per_atom.terms["unresolved"] == () and per_atom.unresolved == 0.0
        assert "embedding_readout" in per_atom.terms["resolved"]


def test_node_e0_lookup_is_exact():
    table = torch.tensor([[-2048.1234567891, -12522.6512345678], [-1.5, -4.75]], dtype=torch.float64)
    one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    assert energy.node_e0_float64(table, one_hot, head=0).tolist() == [
        -2048.1234567891, -12522.6512345678, -12522.6512345678]
    assert energy.node_e0_float64(table, one_hot, head=1).tolist() == [-1.5, -4.75, -4.75]
    assert energy.node_e0_float64(table[0], one_hot).tolist()[0] == -2048.1234567891


# --------------------------------------------------------------------------- float32 models


@pytest.mark.parametrize("family", ["mh1", "omol", pytest.param("polar", marks=needs_polar)])
def test_float32_error_is_the_forward_pass_not_the_e0_sum(family):
    model, atoms = model64(family), structure(family)
    exact = e0_table(CASES[family][0])
    reference = field(calculator(model), atoms).evaluate(atoms)[0]
    calc32 = calculator(model, "float32")
    stock32 = stock(calc32, atoms)[0]
    accumulated = field(calc32, atoms, atomic_energies=exact).evaluate(atoms)[0]
    rounded_table = field(calc32, atoms).evaluate(atoms)[0]  # the model's float32-rounded E0, upcast
    # The forward pass alone: the same model with E0 = 0 in float32 against float64.
    zero = copy.deepcopy(model)
    with torch.no_grad():
        zero.atomic_energies_fn.atomic_energies.zero_()
    forward_error = abs(field(calculator(zero, "float32"), atoms).evaluate(atoms)[0]
                        - field(calculator(zero), atoms).evaluate(atoms)[0])
    error = abs(accumulated - reference)
    stock_error = abs(stock32 - reference)
    assert stock_error > 1e-4  # the E0 sum in float32 (step 2**-9 eV near 1e5 eV)
    assert error <= forward_error + 1e-9
    # Below the stock error; by 20x unless the forward pass itself is that far off (the tiny
    # PolarMACE's float32 SCF: 13 meV on 64 atoms).
    assert error < stock_error / (2 if family == "polar" else 20)
    # The rounded float32 table is off by the E0 rounding; only the checkpoint's table removes it.
    table32 = exact.float().double()
    numbers = [sorted(CASES[family][0]).index(z) for z in atoms.numbers]
    e0_rounding = abs(float((table32 - exact)[numbers].sum()))
    np.testing.assert_allclose(abs(rounded_table - accumulated), e0_rounding, rtol=1e-6, atol=1e-9)


def test_float64_energy_off_reproduces_the_stock_float32_energy():
    model, atoms = model64("mh1"), structure("mh1")
    calc32 = calculator(model, "float32")
    ff = field(calc32, atoms, float64_energy=False)
    assert not ff.float64_energy
    stock32 = stock(calc32, atoms)[0]
    for _ in range(2):  # a throwaway first evaluation (docs/development.md)
        value = ff.evaluate(atoms)[0]
    assert value == stock32


# --------------------------------------------------------------------------- forces and stress bitwise


def _forces_and_stress(ff, atoms):
    positions = torch.tensor(atoms.positions + 0.02 * np.random.default_rng(3).standard_normal(atoms.positions.shape))
    out = None
    for _ in range(2):  # a throwaway first evaluation (docs/development.md)
        ff.compact(positions)
        ff.use_edges(int(ff.edge_count))
        out = ff(positions)
    return out.forces.detach().clone(), None if out.stress is None else out.stress.detach().clone()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("family", ["mh1", "omol", pytest.param("polar", marks=needs_polar)])
def test_forces_and_stress_are_bitwise_unchanged(family, dtype):
    model, atoms = model64(family), structure(family)
    calc = calculator(model, dtype)
    on = field(calc, atoms, compute_stress=True)
    off = field(calc, atoms, compute_stress=True, float64_energy=False)
    forces_on, stress_on = _forces_and_stress(on, atoms)
    forces_off, stress_off = _forces_and_stress(off, atoms)
    assert torch.equal(forces_on, forces_off)
    assert torch.equal(stress_on, stress_off)


@pytest.fixture
def one_thread():
    """One intra-op thread: with more, two compilations of the same model differ by ~1e-18 (parallel CPU
    reductions in Inductor's kernels), with or without the float64 sum (1b-T03, OMP_NUM_THREADS=2)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compiled_forces_and_stress_are_bitwise_unchanged(dtype, one_thread):
    """Compiled, with and without the float64 sum: forces and stress bitwise equal.

    Runs with one intra-op thread (``one_thread``): with two, already two compilations of the same
    model differ by about 3e-18 eV/Å, so bitwise equality is defined only single-threaded.
    """
    pytest.importorskip("torch._inductor")
    model, atoms = model64("mh1"), structure("mh1")
    calc = calculator(model, dtype)
    # compile_model rebuilds e3nn models from their configuration in torch's default dtype, which
    # building the tiny model set to float64 (tests/conftest.py restores it).
    torch.set_default_dtype(getattr(torch, dtype))
    engine = EngineConfig(compile_mode="default")
    cache = fc.CompiledModelCache()
    on = field(calc, atoms, engine=engine, compile_cache=cache, compute_stress=True)
    off = field(calc, atoms, engine=engine, compile_cache=cache, compute_stress=True, float64_energy=False)
    assert len(cache) == 2  # the float64 forward is part of the compiled model's key
    forces_on, stress_on = _forces_and_stress(on, atoms)
    forces_off, stress_off = _forces_and_stress(off, atoms)
    assert torch.equal(forces_on, forces_off)
    assert torch.equal(stress_on, stress_off)
    eager = field(calc, atoms, compute_stress=True)
    np.testing.assert_allclose(on.evaluate(atoms, stress=True)[0], eager.evaluate(atoms, stress=True)[0],
                               rtol=1e-6 if dtype == "float32" else 1e-12)


def test_compile_cache_key_includes_float64_energy(monkeypatch):
    model, atoms = model64("mh1"), structure("mh1")
    calc = calculator(model)
    compiled = []

    def fake_compile(source, mode, *, prepare=None, **kwargs):
        private = fc.private_copy(source)
        if prepare is not None:
            prepare(private)
        compiled.append(private)
        return private

    monkeypatch.setattr(fc, "compile_model", fake_compile)
    cache = fc.CompiledModelCache()
    engine = EngineConfig(compile_mode="default")
    on = field(calc, atoms, engine=engine, compile_cache=cache)
    off = field(calc, atoms, engine=engine, compile_cache=cache, float64_energy=False)
    again = field(calc, atoms, engine=engine, compile_cache=cache)
    assert len(compiled) == 2 and on.model is again.model and on.model is not off.model
    assert "forward" in on.model.__dict__ and "forward" not in off.model.__dict__
    assert "forward" not in calc.models[0].__dict__  # the caller's model is never changed
    np.testing.assert_allclose(on.evaluate(atoms)[0], stock(calc, atoms)[0], rtol=1e-12)


# --------------------------------------------------------------------------- qualification


def test_the_rewritten_forwards_are_qualified_here():
    assert energy.source_problems() == []


def test_an_unqualified_forward_is_refused(monkeypatch):
    model, atoms = model64("mh1"), structure("mh1")
    monkeypatch.setitem(energy.SOURCE_HASHES, "mace.modules.models:ScaleShiftMACE.forward", "0" * 64)
    monkeypatch.setattr(energy, "_FORWARDS", {})
    with pytest.raises(energy.EnergyReductionError, match="ScaleShiftMACE.forward"):
        field(calculator(model), atoms)
    field(calculator(model), atoms, float64_energy=False)  # 1a's energy needs no rewritten source


def test_edits_must_match_exactly_once():
    with pytest.raises(energy.EnergyReductionError, match="not MACE 0.3.16"):
        energy.apply_edits("def forward(self):\n    return 1\n", energy.EDITS["ScaleShiftMACE"], "X.forward")


def test_other_model_classes_are_refused():
    with pytest.raises(energy.EnergyReductionError, match="Linear"):
        energy.float64_forward(torch.nn.Linear(1, 1))


# --------------------------------------------------------------------------- loading and the planner


@pytest.fixture
def e0_checkpoint(tmp_path, monkeypatch):
    from fastmace.model import loading

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    path = tmp_path / "tiny-mh1-e0.model"
    torch.save(model64("mh1"), path)
    return path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_load_keeps_the_checkpoint_e0_table_in_float64(e0_checkpoint):
    from fastmace.model import loading

    exact = e0_table(CASES["mh1"][0])
    loaded = loading.load(str(e0_checkpoint), backend="e3nn", dtype="float32", device="cpu", trusted=True)
    assert loaded.atomic_energies.dtype == torch.float64
    assert torch.equal(loaded.atomic_energies.reshape(-1), exact)
    rounded = loaded.model.atomic_energies_fn.atomic_energies
    assert rounded.dtype == torch.float32 and not torch.equal(rounded.double().reshape(-1), exact)
    atoms = structure("mh1")
    ff = field(loaded, atoms)
    assert torch.equal(ff.atomic_energies, loaded.atomic_energies)
    reference = field(calculator(model64("mh1")), atoms).evaluate(atoms)[0]
    stock32 = stock(loaded.calculator, atoms)[0]
    assert abs(ff.evaluate(atoms)[0] - reference) < abs(stock32 - reference) / 20


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("enabled", [True, False])
def test_fixed_planner_passes_float64_energy_through(e0_checkpoint, enabled):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.config import IntegratorConfig, PrecisionConfig
    from fastmace.engine import constraints
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"),
                           precision=PrecisionConfig(float64_energy=enabled), environ={})
    loaded = planner.load(str(e0_checkpoint), trusted=True)
    atoms = structure("mh1")
    plan = planner.plan(loaded, atoms, IntegratorConfig(), Hardware("cpu", (0, 0), 0))
    assert plan.precision.float64_energy is enabled
    engine = planner.build(plan, loaded, atoms, temperature_K=300.0, constraints=constraints.ProcessRecord())
    assert engine.force_field.float64_energy is enabled
    assert engine.force_field.info()["float64_energy"] is enabled
