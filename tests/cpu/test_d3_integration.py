"""D3 in the force field, the planner, the engine, the gate's reference and MC (plan task 3b-T04; SPEC §11.2, §12.1).

A tiny MH-1-class model on NaCl, the minimal planner on the CPU without compilation; the D3 term runs its
torch tier (``D3Kernel._torch``) on 3b-T02's list.

- plans carry the record (``Plan.dispersion``, a decision, rule 11's D3 skin), and a D3-free plan is as before:
  its force field is the plain ``MACEForceField``;
- ``fm.ForceField(dispersion=...).evaluate`` and the engine's forces, energy and stress are MACE's plus
  torch-dftd's (the §12.1 reference) on the same record;
- the D3 list rebuilds on its own test, and every list at a canonical point;
- a species event needs no D3 refresh (the kernel reads its tables through the state's species);
- a rejected MC displacement leaves the D3 list and kernel buffers bit for bit; the float64 MC copy adds D3;
- the reference record and calculator: MACE plus torch-dftd, labelled non-stock.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from _support.structures import nacl
from fastmace import integrators
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.longrange.d3 import D3
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

RECORD = D3(functional="r2scan")
NA_CL = (11, 17)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded

    return tiny_loaded(tmp_path_factory.mktemp("models"), NA_CL)


def _planner(d3_skin="auto"):
    from fastmace.engine import constraints
    from fastmace.planner import minimal

    return minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"),
                                  neighbours=NeighbourConfig(d3_skin=d3_skin), environ={},
                                  process=constraints.ProcessRecord())


@pytest.fixture
def planner(monkeypatch):
    from fastmace.planner import minimal

    made = _planner()
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: made)
    return made


def _field(loaded, atoms, dispersion=RECORD, precision="float64"):
    from fastmace.model.forcefield import ForceField

    return ForceField(loaded, atoms, precision=precision, species=list(NA_CL), dispersion=dispersion)


def _nvt():
    return integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01).to_config(seed=1)


def _npt():
    return integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01,
                                     pressure_GPa=0.0).to_config(seed=1)


def _engine(loaded, atoms, *, integrator=None, d3_skin="auto", dispersion=RECORD, precision="float64"):
    planner = _planner(d3_skin)
    field = _field(loaded, atoms, dispersion, precision)
    plan = planner.plan(field, atoms, integrator or _nvt(), Workload(max_temperature_K=300.0))
    return planner.build(plan, field, atoms)


def _reference(atoms, loaded, monkeypatch):
    """MACE alone (a D3-free force field's one-off evaluation) plus torch-dftd on RECORD."""
    pytest.importorskip("torch_dftd")
    from fastmace.planner import minimal
    from fastmace.reference.d3 import torch_dftd3

    planner = _planner()
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    energy, forces, stress = _field(loaded, atoms, None).evaluate(atoms, stress=True)
    e3, f3, s3 = torch_dftd3(atoms, RECORD)
    return energy + e3, forces + f3, stress + s3


def _assert_close(ours, expected, natoms):
    energy, forces, stress = ours
    assert abs(energy - expected[0]) / natoms < 1e-8
    np.testing.assert_allclose(forces, expected[1], rtol=0, atol=1e-7)
    if stress is not None:
        np.testing.assert_allclose(stress, expected[2], rtol=0, atol=1e-8)


# --------------------------------------------------------------------------- plans


def test_plan_carries_the_record_and_the_d3_skin(loaded, planner):
    atoms = nacl(1, rattle=0.02, seed=1)
    plan = planner.plan(_field(loaded, atoms), atoms, _nvt(), Workload(max_temperature_K=300.0))
    assert plan.dispersion == RECORD
    assert isinstance(plan.neighbours.d3_skin, float) and plan.neighbours.d3_skin >= 1.0
    key, choice, reason = plan.decision("dispersion")
    assert "functional=r2scan" in choice and "non-stock" in str(reason)
    assert "dispersion: D3(functional=r2scan" in plan.describe()
    plain = planner.plan(_field(loaded, atoms, None), atoms, _nvt(), Workload(max_temperature_K=300.0))
    assert plain.dispersion is None
    with pytest.raises(KeyError):
        plain.decision("dispersion")
    assert "dispersion" not in plain.describe()


def test_the_restart_record_keeps_the_dispersion(loaded, planner):
    from fastmace.planner.plan import Plan

    atoms = nacl(1)
    plan = planner.plan(_field(loaded, atoms), atoms, _nvt(), Workload(max_temperature_K=300.0))
    record = plan.to_restart(edge_bucket=64, neighbour_capacity=128)
    assert Plan.from_restart(record).dispersion == RECORD
    plain = planner.plan(_field(loaded, atoms, None), atoms, _nvt(), Workload(max_temperature_K=300.0))
    record = plain.to_restart(edge_bucket=64, neighbour_capacity=128)
    assert "dispersion" not in record and Plan.from_restart(record).dispersion is None


def test_without_d3_the_engine_has_the_plain_mace_force_field(loaded):
    from fastmace.model.forces import MACEForceField

    engine = _engine(loaded, nacl(1), dispersion=None)
    assert type(engine.force_field) is MACEForceField
    assert not any(name.startswith("force_field.d3.") for name in engine.registry.entries())


# --------------------------------------------------------------------------- MACE plus D3


def test_one_off_evaluation_is_mace_plus_torch_dftd(loaded, planner, monkeypatch):
    from _support.d3_cases import strained

    atoms = strained(nacl(1, rattle=0.05, seed=2), +1)
    ours = _field(loaded, atoms).evaluate(atoms, stress=True)
    _assert_close(ours, _reference(atoms, loaded, monkeypatch), len(atoms))


def test_engine_forces_and_npt_stress_are_mace_plus_torch_dftd(loaded, monkeypatch):
    from fastmace.model.dispersed import DispersedForceField

    atoms = nacl(1, rattle=0.05, seed=3)
    engine = _engine(loaded, atoms, integrator=_npt())
    assert isinstance(engine.force_field, DispersedForceField)
    state = engine.state
    ours = (float(state.energy[0]), state.forces.detach().numpy(), state.stress.detach().numpy())
    _assert_close(ours, _reference(atoms, loaded, monkeypatch), len(atoms))


def test_md_steps_rebuild_the_d3_list_on_its_own_test(loaded, monkeypatch):
    atoms = nacl(1, rattle=0.05, seed=4)
    engine = _engine(loaded, atoms, d3_skin=0.02)  # a tiny skin: the D3 list rebuilds within a few steps
    engine.run(20)
    engine.flush()
    counters = engine.force_field.d3.counters()
    assert counters["builds"] > 1 and counters["d3_skin_A"] == 0.02
    current = atoms.copy()
    current.positions = engine.state.positions.detach().numpy()
    ours = (float(engine.state.energy[0]), engine.state.forces.detach().numpy(), None)
    _assert_close(ours, _reference(current, loaded, monkeypatch), len(atoms))
    assert "d3" in engine.force_field.info() and "functional=r2scan" in engine.force_field.info()["dispersion"]


def test_a_mace_rebuild_leaves_the_d3_list_when_its_test_is_quiet(loaded):
    engine = _engine(loaded, nacl(1, rattle=0.05, seed=5), d3_skin=3.0)
    field = engine.force_field
    before = field.d3.builds
    assert field.rebuild(engine.state.positions) is False
    assert field.d3.builds == before
    field.rebuild_all(engine.state.positions)
    assert field.d3.builds == before + 1


def test_a_canonical_point_rebuilds_the_d3_list(loaded):
    engine = _engine(loaded, nacl(1, rattle=0.05, seed=6), d3_skin=3.0)
    before = engine.force_field.d3.builds
    engine.canonicalise()
    assert engine.force_field.d3.builds == before + 1


def test_registered_d3_buffers_have_policies(loaded):
    engine = _engine(loaded, nacl(1), integrator=_npt())
    names = [name for name in engine.registry.entries() if name.startswith("force_field.d3.")]
    assert "force_field.d3.list.cell" in names and "force_field.d3.list.index" in names
    assert any(name.startswith("force_field.d3.kernel.0.") for name in names)


# --------------------------------------------------------------------------- species events and MC


def test_a_species_event_needs_no_d3_refresh(loaded, monkeypatch):
    from fastmace.events import Event, Transaction

    atoms = nacl(1, rattle=0.05, seed=7)
    engine = _engine(loaded, atoms)
    numbers = engine.state.species.detach().numpy().copy()
    numbers[[0, 1]] = numbers[[1, 0]]
    assert numbers[0] != numbers[1]
    assert Transaction(engine, [Event("species", values=numbers)]).run() == "accepted"
    assert engine.force_field.d3.kernel.numbers is engine.state.species
    out = engine.force_field(engine.state.positions)
    swapped = atoms.copy()
    swapped.numbers = numbers
    ours = (float(out.energy[0]), out.forces.detach().numpy(), None)
    _assert_close(ours, _reference(swapped, loaded, monkeypatch), len(atoms))


def test_a_rejected_displacement_leaves_the_d3_list(loaded, monkeypatch):
    from fastmace.mc import Displace, MCDriver, moves

    monkeypatch.setattr(moves, "metropolis", lambda u, excess, kT: torch.full_like(u, False, dtype=torch.bool))
    engine = _engine(loaded, nacl(1, rattle=0.05, seed=8), d3_skin=0.1)
    driver = MCDriver([Displace(delta_A=0.2, every=1, attempts=4)], seed=7)
    driver.attach(engine, max_displacement_A=0.2)
    term = engine.force_field.d3
    before = {name: tensor.clone() for name, tensor in term.buffers().items()}
    positions = engine.state.positions.clone()
    engine.integrator.set_temperature(300.0)
    entry = driver.run_block(engine, max(1, int(engine.state.step)))["moves"][0]
    assert entry["accepted"] == 0 and entry["trial_rebuilds"] > 0
    after = term.buffers()
    assert before.keys() == after.keys()
    for name, tensor in before.items():
        torch.testing.assert_close(after[name], tensor, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(engine.state.positions, positions, rtol=0, atol=0)


def test_the_float64_mc_copy_adds_d3(loaded, tmp_path):
    from fastmace.mc.evaluator import float64_copy
    from fastmace.model import loading

    atoms = nacl(1, rattle=0.05, seed=9)
    engine = _engine(loaded, atoms, precision="float32")
    field = engine.force_field
    model = loading.load(str(loaded.path), None, backend="cueq", dtype="float64", device="cpu", trusted=True,
                         allow_naive_cueq=True)
    copy = float64_copy(field, model)
    assert copy._leader is field and copy.d3 is field.d3 and copy.kernel.dtype == torch.float64
    energy = float(copy.energy_only(engine.state.positions)[0])
    mace_only = float(copy.mace.energy_only(engine.state.positions)[0])
    d3_energy = float(field.d3.kernel_for(torch.float64).evaluate_energy(engine.state.positions, field.d3.cell))
    assert energy == pytest.approx(mace_only + d3_energy, rel=0, abs=1e-12)
    assert d3_energy < 0


# --------------------------------------------------------------------------- the reference


def test_the_reference_record_has_the_d3_entry_and_is_non_stock():
    from fastmace.reference import records

    record = records.record_for("0.3.16", "mh1", d3=RECORD)
    entry = next(entry for entry in record.optional_physics if entry["name"] == "d3")
    assert entry["status"] == "accepted" and entry["parameters"]["s8"] == RECORD.s8
    assert not record.is_stock
    assert record.id != records.record_for("0.3.16", "mh1").id
    with pytest.raises(records.ReferencePolicyError, match="PolarMACE"):
        records.record_for("0.3.16", "polar", d3=RECORD)


def test_the_reference_calculator_is_mace_plus_torch_dftd(loaded, monkeypatch):
    pytest.importorskip("torch_dftd")
    from fastmace.reference import calculators

    atoms = nacl(1, rattle=0.05, seed=10)
    plain = calculators.load(str(loaded.path), atoms, dtype="float64", device="cpu", trusted=True)
    with_d3 = calculators.load(str(loaded.path), atoms, dtype="float64", device="cpu", trusted=True, d3=RECORD)
    assert with_d3.d3 == RECORD and "non-stock" in with_d3.info()["dispersion"]
    assert with_d3.info()["stock"] is False and plain.info()["dispersion"] is None
    from fastmace.reference.d3 import torch_dftd3

    e0, f0, s0 = plain.evaluate(atoms)
    e3, f3, s3 = torch_dftd3(atoms, RECORD)
    _assert_close(with_d3.evaluate(atoms), (e0 + e3, f0 + f3, s0 + s3), len(atoms))


def test_the_gate_job_carries_the_record(loaded, planner):
    from fastmace.planner import selfcheck

    atoms = nacl(1)
    plan = planner.plan(_field(loaded, atoms), atoms, _nvt(), Workload(max_temperature_K=300.0))
    job = selfcheck._job(plan, _field(loaded, atoms), [selfcheck.GateCopy(kind="production", atoms=atoms)])
    assert selfcheck._d3_record(job) == RECORD
    plain = dataclasses.replace(plan, dispersion=None)
    assert selfcheck._d3_record(selfcheck._job(plain, _field(loaded, atoms, None), [])) is None
