"""Canonical restart points in the engine (plan task 1b-T14, SPEC §10.6, §8.1, §12.2 restart row).

A restart write is a canonicalisation point in interrupted and uninterrupted runs alike: the engine
quiesces the pipeline, rebuilds every neighbour list from the current positions, cell and species,
and resets the pipeline floors and the ``reset`` buffers of the registry. A resumed engine, built
fresh and given the restart's state and its edge bucket and neighbour capacity, is then in exactly
the state of the uninterrupted one. Checked here on the CPU in float64:

- after ``canonicalise()`` the registered state (and the engine's host bookkeeping: edge bucket,
  pipeline floor and next size, neighbour capacity) equals a freshly built engine's at the same
  positions, cell and species;
- interrupted against uninterrupted runs with the same restart cadence agree bit for bit, across a
  partly consumed noise batch, a pipelined count-overflow redo just before the restart, a rebuild
  in the step after it and a full-cell BAOAB run with h != h0 at the restart step; Lennard-Jones on
  the skin list (``test_engine.py``) and tiny MACE models through the fixed planner;
- ``canonicalise`` writes only registered tensors, and no registered buffer is left without a
  restart policy;
- ``canonicalise(reevaluate=True)`` (host writes, owner decision D4) equals a fresh engine at the
  written positions, forces and the BAOAB cell force included.
"""

import dataclasses

import numpy as np
import pytest
import torch
from test_engine import ReplayEngine, TorchLJ, argon

from _support.structures import nacl
from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.engine.registry import RESTART_POLICIES, unregistered_mutations
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.io import restart
from fastmace.state import SimulationState

NEIGHBOURS = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off")
# Integrator scratch written before it is read in every step (registry: "derived" without recompute);
# a fresh engine holds zeros there, a running one the last step's values, and neither is ever read.
SCRATCH = {"integrator.x_old", "integrator.rnd_pos", "integrator.rnd_vel", "integrator.xi", "integrator.eta",
           "integrator.zeta"}
MODES = {
    "checked": dict(engine=EngineConfig(pipelined="off", device="cpu"), replay=False),
    "pipelined": dict(engine=EngineConfig(pipelined="on", device="cpu"), replay=True),
    "speculative": dict(engine=EngineConfig(pipelined="off", speculative_graph=True, device="cpu"), replay=True),
}


class LJ(TorchLJ):
    """The Lennard-Jones force field of ``test_engine.py``, its skin list under the MACE force field's name."""

    @property
    def neighbours(self):
        return self.nl


def lj_engine(atoms=None, *, mode="pipelined", npt=False, cell_mode="hydrostatic", neighbours=NEIGHBOURS,
              noise_batch=7, seed=7):
    """Force field, state, integrator and engine for ``atoms`` (default the rattled argon of test_engine)."""
    atoms = argon() if atoms is None else atoms
    options = MODES[mode]
    field = LJ(atoms, neighbours, compute_stress=npt)
    state = SimulationState.from_atoms(atoms, "cpu", seed, compute_stress=npt)
    if npt:
        config = IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=0.02, noise_batch=noise_batch,
                                  pressure_GPa=0.5, barostat_mass_factor=0.01, cell_mode=cell_mode)
        integrator = LangevinBAOAB(state, config)
    else:
        config = IntegratorConfig(timestep_fs=2.0, friction=0.02, fixcm=True, noise_batch=noise_batch)
        integrator = Langevin(state, config)
    cls = ReplayEngine if options["replay"] else GraphEngine
    engine = cls(options["engine"], field, integrator, state, neighbours=neighbours, capture=options["replay"],
                 constraints=constraints.ProcessRecord())
    if options["replay"]:
        engine.captures = []
    return engine


def restart_atoms(template, stored):
    """The structure a resume builds from: the template's species with the restart's positions and cell."""
    atoms = template.copy()
    atoms.set_cell(stored["cell_state"] if "cell_state" in stored else stored["cell"])
    atoms.set_positions(stored["positions"])
    return atoms


def target(step):
    return min(2.0 * step, 300.0)


def heat(engine, steps, start=0):
    for step in range(start + 1, start + steps + 1):
        engine.integrator.set_temperature(target(step))
        engine.step()


def stored(engine, tmp_path, name="restart.npz"):
    path = tmp_path / name
    restart.write_engine_restart(path, engine, name="fastmace")
    return restart.read_restart(path)


def registered(engine):
    return {name: (policy, tensor.detach().clone()) for name, (policy, tensor) in engine.registry.entries().items()}


def bookkeeping(engine):
    return dict(edge_bucket=engine.edge_bucket, pipe_floor=engine._pipe_floor, pipe_edges=engine._pipe_edges,
                spec_edges=engine._spec_edges, capacity=engine.force_field.edge_capacity,
                pending=(len(engine._pending), engine._pipe_pending))


def assert_same_registered_state(got, want, skip=()):
    assert got.keys() == want.keys()
    for name, (policy, tensor) in got.items():
        if name in skip:
            continue
        assert policy == want[name][0], name
        expected = want[name][1]
        assert tensor.shape == expected.shape, name
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0, msg=f"{name} ({policy.restart})")


def assert_same_values(got, want):
    assert got.keys() == want.keys()
    for key in got:
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


# --------------------------------------------------------------------------- the canonical state


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), [
    ("checked", False, "hydrostatic"), ("pipelined", False, "hydrostatic"), ("speculative", False, "hydrostatic"),
    ("pipelined", True, "full"), ("checked", True, "hydrostatic")])
def test_canonical_state_equals_a_fresh_engine(tmp_path, mode, npt, cell_mode):
    engine = lj_engine(mode=mode, npt=npt, cell_mode=cell_mode)
    heat(engine, 23)
    if mode == "pipelined":
        engine.flush()  # validates the step in flight, which would set the next size itself
        engine._pipe_edges = 1  # the step below is redone for its count; the larger size becomes the floor
    heat(engine, 1, start=23)
    file = stored(engine, tmp_path)
    if mode == "pipelined":
        assert engine.pipelined_redos >= 1
    fresh = lj_engine(restart_atoms(argon(), file), mode=mode, npt=npt, cell_mode=cell_mode)
    restart.resume_engine(file, fresh)
    fresh.integrator.set_temperature(target(24))  # the protocol's target at the restart step
    assert_same_registered_state(registered(engine), registered(fresh), skip=SCRATCH)
    assert bookkeeping(engine) == bookkeeping(fresh)


def test_canonicalise_resets_the_pipeline_floor_and_the_build_cell():
    # A wide skin: no rebuild, which would reset the floor itself.
    engine = lj_engine(npt=True, cell_mode="full", neighbours=dataclasses.replace(NEIGHBOURS, skin=1.0))
    heat(engine, 30)
    engine.flush()
    engine._pipe_edges = 1
    heat(engine, 1, start=30)
    engine.flush()
    neighbours = engine.force_field.neighbours
    assert engine._pipe_floor > 0
    assert not torch.equal(neighbours.build_cell, engine.state.cell)  # h != h0 at the restart step
    engine.canonicalise()
    assert engine._pipe_floor == 0
    assert engine._pipe_edges == engine._padded(int(engine.force_field.edge_count))
    torch.testing.assert_close(neighbours.build_cell, engine.state.cell, rtol=0, atol=0)
    torch.testing.assert_close(neighbours.reference_positions, engine.state.positions, rtol=0, atol=0)
    for name in ("rebuild_needed", "_flags", "_pipe_flags", "_spec_flags", "_flags_host", "_pipe_flags_host"):
        assert not bool(getattr(engine, name).any()), name
    assert engine.info()["canonical_points"] == 1


def test_canonicalise_writes_only_registered_tensors():
    engine = lj_engine(npt=True, cell_mode="full")
    heat(engine, 12)
    assert unregistered_mutations(engine, engine.canonicalise) == []
    for name, (policy, _) in engine.registry.entries().items():
        assert policy.restart in RESTART_POLICIES, name
        assert policy.restart != "reset" or policy.recompute, name


def test_a_capacity_grown_by_the_canonical_rebuild_drops_the_graphs():
    engine = lj_engine(mode="pipelined")
    heat(engine, 5)
    assert engine._graphs
    neighbours = engine.force_field.neighbours
    neighbours.set_capacity(neighbours.num_candidates)  # exactly full: the next build may grow it
    engine._clear_graphs()
    heat(engine, 2, start=5)
    assert engine._graphs
    original = neighbours.build

    def build(positions):
        original(positions)
        return True  # as if the capacity grew: every buffer moved

    neighbours.build = build
    engine.canonicalise()
    assert not engine._graphs
    heat(engine, 2, start=7)  # captured again on demand


def test_the_restart_file_holds_the_buckets(tmp_path):
    engine = lj_engine()
    heat(engine, 4)
    file = stored(engine, tmp_path)
    assert restart.plan_record(file) == engine.restart_buckets()
    assert engine.restart_buckets() == {"edge_bucket": engine.edge_bucket,
                                        "neighbour_capacity": engine.force_field.edge_capacity}


def test_resume_restores_a_capacity_smaller_than_a_fresh_build(tmp_path):
    """The fresh build allocates with headroom for its count; the resumed engine takes the stored capacity."""
    engine = lj_engine(neighbours=dataclasses.replace(NEIGHBOURS, capacity_headroom=1.0))
    heat(engine, 4)
    file = stored(engine, tmp_path)
    fresh = lj_engine(restart_atoms(argon(), file), neighbours=dataclasses.replace(NEIGHBOURS, capacity_headroom=1.5))
    assert fresh.force_field.edge_capacity > engine.force_field.edge_capacity
    restart.resume_engine(file, fresh)
    assert fresh.force_field.edge_capacity == engine.force_field.edge_capacity


def test_set_capacity_refuses_less_than_the_candidates():
    engine = lj_engine()
    neighbours = engine.force_field.neighbours
    with pytest.raises(ValueError, match="candidates"):
        neighbours.set_capacity(neighbours.num_candidates - 1)
    assert neighbours.set_capacity(neighbours.capacity) is False  # unchanged: no buffer moved
    assert neighbours.set_capacity(neighbours.capacity + 256) is True


# --------------------------------------------------------------------------- interrupted against uninterrupted


def lj_run(tmp_path, *, total, cadence, interrupt_at=None, redo_before=None, **options):
    """``total`` steps with a restart every ``cadence``; ``interrupt_at``: resume a fresh engine from that restart."""
    engine = lj_engine(**options)
    info = {"rebuilds_after_restart": []}
    step = 0
    while step < total:
        if redo_before is not None and step + 1 == redo_before:
            engine.flush()  # validates the step in flight, which would set the next size itself
            engine._pipe_edges = 1
        heat(engine, 1, start=step)
        step += 1
        if step % cadence == 0 and step < total:
            file = stored(engine, tmp_path, f"restart-{step}.npz")
            if step == interrupt_at:
                del engine
                engine = lj_engine(restart_atoms(argon(), file), **options)
                restart.resume_engine(file, engine)
            rebuilds = engine.force_field.neighbours.rebuilds
            heat(engine, 1, start=step)
            engine.flush()  # a pipelined step's rebuild happens in its validation
            step += 1
            info["rebuilds_after_restart"].append(engine.force_field.neighbours.rebuilds > rebuilds)
    return engine.sync(), engine, info


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), [
    ("checked", False, "hydrostatic"), ("pipelined", False, "hydrostatic"), ("speculative", False, "hydrostatic"),
    ("pipelined", True, "hydrostatic"), ("pipelined", True, "full"), ("checked", True, "full")])
def test_interrupted_equals_uninterrupted(tmp_path, mode, npt, cell_mode):
    options = dict(mode=mode, npt=npt, cell_mode=cell_mode, noise_batch=7,
                   neighbours=dataclasses.replace(NEIGHBOURS, skin=0.005))
    # A skin of 0.005 Å rebuilds the list in most steps, the step after a restart included.
    # A restart every 10 steps with a 7-step noise batch: the restart at step 30 is 2 draws into a batch.
    whole, reference, info = lj_run(tmp_path / "whole", total=60, cadence=10, redo_before=30, **options)
    part, resumed, _ = lj_run(tmp_path / "part", total=60, cadence=10, interrupt_at=30, redo_before=30, **options)
    assert_same_values(part, whole)
    assert any(info["rebuilds_after_restart"])  # a rebuild in the step after a restart
    if mode == "pipelined":
        assert reference.pipelined_redos >= 1
    if npt and cell_mode == "full":
        assert not np.allclose(whole["cell"], argon().cell.array)


@pytest.fixture(autouse=True)
def _directories(tmp_path):
    (tmp_path / "whole").mkdir(exist_ok=True)
    (tmp_path / "part").mkdir(exist_ok=True)


def test_without_canonical_points_a_resume_is_not_bitwise(tmp_path, monkeypatch):
    """The negative control: restart writes that skip canonicalise give a different trajectory after resume."""
    options = dict(mode="pipelined", npt=True, cell_mode="full", neighbours=dataclasses.replace(NEIGHBOURS, skin=0.05))
    whole, _, _ = lj_run(tmp_path / "whole", total=60, cadence=10, **options)
    monkeypatch.setattr(GraphEngine, "canonicalise", lambda self, reevaluate=False: self.flush())
    part, _, _ = lj_run(tmp_path / "part", total=60, cadence=10, interrupt_at=30, **options)
    assert not all(np.array_equal(part[key], whole[key]) for key in whole)


# --------------------------------------------------------------------------- host writes (owner decision D4)


@pytest.mark.parametrize("npt", [False, True])
def test_reevaluate_equals_a_fresh_engine_at_the_written_positions(tmp_path, npt):
    engine = lj_engine(npt=npt, cell_mode="full")
    heat(engine, 15)
    engine.flush()
    moved = engine.state.positions + 0.01
    engine.state.positions.copy_(moved)
    engine.canonicalise(reevaluate=True)
    atoms = argon()
    atoms.set_cell(engine.state.cell.numpy())
    atoms.set_positions(moved.numpy())
    fresh = lj_engine(atoms, npt=npt, cell_mode="full")
    # The cell force has a kinetic part; Atoms.set_velocities would round through the momenta.
    fresh.state.velocities.copy_(engine.state.velocities)
    fresh.integrator.start(fresh.state)
    names = ("forces", "energy", "stress", "force_eps") if npt else ("forces", "energy")
    for name in names:
        torch.testing.assert_close(getattr(engine.state, name), getattr(fresh.state, name), rtol=0, atol=0,
                                   msg=name)
    torch.testing.assert_close(engine.force_field.neighbours.reference_positions, moved, rtol=0, atol=0)


# --------------------------------------------------------------------------- tiny MACE models


@pytest.fixture
def mace_builder(tmp_path, monkeypatch):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading
    from fastmace.planner.fixed import FixedPlanner
    from fastmace.planner.plan import Hardware

    monkeypatch.setattr(loading, "_LICENCES_SHOWN", set())
    loaded = {}

    def make(family, atoms, npt=False, cell_mode="hydrostatic", pipelined="on"):
        planner = FixedPlanner(engine=EngineConfig(device="cpu", backend="cueq", compile_mode="none"), environ={})
        if family not in loaded:
            path = tmp_path / f"tiny-{family}.model"
            torch.save(tiny_mace_model([11, 17], **FAMILIES[family]), path)
            loaded[family] = planner.load(str(path), trusted=True)
        integrator = (IntegratorConfig(ensemble="npt", pressure_GPa=0.5, cell_mode=cell_mode, noise_batch=7)
                      if npt else IntegratorConfig(noise_batch=7))
        plan = planner.plan(loaded[family], atoms, integrator, Hardware("cpu", (0, 0), 0))
        plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, pipelined=pipelined),
                                   neighbours=dataclasses.replace(plan.neighbours, skin=0.1))
        make.plans.append(plan)
        return planner.build(plan, loaded[family], atoms, temperature_K=300.0, constraints=constraints.ProcessRecord())

    make.plans = []
    return make


def mace_run(make, family, directory, *, total, cadence, interrupt_at=None, **options):
    template = nacl(rattle=0.05, seed=1)
    engine = make(family, template, **options)
    for step in range(1, total + 1):
        engine.step()
        if step % cadence == 0 and step < total:
            file = stored(engine, directory, f"restart-{step}.npz")
            if step == interrupt_at:
                engine = make(family, restart_atoms(template, file), **options)
                restart.resume_engine(file, engine)
    return engine.sync()


MACE_CASES = [("mh1", dict()), ("mh1", dict(npt=True, cell_mode="full")), ("mh1", dict(pipelined="off")),
              ("polar", dict()), ("polar", dict(npt=True, cell_mode="full"))]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("family", "options"), MACE_CASES)
def test_mace_interrupted_equals_uninterrupted(mace_builder, tmp_path, family, options):
    if family == "polar" and not polar_available():
        pytest.skip("PolarMACE needs MACE's polar module")
    mace_builder(family, nacl(rattle=0.05, seed=1), **options).run(1)  # throwaway evaluation (docs/development.md)
    whole = mace_run(mace_builder, family, tmp_path / "whole", total=24, cadence=8, **options)
    part = mace_run(mace_builder, family, tmp_path / "part", total=24, cadence=8, interrupt_at=16, **options)
    assert_same_values(part, whole)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("family", "options"), [("mh1", dict(npt=True, cell_mode="full")), ("polar", dict(npt=True))])
def test_mace_canonical_state_equals_a_fresh_engine(mace_builder, tmp_path, family, options):
    if family == "polar" and not polar_available():
        pytest.skip("PolarMACE needs MACE's polar module")
    template = nacl(rattle=0.05, seed=1)
    engine = mace_builder(family, template, **options)
    engine.run(12)
    file = stored(engine, tmp_path)
    fresh = mace_builder(family, restart_atoms(template, file), **options)
    restart.resume_engine(file, fresh)
    assert_same_registered_state(registered(engine), registered(fresh), skip=SCRATCH)
    assert bookkeeping(engine) == bookkeeping(fresh)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_polar_overflow_is_recorded_and_cleared(mace_builder):
    if not polar_available():
        pytest.skip("PolarMACE needs MACE's polar module")
    engine = mace_builder("polar", nacl(rattle=0.05, seed=1))
    engine.run(3)
    engine.registry.tensors()["force_field.polar.overflow"].fill_(True)
    engine.canonicalise()
    assert not bool(engine.registry.tensors()["force_field.polar.overflow"])
    assert engine.info()["polar_overflow_steps"] == [engine.state.step]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_plan_record_comes_from_the_plan(mace_builder, tmp_path):
    engine = mace_builder("mh1", nacl(rattle=0.05, seed=1))
    plan = mace_builder.plans[-1]
    engine.run(2)
    path = tmp_path / "restart.npz"
    restart.write_engine_restart(path, engine, name="fastmace", plan=plan)
    record = restart.plan_record(restart.read_restart(path))
    assert record == plan.to_restart(**engine.restart_buckets())
