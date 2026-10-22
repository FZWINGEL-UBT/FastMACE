"""D3 in the captured force graph on CUDA (plan task 3b-T04; SPEC §11.2, §8.1).

A tiny MH-1-like model on NaCl 2x2x2 (64 atoms), float32 with D3(BJ) r2scan, through the minimal planner on
``cuda:0`` in a fresh process, with a small D3 skin so that the D3 list rebuilds during the run:

- the engine's force field is MACE plus D3 (Triton passes), and its first forces, energy and NPT stress
  equal a D3-free engine's plus torch-dftd's (float64) within the gate's floors;
- captured graphs (checked; pipelined with early rebuilds beside the force call) and an uncaptured engine run
  60 steps with D3 list rebuilds: the committed forces equal an eager force call at the committed positions,
  the D3 term on the engine's list equals one built fresh there, and no rebuild drops a graph;
- the fused integrator stage ORs the D3 list's test in (its rebuilds happen with the fused stage on);
- a float64 MC energy copy adds D3 on the same list; a rejected displacement block leaves the D3 list as it was.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from _support.tiny_models import FAMILIES, tiny_mace_model

pytest.importorskip("triton")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_d3_integration_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import dataclasses, json
import numpy as np
import torch
from fastmace import integrators
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.longrange.d3 import D3
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import minimal
from fastmace.planner.workload import Workload

RECORD = D3(functional="r2scan")

def structure():
    from ase.build import bulk
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=3)
    return atoms

loaded = loading.load_pristine(SPEC["path"], trusted=True)

def planner(**engine):
    return minimal.MinimalPlanner(engine=EngineConfig(device="cuda:0", compile_mode="none", expandable_segments="off",
                                                      **engine),
                                  neighbours=NeighbourConfig(d3_skin=SPEC["d3_skin"]), environ={},
                                  process=constraints.ProcessRecord())

def build(npt=False, dispersion=RECORD, capture=None, **engine):
    atoms = structure()
    made = planner(**engine)
    field = ForceField(loaded, atoms, precision="float32", dispersion=dispersion)
    if npt:
        record = integrators.LangevinBAOAB(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.01,
                                           pressure_GPa=0.0).to_config(seed=7)
    else:
        record = integrators.Langevin(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.01).to_config(seed=7)
    plan = made.plan(field, atoms, record, Workload(max_temperature_K=600.0))
    engine = made.build(plan, field, atoms)
    if capture is False:
        engine.capture = False
    return engine

out = {}
# 1. MACE plus D3 against a D3-free engine plus torch-dftd (float64), NPT so the stress is there too.
from fastmace.reference.d3 import torch_dftd3
with_d3, without = build(npt=True), build(npt=True, dispersion=None)
e3, f3, s3 = torch_dftd3(structure(), RECORD)
state, plain = with_d3.state, without.state
out["type"] = type(with_d3.force_field).__name__
out["energy_per_atom"] = abs(float(state.energy[0]) - float(plain.energy[0]) - e3) / 64
out["force"] = float(np.abs(state.forces.double().cpu().numpy() - plain.forces.double().cpu().numpy() - f3).max())
out["stress"] = float(np.abs(state.stress.double().cpu().numpy() - plain.stress.double().cpu().numpy() - s3).max())
del with_d3, without

# 2. captured graphs (checked, pipelined with early rebuilds beside B, uncaptured) over steps with D3 list rebuilds:
# the committed forces equal an eager force call at the committed positions, and the D3 term on the engine's list
# equals a D3 term built fresh at those positions.
from fastmace.longrange.d3_term import D3Term

def run(engine, steps=60):
    for _ in range(steps):
        engine.step()
    engine.flush()

out["modes"] = {}
for mode, options in (("checked", dict(pipelined="off")), ("pipelined", dict(pipelined="on", overlap_rebuild=True)),
                      ("uncaptured", dict(pipelined="on"))):
    engine = build(capture=False if mode == "uncaptured" else None, **options)
    run(engine)
    field, state = engine.force_field, engine.state
    eager = field(state.positions)  # MACE's forces need autograd
    with torch.no_grad():
        fresh = D3Term(RECORD, state.species, np.asarray(field.d3.list.cell), skin=field.d3.list.skin,
                       dtype=torch.float32)
        fresh.build(state.positions, None)
        ours, theirs = field.d3.evaluate(state.positions), fresh.evaluate(state.positions)
    info = engine.info()
    out["modes"][mode] = {
        "force_vs_eager": float((eager.forces - state.forces.to(eager.forces.dtype)).abs().max()),
        "d3_force_vs_fresh": float((ours.forces - theirs.forces).abs().max()),
        "d3_energy_vs_fresh": abs(float(ours.energy) - float(theirs.energy)),
        "d3_builds": field.d3.builds, "d3_regrowths": field.d3.regrowths,
        "invalidations": [entry.get("reason") for entry in info["invalidations"]],
        "early": info["pipelined_early_rebuilds"], "beside": info.get("rebuilds_beside"),
        "fused": info.get("fused_integrator"), "graphs": info["graphs"]}
    del engine, field, state

# 3. MC: the float64 energy copy adds D3; a rejected displacement block leaves the D3 list.
from fastmace.mc import Displace, MCDriver, moves
from fastmace.mc.evaluator import float64_copy
engine = build()
model64 = loading.load(SPEC["path"], None, backend="cueq", dtype="float64", device="cuda:0", trusted=True)
copy = float64_copy(engine.force_field, model64)
positions = engine.state.positions
mace64 = float(copy.mace.energy_only(positions)[0])
d364 = float(engine.force_field.d3.kernel_for(torch.float64).evaluate_energy(positions, engine.force_field.d3.cell))
out["copy_energy_diff"] = abs(float(copy.energy_only(positions)[0]) - mace64 - d364)
moves.metropolis = lambda u, excess, kT: torch.zeros_like(u, dtype=torch.bool)  # before attach: all rejected
driver = MCDriver([Displace(delta_A=0.3, every=1, attempts=8)], seed=6)
driver.attach(engine, max_displacement_A=0.3)
before = {name: tensor.clone() for name, tensor in engine.force_field.d3.buffers().items()}
engine.integrator.set_temperature(300.0)
out["rejected"] = driver.run_block(engine, 10)["moves"][0]
after = engine.force_field.d3.buffers()
out["d3_list_same"] = all(torch.equal(before[name], after[name]) for name in before)
print(json.dumps(out, default=str))
"""


def run(tmp_path, d3_skin):
    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    out = ENGINE.run_fresh(f"SPEC = {{'path': {str(path)!r}, 'd3_skin': {d3_skin!r}}}\n" + CHILD)
    print(out)
    return out


def test_d3_in_the_captured_force_graph(tmp_path):
    pytest.importorskip("torch_dftd")
    out = run(tmp_path, d3_skin=0.05)
    assert out["type"] == "DispersedForceField"
    # The gate's floors for float32 against float64 (SPEC §12.3; tests/_support/d3_cases.GATE).
    assert out["energy_per_atom"] < 1e-5 and out["force"] < 1e-4 and out["stress"] < 1e-6
    for mode, result in out["modes"].items():
        assert result["d3_builds"] > 1, (mode, result)
        assert result["force_vs_eager"] < 1e-5, (mode, result)
        assert result["d3_force_vs_fresh"] < 1e-6 and result["d3_energy_vs_fresh"] < 1e-9, (mode, result)
        assert not result["invalidations"], (mode, result)  # D3 rebuilds without growth keep the graphs
    assert out["modes"]["checked"]["fused"] is True
    assert out["copy_energy_diff"] < 1e-9
    assert out["rejected"]["accepted"] == 0 and out["rejected"]["trial_rebuilds"] > 0
    assert out["d3_list_same"]
