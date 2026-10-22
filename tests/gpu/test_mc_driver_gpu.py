"""The MC driver's captured attempts on the GPU (plan task 3-T09; SPEC §10.4 "Graph capture"), fast tier.

A tiny MH-1-like model (Na, Cl, K superset) on NaCl 2x2x2, plain cuEq float64 through ``Simulation``, in a
fresh process. The attempts of a swap and a transmutation are captured (keys ``("mc.<move>", ...)``):

- with every decision forced to reject, a block leaves the registered state bit for bit as it was;
- with every decision forced to accept, the captured engine and an uncaptured twin end with the same
  species, masses and velocities (bitwise; the energies agree within the GPU's summation noise), the
  same counters and stream offsets;
- with the Metropolis rule at a high temperature, the counters add up, a swap keeps the composition,
  and a first capture leaves no trace (the twin's counters after the same blocks are equal);
- the block's device time is measured (CUDA events).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_mc_driver_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import json, pathlib, tempfile
import torch
import fastmace
from _support.mc_fixtures import tiny_loaded, mc_simulation
from _support.structures import nacl
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.engine import constraints
from fastmace.planner import minimal
from fastmace.mc import MCDriver, Swap, Transmute, moves

planner = minimal.MinimalPlanner(engine=EngineConfig(backend="cueq", compile_mode="none", expandable_segments="off"),
                                 environ={},
                                 process=constraints.ProcessRecord())
minimal.default_planner = lambda force_field: planner
loaded = tiny_loaded(pathlib.Path(tempfile.mkdtemp()), (11, 17, 19))
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=3000.0, friction_per_fs=0.01)
METROPOLIS = moves.metropolis
MOVES = [Swap([("Na", "Cl")], every=2, attempts=4), Transmute(["Na", "K"], every=2, attempts=3)]

def build(capture):
    sim = mc_simulation(loaded, nacl(repeat=2, rattle=0.05, seed=3), (11, 17, 19), integrator=LANGEVIN)
    sim._engine.capture = capture
    driver = MCDriver(MOVES, seed=5)
    driver.attach(sim._engine)
    return sim, driver

def registered(engine):
    return {name: tensor.detach().clone() for name, (_, tensor) in engine.registry.entries().items()}

out = {}
# Forced rejections: the state stays bit for bit.
moves.metropolis = lambda u, excess, kT: torch.zeros_like(u, dtype=torch.bool)
sim, driver = build(True)
sim.run(steps=2)
sim._engine.canonicalise()
before = registered(sim._engine)
record = driver.run_block(sim._engine, 2)
after = registered(sim._engine)
out["reject_differs"] = sorted(name for name in before if not torch.equal(before[name], after[name]))
out["reject_moves"] = record["moves"]
out["device_ms"] = record["device_ms"]
out["mc_graphs"] = sorted(str(key) for key in sim._engine._graphs if str(key).startswith("('mc."))

# Forced acceptances: captured against an uncaptured twin.
moves.metropolis = lambda u, excess, kT: torch.ones_like(u, dtype=torch.bool)
pair = [build(True), build(False)]
for step in (2, 4):
    for sim, driver in pair:
        sim.run(steps=2)
        driver.run_block(sim._engine, step)
(a, da), (b, db) = pair
sa, sb = a._engine.state, b._engine.state
out["accept_same"] = {name: bool(torch.equal(getattr(sa, name), getattr(sb, name)))
                      for name in ("species", "masses", "velocities")}
out["accept_positions"] = float((sa.positions - sb.positions).abs().max())
out["accept_counters"] = [da.counters, db.counters]
out["accept_offsets"] = [da.offsets, db.offsets]
out["accept_energy"] = [da.last_block["energy_eV"], db.last_block["energy_eV"]]

# The Metropolis rule at 3000 K.
moves.metropolis = METROPOLIS
sim, driver = build(True)
counts = []
for step in (2, 4, 6, 8):
    sim.run(steps=2)
    driver.run_block(sim._engine, step)
    counts.append(int((sim._engine.state.species == 17).sum()))
out["metropolis_counters"] = driver.counters
out["chlorine"] = counts
out["records"] = driver.state_dict()
print(json.dumps(out, default=str))
"""


def test_captured_attempts_tiny_plain_cueq():
    out = ENGINE.run_fresh(CHILD)
    print(out)
    assert out["reject_differs"] == []
    assert [entry["accepted"] for entry in out["reject_moves"]] == [0, 0]
    assert out["device_ms"] is not None and out["device_ms"] > 0
    assert len(out["mc_graphs"]) == 2  # one captured attempt per move at this edge size
    assert out["accept_same"] == {"species": True, "masses": True, "velocities": True}
    assert out["accept_positions"] < 1e-9
    assert out["accept_counters"][0] == out["accept_counters"][1]
    assert out["accept_offsets"][0] == out["accept_offsets"][1]
    assert abs(out["accept_energy"][0] - out["accept_energy"][1]) < 1e-9 * abs(out["accept_energy"][0])
    counters = out["metropolis_counters"]
    assert counters["swap"]["attempted"] == 16 and counters["transmute"]["attempted"] == 12
    assert counters["swap"]["impossible"] == 0
    assert out["chlorine"] == [32, 32, 32, 32]  # swaps and Na <-> K keep the chlorine count
