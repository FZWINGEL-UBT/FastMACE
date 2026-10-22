"""Displacement moves on the GPU (plan task 3-T10; SPEC §10.4 "Graph capture", §8.5 step 4, §12.2 "Mutation events").

A tiny MH-1-like model on NaCl 2x2x2, plain cuEq float32 through the fixed planner with a 0.1 A skin, in a
fresh process. A captured engine and an uncaptured twin (``capture=False``, same GPU) run the same blocks:

- the attempt graph is captured (key ``mc.displace``) and its warm-up leaves the block's counters alone:
  both engines take the same decisions and end at the same positions;
- small displacements stay in the graph; displacements beyond skin/2 take the host path (trial builds
  around the trial geometry, GPU neighbour build), accepted and rejected, without looping;
- a block of rejections leaves the committed list and positions bit for bit.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from _support.tiny_models import FAMILIES, tiny_mace_model


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_mc_displace_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import dataclasses, json
import numpy as np
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig, PrecisionConfig
from fastmace.planner import fixed

def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    return atoms

planner = fixed.FixedPlanner(engine=EngineConfig(backend="cueq"), precision=PrecisionConfig(dtype="float32"))
loaded = planner.load(SPEC["path"], trusted=True)
plan = planner.plan(loaded, structure(), IntegratorConfig(timestep_fs=1.0, friction=0.02, seed=7), planner.hardware())
plan = dataclasses.replace(plan, neighbours=dataclasses.replace(plan.neighbours, skin=0.1))
import torch
from fastmace.mc import Displace, MCDriver, moves

captured = planner.build(plan, loaded, structure(), temperature_K=300.0)
engines = (captured,)
if SPEC["twin"]:
    engines += (planner.build(plan, loaded, structure(), temperature_K=300.0, capture=False),)
names = ("captured", "twin")[:len(engines)]
for engine in engines:
    for _ in range(10):
        engine.step()
    engine.flush()

def lists(engine):
    nl = engine.force_field.neighbours
    return {name: tensor.clone() for name, tensor in nl.buffers().items()}

out = {"blocks": []}
drivers = {}
DELTAS = (0.01, 0.2) if SPEC["twin"] else (0.2,)
for delta in DELTAS:
    for name, engine in zip(names, engines):
        drivers[name, delta] = MCDriver([Displace(delta_A=delta, every=1, attempts=8)], seed=5)
        drivers[name, delta].attach(engine, max_displacement_A=0.2)
for delta in DELTAS:
    for temperature in (5.0, 300.0):
        records = []
        for name, engine in zip(names, engines):
            engine.integrator.set_temperature(temperature)
            records.append(drivers[name, delta].run_block(engine, 10)["moves"][0])
        out["blocks"].append(records)
out["positions_diff"] = float((captured.state.positions - engines[-1].state.positions).abs().max())
# A block of rejections with trial builds leaves the list and positions as they were.
moves.metropolis = lambda u, excess, kT: torch.zeros_like(u, dtype=torch.bool)  # before attach: all rejected
rejecting = MCDriver([Displace(delta_A=0.2, every=1, attempts=8)], seed=6)
rejecting.attach(captured, max_displacement_A=0.2)
before, positions = lists(captured), captured.state.positions.clone()
out["rejected"] = rejecting.run_block(captured, 10)["moves"][0]
after = lists(captured)
out["list_same"] = all(torch.equal(before[name], after[name]) for name in before)
out["positions_same"] = bool(torch.equal(positions, captured.state.positions))
out["graphs"] = [str(key) for key in captured._graphs]
out["twin_graphs"] = [str(key) for key in engines[-1]._graphs] if SPEC["twin"] else []
print(json.dumps(out, default=str))
"""


def run(tmp_path, twin):
    path = tmp_path / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    out = ENGINE.run_fresh(f"SPEC = {{'path': {str(path)!r}, 'twin': {twin!r}}}\n" + CHILD)
    print(out)
    return out


def test_displacement_attempts_captured_with_trial_builds(tmp_path):
    """The fast tier's cheaper check: the captured engine alone, displacements beyond skin/2."""
    out = run(tmp_path, twin=False)
    assert any("mc.displace" in key for key in out["graphs"])
    assert all(block[0]["attempted"] == 8 for block in out["blocks"])
    assert sum(block[0]["trial_rebuilds"] for block in out["blocks"]) > 0
    assert out["rejected"]["accepted"] == 0 and out["rejected"]["trial_rebuilds"] > 0
    assert out["list_same"] and out["positions_same"]


@pytest.mark.slow
def test_displacement_attempts_captured_and_on_the_host_path(tmp_path):
    """Against an uncaptured twin on the same GPU, small and large displacements, two temperatures."""
    out = run(tmp_path, twin=True)
    assert any("mc.displace" in key for key in out["graphs"]) and not out["twin_graphs"]
    small, large = out["blocks"][:2], out["blocks"][2:]
    for captured, twin in out["blocks"]:
        for key in ("attempted", "accepted", "trial_rebuilds"):
            assert captured[key] == twin[key], (key, captured, twin)
        assert captured["attempted"] == 8
    assert all(block[0]["trial_rebuilds"] == 0 for block in small)
    assert sum(block[0]["trial_rebuilds"] for block in large) > 0
    assert out["positions_diff"] <= 1e-6
    assert out["rejected"]["accepted"] == 0 and out["rejected"]["trial_rebuilds"] > 0
    assert out["list_same"] and out["positions_same"]
