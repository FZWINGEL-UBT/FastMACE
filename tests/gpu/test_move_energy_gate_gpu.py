"""The move-energy gate on the GPU (plan task 3-T11; SPEC §12.3 "move-energy differences", §10.4).

A tiny MH-1-like model over the superset {O, Na, Cl} on NaCl 2x2x2 (O absent), plain cuEq float32 through
the minimal planner, in a fresh process: the float64 references run in the gate's child process, the
production ΔE come from the captured energy-only graph; the gate passes, leaves the registered state
unchanged, samples the absent species, and a species-dependent constant error in the float32 energies
moves "auto" to the float64 copy.
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


ENGINE = _module("_move_gate_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")


def tiny_mh1(directory):
    path = directory / "tiny-mh1-ona-cl.model"
    torch.save(tiny_mace_model([8, 11, 17], **FAMILIES["mh1"]), path)
    return str(path)


CHILD = """
import json
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.planner.minimal import MinimalPlanner

planner = MinimalPlanner(engine=EngineConfig(backend="cueq"))
hardware = planner.hardware()
import torch
from ase.build import bulk
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import selfcheck
from fastmace.planner.workload import Workload

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
atoms.rattle(0.05, seed=1)
loaded = loading.load_pristine(SPEC["path"], trusted=True)
ff = ForceField(loaded, atoms, precision="float32", species=(8, 11, 17))
workload = Workload(max_temperature_K=1000.0, species_changes=True, mc_moves=("swap", "transmute", "displace"),
                    min_mc_temperature_K=250.0, max_displacement_A=0.1)
integrator = IntegratorConfig(temperature_K=300.0)
plan = planner.plan(ff, atoms, integrator, workload, hardware)
moves = selfcheck.move_references(plan, ff, atoms, workload, seed=3)
engine = planner.build(plan, ff, atoms.copy(), temperature_K=300.0)
engine.prepare()
before = {name: tensor.detach().clone() for name, tensor in engine.registry.tensors().items()}
record = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
after = engine.registry.tensors()
out = {"ran": moves.ran, "capture": bool(engine.capture), "part": record.move_energy,
       "changed": [name for name in before if not torch.equal(before[name], after[name])],
       "graphs": [str(key) for key in engine._graphs]}

field = engine.force_field
original = field.energy_only
def shifted(positions):
    return original(positions) + 1e-3 * (field.atomic_numbers == 11).sum().to(torch.float64)
field.energy_only = shifted
engine.invalidate_graphs("3-T11 test: the injected error must reach the captured energy graph")
injected = selfcheck.move_gate(engine, moves, plan=plan, force_field=ff)
out["injected"] = injected.move_energy
print(json.dumps(out, default=str))
"""


@pytest.mark.slow  # 64 s on the local RTX 5060 Ti (fresh process, reference child); the logic is in the CPU tests
def test_move_energy_gate_captured_tiny_plain_cueq(tmp_path):
    out = ENGINE.run_fresh(f"SPEC = {{'path': {tiny_mh1(tmp_path)!r}}}\n" + CHILD)
    part = out["part"]
    print({key: part.get(key) for key in ("status", "precision", "reason", "moves", "tried", "mode", "references")})
    assert out["ran"] == "child process" and out["capture"]
    assert part["status"] == "passed" and part["precision"] == "float32", part["tried"]
    assert part["mode"] == "captured energy-only graph"
    assert any("'E'" in key for key in out["graphs"])
    assert out["changed"] == []
    assert any("-> O" in row["label"] for row in part["rows"])
    injected = out["injected"]
    assert [entry["precision"] for entry in injected["tried"]] == ["float32", "float64"]
    assert [entry["status"] for entry in injected["tried"]] == ["failed", "passed"]
    assert abs(injected["tried"][0]["moves"]["transmute"]["worst_eV"] - 1e-3) < 1e-4
