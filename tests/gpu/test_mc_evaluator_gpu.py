"""The energy-only graph and the MC evaluator on the GPU (plan task 3-T07; SPEC §8.1 "Energy-only graph", §10.4).

A tiny MH-1-like model on NaCl 2x2x2, plain cuEq float32 through the fixed planner, in fresh processes:

- the captured energy-only graph (key ``"E"``) gives the force path's energy within float32 rounding of
  the forward, its output has no ``grad_fn``, and the float64 copy (eager, captured) equals a fresh
  float64 force field's energy;
- a later force capture is unaffected: an engine that evaluated energies between its steps and then
  re-captured its force graphs runs as a twin that never did (same positions and forces, up to the
  reduction-order noise of the GPU kernels);
- the baseline after MD equals a fresh evaluation with the MC evaluator, never the MD energy.
"""

import importlib.util
from pathlib import Path

import torch

from _support.tiny_models import FAMILIES, tiny_mace_model


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_mc_evaluator_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")


def tiny_mh1(directory):
    path = directory / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    return str(path)


CHILD = """
import json
import numpy as np
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig, PrecisionConfig
from fastmace.planner import fixed

def planner(dtype):
    return fixed.FixedPlanner(engine=EngineConfig(backend="cueq"), precision=PrecisionConfig(dtype=dtype))

def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
    return atoms

single = planner("float32")
hardware = single.hardware()
loaded = single.load(SPEC["path"], trusted=True)
loaded64 = planner("float64").load(SPEC["path"], trusted=True)
import torch
from fastmace.mc.evaluator import MCEvaluator
from fastmace.model.forces import MACEForceField

integrator = IntegratorConfig(timestep_fs=0.5, friction=0.02, seed=7)
plan = single.plan(loaded, structure(), integrator, hardware)
engines = [single.build(plan, loaded, structure(), temperature_K=600.0) for _ in range(2)]
probe, twin = engines
for engine in engines:
    for _ in range(20):
        engine.step()
    engine.flush()
out = {}

ev32 = MCEvaluator(probe)
ev64 = MCEvaluator(probe, "float64", model=loaded64)
e32 = ev32.baseline()
e64 = ev64.baseline()
torch.cuda.synchronize()
out["grad_fn"] = [repr(e32.grad_fn), repr(e64.grad_fn), bool(e32.requires_grad), bool(e64.requires_grad)]
out["md_energy"] = float(probe.state.energy[0])
out["e32"], out["e64"] = float(e32[0]), float(e64[0])
atoms = structure()
atoms.positions = probe.state.positions.detach().cpu().numpy()
fresh = MACEForceField(loaded64, atoms, neighbours=plan.neighbours)
out["fresh64"] = float(fresh.energy_only(torch.as_tensor(atoms.positions, device="cuda:0"))[0])
out["info32"], out["info64"] = ev32.info(), ev64.info()
out["graphs"] = [str(key) for key in probe._graphs]

# Energies between steps, then a re-capture of the force graphs; the twin never evaluates energies.
for _ in range(10):
    for evaluator in (ev32, ev64):
        evaluator.evaluate()
    probe.step()
    twin.step()
for engine in engines:
    engine.flush()
    engine.invalidate_graphs("3-T07 test: force graphs re-captured after energy-only graphs")
for _ in range(30):
    probe.step()
    twin.step()
    if _ % 10 == 0:
        ev32.evaluate()
for engine in engines:
    engine.flush()
out["positions_diff"] = float((probe.state.positions - twin.state.positions).abs().max())
out["forces_diff"] = float((probe.state.forces - twin.state.forces).abs().max())
# The baseline after MD: a fresh evaluation, never the MD energy.
out["after_md"] = [float(ev64.baseline()[0]), float(ev64.evaluate()[0])]
print(json.dumps(out, default=str))
"""


def test_energy_only_graph_and_evaluator_tiny_plain_cueq(tmp_path):
    spec = {"path": tiny_mh1(tmp_path)}
    out = ENGINE.run_fresh(f"SPEC = {spec!r}\n" + CHILD)
    print(out)
    assert out["grad_fn"] == ["None", "None", False, False]
    assert abs(out["e32"] - out["md_energy"]) <= 1e-5 * abs(out["md_energy"])  # float32 forward, two graphs
    assert abs(out["e64"] - out["fresh64"]) <= 1e-12 * abs(out["fresh64"])
    assert abs(out["e32"] - out["e64"]) <= 1e-5 * abs(out["e64"])
    assert out["info32"]["captured"] and out["info64"]["captured"]
    assert out["info64"]["copy"] and out["info64"]["compile_mode"] == "none" and out["info64"]["memory_bytes"] > 0
    assert any("'E'" in key for key in out["graphs"])
    # Bitwise where the GPU kernels are deterministic; within float32 reduction-order noise otherwise.
    assert out["positions_diff"] <= 1e-6 and out["forces_diff"] <= 1e-4, out
    assert out["after_md"][0] == out["after_md"][1]
