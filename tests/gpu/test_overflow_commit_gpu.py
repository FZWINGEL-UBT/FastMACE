"""Overflow before commit on the GPU (plan task 3-T04, SPEC §8.1, §12.2 "replay and transitions"), fast tier.

Tiny models (``_support.tiny_models``) on NaCl 2x2x2, planned and built by the fixed planner and stepped
as captured graphs, in fresh processes. Each run steps a reference engine and an engine whose steps are
made to overflow, side by side, and compares them:

- **edges**, tiny MH-1 on plain cuEq float64 with the production defaults (pipelined, the fused
  integrator, CUDA graphs): after the validation of the listed steps the next force stage's padded
  size is shrunk below the coming count (a monkeypatched capacity). Each such step is redone from stage
  A's output with the same noise (conductor answer 2), so the run stays within 1e-10 Angstrom of the
  reference (cuEq is not bitwise reproducible, SPEC §18);
- **k-vectors**, tiny PolarMACE NPT (eager-captured: compiled plain cuEq is refused for PolarMACE,
  rule 3), checked and pipelined: the k-space reference cell's inverse (a device buffer the captured
  force graph reads) is scaled, so the next force stage's deformation check sets the sticky flag inside
  the graph. The per-step check (one host sync after B) catches it in that step, rebuilds the k-vector
  set for the step's cell and redoes the step; no canonical point is involved.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from _support.tiny_models import FAMILIES, polar_available, tiny_mace_model


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_overflow_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import json
import numpy as np
import fastmace
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.planner import fixed

planner = fixed.FixedPlanner(engine=EngineConfig(**SPEC["engine"]))
hardware = planner.hardware()
loaded = planner.load(SPEC["path"], trusted=True)
import torch
from fastmace.model import compile as fc


def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
    return atoms


cache = fc.CompiledModelCache()
config = IntegratorConfig(timestep_fs=1.0, friction=0.02, seed=7, **SPEC["integrator"])
plan = planner.plan(loaded, structure(), config, hardware)
if SPEC["pipelined"] is not None:
    import dataclasses
    plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, pipelined=SPEC["pipelined"]))
build = lambda: planner.build(plan, loaded, structure(), temperature_K=600.0, compile_cache=cache)
reference, engine = build(), build()
for step in range(1, SPEC["steps"] + 1):
    reference.step()
    engine.step()
    if step in SPEC["forced"]:
        engine.flush()
        if SPEC["kind"] == "edges":
            engine._pipe_edges = int(engine.force_field.edge_count) - SPEC["shrink"]
        else:
            engine.force_field._polar_state().inverse_cell0.mul_(1.5)
engine.flush()
reference.flush()
got, want = engine.sync(), reference.sync()
info = engine.info()
out = {key: info.get(key) for key in ("mode", "fused_integrator", "overflow_check", "overflows_redone",
                                      "overflow_log", "polar_overflow_steps", "canonical_points", "recaptures",
                                      "recapture_budget", "pipelined_redos", "capture")}
out["decisions"] = [item for item, _, _ in plan.decisions]
out["uncommitted"] = engine.uncommitted()
out["max_dx"] = float(np.abs(got["positions"] - want["positions"]).max())
out["max_de"] = float(abs(got["energy"][0] - want["energy"][0]))
out["reference_redone"] = reference.info()["overflows_redone"]
print(json.dumps(out))
"""


def _run(directory, family, **spec):
    path = directory / f"tiny-{family}.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES[family]), path)
    spec = {"path": str(path), "steps": 30, "pipelined": None, "integrator": {}, **spec}
    return ENGINE.run_fresh(f"SPEC = {spec!r}\n" + CHILD)


def test_an_edge_overflow_is_redone_with_the_fused_integrator(tmp_path):
    forced = [6, 13, 20]
    out = _run(tmp_path, "mh1", engine={"backend": "cueq"}, kind="edges", forced=forced, shrink=40)
    print(out)
    assert out["mode"] == "pipelined" and out["capture"]
    assert out["fused_integrator"] is True  # the default AC/ACP stage
    assert out["overflow_check"] == "edges"
    assert out["reference_redone"] == 0
    assert out["overflows_redone"] == len(forced)
    assert [entry["step"] for entry in out["overflow_log"]] == [step + 1 for step in forced]
    assert not out["uncommitted"]
    assert out["max_dx"] < 1e-10 and out["max_de"] < 1e-9


@pytest.mark.parametrize("pipelined", ["off", "on"])
def test_a_kvector_overflow_is_caught_in_its_own_step(tmp_path, pipelined):
    if not polar_available():
        pytest.skip("PolarMACE needs MACE's polar module")
    out = _run(tmp_path, "polar", engine={"backend": "cueq", "compile_mode": "none"}, kind="kvectors",
               forced=[8], pipelined=pipelined, integrator={"ensemble": "npt", "pressure_GPa": 0.5})
    print(out)
    assert out["capture"]
    assert out["overflow_check"] == "per step"
    assert "overflow check" in out["decisions"]
    kvectors = [entry["step"] for entry in out["overflow_log"] if "k-vectors" in entry["kinds"]]
    assert kvectors == [9]
    assert out["overflows_redone"] == len(out["overflow_log"])  # pipelined: edge-count redos too
    assert out["polar_overflow_steps"] == [9]
    assert out["canonical_points"] == 0
    assert out["max_dx"] < 1e-9


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
