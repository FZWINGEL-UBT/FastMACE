"""``reproducible=True`` on the GPU (plan task 2-T16; SPEC §9.3, §8.7; phase 2 acceptance item 6).

MH-1 (``mh1-r2scan``) on NaCl 3x3x3 and NbCl5 2x2x2, float64 and float32, Langevin at 300 K, 2 fs,
every run in a fresh process (``FASTMACE_MODELS``):

- **repeated and interrupted runs.** Three fresh runs of 500 steps with a restart write at step 250
  (the same restart cadence, §9.3's scope), and the first run's restart resumed in a fresh process
  for the last 250 steps: positions and momenta bitwise equal. Each process is reproducible: the plan
  is plain cuEq with the unfused convolution, in-process, qualified, with the three variables set;
- **the runner trigger** (2-T07's offer, its GPU case): ``reproducible=True`` asked for after this
  process initialised CUDA runs in a child, and equals a fresh reproducible process with a restart
  write at the same step bitwise.

Marked ``slow`` and ``acceptance``: about 25 minutes on an RTX 4000 Ada.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_reproducible_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow, pytest.mark.acceptance,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

STEPS, HALF = 500, 250

DRIVER = """
import json, os, time
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl, nbcl5
from fastmace.planner import reproducible

SPEC = {spec!r}
if SPEC["touch_cuda"]:
    torch.zeros(1, device="cuda")  # CUDA started before the plan: the runner trigger
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
started = time.perf_counter()
if SPEC["action"] == "resume":
    sim = fm.Simulation.resume(SPEC["directory"])
else:
    atoms = nacl(repeat=3) if SPEC["structure"] == "NaCl" else nbcl5(repeat=2)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    ff = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision=SPEC["dtype"])
    sim = fm.Simulation(atoms, ff, integrator=integrator, seed=20261020,
                        workload=fm.Workload(max_temperature_K=300.0, reproducible=True))
    if SPEC["save_at_start"]:
        sim.save_restart(SPEC["directory"])
for index, steps in enumerate(SPEC["runs"]):
    sim.run(steps=steps)
    if SPEC["save_after_first"] and index == 0:
        sim.save_restart(SPEC["directory"])
committed = sim.committed()
plan = sim.plan
out = {{
    "step": committed.step, "positions": committed.positions.tobytes().hex(),
    "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy,
    "backend": plan.engine.backend, "fusion": plan.engine.cueq_conv_fusion, "dtype": plan.precision.dtype,
    "runner": plan.runner, "decisions": [[key, choice] for key, choice, _ in plan.decisions
                                         if key.startswith("reproducible") or key == "runner"],
    "flag": torch.are_deterministic_algorithms_enabled(),
    "environment": {{name: os.environ.get(name) for name in reproducible.PROCESS_ENVIRONMENT}},
    "seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(),
}}
sim.close()
print(json.dumps(out))
"""


def driver(directory, *, structure="NaCl", dtype="float64", action="new", runs=(HALF, STEPS - HALF),
           save_after_first=True, save_at_start=False, touch_cuda=False):
    spec = dict(structure=structure, dtype=dtype, action=action, runs=list(runs), directory=str(directory),
                save_after_first=save_after_first, save_at_start=save_at_start, touch_cuda=touch_cuda)
    return ENGINE.run_fresh(DRIVER.format(spec=spec))


def _same(a, b):
    return (a["step"], a["positions"], a["momenta"]) == (b["step"], b["positions"], b["momenta"])


def _reproducible_process(run, dtype):
    from fastmace.planner import reproducible

    assert (run["backend"], run["fusion"], run["dtype"]) == ("cueq", "off", dtype)
    assert run["flag"] is True and run["environment"] == dict(reproducible.PROCESS_ENVIRONMENT)
    assert any(key == "reproducible" and choice.startswith("qualified") for key, choice in run["decisions"])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("structure", ["NaCl", "NbCl5"])
def test_repeated_and_interrupted_runs_are_bitwise_equal(tmp_path, structure, dtype):
    runs = [driver(tmp_path / f"run{index}", structure=structure, dtype=dtype) for index in range(3)]
    resumed = driver(tmp_path / "run0", structure=structure, dtype=dtype, action="resume", runs=(STEPS - HALF,),
                     save_after_first=False)
    for run in (*runs, resumed):
        _reproducible_process(run, dtype)
        assert run["runner"] == "in-process" and run["step"] == STEPS
    print(f"{structure} {dtype}: 3 x {STEPS} steps and a resume at {HALF}; energy {runs[0]['energy']:.10f} eV; "
          f"seconds {[round(run['seconds'], 1) for run in (*runs, resumed)]}; {runs[0]['gpu']}")
    assert all(_same(run, runs[0]) for run in runs[1:]), "repeated runs differ"
    assert _same(resumed, runs[0]), "the interrupted run differs from the uninterrupted one"


def test_reproducible_after_cuda_started_runs_in_a_child_bitwise(tmp_path):
    child = driver(tmp_path / "child", runs=(100,), save_after_first=False, touch_cuda=True)
    fresh = driver(tmp_path / "fresh", runs=(100,), save_after_first=False, save_at_start=True)
    assert child["runner"] == "subprocess" and fresh["runner"] == "in-process"
    assert ["runner", "subprocess"] in child["decisions"]
    _reproducible_process(fresh, "float64")
    assert _same(child, fresh), "the child's run differs from a fresh reproducible process"
