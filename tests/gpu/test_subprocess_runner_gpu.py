"""The subprocess runner on the GPU (plan task 2-T07; SPEC §8.7; phase 2 acceptance item 4), with MH-1.

Each case is a fresh process in the hybrid environment (``FASTMACE_HYBRID_PYTHON``, which has cuEq
and OpenEquivariance), NaCl 3x3x3, Langevin at 300 K, 2 fs:

- **plain cuEq float64 after a compiled hybrid.** The process runs a hybrid ``Simulation`` first
  (its compile fixes the process's family), then asks for a plain-cuEq float64 plan
  (``EngineConfig(backend="cueq")``). The planner marks it ``runner="subprocess"``, and it runs in
  children: setup, then three ``run(100)`` calls (two hand-offs between runs). The output chunks
  hold every thermo row and frame once, and the trajectory equals a plain-cuEq run started in a
  fresh process, with restart writes at the same steps, within 1e-12 Å (1b's bound: plain cuEq is
  not bitwise even in one process, SPEC §18);
- **hybrid after a compiled plain cuEq** (the other direction): the hybrid plan runs in children;
  the run completes, the parent reports the child's hybrid plan, and the chunks are complete. The
  statistical comparison of long hybrid runs is 2-T23's (as 1b-T14's 20,000 steps).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_runner_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.slow, pytest.mark.acceptance,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
    ENGINE.needs_oeq,
]

DRIVER = """
import fastmace
import fastmace as fm
import torch
from ase import units
from fastmace.config import EngineConfig
from fastmace.io import output as output_io

STEPS, RUNS, EVERY, SEED, TEMPERATURE, TIMESTEP, FRICTION = 100, 3, 10, 20261020, 300.0, 2.0, 0.02
template = structure()
natoms = len(template)
integrator = fm.integrators.Langevin(timestep_fs=TIMESTEP, temperature_K=TEMPERATURE,
                                     friction_per_fs=FRICTION * units.fs)
model = fm.load("mh1-r2scan")


def first(backend, precision):
    # the plan whose compile fixes this process's family
    ff = fm.ForceField(model, template, precision=precision)
    sim = fm.Simulation(template.copy(), ff, integrator=integrator, seed=SEED, engine=EngineConfig(backend=backend))
    sim.run(steps=20)
    out = {"backend": sim.plan.engine.backend, "runner": sim.plan.runner}
    sim.close()
    return out


def run(directory, backend, precision, save_points=None):
    ff = fm.ForceField(model, template, precision=precision)
    output = fm.Output(directory, thermo_every=EVERY, trajectory_every=EVERY, restart_every=None)
    started = time.perf_counter()
    sim = fm.Simulation(template.copy(), ff, integrator=integrator, output=output, seed=SEED,
                        engine=EngineConfig(backend=backend))
    setup_s = time.perf_counter() - started
    if save_points:
        sim.save_restart(save_points)
    rows = []
    for _ in range(RUNS):
        started = time.perf_counter()
        sim.run(steps=STEPS)
        seconds = time.perf_counter() - started
        if save_points:
            sim.save_restart(save_points)
        committed = sim.committed()
        rows.append({"step": committed.step, "energy": committed.energy / natoms, "seconds": seconds,
                     "positions": committed.positions.tolist()})
    metadata = sim.metadata()
    sim.close()
    header, thermo = output_io.read_thermo(directory)
    frames = output_io.read_frames(directory)
    return {"rows": rows, "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype,
            "runner": sim.plan.runner, "setup_s": setup_s, "report": sim.report.text(),
            "subprocess": metadata.get("subprocess"),
            "restart_points": [point["step"] for point in metadata["restart_points"]],
            "thermo_steps": [int(row[0]) for row in thermo], "frame_count": len(frames),
            "handoff_left": (Path(directory) / ".fastmace-handoff").exists()}


# Nothing touches CUDA before the first Simulation: the planner sets the allocator before the first CUDA use.
out = {}
if SPEC["case"] == "reference":
    out["run"] = run(SPEC["directory"] + "/reference", SPEC["second"], SPEC["precision"],
                     save_points=SPEC["directory"] + "/points")
else:
    out["first"] = first(SPEC["first"], "float32")
    out["run"] = run(SPEC["directory"] + "/runner", SPEC["second"], SPEC["precision"])
out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fastmace=fastmace.__version__)
print(json.dumps(out))
"""


def driver(tmp_path, case, *, first, second, precision):
    spec = ENGINE.spec("hybrid")
    spec.update(case=case, first=first, second=second, precision=precision, directory=str(tmp_path))
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + "\nfrom pathlib import Path\n" + DRIVER,
                            python=ENGINE.HYBRID_PYTHON)


def _complete_chunks(run):
    steps = list(range(0, 301, 10))
    assert run["thermo_steps"] == steps  # every row once
    assert run["frame_count"] == len(steps)  # every frame once
    assert run["runner"] == "subprocess" and run["subprocess"]["children"] == 4
    assert not run["handoff_left"]


def test_plain_cueq_float64_after_a_hybrid_runs_in_a_child_within_1e_12(tmp_path):
    result = driver(tmp_path, "runner", first="hybrid", second="cueq", precision="float64")
    reference = driver(tmp_path, "reference", first=None, second="cueq", precision="float64")
    assert result["first"] == {"backend": "hybrid", "runner": "in-process"}
    run, ref = result["run"], reference["run"]
    assert (run["backend"], run["dtype"]) == ("cueq", "float64") == (ref["backend"], ref["dtype"])
    _complete_chunks(run)
    assert run["restart_points"] == ref["restart_points"] == [0, 100, 200, 300]
    deviation = max(float(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"])).max())
                    for a, b in zip(run["rows"], ref["rows"], strict=True))
    print(f"plain cuEq float64 in a child vs a fresh process: {deviation:.3g} Å; setup {run['setup_s']:.1f} s; "
          f"runs {[round(row['seconds'], 1) for row in run['rows']]} s; {result['gpu']}")
    assert deviation <= 1e-12


def test_hybrid_after_a_plain_cueq_runs_in_a_child(tmp_path):
    result = driver(tmp_path, "runner", first="cueq", second="hybrid", precision="float32")
    assert result["first"] == {"backend": "cueq", "runner": "in-process"}
    run = result["run"]
    assert (run["backend"], run["dtype"]) == ("hybrid", "float32")
    _complete_chunks(run)
    assert all(np.isfinite(row["energy"]) for row in run["rows"])
    assert "subprocess" in run["report"]
