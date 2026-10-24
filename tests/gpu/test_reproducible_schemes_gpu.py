"""Plan task 4-T13 (phase 4 acceptance, item 4): the new integrators' restarts on the GPU, bitwise.

``test_reproducible_gpu.py`` covers Langevin. This file runs each phase 4 scheme the same way: MH-1
(``mh1-r2scan``) on NaCl 3x3x3, ``reproducible=True``, float64 and float32, every run in a fresh process. An
uninterrupted run of 200 steps writes a restart at step 100 (the same restart cadence, §9.3's scope); a fresh
process resumes that restart for the last 100 steps. Positions, momenta and the cell are bitwise equal.

- The schemes are 4-T02's and 4-T03's records (``test_nve_csvr_berendsen_gpu.py``,
  ``test_nose_hoover_mtk_mh1_gpu.py``), 0.5 fs at 300 K, and Langevin with ``FixAtoms`` on 10 atoms
  (``fixcm="auto"``: off).
- CSVR's restart falls inside a noise batch (cursor 100 of 256); the NPT schemes' cell at the restart differs
  from the starting cell (h ≠ h₀), both asserted.

Marked ``slow`` and ``acceptance``.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).parent
ENGINE = _module("_reproducible_schemes_engine_helpers", HERE / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow, pytest.mark.acceptance,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

STEPS, HALF = 200, 100
FIXED = [0, 5, 17, 42, 64, 99, 128, 150, 177, 211]
NPT = ("npt-berendsen", "inhomogeneous-berendsen", "mtk-hydrostatic", "mtk-full")
SCHEMES = {
    "nve": ("test_nve_csvr_berendsen_gpu.py", "nve"),
    "csvr": ("test_nve_csvr_berendsen_gpu.py", "csvr"),
    "nvt-berendsen": ("test_nve_csvr_berendsen_gpu.py", "nvt-berendsen"),
    "npt-berendsen": ("test_nve_csvr_berendsen_gpu.py", "npt-berendsen"),
    "inhomogeneous-berendsen": ("test_nve_csvr_berendsen_gpu.py", "inhomogeneous-berendsen"),
    "nhc": ("test_nose_hoover_mtk_mh1_gpu.py", "nhc"),
    "mtk-hydrostatic": ("test_nose_hoover_mtk_mh1_gpu.py", "hydrostatic"),
    "mtk-full": ("test_nose_hoover_mtk_mh1_gpu.py", "full"),
    "langevin-fix-atoms": (None, None),
}

DRIVER = """
import importlib.util, json, time
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from _support.structures import nacl

SPEC = {spec!r}


def record():
    if SPEC["helpers"] is None:
        return fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
    spec = importlib.util.spec_from_file_location("_scheme_helpers", SPEC["helpers"])
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    if hasattr(helpers, "records"):  # 4-T02's file
        return helpers.records()[SPEC["name"]]
    return helpers.record(SPEC["name"])  # 4-T03's


started = time.perf_counter()
if SPEC["action"] == "resume":
    sim = fm.Simulation.resume(SPEC["directory"])
else:
    atoms = nacl(repeat=3)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    Stationary(atoms)
    if SPEC["fixed"]:
        atoms.set_constraint(FixAtoms(indices=SPEC["fixed"]))
    ff = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision=SPEC["dtype"])
    sim = fm.Simulation(atoms, ff, integrator=record(), seed=20261023,
                        workload=fm.Workload(max_temperature_K=600.0, reproducible=True))
cells = {{}}
for index, steps in enumerate(SPEC["runs"]):
    sim.run(steps=steps)
    if SPEC["save_after_first"] and index == 0:
        sim.save_restart(SPEC["directory"])
        cells["restart"] = sim.committed().cell.tolist()
        cells["noise_cursor"] = getattr(sim._engine.state.rng, "cursor", None)
committed = sim.committed()
out = {{
    "step": committed.step, "positions": committed.positions.tobytes().hex(),
    "momenta": committed.momenta.tobytes().hex(), "cell": committed.cell.tobytes().hex(),
    "energy": committed.energy, "cells": cells, "start_cell": None if SPEC["action"] == "resume" else
    atoms.cell.array.tolist(),
    "dtype": sim.plan.precision.dtype, "runner": sim.plan.runner, "scheme": sim.plan.integrator.scheme,
    "seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(),
}}
sim.close()
print(json.dumps(out))
"""


def driver(directory, scheme, dtype, *, action="new", runs=(HALF, STEPS - HALF), save_after_first=True):
    helpers, name = SCHEMES[scheme]
    spec = dict(helpers=None if helpers is None else str(HERE / helpers), name=name, dtype=dtype, action=action,
                runs=list(runs), directory=str(directory), save_after_first=save_after_first,
                fixed=FIXED if scheme == "langevin-fix-atoms" else [])
    return ENGINE.run_fresh(DRIVER.format(spec=spec))


def _same(a, b):
    return [(run["step"], run["positions"], run["momenta"], run["cell"]) for run in (a, b)]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_an_interrupted_run_equals_the_uninterrupted_one_bitwise(tmp_path, scheme, dtype):
    full = driver(tmp_path / "run", scheme, dtype)
    resumed = driver(tmp_path / "run", scheme, dtype, action="resume", runs=(STEPS - HALF,), save_after_first=False)
    print(f"{scheme} {dtype}: {STEPS} steps, resumed at {HALF}; energy {full['energy']:.10f} eV; "
          f"seconds {round(full['seconds'], 1)}, {round(resumed['seconds'], 1)}; {full['gpu']}")
    for run in (full, resumed):
        assert run["runner"] == "in-process" and run["step"] == STEPS and run["dtype"] == dtype, run
    if scheme == "csvr":
        assert 0 < full["cells"]["noise_cursor"] < 256, "the restart is not inside a noise batch"
    if scheme in NPT:
        assert full["cells"]["restart"] != full["start_cell"], "the cell did not move before the restart"
    first, second = _same(full, resumed)
    assert first == second, "the interrupted run differs from the uninterrupted one"
