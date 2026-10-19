"""Canonical restart points on the GPU (plan task 1b-T14, SPEC §10.6, §12.2 restart row), with MH-1.

Each run is a fresh process that plans and builds the production engine for NaCl 3x3x3 with the
fixed planner, as ``main.py`` does, ramps the thermostat target and writes a restart file every
``cadence`` steps (``fastmace.io.restart.write_engine_restart``: a canonical restart point). An
interrupted run resumes a freshly built engine from the file at the listed steps
(``resume_engine``); the uninterrupted run keeps its engine. Same restart cadence on both sides.

- **plain cuEq float64**: 500 steps, a restart every 100, resumed at every restart; positions within
  1e-12 Å of the uninterrupted run at every output step. Plain cuEq is not bitwise reproducible even
  uninterrupted (cuEq picks kernel variants at run time, SPEC §18), so a second uninterrupted run
  ("repeat") gives the noise floor, reported alongside;
- **hybrid** (cuEq + OEq, float32): statistically, as 1a-T17 compared ``main.py`` on FastMACE with
  the research engine. 20,000 steps from 300 to 600 K, a restart every 2,000, resumed at step
  10,000, two seeds: the mean temperature and potential energy and the potential energy in 50 K
  target bins agree between interrupted and uninterrupted runs as closely as between the two seeds.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_canonical_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import fastmace
import torch
from ase import units
from fastmace.config import EngineConfig, IntegratorConfig
from fastmace.io import restart
from fastmace.planner import fixed
from fastmace.model import compile as fc

planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["planner_backend"]))
loaded = planner.load(SPEC["label"])
cache = fc.CompiledModelCache()
template = structure()
natoms = len(template)


def target(step):
    first, last = SPEC["targets"]
    return first + (last - first) * step / SPEC["steps"]


def run(tag, seed, interrupt_at):
    config = IntegratorConfig(timestep_fs=SPEC["timestep_fs"], friction=0.02, seed=seed)
    plan = planner.plan(loaded, template, config, planner.hardware())

    def build(atoms):
        return planner.build(plan, loaded, atoms, temperature_K=target(0), compile_cache=cache)

    engine = build(template.copy())
    rows, resumed = [], []
    for step in range(1, SPEC["steps"] + 1):
        engine.integrator.set_temperature(target(step))
        engine.step()
        if step % SPEC["every"] == 0:
            values = engine.sync()
            kinetic = 0.5 * float((values["momenta"] ** 2 / template.get_masses()[:, None]).sum())
            rows.append({"step": step, "target": target(step), "energy": float(values["energy"][0]) / natoms,
                         "temperature": 2.0 * kinetic / (3.0 * natoms * units.kB),
                         "positions": values["positions"].tolist() if SPEC["positions"] else None})
        if step % SPEC["cadence"] == 0 and step < SPEC["steps"]:
            path = f"{SPEC['directory']}/{tag}-{step}.npz"
            restart.write_engine_restart(path, engine, name="fastmace", plan=plan)
            if step in interrupt_at:
                stored = restart.read_restart(path)
                atoms = template.copy()
                atoms.set_cell(stored["cell"])
                atoms.set_positions(stored["positions"])
                del engine
                gc.collect()
                torch.cuda.empty_cache()
                engine = build(atoms)
                restart.resume_engine(stored, engine)
                resumed.append(step)
    info = engine.info()
    return {"rows": rows, "resumed": resumed, "mode": info["mode"], "canonical_points": info["canonical_points"],
            "neighbour_rebuilds": info["neighbour_rebuilds"], "backend": plan.engine.backend,
            "dtype": plan.precision.dtype}


out = {"runs": {}, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
for tag, seed, interrupt_at in SPEC["runs"]:
    out["runs"][tag] = run(tag, seed, set(interrupt_at))
print(json.dumps(out))
"""


def canonical_run(tmp_path, backend, python, *, steps, cadence, every, runs, positions, targets=(300.0, 600.0)):
    spec = ENGINE.spec("hybrid" if backend == "auto" else backend)
    spec.update(planner_backend=backend, label="mh1-r2scan", steps=steps, cadence=cadence, every=every,
                runs=runs, positions=positions, targets=list(targets), timestep_fs=2.0, directory=str(tmp_path))
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=python)


def max_deviation(rows, reference):
    return max(float(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"])).max())
               for a, b in zip(rows, reference, strict=True))


@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_plain_cueq_float64_resumes_within_1e_12(tmp_path):
    every_restart = [100, 200, 300, 400]
    result = canonical_run(tmp_path, "cueq", sys.executable, steps=500, cadence=100, every=50, positions=True,
                           runs=[("whole", 20261019, []), ("repeat", 20261019, []),
                                 ("interrupted", 20261019, every_restart)])
    runs = result["runs"]
    summary = {"interrupted_vs_whole_A": max_deviation(runs["interrupted"]["rows"], runs["whole"]["rows"]),
               "repeat_vs_whole_A": max_deviation(runs["repeat"]["rows"], runs["whole"]["rows"]),
               "resumed": runs["interrupted"]["resumed"], "gpu": result["gpu"],
               **{f"{tag}_{key}": runs[tag][key] for tag in runs
                  for key in ("mode", "canonical_points", "neighbour_rebuilds", "backend", "dtype")}}
    ENGINE.report("1b-T14-canonical-mh1-cueq-float64", summary)
    assert (runs["whole"]["backend"], runs["whole"]["dtype"]) == ("cueq", "float64")
    assert runs["interrupted"]["resumed"] == every_restart
    assert summary["interrupted_vs_whole_A"] <= ENGINE.FLOAT64_LIMIT, summary


def binned(rows, width=50.0):
    out = {}
    for row in rows:
        out.setdefault(int(row["target"] // width), []).append(row["energy"])
    return {key: float(np.mean(values)) for key, values in sorted(out.items())}


def statistics(rows):
    return {"temperature_K": float(np.mean([row["temperature"] for row in rows])),
            "energy_eV_per_atom": float(np.mean([row["energy"] for row in rows])), "bins": binned(rows)}


def differences(a, b):
    bins = [abs(a["bins"][key] - b["bins"][key]) * 1000 for key in a["bins"]]
    return {"temperature_K": abs(a["temperature_K"] - b["temperature_K"]),
            "energy_meV_per_atom": abs(a["energy_eV_per_atom"] - b["energy_eV_per_atom"]) * 1000,
            "bins_meV_per_atom_max": max(bins), "bins_meV_per_atom_rms": float(np.sqrt(np.mean(np.square(bins))))}


@pytest.mark.slow
@ENGINE.needs_oeq
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_hybrid_resumes_statistically(tmp_path):
    seeds = (20261001, 20261101)
    runs = [(f"{kind}-{seed}", seed, [10_000] if kind == "interrupted" else [])
            for seed in seeds for kind in ("whole", "interrupted")]
    result = canonical_run(tmp_path, "auto", ENGINE.HYBRID_PYTHON, steps=20_000, cadence=2_000, every=20,
                           positions=False, runs=runs)
    stats = {tag: statistics(run["rows"]) for tag, run in result["runs"].items()}
    resume = {seed: differences(stats[f"interrupted-{seed}"], stats[f"whole-{seed}"]) for seed in seeds}
    between_seeds = differences(stats[f"whole-{seeds[0]}"], stats[f"whole-{seeds[1]}"])
    summary = {"interrupted_vs_whole": resume, "between_seeds": between_seeds, "statistics": stats,
               "gpu": result["gpu"], **{f"{tag}_{key}": run[key] for tag, run in result["runs"].items()
                                        for key in ("mode", "resumed", "canonical_points", "neighbour_rebuilds",
                                                    "backend")}}
    ENGINE.report("1b-T14-canonical-mh1-hybrid", summary)
    assert all(run["backend"] == "hybrid" for run in result["runs"].values())
    for seed in seeds:
        assert result["runs"][f"interrupted-{seed}"]["resumed"] == [10_000]
        # As 1a-T17: a resumed run differs from the uninterrupted one by no more than the seeds do.
        for key in ("temperature_K", "energy_meV_per_atom", "bins_meV_per_atom_rms"):
            assert resume[seed][key] <= max(2.0 * between_seeds[key], {"temperature_K": 1.0}.get(key, 0.05)), summary
