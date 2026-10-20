"""The reference tier on the GPU: MH-1 NaCl 3x3x3, 200 steps against the graph tier (plan task 2-T03).

``fm.Simulation(..., mode="reference")`` runs MH-1 (``mh1-r2scan``) through the corrected reference
calculator on FastMACE's Langevin integrator, and ``fm.Simulation`` on the default graph tier runs the
same structure, starting velocities and seed; each tier in a fresh process (one engine and one backend
family per process, SPEC §8.7). Both are float32 force fields, so the trajectories diverge; the
comparison is statistical: the mean temperature and potential energy over steps 51-200 agree within
the thermal fluctuation of a 216-atom system, and the reference tier's gate passes.
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


ENGINE = _module("_reference_tier_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

STEPS, BURN_IN = 200, 50

CHILD = """
import fastmace
import fastmace as fm
import torch
from ase import units
from fastmace.io import output as output_io

template = structure()
natoms = len(template)
model = fm.load("mh1-r2scan")
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=SPEC["temperature"],
                                     friction_per_fs=0.02 * units.fs)
output = fm.Output(SPEC["directory"], thermo_every=1, trajectory_every=None, restart_every=None)
started = time.perf_counter()
sim = fm.Simulation(template.copy(), model, integrator=integrator, output=output, seed=SPEC["seed"],
                    mode=SPEC["mode"], check=SPEC["check"])
setup_s = time.perf_counter() - started
started = time.perf_counter()
sim.run(steps=SPEC["steps"])
run_s = time.perf_counter() - started
sim.close()
header, thermo = output_io.read_thermo(SPEC["directory"])
plan = sim.plan
print(json.dumps({
    "tier": plan.tier, "backend": plan.engine.backend, "dtype": plan.precision.dtype,
    "gate": sim.gate.status, "setup_s": setup_s, "run_s": run_s, "report": sim.report.text(),
    "steps": [int(row[0]) for row in thermo],
    "temperature_K": [float(row[header.index("temperature_K")]) for row in thermo],
    "potential_eV_atom": [float(row[header.index("potential_eV_atom")]) for row in thermo],
    "positions": sim.committed().positions.tolist(), "engine": sim._engine.info()["engine"],
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "fastmace": fastmace.__version__}))
"""


def _run(tmp_path, mode, check):
    spec = {"temperature": 300.0, "seed": 20261020, "directory": str(tmp_path / mode), "mode": mode,
            "check": check, "steps": STEPS}
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD)


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_reference_tier_agrees_with_the_graph_tier_statistically(tmp_path):
    reference = _run(tmp_path, "reference", "gate")
    graph = _run(tmp_path, "graph", "skip")
    window = slice(BURN_IN + 1, None)  # thermo row 0 is step 0
    temperature = {name: np.asarray(run["temperature_K"][window]) for name, run in
                   (("reference", reference), ("graph", graph))}
    energy = {name: np.asarray(run["potential_eV_atom"][window]) for name, run in
              (("reference", reference), ("graph", graph))}
    natoms = 216
    # The thermal fluctuation of the instantaneous temperature, sqrt(2 / (3 N)) T, over ~10 independent samples.
    temperature_limit = 3 * 300.0 * np.sqrt(2 / (3 * natoms)) / np.sqrt(10)
    summary = {
        "gpu": reference["gpu"], "torch": reference["torch"], "fastmace": reference["fastmace"],
        "reference": {key: reference[key] for key in ("tier", "backend", "dtype", "gate", "setup_s", "run_s")},
        "graph": {key: graph[key] for key in ("tier", "backend", "dtype", "gate", "setup_s", "run_s")},
        "mean_temperature_K": {name: float(values.mean()) for name, values in temperature.items()},
        "std_temperature_K": {name: float(values.std()) for name, values in temperature.items()},
        "mean_potential_eV_atom": {name: float(values.mean()) for name, values in energy.items()},
        "std_potential_eV_atom": {name: float(values.std()) for name, values in energy.items()},
        "temperature_limit_K": float(temperature_limit),
        "final_position_deviation_A": float(np.abs(np.asarray(reference["positions"])
                                                   - np.asarray(graph["positions"])).max()),
        "reference_report": reference["report"],
    }
    ENGINE.report("2-T03-reference-tier-mh1-nacl", summary)
    assert reference["tier"] == "reference" and reference["engine"] == "graph"
    assert graph["tier"] == "graph"
    assert reference["gate"] == "passed"
    assert reference["steps"] == graph["steps"] == list(range(STEPS + 1))
    assert any(line.strip().startswith("plan:") and "reference," in line for line in reference["report"].splitlines())
    delta_t = abs(summary["mean_temperature_K"]["reference"] - summary["mean_temperature_K"]["graph"])
    assert delta_t < temperature_limit, summary
    # Potential energy: within 1 meV/atom, well below the thermal energy (3/2 kT = 39 meV/atom at 300 K).
    delta_e = abs(summary["mean_potential_eV_atom"]["reference"] - summary["mean_potential_eV_atom"]["graph"])
    assert delta_e < 1e-3, summary
