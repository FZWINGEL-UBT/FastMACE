"""``fm.Simulation`` on the GPU: the 1a-T17 smoke run through ``Simulation`` (plan task 1b-T17).

One fresh process plans and runs MH-1 (``mh1-r2scan``) NaCl 3x3x3 for 60 steps of NVT Langevin at
300 K, 2 fs, ``main.py``'s friction (0.02 per ASE time unit), twice through ``FixedPlanner`` as
``main.py``'s adapter builds it (``reference`` and ``repeat``: the float32 noise floor) and once
through ``fm.Simulation`` with output (thermo and frames through the asynchronous chunked writer),
all from the same starting velocities and seed, one backend family (plain cuEq in
``mace_cueq_latest``, the hybrid in ``FASTMACE_HYBRID_PYTHON``). ``Simulation`` must reach the
``FixedPlanner`` trajectory within the float32 divergence of two ``FixedPlanner`` runs.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_simulation_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import fastmace
import fastmace as fm
import torch
from ase import units
from fastmace.config import EngineConfig, IntegratorConfig, PrecisionConfig
from fastmace.io import output as output_io
from fastmace.planner import fixed

STEPS, EVERY, SEED, TEMPERATURE, TIMESTEP, FRICTION = 60, 10, SPEC["seed"], 300.0, 2.0, 0.02
template = structure()
natoms = len(template)


def reference():
    planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["fixed_backend"]),
                                 precision=PrecisionConfig(dtype="float32"))
    loaded = planner.load("mh1-r2scan")
    config = IntegratorConfig(timestep_fs=TIMESTEP, friction=FRICTION, seed=SEED)
    plan = planner.plan(loaded, template, config, planner.hardware())
    engine = planner.build(plan, loaded, template.copy(), temperature_K=TEMPERATURE)
    rows = []
    for step in range(1, STEPS + 1):
        engine.integrator.set_temperature(TEMPERATURE)
        engine.step()
        if step % EVERY == 0:
            values = engine.sync()
            rows.append({"step": step, "energy": float(values["energy"][0]) / natoms,
                         "positions": values["positions"].tolist()})
    out = {"rows": rows, "backend": plan.engine.backend, "dtype": plan.precision.dtype,
           "options": fixed.enabled_options(plan)}
    del engine
    torch.cuda.empty_cache()
    return out


def simulation():
    model = fm.load("mh1-r2scan")
    integrator = fm.integrators.Langevin(timestep_fs=TIMESTEP, temperature_K=TEMPERATURE,
                                         friction_per_fs=FRICTION * units.fs)
    output = fm.Output(SPEC["directory"], trajectory_every=EVERY, thermo_every=EVERY, restart_every=None)
    started = time.perf_counter()
    sim = fm.Simulation(template.copy(), model, integrator=integrator, output=output, seed=SEED)
    setup_s = time.perf_counter() - started
    rows = []
    for _ in range(STEPS // EVERY):
        sim.run(steps=EVERY)
        committed = sim.committed()
        rows.append({"step": committed.step, "energy": committed.energy / natoms,
                     "positions": committed.positions.tolist()})
    sim.close()
    header, thermo = output_io.read_thermo(SPEC["directory"])
    frames = output_io.read_frames(SPEC["directory"])
    plan = sim.plan
    return {"rows": rows, "backend": plan.engine.backend, "dtype": plan.precision.dtype,
            "options": fixed.enabled_options(plan), "gate": sim.gate.status, "setup_s": setup_s,
            "thermo_steps": [int(row[0]) for row in thermo],
            "thermo_energy": [float(row[header.index("potential_eV_atom")]) for row in thermo],
            "frames": len(frames), "frame_positions": frames[-1].get_positions().tolist(),
            "report": sim.report.text(), "speed": sim.report.speed, "mode": sim._engine.info()["mode"]}


# Simulation first, before anything touches CUDA: the planner sets the allocator before the first CUDA
# use, and fm.load prepares the OEq runtime before MACE is imported (1b-T08b), which the hybrid needs.
# FixedPlanner's engines then run in the same backend family.
out = {"simulation": simulation()}
out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fastmace=fastmace.__version__)
out["reference"] = reference()
out["repeat"] = reference()
print(json.dumps(out))
"""


def deviation(rows, reference):
    import numpy as np

    return max(float(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"])).max())
               for a, b in zip(rows, reference, strict=True))


def energy_deviation_meV(rows, reference):
    return max(abs(a["energy"] - b["energy"]) * 1000 for a, b in zip(rows, reference, strict=True))


@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
@pytest.mark.parametrize("backend", ["cueq", "hybrid"])
def test_simulation_reaches_the_main_py_trajectory(tmp_path, backend):
    import numpy as np

    if backend == "hybrid" and not ENGINE._has_oeq():
        pytest.skip("OpenEquivariance is not installed (hybrid environment needed)")
    python = ENGINE.HYBRID_PYTHON if backend == "hybrid" else ENGINE.sys.executable
    spec = {"temperature": 300.0, "seed": 20261019, "directory": str(tmp_path / "run"),
            "fixed_backend": "auto" if backend == "hybrid" else "cueq"}
    result = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=python)
    ref, repeat, sim = result["reference"], result["repeat"], result["simulation"]
    summary = {
        "gpu": result["gpu"], "torch": result["torch"], "fastmace": result["fastmace"], "backend": sim["backend"],
        "dtype": sim["dtype"], "mode": sim["mode"], "gate": sim["gate"],
        "simulation_vs_fixed_A": deviation(sim["rows"], ref["rows"]),
        "repeat_vs_fixed_A": deviation(repeat["rows"], ref["rows"]),
        "simulation_vs_fixed_energy_meV_per_atom": energy_deviation_meV(sim["rows"], ref["rows"]),
        "repeat_vs_fixed_energy_meV_per_atom": energy_deviation_meV(repeat["rows"], ref["rows"]),
        "same_options": sim["options"] == ref["options"], "setup_s": sim["setup_s"], "speed": sim["speed"],
        "report": sim["report"],
    }
    ENGINE.report(f"1b-T17-simulation-smoke-{backend}", summary)
    assert (sim["backend"], sim["dtype"]) == (ref["backend"], ref["dtype"]) == (
        ("hybrid", "float32") if backend == "hybrid" else ("cueq", "float32"))
    assert sim["options"] == ref["options"]
    assert sim["gate"] == "skipped"
    assert sim["thermo_steps"] == list(range(0, 61, 10)) and sim["frames"] == 7
    np.testing.assert_allclose(sim["thermo_energy"][1:], [row["energy"] for row in sim["rows"]], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sim["frame_positions"], sim["rows"][-1]["positions"])
    # Within the float32 divergence: no further from FixedPlanner than a second FixedPlanner run, with a floor
    # for the case where the two FixedPlanner runs happen to agree to the last bit.
    assert summary["simulation_vs_fixed_A"] <= max(10 * summary["repeat_vs_fixed_A"], 1e-4), summary
    assert summary["simulation_vs_fixed_energy_meV_per_atom"] <= max(
        10 * summary["repeat_vs_fixed_energy_meV_per_atom"], 0.01), summary
