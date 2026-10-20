"""The eager tier on the GPU (plan task 2-T02; SPEC §8.3, §9.3 rule 3, §9.4 step 2).

- MH-1 (``mh1-r2scan``) hybrid float32 on NaCl 3x3x3, NVT Langevin at 300 K, 100 steps through
  ``fm.Simulation`` in one fresh hybrid process: eager (``check="gate"``, references in-process
  before any production engine), then graph twice from the same seed. Eager must reach the graph
  trajectory within the float32 divergence of the two graph runs, and pass the gate.
- POLAR-1-S on plain cuEq in eager mode (rule 3's "eager plain"; plain compiled cuEq gives zero
  forces): a short gated run in a fresh ``mace_cueq_latest`` process passes the gate and stays finite.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_eager_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import fastmace
import fastmace as fm
import torch
from ase import units

STEPS, EVERY, SEED = SPEC["steps"], 10, SPEC["seed"]
template = structure()
natoms = len(template)


def run(mode, check):
    model = fm.load(SPEC["label"])
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=SPEC["temperature"],
                                         friction_per_fs=0.02 * units.fs)
    from fastmace.config import EngineConfig
    options = {"engine": EngineConfig(backend=SPEC["backend"])} if SPEC["backend"] != "auto" else {}
    sim = fm.Simulation(template.copy(), model, integrator=integrator, seed=SEED, mode=mode, check=check, **options)
    rows = []
    for _ in range(STEPS // EVERY):
        sim.run(steps=EVERY)
        committed = sim.committed()
        rows.append({"step": committed.step, "energy": committed.energy / natoms,
                     "positions": committed.positions.tolist()})
    plan, info = sim.plan, sim._engine.info()
    out = {"rows": rows, "tier": plan.tier, "backend": plan.engine.backend, "dtype": plan.precision.dtype,
           "compile_mode": plan.engine.compile_mode, "capture": info["capture"], "gate": sim.gate.status,
           "gate_record": sim.gate.to_metadata(), "report": sim.report.text(), "speed": sim.report.speed}
    sim.close()
    del sim
    gc.collect()
    torch.cuda.empty_cache()
    return out


out = {"eager": run("eager", SPEC["check"])}
for name in SPEC["graph_runs"]:
    out[name] = run("graph", "skip")
out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fastmace=fastmace.__version__)
print(json.dumps(out))
"""


def deviation(rows, reference):
    import numpy as np

    return max(float(np.abs(np.asarray(a["positions"]) - np.asarray(b["positions"])).max())
               for a, b in zip(rows, reference, strict=True))


def energy_deviation_meV(rows, reference):
    return max(abs(a["energy"] - b["energy"]) * 1000 for a, b in zip(rows, reference, strict=True))


def _rows_finite(rows):
    import numpy as np

    return all(np.isfinite(row["energy"]) and np.isfinite(row["positions"]).all() for row in rows)


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_hybrid_eager_equals_graph_within_float32_noise():
    if not ENGINE._has_oeq():
        pytest.skip("OpenEquivariance is not installed (hybrid environment needed)")
    spec = {"temperature": 300.0, "seed": 20261020, "steps": 100, "label": "mh1-r2scan", "backend": "auto",
            "check": "gate", "graph_runs": ["graph", "repeat"]}
    result = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=ENGINE.HYBRID_PYTHON)
    eager, graph, repeat = result["eager"], result["graph"], result["repeat"]
    summary = {
        "gpu": result["gpu"], "torch": result["torch"], "fastmace": result["fastmace"],
        "eager": {key: eager[key] for key in ("tier", "backend", "dtype", "compile_mode", "capture", "gate")},
        "graph": {key: graph[key] for key in ("tier", "backend", "dtype", "compile_mode", "capture")},
        "gate_record": eager["gate_record"],
        "eager_vs_graph_A": deviation(eager["rows"], graph["rows"]),
        "repeat_vs_graph_A": deviation(repeat["rows"], graph["rows"]),
        "eager_vs_graph_energy_meV_per_atom": energy_deviation_meV(eager["rows"], graph["rows"]),
        "repeat_vs_graph_energy_meV_per_atom": energy_deviation_meV(repeat["rows"], graph["rows"]),
        "speed_eager": eager["speed"], "speed_graph": graph["speed"], "report_eager": eager["report"],
    }
    ENGINE.report("2-T02-eager-vs-graph-mh1-hybrid", summary)
    assert (eager["tier"], eager["capture"], eager["compile_mode"]) == ("eager", False, "none")
    assert (graph["tier"], graph["capture"]) == ("graph", True)
    assert (eager["backend"], eager["dtype"]) == (graph["backend"], graph["dtype"]) == ("hybrid", "float32")
    assert eager["gate"] == "passed", eager["gate_record"]
    assert "eager tier: no CUDA graphs" in eager["report"]
    # Within the float32 divergence of two graph runs, with a floor for runs that agree to the last bit.
    assert summary["eager_vs_graph_A"] <= max(10 * summary["repeat_vs_graph_A"], 1e-4), summary
    assert summary["eager_vs_graph_energy_meV_per_atom"] <= max(
        10 * summary["repeat_vs_graph_energy_meV_per_atom"], 0.01), summary


POLAR = _module("_eager_gpu_polar_helpers", Path(__file__).parent / "test_polar_gpu.py")


@pytest.mark.slow
@pytest.mark.skipif(not POLAR.polar_available(), reason="PolarMACE needs graph_longrange")
@pytest.mark.skipif(not POLAR.has_checkpoint("polar1-s"), reason="POLAR-1-S not in FASTMACE_MODELS")
def test_polar1_s_plain_cueq_runs_gated_in_eager_mode():
    spec = {"temperature": 300.0, "seed": 20261020, "steps": 20, "label": "polar1-s", "backend": "cueq",
            "check": "gate", "graph_runs": []}
    result = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD)
    eager = result["eager"]
    summary = {"gpu": result["gpu"], "torch": result["torch"], "fastmace": result["fastmace"],
               **{key: eager[key] for key in ("tier", "backend", "dtype", "compile_mode", "capture", "gate")},
               "gate_record": eager["gate_record"], "speed": eager["speed"], "report": eager["report"],
               "energies_eV_per_atom": [row["energy"] for row in eager["rows"]]}
    ENGINE.report("2-T02-eager-polar1-s-plain-cueq", summary)
    assert (eager["tier"], eager["backend"], eager["capture"], eager["compile_mode"]) == (
        "eager", "cueq", False, "none")
    assert eager["gate"] == "passed", eager["gate_record"]
    assert _rows_finite(eager["rows"])
