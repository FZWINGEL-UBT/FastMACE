"""The compiled MD tier on the GPU (plan task 3-T06; SPEC §8.2, §9.3 rules 4 and 9).

NaCl 3x3x3, NVT Langevin at 300 K, 100 steps through ``fm.Simulation`` in one fresh process per
model: compiled (``check="gate"``, references in-process before any production engine), then graph
twice from the same seed. The compiled tier must reach the graph trajectory within the float32
divergence of the two graph runs, pass the gate, and compile once: the neighbour rebuilds of the
run change the edge count, and the edge dimension is dynamic from the first call.

- MH-1 (``mh1-r2scan``) hybrid float32, in the OEq environment;
- MP-0 medium (``mp0-medium``, a generic model) on the default backend of the test environment.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_compiled_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import fastmace
import fastmace as fm
import torch
from ase import units
from torch._dynamo.utils import counters

STEPS, EVERY, SEED = SPEC["steps"], 10, SPEC["seed"]
template = structure()
natoms = len(template)


def run(mode, check):
    model = fm.load(SPEC["label"])
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=SPEC["temperature"],
                                         friction_per_fs=0.02 * units.fs)
    sim = fm.Simulation(template.copy(), model, integrator=integrator, seed=SEED, mode=mode, check=check)
    rows, graphs = [], []
    for _ in range(STEPS // EVERY):
        sim.run(steps=EVERY)
        committed = sim.committed()
        rows.append({"step": committed.step, "energy": committed.energy / natoms,
                     "positions": committed.positions.tolist()})
        graphs.append(counters["stats"]["unique_graphs"])
    plan, info = sim.plan, sim._engine.info()
    out = {"rows": rows, "tier": plan.tier, "backend": plan.engine.backend, "dtype": plan.precision.dtype,
           "compile_mode": plan.engine.compile_mode, "dynamic_edges": plan.engine.dynamic_edges,
           "capture": info["capture"], "edge_sizes_used": info["edge_sizes_used"],
           "neighbour_rebuilds": info["neighbour_rebuilds"], "unique_graphs": graphs, "gate": sim.gate.status,
           "gate_record": sim.gate.to_metadata(), "report": sim.report.text(), "speed": sim.report.speed}
    sim.close()
    del sim
    gc.collect()
    torch.cuda.empty_cache()
    return out


counters.clear()
out = {"compiled": run("compiled", SPEC["check"])}
for name in SPEC["graph_runs"]:
    out[name] = run("graph", "skip")
out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fastmace=fastmace.__version__)
print(json.dumps(out))
"""

EAGER = _module("_compiled_gpu_eager_helpers", Path(__file__).parent / "test_eager_gpu.py")


def _label_path(label):
    if not ENGINE.MODELS:
        return None
    from fastmace.model import labels

    path = Path(ENGINE.MODELS) / labels.LABELS[label].checkpoint.file_name
    return path if path.is_file() else None


def _compare(name, label, python):
    spec = {"temperature": 300.0, "seed": 20261022, "steps": 100, "label": label, "check": "gate",
            "graph_runs": ["graph", "repeat"]}
    result = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=python)
    compiled, graph, repeat = result["compiled"], result["graph"], result["repeat"]
    summary = {
        "gpu": result["gpu"], "torch": result["torch"], "fastmace": result["fastmace"],
        "compiled": {key: compiled[key] for key in ("tier", "backend", "dtype", "compile_mode", "dynamic_edges",
                                                     "capture", "gate", "edge_sizes_used", "neighbour_rebuilds",
                                                     "unique_graphs")},
        "graph": {key: graph[key] for key in ("tier", "backend", "dtype", "compile_mode", "capture")},
        "gate_record": compiled["gate_record"],
        "compiled_vs_graph_A": EAGER.deviation(compiled["rows"], graph["rows"]),
        "repeat_vs_graph_A": EAGER.deviation(repeat["rows"], graph["rows"]),
        "compiled_vs_graph_energy_meV_per_atom": EAGER.energy_deviation_meV(compiled["rows"], graph["rows"]),
        "repeat_vs_graph_energy_meV_per_atom": EAGER.energy_deviation_meV(repeat["rows"], graph["rows"]),
        "speed_compiled": compiled["speed"], "speed_graph": graph["speed"], "report_compiled": compiled["report"],
    }
    ENGINE.report(name, summary)
    assert (compiled["tier"], compiled["capture"], compiled["compile_mode"]) == ("compiled", False, "default")
    assert compiled["dynamic_edges"] is True
    assert (graph["tier"], graph["capture"]) == ("graph", True)
    assert (compiled["backend"], compiled["dtype"]) == (graph["backend"], graph["dtype"])
    assert compiled["gate"] == "passed", compiled["gate_record"]
    assert "compiled tier: no CUDA graphs" in compiled["report"]
    # One compile for every edge count after the first block (2-T13's dynamic edge dimension).
    assert len(set(compiled["unique_graphs"])) == 1, compiled["unique_graphs"]
    # Within the float32 divergence of two graph runs, with a floor for runs that agree to the last bit.
    assert summary["compiled_vs_graph_A"] <= max(10 * summary["repeat_vs_graph_A"], 1e-4), summary
    assert summary["compiled_vs_graph_energy_meV_per_atom"] <= max(
        10 * summary["repeat_vs_graph_energy_meV_per_atom"], 0.01), summary
    return summary


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_mh1_hybrid_compiled_equals_graph_within_float32_noise():
    if not ENGINE._has_oeq():
        pytest.skip("OpenEquivariance is not installed (hybrid environment needed)")
    summary = _compare("3-T06-compiled-vs-graph-mh1-hybrid", "mh1-r2scan", ENGINE.HYBRID_PYTHON)
    assert summary["compiled"]["backend"] == "hybrid" and summary["compiled"]["dtype"] == "float32"


@pytest.mark.slow
@pytest.mark.skipif(_label_path("mp0-medium") is None, reason="needs FASTMACE_MODELS with MP-0 medium")
def test_mp0_compiled_equals_graph_within_float32_noise():
    import sys

    summary = _compare("3-T06-compiled-vs-graph-mp0-medium", "mp0-medium", sys.executable)
    assert summary["compiled"]["dtype"] == "float32"
