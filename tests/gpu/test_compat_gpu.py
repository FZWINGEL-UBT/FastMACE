"""The GPU compatibility tier's own tests (plan task 4-T13; SPEC decision 39, amended 23 October 2026).

``-m compat`` (``_support.tiers``) is the fast GPU tier plus the slow tests listed in ``tiers.COMPAT_TESTS``,
which already check most of what differs between GPU architectures. These two add what nothing else covers in
under a few minutes, MH-1 ``matpes_r2scan`` with the planner's default plan (hybrid float32 where the
OpenEquivariance runtime is there), NaCl 3x3x3, each in a fresh process:

- **captured graphs and their kernels:** 200 Langevin steps in graph mode; every captured graph's node count,
  its kernel count and its cuEquivariance kernels are recorded (``FASTMACE_RESULTS``), so that a new
  architecture's graphs can be compared with the reference GPU's (phase-3.md open finding 8: cuEq picks its
  kernel variants at run time, so the counts are recorded, not judged). Replay against eager is
  ``test_eager_gpu.py``'s, also in the tier;
- **a short MD smoke:** NVE, NVT (Langevin) and NPT (hydrostatic BAOAB at 0 GPa), 2 ps each at 1 fs from
  300 K. Limits loose enough for any GPU and tight enough for a broken kernel or integrator: NVE's conserved
  energy drifts by less than 0.05 meV/atom/ps and stays within a 0.1 meV/atom range (4-T03's NHC drifted
  1e-4 meV/atom/ps over 20 ps); the NVT temperature over the last 1.5 ps is within 10 % of 300 K; NPT stays
  finite and its volume within 5 % of the start. Values are recorded.
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


ENGINE = _module("_compat_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow, pytest.mark.compat,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

GRAPHS_CHILD = """
import json
import fastmace
import fastmace as fm
import torch
count_graph_nodes(torch)
from ase import units
atoms = structure()
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms),
                    integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                       friction_per_fs=0.01),
                    seed=1, workload=fm.Workload(max_temperature_K=300.0))
sim.run(steps=200)
graphs = {str(key): {"nodes": getattr(graph, "node_count", None),
                     "kernels": sum(1 for kind in (getattr(graph, "node_kinds", None) or [])
                                    if not kind.startswith("CU_GRAPH_NODE_TYPE")),
                     "cueq_kernels": sorted({kind for kind in (getattr(graph, "node_kinds", None) or [])
                                             if "segmented_polynomial" in kind})}
          for key, graph in sim._engine._graphs.items()}
out = {"graphs": graphs, "tier": sim.plan.tier, "engine_mode": sim._engine.mode, "backend": sim.plan.engine.backend,
       "dtype": sim.plan.precision.dtype, "gpu": torch.cuda.get_device_name(),
       "capability": list(torch.cuda.get_device_capability()), "energy": sim.committed().energy}
sim.close()
print(json.dumps(out, default=str))
"""

SMOKE_CHILD = """
import json, math
import fastmace
import fastmace as fm
import numpy as np
import torch
from ase import units
from fastmace.integrators.base import CONSERVED_KEY
from fastmace.quick import nve_max_temperature_K

def simulation(integrator, max_temperature_K):
    atoms = structure()
    sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms), integrator=integrator, seed=1,
                        workload=fm.Workload(max_temperature_K=max_temperature_K))
    return sim, len(atoms)

out = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}
sim, n = simulation(fm.integrators.NVE(timestep_fs=1.0), nve_max_temperature_K(structure()))
engine = sim._engine
rows = []
for chunk in range(21):
    if chunk:
        sim.run(steps=100)
    values = engine.integrator.output_values(engine.state)
    rows.append([chunk * 0.1, float(values[CONSERVED_KEY][0]) / n])
times, conserved = np.array(rows).T
slope = float(np.polyfit(times, conserved, 1)[0])
out["nve"] = {"drift_meV_per_atom_per_ps": 1e3 * slope, "range_meV_per_atom": 1e3 * float(np.ptp(conserved)),
              "plan": sim.plan.describe()}
sim.close()

sim, n = simulation(fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01), 300.0)
temperatures = []
for chunk in range(20):
    sim.run(steps=100)
    if chunk >= 5:
        kinetic = float(sim._engine.integrator.output_values(sim._engine.state)["kinetic"][0])
        dof = sim._engine.integrator.degrees_of_freedom(sim._engine.state)
        temperatures.append(2.0 * kinetic / (dof * units.kB))
out["nvt"] = {"mean_T_K": float(np.mean(temperatures)), "samples": len(temperatures)}
sim.close()

sim, n = simulation(fm.integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01,
                                                 pressure_GPa=0.0), 300.0)
volume0 = float(abs(np.linalg.det(sim.committed().cell)))
sim.run(steps=2000)
committed = sim.committed()
out["npt"] = {"volume_ratio": float(abs(np.linalg.det(committed.cell))) / volume0,
              "finite": bool(np.isfinite(committed.positions).all() and math.isfinite(committed.energy))}
sim.close()
print(json.dumps(out, default=str))
"""


def _child(code):
    spec = {"temperature": 300.0, "steps": 0, "timed": 0, "every": 1}
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + code)


def test_captured_graphs_and_their_kernel_counts_are_recorded():
    out = _child(GRAPHS_CHILD)
    ENGINE.report("compat-graph-kernels", out)
    print(out)
    assert out["graphs"], "no graph was captured"
    assert all(graph["nodes"] and graph["kernels"] for graph in out["graphs"].values()), out["graphs"]
    assert np.isfinite(out["energy"])


def test_a_short_md_smoke_in_nve_nvt_and_npt():
    out = _child(SMOKE_CHILD)
    ENGINE.report("compat-md-smoke", out)
    print(out)
    assert abs(out["nve"]["drift_meV_per_atom_per_ps"]) < 0.05, out["nve"]
    assert out["nve"]["range_meV_per_atom"] < 0.1, out["nve"]
    assert abs(out["nvt"]["mean_T_K"] - 300.0) < 30.0, out["nvt"]
    assert out["npt"]["finite"] and abs(out["npt"]["volume_ratio"] - 1.0) < 0.05, out["npt"]
