"""Plan task 4-T13 (phase 4 acceptance, item 1): the new integrators on the graph engine against ASE.

4-T02's and 4-T03's ASE-equivalence tests (``test_nve_csvr_berendsen_gpu.py``,
``test_nose_hoover_mtk_mh1_gpu.py``) drive the integrators eagerly, with the forces from a second MACE
calculator; the engine's run of the new schemes was left to 4-T10 and 4-T13. This test runs each scheme as
production does: planned by the fixed planner (plain cuEq float64) and stepped by the graph engine, with its
captured stages, neighbour list and rebuilds, in a fresh process, against the same ASE 3.27 class on MACE's
calculator in another fresh process (as ``test_ase_equivalence_gpu.py`` runs Langevin).

The setting is those tests' (conductor's choice 2): MH-1 ``matpes_r2scan``, NaCl 3x3x3, 4,000 steps of 0.5 fs at
300 K from a 300 K zero-momentum start, ``dof="ase"``; the records, the ASE classes (NPT Berendsen against ASE's
class reordered to one force call per step) and the time constants are taken from those files. Limits per
500-step block, over all 4,000 steps (the runbook uses no shortened horizon): |ΔT| < 1e-6 K and
|ΔPE| < 1e-7 meV/atom. Volume and the final position and cell differences are recorded.

Results go to ``FASTMACE_RESULTS`` when set (``docs/validation/phase-4.md``).
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


HERE = Path(__file__).parent
ENGINE = _module("_4t13_engine_helpers", HERE / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow, pytest.mark.acceptance,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

# scheme: (helper file, the name it uses)
SCHEMES = {
    "nve": ("test_nve_csvr_berendsen_gpu.py", "nve"),
    "csvr": ("test_nve_csvr_berendsen_gpu.py", "csvr"),
    "nvt-berendsen": ("test_nve_csvr_berendsen_gpu.py", "nvt-berendsen"),
    "npt-berendsen": ("test_nve_csvr_berendsen_gpu.py", "npt-berendsen"),
    "inhomogeneous-berendsen": ("test_nve_csvr_berendsen_gpu.py", "inhomogeneous-berendsen"),
    "nhc": ("test_nose_hoover_mtk_mh1_gpu.py", "nhc"),
    "mtk-hydrostatic": ("test_nose_hoover_mtk_mh1_gpu.py", "hydrostatic"),
    "mtk-full": ("test_nose_hoover_mtk_mh1_gpu.py", "full"),
}
STEPS, BLOCK = 4000, 500
PE_LIMIT = 1e-10     # eV/atom: 1e-7 meV/atom (SPEC §12.2)
T_LIMIT = 1e-6       # K, per block mean
TIMEOUT = 2 * 3600

COMMON = """
import importlib.util, json, time, warnings
import numpy as np
SPEC = {spec!r}

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helpers = _module("_4t13_scheme_helpers", SPEC["helpers"])

def record():
    if hasattr(helpers, "records"):  # 4-T02's file
        return helpers.records()[SPEC["name"]]
    return helpers.record(SPEC["name"])  # 4-T03's: dof="ase", fixcm=False

def structure():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=helpers.TEMPERATURE_K, rng=np.random.default_rng(11))
    Stationary(atoms)
    return atoms
"""

ASE_CHILD = """
import torch
from ase import units
atoms = structure()
atoms.calc = helpers.mh1_calculator()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dyn = helpers.ase_dynamics(SPEC["name"], atoms)
n = len(atoms)
rows = []
start = time.perf_counter()
for step in range(1, SPEC["steps"] + 1):
    dyn.run(1)
    rows.append([step, 2 * atoms.get_kinetic_energy() / (3 * n * units.kB), atoms.get_potential_energy() / n,
                 atoms.get_volume()])
out = {"rows": rows, "positions": atoms.positions.tolist(), "cell": atoms.cell.array.tolist(),
       "seconds": time.perf_counter() - start, "gpu": torch.cuda.get_device_name()}
print(json.dumps(out))
"""

FASTMACE_CHILD = """
import fastmace
from fastmace.config import EngineConfig
from fastmace.planner import fixed
from fastmace.model import compile as fc
from ase import units
import torch
planner = fixed.FixedPlanner(engine=EngineConfig(backend="cueq"))
hardware = planner.hardware()
loaded = planner.load("mh1-r2scan")
atoms = structure()
config = record().to_config(seed=helpers.SEED)
plan = planner.plan(loaded, atoms, config, hardware)
engine = planner.build(plan, loaded, atoms, temperature_K=helpers.TEMPERATURE_K,
                       compile_cache=fc.CompiledModelCache())
n = len(atoms)
rows = []
start = time.perf_counter()
for step in range(1, SPEC["steps"] + 1):
    engine.step()
    values = engine.sync()
    volume = float(abs(np.linalg.det(values["cell"]))) if "cell" in values else float(atoms.get_volume())
    rows.append([step, 2 * float(values["kinetic"]) / (3 * n * units.kB), float(values["energy"][0]) / n, volume])
out = {"rows": rows, "positions": values["positions"].tolist(),
       "cell": values["cell"].tolist() if "cell" in values else atoms.cell.array.tolist(),
       "seconds": time.perf_counter() - start, "gpu": torch.cuda.get_device_name(), "mode": engine.mode,
       "records": list(engine.force_field.resolution.enabled), "plan": plan.describe(),
       "engine": type(engine).__name__, "integrator": type(engine.integrator).__name__}
print(json.dumps(out, default=str))
"""


def run(side, scheme, steps=STEPS):
    helpers, name = SCHEMES[scheme]
    spec = {"helpers": str(HERE / helpers), "name": name, "steps": steps}
    child = ASE_CHILD if side == "ase" else FASTMACE_CHILD
    return ENGINE.run_fresh(COMMON.format(spec=spec) + child, timeout=TIMEOUT)


def blocks(rows):
    rows = np.asarray(rows)
    return [rows[i:i + BLOCK, 1:4].mean(axis=0) for i in range(0, len(rows), BLOCK)]


def compare(ase, ours):
    """Per block: |ΔT| (K), |ΔPE| (eV/atom) and ΔV (Å³), and the final position and cell differences."""
    pairs = list(zip(blocks(ase["rows"]), blocks(ours["rows"])))
    return {"blocks_dT_K": [float(abs(o[0] - a[0])) for a, o in pairs],
            "blocks_dPE_eV_per_atom": [float(abs(o[1] - a[1])) for a, o in pairs],
            "blocks_dV_A3": [float(o[2] - a[2]) for a, o in pairs],
            "final_positions_max_diff_A": float(np.abs(np.subtract(ours["positions"], ase["positions"])).max()),
            "final_cell_max_diff_A": float(np.abs(np.subtract(ours["cell"], ase["cell"])).max())}


def passes(comparison):
    return (all(d < T_LIMIT for d in comparison["blocks_dT_K"])
            and all(d < PE_LIMIT for d in comparison["blocks_dPE_eV_per_atom"]))


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_scheme_on_the_graph_engine_reproduces_ase(scheme):
    ase, ours = run("ase", scheme), run("fastmace", scheme)
    comparison = compare(ase, ours)
    summary = {"scheme": scheme, "steps": STEPS, **comparison,
               "blocks_dPE_meV_per_atom": [1e3 * d for d in comparison["blocks_dPE_eV_per_atom"]],
               "final_T_K": [ase["rows"][-1][1], ours["rows"][-1][1]],
               "final_volume_A3": [ase["rows"][-1][3], ours["rows"][-1][3]],
               "seconds": {"ase": ase["seconds"], "fastmace": ours["seconds"]}, "gpu": ours["gpu"],
               "mode": ours["mode"], "engine": ours["engine"], "integrator": ours["integrator"],
               "records": ours["records"], "plan": ours["plan"]}
    ENGINE.report(f"4-T13-{scheme}-engine-ase-equivalence", summary)
    assert ours["engine"] == "GraphEngine", summary
    assert passes(comparison), summary
