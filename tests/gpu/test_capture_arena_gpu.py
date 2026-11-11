"""The capture arena (``EngineConfig.capture_arena``, results/capture-arena-20261110, option A) on a GPU.

The reproducible MH-1 plan (``mh1-r2scan``, plain cuEq float64, NaCl 3x3x3, NVT 1000 K) in fresh processes: the
arena's build passes (recording capture, plan, planned capture matching its plan, replay against eager runs) and the
trajectory and its forces and stress are bitwise those of the default capture on the same plan. A second side
retires the arena on purpose (a later capture) and must stay bitwise too, with the arena's memory returned.

Linux only (the arena is refused on WSL: results/capture-arena-20261110, HYPERVISOR_ERROR bugchecks).
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_capture_arena_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")


def _on_wsl():
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


CHILD = """
import fastmace as fm
import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig

atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"),
                    integrator=integrator, seed=20261111,
                    workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(capture_arena=SPEC["arena"]))
engine = sim._engine
built = dict(engine.capture_arena_info)
if SPEC["retire"]:
    sim.run(steps=SPEC["steps"] // 2)
    engine.invalidate_graphs("test: retire the arena")
    sim.run(steps=SPEC["steps"] - SPEC["steps"] // 2)
else:
    sim.run(steps=SPEC["steps"])
committed = sim.committed()
info = engine.info()
stress = None if committed.stress is None else np.asarray(committed.stress).tobytes().hex()
out = {"step": committed.step, "positions": committed.positions.tobytes().hex(),
       "forces": committed.forces.tobytes().hex(), "stress": stress,
       "energy": committed.energy, "tier": sim.plan.tier, "capture_arena": sim.plan.engine.capture_arena,
       "built": built, "after": info.get("capture_arena"), "arena_pools": len(engine._process_record.arena_pools),
       "rebuilds": info["neighbour_rebuilds"]}
sim.close()
print(json.dumps(out, default=str))
"""


def _run(arena, retire=False, steps=200):
    spec = {"arena": arena, "retire": retire, "steps": steps}
    return ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD)


@pytest.mark.slow
@pytest.mark.skipif(not sys.platform.startswith("linux") or _on_wsl(), reason="the capture arena needs native Linux")
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_capture_arena_is_bitwise_the_default_capture():
    default, arena, retired = _run(False), _run(True), _run(True, retire=True)
    summary = {name: {k: v for k, v in run.items() if k not in ("positions", "forces", "stress")}
               for name, run in (("default", default), ("arena", arena), ("retired", retired))}
    ENGINE.report("capture-arena-bitwise", summary)
    assert default["tier"] == "graph" and default["capture_arena"] is False, summary
    assert arena["capture_arena"] is True, summary
    assert arena["built"]["status"] == "active", summary
    assert arena["built"]["plan_check"] == "ok", summary
    assert arena["built"]["verify"]["bitwise"], summary  # deterministic plan: replay == eager, bit for bit
    assert arena["built"]["arena_MiB"] <= arena["built"]["reserved_MiB"], summary
    for key in ("step", "positions", "forces", "stress", "energy"):
        assert arena[key] == default[key], (key, summary)
        assert retired[key] == default[key], (key, summary)
    assert retired["after"]["status"] == "retired", summary
    assert retired["arena_pools"] == 0, summary  # the retired arena's pool is gone
