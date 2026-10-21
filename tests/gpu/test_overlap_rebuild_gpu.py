"""``EngineConfig.overlap_rebuild`` on the GPU (opt-2 OPT2-T04): exact, bitwise in the reproducible mode.

MH-1 (``mh1-r2scan``) on NaCl 3x3x3 in ``reproducible=True`` (plain cuEq, unfused convolution, the
deterministic process), pipelined, Langevin at 1000 K and 2 fs so that early neighbour rebuilds occur,
in two fresh processes: the option off and on. Positions and momenta after the run are bitwise equal,
and the second run did rebuild beside the force graph in flight (``rebuilds_beside``).
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_overlap_rebuild_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

DRIVER = """
import json
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.config import EngineConfig
SPEC = {spec!r}
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261021, workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(pipelined="on", overlap_rebuild=SPEC["overlap"]))
sim.run(steps=SPEC["steps"])
committed = sim.committed()
info = dict(sim._engine.info())
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy,
        "pipelined": sim._engine.pipelined, "early_rebuilds": info["pipelined_early_rebuilds"],
        "rebuilds": info["neighbour_rebuilds"], "beside": info.get("rebuilds_beside"),
        "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier,
        "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""


def test_overlapped_rebuilds_are_bitwise_equal_to_waiting_ones():
    runs = {overlap: ENGINE.run_fresh(DRIVER.format(spec={"overlap": overlap, "steps": 600}))
            for overlap in (False, True)}
    off, on = runs[False], runs[True]
    summary = {key: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for key, run in runs.items()}
    ENGINE.report("opt2-T04-overlap-rebuild-mh1-nacl", summary)
    assert on["pipelined"] and on["tier"] == "graph" and on["early_rebuilds"] > 0, summary
    assert on["beside"] == on["early_rebuilds"] and off["beside"] is None, summary
    assert (on["step"], on["positions"], on["momenta"]) == (off["step"], off["positions"], off["momenta"]), summary
