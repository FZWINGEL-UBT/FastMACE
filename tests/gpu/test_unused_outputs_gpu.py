"""Capacity screen A on CUDA: the compiled model without ``node_feats`` gives the same MD, bitwise.

MH-1, NaCl 3x3x3 at 1000 K, 300 steps with rebuilds, ``reproducible=True`` (the deterministic process), in float32
and float64, each run in a fresh process: ``fastmace.model.compile.UNUSED_OUTPUTS`` at its default and set to ``()``
(every output returned). Positions and momenta bitwise equal.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_unused_outputs_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
pytestmark = [
    pytest.mark.gpu, pytest.mark.slow,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

MD_DRIVER = """
import json
import fastmace  # noqa: F401  (before torch)
import fastmace as fm
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from _support.structures import nacl
from fastmace.model import compile as compiling
SPEC = {spec!r}
compiling.UNUSED_OUTPUTS = tuple(SPEC["unused"])
atoms = nacl(repeat=3)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(11))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision=SPEC["precision"]),
                    integrator=integrator, seed=20261030,
                    workload=fm.Workload(max_temperature_K=1000.0, reproducible=True))
sim.run(steps=300)
committed = sim.committed()
info = dict(sim._engine.info())
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "rebuilds": info["neighbour_rebuilds"],
        "compile_mode": sim.plan.engine.compile_mode, "backend": sim.plan.engine.backend,
        "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier, "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_md_is_bitwise_equal_without_node_feats(precision):
    runs = {name: ENGINE.run_fresh(MD_DRIVER.format(spec={"unused": unused, "precision": precision}))
            for name, unused in (("lean", ["node_feats"]), ("all", []))}
    summary = {name: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for name, run in runs.items()}
    ENGINE.report(f"cap-outputs-mh1-nacl-{precision}", summary)
    lean, full = runs["lean"], runs["all"]
    assert lean["step"] == 300 and lean["rebuilds"] > 1 and lean["tier"] == "graph", summary
    assert lean["compile_mode"] not in (None, "none"), summary
    assert (lean["positions"], lean["momenta"]) == (full["positions"], full["momenta"]), summary
