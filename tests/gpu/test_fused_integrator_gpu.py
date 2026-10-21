"""OPT3-T03's fused integrator stage on the GPU: bitwise the PyTorch stage.

1. Stage level, the production plan (hybrid FP32 with ``FASTMACE_HYBRID_PYTHON``, else the planner's
   choice in this interpreter; NaCl 3x3x3, pipelined, ``fused_integrator=True``):
   after 60 steps, for several fresh noise draws, ``_stage_acp`` and ``_stage_ac`` run once through
   PyTorch and once fused from the same buffers; every buffer the stage writes (positions, velocities,
   ``x_old``, the noise terms, the compacted edges and their count, the flags and the saved
   velocities) is compared byte for byte. Also with an atom moved past the early and the hard
   rebuild limit (so the flags are set), and with ``fixcm=False``.
2. Trajectories in ``reproducible=True`` (plain cuEq float64, the deterministic process), NaCl at
   1000 K and 2 fs, 600 steps, each in a fresh process: the default (the option on) and the option off,
   pipelined and checked, a save/resume at step 300, and NPT (``LangevinBAOAB``, which keeps the PyTorch
   stage). Positions and momenta bitwise equal to the option off: it stays on in reproducible plans.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_fused_integrator_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [
    pytest.mark.gpu, pytest.mark.slow,
    pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1"),
]

STAGE_DRIVER = """
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
MaxwellBoltzmannDistribution(atoms, temperature_K=1000.0, rng=np.random.default_rng(5))
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs,
                                     fixcm=SPEC["fixcm"])
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms), integrator=integrator, seed=7,
                    workload=fm.Workload(max_temperature_K=1000.0),
                    engine=EngineConfig(pipelined="on", fused_integrator=True))
sim.run(steps=60)
eng = sim._engine
eng.flush()
torch.cuda.synchronize()
assert eng._fused is not None, eng.info()
state, integ, nb = eng.state, eng.integrator, eng.force_field.neighbours
written = dict(positions=state.positions, velocities=state.velocities, x_old=integ.x_old, rnd_pos=integ.rnd_pos,
               rnd_vel=integ.rnd_vel, edge_index=nb.edge_index, unit_shifts=nb.unit_shifts, shifts=nb.shifts,
               count=nb.count, rebuild_needed=eng.rebuild_needed, flags=eng._flags, pipe_flags=eng._pipe_flags,
               mid=eng._mids[0])
as_bytes = lambda t: t.reshape(-1).view(torch.uint8)
fused = eng._fused
out = dict(trials=[], dtype=str(nb.shifts.dtype), backend=sim.plan.engine.backend, gpu=torch.cuda.get_device_name())
half_skin = nb.skin / 2
for trial, move in enumerate([0.0, 0.0, 0.0, 0.9 * half_skin, 1.2 * half_skin]):
    integ.prepare_step(state)
    reference = nb.reference_positions.clone()
    if trial >= 2:  # displacements of exactly 0 (2), 0.9 (3) and 1.2 (4) half skins: no flag, early, hard
        before = {{k: v.clone() for k, v in written.items()}}
        eng._fused = None
        eng._stage_acp()
        nb.reference_positions.copy_(state.positions)  # the positions this stage moves to
        for k, v in written.items():
            v.copy_(before[k])
        nb.reference_positions[0, 0] -= move
    for pipelined in (True, False):
        before = {{k: v.clone() for k, v in written.items()}}
        eng._fused = None
        (eng._stage_acp if pipelined else eng._stage_ac)()
        expected = {{k: v.clone() for k, v in written.items()}}
        for k, v in written.items():
            v.copy_(before[k])
        eng._fused = fused
        (eng._stage_acp if pipelined else eng._stage_ac)()
        torch.cuda.synchronize()
        bad = {{k: int((as_bytes(v) != as_bytes(expected[k])).sum()) for k, v in written.items()}}
        out["trials"].append(dict(trial=trial, move=move, pipelined=pipelined, mismatched_bytes=bad,
                                  flags=eng._flags.tolist(), pipe_flags=eng._pipe_flags.tolist()))
    nb.reference_positions.copy_(reference)
sim.close()
print(json.dumps(out))
"""

MD_DRIVER = """
import json, tempfile
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
if SPEC["npt"]:
    integrator = fm.integrators.LangevinBAOAB(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs,
                                              pressure_GPa=0.0)
else:
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=1000.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float64"), integrator=integrator,
                    seed=20261021,
                    workload=fm.Workload(max_temperature_K=1000.0, reproducible=True),
                    engine=EngineConfig(pipelined=SPEC["pipelined"], **SPEC["engine"]))
if SPEC["restart"]:
    sim.run(steps=SPEC["steps"] // 2)
    directory = tempfile.mkdtemp(dir=SPEC["scratch"])
    sim.save_restart(directory)
    sim.close()
    sim = fm.Simulation.resume(directory)
    sim.run(steps=SPEC["steps"] - SPEC["steps"] // 2)
else:
    sim.run(steps=SPEC["steps"])
committed = sim.committed()
info = dict(sim._engine.info())
out = {{"step": committed.step, "positions": committed.positions.tobytes().hex(),
        "momenta": committed.momenta.tobytes().hex(), "energy": committed.energy,
        "fused": info.get("fused_integrator"), "rebuilds": info["neighbour_rebuilds"], "graphs": info["graphs"],
        "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype, "tier": sim.plan.tier,
        "gpu": torch.cuda.get_device_name()}}
sim.close()
print(json.dumps(out))
"""


@pytest.mark.parametrize("fixcm", [True, False])
def test_the_fused_stage_writes_the_pytorch_stage_bytes(fixcm):
    run = ENGINE.run_fresh(STAGE_DRIVER.format(spec={"fixcm": fixcm}), python=ENGINE.HYBRID_PYTHON)
    ENGINE.report(f"opt3-T03-fused-stage-fixcm-{fixcm}", run)
    trials = run["trials"]
    assert len(trials) == 10, run
    for trial in trials:
        assert all(count == 0 for count in trial["mismatched_bytes"].values()), trial
    none, early, hard = trials[4], trials[6], trials[8]  # pipelined trials 2, 3, 4
    assert none["pipe_flags"][0] == 0 and none["pipe_flags"][2] == 0, none
    assert early["pipe_flags"][0] == 0 and early["pipe_flags"][2] == 1, early
    assert hard["pipe_flags"][0] == 1 and hard["flags"][0] == 1, hard


def _engine(fused):
    """The default engine (on since the owner's decision of 21 October 2026) or the option switched off."""
    return {} if fused else {"fused_integrator": False}


GROUPS = {
    "nvt": {f"{pipelined}-{fused}": dict(pipelined=pipelined, engine=_engine(fused), restart=False,
                                         npt=False) for pipelined in ("on", "off") for fused in (False, True)},
    "restart-npt": {
        **{f"restart-{fused}": dict(pipelined="on", engine=_engine(fused), restart=True, npt=False)
           for fused in (False, True)},
        **{f"npt-{fused}": dict(pipelined="off", engine=_engine(fused), restart=False, npt=True)
           for fused in (False, True)}},
}


@pytest.mark.parametrize("group", list(GROUPS))
def test_trajectories_are_bitwise_equal_with_the_option(tmp_path, group):
    runs = {name: ENGINE.run_fresh(MD_DRIVER.format(spec={**case, "steps": 600, "scratch": str(tmp_path)}))
            for name, case in GROUPS[group].items()}
    summary = {name: {k: v for k, v in run.items() if k not in ("positions", "momenta")} for name, run in runs.items()}
    ENGINE.report(f"opt3-T03-fused-integrator-mh1-nacl-{group}", summary)
    for name, run in runs.items():
        off = runs[name.rsplit("-", 1)[0] + "-False"]
        assert run["step"] == 600 and run["rebuilds"] > 0, (name, summary)
        if name.endswith("True"):
            assert run["fused"] == (True if not name.startswith("npt") else
                                    "variable cell (LangevinBAOAB keeps the PyTorch stage)"), (name, summary)
        else:
            assert run["fused"] is None, (name, summary)
        assert (run["positions"], run["momenta"]) == (off["positions"], off["momenta"]), (name, summary)
        assert run["tier"] == "graph", (name, summary)
