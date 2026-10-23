"""The tier switch on the GPU (plan task 5-T07; SPEC §9.6, §18 decisions 52 and 53), untimed.

A fresh process runs MH-1 through ``Simulation`` with the production defaults on NaCl 3x3x3 at 600 K, a
restart every 50 steps. ``EngineConfig(graph_variant_cap=1)`` forces a re-capture storm (every step
re-captures, as in 5-T06's smoke), so the budget is exceeded and re-capturing takes most of the wall time:
the run must switch graph -> compiled once, at a committed step, as a plan revision recorded in
``counters.tier_switches`` and the report. Against an unswitched compiled run (``mode="compiled"``, the same
seed) its energies and temperatures after the switch agree statistically. A resume from a restart after the
switch runs the compiled revision, does not switch again, and agrees statistically with the uninterrupted
run (bit for bit only on the CPU: this plan is not reproducible; ``tests/cpu/test_adapt_switch.py``). The
step time is not judged here: the timed storm and the production profiles are 5-T11's quiet window.
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


ENGINE = _module("_adapt_switch_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import json

import fastmace
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.io.output import Output
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import adapt
from fastmace.planner.workload import Workload
from fastmace.simulation import Simulation

SPEC = {spec!r}
loaded = loading.load_pristine(SPEC["path"], trusted=True)
out = {{"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "runs": {{}}}}


def thermo(directory):
    from fastmace.io.output import read_thermo

    header, rows = read_thermo(directory)
    columns = [header.index(name) for name in ("step", "temperature_K", "potential_eV_atom")]
    return [[float(row[column]) for column in columns] for row in rows]


def record(tag, sim, directory):
    sim.close()
    metadata = json.load(open(directory + "/metadata.json"))
    committed = sim.committed()
    out["runs"][tag] = {{
        "tier": sim.plan.tier, "backend": sim.plan.engine.backend, "revision": sim.plan.revision,
        "switches": [switch.record() for switch in adapt.tier_switches(sim.plan)],
        "metadata_switches": metadata["counters"]["tier_switches"], "step": committed.step,
        "energy_per_atom": committed.energy / len(committed.positions), "thermo": thermo(directory),
        "report": sim.report.text()}}


def new(tag, cap=None, mode=None):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(11))
    options = {{}} if cap is None else {{"engine": EngineConfig(graph_variant_cap=cap)}}
    directory = SPEC["directory"] + "/" + tag
    sim = Simulation(atoms, ForceField(loaded, atoms, precision="float32"),
                     integrator=integrators.Langevin(timestep_fs=2.0, temperature_K=600.0, friction_per_fs=0.02),
                     seed=7, workload=Workload(max_temperature_K=600.0), mode=mode,
                     output=Output(directory, thermo_every=5, trajectory_every=None, restart_every=50), **options)
    return sim, directory


sim, directory = new("storm", cap=1)
sim.run(steps=SPEC["steps"])
record("storm", sim, directory)

sim, directory = new("compiled", mode="compiled")
sim.run(steps=SPEC["steps"])
record("compiled", sim, directory)

sim, directory = new("first", cap=1)
sim.run(steps=SPEC["steps"] // 2)
sim.close()
resumed = Simulation.resume(directory)
out["resumed_tier"] = resumed.plan.tier
resumed.run(steps=SPEC["steps"] - SPEC["steps"] // 2)
record("resumed", resumed, directory)
print(json.dumps(out))
"""


def _mean(rows, column, first_step):
    values = [row[column] for row in rows if row[0] > first_step]
    return sum(values) / len(values)


@pytest.mark.slow
@pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
def test_a_forced_recapture_storm_switches_once_to_the_compiled_tier(tmp_path):
    spec = {"path": str(ENGINE._mh1_path()), "steps": 200, "directory": str(tmp_path)}
    result = ENGINE.run_fresh(CHILD.format(spec=spec), python=sys.executable)
    runs = result["runs"]
    print({tag: (run["tier"], run["backend"], run["switches"]) for tag, run in runs.items()}, result["gpu"])

    storm, compiled, resumed = runs["storm"], runs["compiled"], runs["resumed"]
    assert storm["tier"] == "compiled" and storm["step"] == 200
    (switch,) = storm["switches"]
    assert (switch["from"], switch["to"]) == ("graph", "compiled") and 0 < switch["step"] < 100
    assert storm["metadata_switches"] == storm["switches"]
    assert f"(tier switch at step {switch['step']:,}): graph -> compiled" in storm["report"]
    assert storm["backend"] == compiled["backend"] and compiled["switches"] == []

    # After the switch, statistically the compiled run (thermo rows: step, temperature, potential per atom).
    header_t, header_e = 1, 2
    after = switch["step"] + 20
    for column, tolerance in ((header_t, 60.0), (header_e, 0.005)):
        assert abs(_mean(storm["thermo"], column, after) - _mean(compiled["thermo"], column, after)) < tolerance

    # Interrupted after the switch (restart at step 100): compiled, no second switch, statistically equal.
    assert result["resumed_tier"] == "compiled" and len(resumed["switches"]) == 1
    assert resumed["switches"][0]["step"] == switch["step"] and resumed["step"] == 200
    assert abs(resumed["energy_per_atom"] - storm["energy_per_atom"]) < 0.01
    assert abs(_mean(resumed["thermo"], header_t, 120) - _mean(storm["thermo"], header_t, 120)) < 60.0
