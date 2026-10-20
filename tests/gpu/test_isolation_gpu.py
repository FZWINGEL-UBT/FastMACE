"""Instance isolation on the GPU (plan task 2-T08; SPEC §8.7, §12.2 "Isolation", phase 2 acceptance 5).

A -> B -> A with two crystals (NaCl 3x3x3 and NbCl5 2x2x2) plus a second NaCl (2x2x2, A2: the
same species as A, so on the MH-1 paths it shares A's compiled model and its streaming
interactions), on the released models, one fresh process per path:

- plain cuEquivariance MH-1, float64 (``mh1-r2scan``, compiled, this interpreter);
- the MH-1 hybrid, float32 (OpenEquivariance with the streaming convolutions,
  ``FASTMACE_HYBRID_PYTHON``);
- POLAR-1-S hybrid, float32 (each force field compiles its own copy after ``torch._dynamo.reset()``,
  ``FASTMACE_HYBRID_PYTHON``).

In each process:

1. evaluations: every crystal on a fresh force field with its own compiled-model cache, then all
   three built together on one cache and evaluated A, B, A2, A, B, A2, A; each against its fresh
   value (the §6.4 threshold of the dtype). The last A repeats the first A's positions, so dynamo
   compiles nothing there unless a swapped buffer or a reset invalidated A's code: the number of
   graphs compiled is counted, and for POLAR-1-S also the recompilation a later polar build
   causes ("correct, slower", conductor answer Q3);
2. engines: each crystal's engine run alone, then the three built together and stepped in turn
   (captured graphs, across neighbour rebuilds and new edge buckets). Each engine's last forces
   equal a fresh force field's at its own positions; plain cuEq float64 trajectories equal the
   engines run alone within 1e-8 Å (not bitwise: cuEquivariance's scatter is not deterministic).

The audit is docs/design/isolation.md. Each test runs minutes (three compiles and captures per
path), so they are in the acceptance tier.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from fastmace.rewrites import registry as reg

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.acceptance,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable


def _has_checkpoint(label):
    if not MODELS:
        return False
    from fastmace.model import labels

    return (Path(MODELS) / labels.LABELS[label].checkpoint.file_name).is_file()


def _has_oeq():
    if HYBRID_PYTHON == sys.executable:
        return importlib.util.find_spec("openequivariance") is not None
    code = "import importlib.util, sys; sys.exit(importlib.util.find_spec('openequivariance') is None)"
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


def run_fresh(code, python, timeout=3600):
    """Run code in a fresh interpreter; returns the JSON its last stdout line prints."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    result = subprocess.run([python, "-c", textwrap.dedent(code)], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


CHILD = """
import gc, json
import numpy as np
import fastmace
import torch
from torch._dynamo.utils import counters
from fastmace.config import EngineConfig, IntegratorConfig, PolarConfig
from fastmace.model import compile as fc
from fastmace.model.forces import MACEForceField
from fastmace.planner import fixed
from fastmace.rewrites import registry as reg
from _support.structures import nacl, nbcl5

planner = fixed.FixedPlanner(engine=EngineConfig(backend=SPEC["backend"]))
loaded = planner.load(SPEC["label"])
hardware = planner.hardware()
registry = reg.default_registry()
CRYSTALS = {"A": nacl(repeat=3, rattle=0.05, seed=1), "B": nbcl5(repeat=2, rattle=0.05, seed=2),
            "A2": nacl(repeat=2, rattle=0.05, seed=3)}
ORDER = ("A", "B", "A2")
SEEDS = {"A": 11, "B": 12, "A2": 13}
plans = {name: planner.plan(loaded, atoms, IntegratorConfig(timestep_fs=2.0, friction=0.02, seed=SEEDS[name]),
                            hardware) for name, atoms in CRYSTALS.items()}


def graphs():
    return int(counters["stats"]["unique_graphs"])


def field(name, cache, atoms=None):
    # As FixedPlanner.build makes it, without the state and the engine.
    plan = plans[name]
    resolution = planner.rewrite_resolution(plan, loaded.model, registry)
    return MACEForceField(loaded, CRYSTALS[name] if atoms is None else atoms, rewrites=resolution,
                          engine=plan.engine, neighbours=plan.neighbours, polar=plan.polar or PolarConfig(),
                          compile_cache=cache, registry=registry, float64_energy=plan.precision.float64_energy)


def moved(name, seed):
    atoms = CRYSTALS[name].copy()
    atoms.positions += 0.02 * np.random.default_rng(seed).standard_normal(atoms.positions.shape)
    return atoms


POSITIONS = {name: [moved(name, seed) for seed in (1, 2)] for name in ORDER}


def evaluate(ff, atoms):
    energy, forces, _ = ff.evaluate(atoms)
    return np.array([energy]), forces


def error(got, expected):
    return max(float(np.abs(a - b).max()) / float(np.abs(b).max()) for a, b in zip(got, expected))


out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "dtype": plans["A"].precision.dtype,
       "backend": plans["A"].engine.backend}

# 1. Evaluations. Fresh: each crystal alone, its own cache.
fresh_fields, fresh = {}, {}
for name in ORDER:
    ff = fresh_fields[name] = field(name, fc.CompiledModelCache())
    fresh[name] = [evaluate(ff, atoms) for atoms in POSITIONS[name]]
out["polar"] = bool(fresh_fields["A"].is_polar)
if out["polar"]:
    # A polar build resets dynamo: the earlier fields recompile at their next call (correct, slower).
    before = graphs()
    again = evaluate(fresh_fields["A"], POSITIONS["A"][0])
    out["recompiled_after_polar_reset"] = graphs() - before
    out["fresh_after_reset"] = error(again, fresh["A"][0])

# Together on one cache: A, B, A2, A, B, A2, then A at its first positions again.
cache = fc.CompiledModelCache()
together = {name: field(name, cache) for name in ORDER}
out["shared_A_A2"] = together["A"].model is together["A2"].model
out["shared_A_B"] = together["A"].model is together["B"].model
out["streaming"] = {name: together[name].info()["streaming_interactions"] for name in ORDER}
out["cache_entries"] = len(cache)
errors = {name: [] for name in ORDER}
compiled = []
for index in (0, 1):
    for name in ORDER:
        errors[name].append(error(evaluate(together[name], POSITIONS[name][index]), fresh[name][index]))
    compiled.append(graphs())
errors["A"].append(error(evaluate(together["A"], POSITIONS["A"][0]), fresh["A"][0]))
out["evaluation_errors"] = errors
out["compiled_in_last_round"] = graphs() - compiled[-1]
del together
gc.collect()


# 2. Engines: alone, then three stepped in turn on one cache.
STEPS = SPEC["steps"]


def run_alone(name):
    engine = planner.build(plans[name], loaded, CRYSTALS[name].copy(), temperature_K=SPEC["temperature"],
                           compile_cache=cache)
    for _ in range(STEPS):
        engine.step()
    values = engine.sync()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return values


alone = {name: run_alone(name) for name in ORDER}
engines = {name: planner.build(plans[name], loaded, CRYSTALS[name].copy(), temperature_K=SPEC["temperature"],
                               compile_cache=cache) for name in ORDER}
for _ in range(STEPS):
    for name in ORDER:
        engines[name].step()
out["engines"] = {}
for name in ORDER:
    engine = engines[name]
    values = engine.sync()
    info = engine.info()
    atoms = CRYSTALS[name].copy()
    atoms.positions = values["positions"]
    _, forces = evaluate(fresh_fields[name], atoms)
    out["engines"][name] = {
        "mode": info["mode"], "capture": info["capture"],
        "neighbour_rebuilds": int(engine.force_field.neighbours.rebuilds),
        "edge_sizes": len(engine.edge_sizes_used), "captured": len(engine._graphs),
        "force_error": error([values["forces"]], [forces]),
        "position_deviation": float(np.abs(values["positions"] - alone[name]["positions"]).max()),
    }
print(json.dumps(out))
"""

PATHS = {
    "mh1-plain-cueq": dict(label="mh1-r2scan", backend="cueq", hybrid=False),
    "mh1-hybrid": dict(label="mh1-r2scan", backend="hybrid", hybrid=True),
    "polar1-s-hybrid": dict(label="polar1-s", backend="hybrid", hybrid=True),
}


@pytest.mark.parametrize("path", list(PATHS))
def test_a_b_a_equals_fresh_instances(path):
    options = PATHS[path]
    if not _has_checkpoint(options["label"]):
        pytest.skip(f"{options['label']} is not in FASTMACE_MODELS")
    if options["hybrid"] and not _has_oeq():
        pytest.skip("OpenEquivariance is not installed (FASTMACE_HYBRID_PYTHON)")
    spec = {"label": options["label"], "backend": options["backend"], "steps": 100, "temperature": 600.0}
    out = run_fresh(f"SPEC = {spec!r}\n" + CHILD, HYBRID_PYTHON if options["hybrid"] else sys.executable)
    print(json.dumps(out, indent=1))
    threshold = reg.THRESHOLDS[out["dtype"]]
    polar = out["polar"]
    assert polar is (path == "polar1-s-hybrid")
    # Sharing as the audit says: MH-1 shares one compiled model between same-species crystals, a
    # polar model is never shared.
    assert out["shared_A_A2"] is not polar and out["shared_A_B"] is False
    if path == "mh1-hybrid":
        assert all(count > 0 for count in out["streaming"].values())
    for name, errors in out["evaluation_errors"].items():
        assert max(errors) <= threshold, (name, errors)
    assert out["compiled_in_last_round"] == 0  # A again: no guard failed on another field's state
    if polar:
        assert out["fresh_after_reset"] <= threshold
    for name, engine in out["engines"].items():
        assert engine["capture"] and engine["captured"] > 0, name
        assert engine["neighbour_rebuilds"] > 1, name  # the steps crossed a rebuild
        assert engine["force_error"] <= threshold, (name, engine)
        if out["dtype"] == "float64":
            assert engine["position_deviation"] <= 1e-8, (name, engine)
