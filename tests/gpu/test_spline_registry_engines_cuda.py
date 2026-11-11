"""Several graph engines of different sizes in one process share the SPLINE_W forward (``spline_forward.shared``).

The custom ops launch the registered ``SplineForward`` of their content id. Equal builds (one per force field) used to
replace the entry, so an engine that captured after a later equal build launched through that build's object (its
BLOCK_ORDER order, or the identity order it cached for this engine's row count); once that build's simulation was
closed, the object and its tensors were freed and the next replay read freed memory (CUDA illegal address in a
five-engine A/B process, RTX 4000 Ada, 10 November 2026). Here, with MACE-OMAT-0 medium on SrTiO3 2x2x2, 4x4x4 and
3x3x3 (rattled and strained): "small" is built, then "large" is built and run; "small" captures only now (before the
fix: through large's object); "middle" is built and run; "large" is closed, the allocator's cache emptied and the
freed blocks refilled with out-of-range indices; "small" and "middle" then keep replaying, interleaved. Each
simulation's committed positions and forces must match the same simulation run alone in a fresh process (float32
atomics: to rounding, not bitwise), and every engine must launch the first registered forward. SPLINE_W and
BLOCK_ORDER forced on; skips where the w' path is off. Needs ``FASTMACE_MODELS`` (the checkpoint) and the hybrid
backend's kernels.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastmace.kernels import streaming_available

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")

pytestmark = [pytest.mark.skipif(not streaming_available(), reason=streaming_available().reason),
              pytest.mark.skipif(not MODELS or not Path(MODELS, "mace-omat-0-medium.model").exists(),
                                 reason="needs FASTMACE_MODELS with mace-omat-0-medium.model")]

CHILD = r"""
import fastmace  # before torch: soft file locks (SPEC §14)
import gc, json, sys

import numpy as np
import torch

import fastmace as fm
from fastmace.config import NeighbourConfig
from fastmace.kernels.oeq_streaming import spline_forward

sys.path.insert(0, "tools/bench")
import tiled_capacity

spline_forward.SPLINE_W, spline_forward.BLOCK_ORDER = True, True
SIZES = {"small": (2, 2, 2), "large": (4, 4, 4), "middle": (3, 3, 3)}
RUNS = {"small": [10, 10, 10, 10], "large": [10], "middle": [10, 10, 10]}
ONLY = sys.argv[1] if len(sys.argv) > 1 else None


def simulation(name):
    atoms = tiled_capacity.supercell("SrTiO3", SIZES[name], 0.05, 0.01)
    field = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
    return fm.Simulation(atoms, field, integrator=fm.integrators.NVE(timestep_fs=1.0),
                         workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                         neighbours=NeighbourConfig(skin=0.5))


def kernels(sim):
    return sim._engine.force_field._ordered_spline_kernels()


report, sims, done = {}, {}, {}


def build(name):
    sims[name], done[name] = simulation(name), 0
    if not kernels(sims[name]):
        print(json.dumps({"skip": "no BLOCK_ORDER spline forward on this GPU (w' path off)"}))
        raise SystemExit(0)


def run(name):
    sims[name].run(steps=RUNS[name][done[name]])
    done[name] += 1


def finish(name):
    c = sims[name].committed()
    report[name] = dict(step=c.step, positions=np.asarray(c.positions).tolist(), forces=np.asarray(c.forces).tolist())


if ONLY is not None:  # the single-engine reference
    build(ONLY)
    while done[ONLY] < len(RUNS[ONLY]):
        run(ONLY)
    finish(ONLY)
else:
    build("small")
    build("large")
    run("large")
    run("small")  # captured while a later equal build ("large") exists
    build("middle")
    run("middle")
    report["shared"] = all(a is b for a, b in zip(kernels(sims["small"]), kernels(sims["middle"])))
    report["registered"] = all(spline_forward._REGISTRY[k.id] is k for k in kernels(sims["small"]))
    finish("large")
    sims.pop("large").close()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    spray = [torch.full((n,), 2 ** 30, dtype=torch.int32, device="cuda") for n in (40, 135, 320) for _ in range(200)]
    while done["small"] < len(RUNS["small"]) or done["middle"] < len(RUNS["middle"]):
        for name in ("small", "middle"):
            if done[name] < len(RUNS[name]):
                run(name)
    finish("small")
    finish("middle")
    torch.cuda.synchronize()
    del spray
print(json.dumps(report))
"""


def _child(*args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD), *args], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_engines_of_different_sizes_in_one_process_match_each_run_alone():
    import numpy as np

    together = _child()
    if "skip" in together:
        pytest.skip(together["skip"])
    assert together["shared"] and together["registered"]
    for name in ("small", "large", "middle"):
        alone = _child(name)
        assert together[name]["step"] == alone[name]["step"], name
        positions = np.abs(np.asarray(together[name]["positions"]) - np.asarray(alone[name]["positions"])).max()
        forces = np.abs(np.asarray(together[name]["forces"]) - np.asarray(alone[name]["forces"])).max()
        print(name, "positions", positions, "forces", forces)
        assert positions <= 1e-4, (name, positions)  # Angstrom, after 20-40 float32 MD steps with atomics
        assert forces <= 1e-3, (name, forces)  # eV/Angstrom
