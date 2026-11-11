"""EDGE_FORCE (idea 13 step 2; ``wprime_backward.EDGE_FORCE``): the w' backward contracts the edge features' gradient
with edge_sh3's Jacobian in its epilogue and returns the edge vectors' gradient instead of dsh [E, 16].

MACE-OMAT-0 medium on SrTiO3 3x3x3 (rattled and strained), float32, two NVE steps: the forces with EDGE_FORCE on
match it off to float32 rounding (stress: ``tools/force_stress_gate.py``), for each end-of-edge reduction (plain
shuffle-down, BUTTERFLY, REDUX), and the on side built and launched the EDGE_FORCE kernel with that reduction.
Skips where the SPLINE_W w' path is off. Needs ``FASTMACE_MODELS`` and the hybrid backend's kernels.
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
import json, sys

import numpy as np
import torch

import fastmace as fm
from fastmace.config import NeighbourConfig
from fastmace.kernels.oeq_streaming import wprime_backward as wp

sys.path.insert(0, "tools/bench")
import tiled_capacity

edge_force, reduction = sys.argv[1] == "on", sys.argv[2]
wp.EDGE_FORCE = edge_force
if reduction != "default":
    wp.REDUX = reduction == "redux"
    wp.BUTTERFLY = reduction == "butterfly"
atoms = tiled_capacity.supercell("SrTiO3", (3, 3, 3), 0.05, 0.01)
field = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
sim = fm.Simulation(atoms, field, integrator=fm.integrators.NVE(timestep_fs=1.0),
                    workload=fm.Workload(max_temperature_K=300.0), output=None, seed=1,
                    neighbours=NeighbourConfig(skin=0.5))
sim.run(steps=2)
c = sim.committed()
force_kernels = [k for k in wp._REGISTRY.values() if getattr(k, "edge_force", False)]
dsh_kernels = [k for k in wp._REGISTRY.values() if not getattr(k, "edge_force", False)]
spline = bool(dsh_kernels) and any(k.spline is not None for k in dsh_kernels)
print(json.dumps(dict(step=c.step, forces=np.asarray(c.forces).tolist(),
                      stress=None if c.stress is None else np.asarray(c.stress).tolist(),
                      force_kernels=len(force_kernels), spline=spline,
                      force_launches=sum(k.launches for k in force_kernels),
                      reductions=sorted({("redux" if k.redux else "butterfly" if k.butterfly else "plain")
                                         for k in force_kernels}))))
"""


def _child(*args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(CHILD), *args], cwd=REPOSITORY_ROOT, env=env,
                            capture_output=True, text=True, timeout=1800, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("reduction", ["plain", "butterfly", "redux"])
def test_edge_force_matches_dsh_to_rounding(reduction):
    import numpy as np
    import torch

    if reduction == "redux" and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("REDUX needs compute capability 8.0")
    off, on = _child("off", reduction), _child("on", reduction)
    if not off["spline"]:
        pytest.skip("the SPLINE_W w' path is off on this GPU")
    assert off["force_kernels"] == 0
    assert on["force_kernels"] >= 1 and on["reductions"] == [reduction], on["reductions"]
    assert on["force_launches"] >= 1  # captured or run: the model's EdgeSH handed its vectors over
    assert on["step"] == off["step"]
    f_on, f_off = np.asarray(on["forces"]), np.asarray(off["forces"])
    scale = np.abs(f_off).max()
    assert np.abs(f_on - f_off).max() <= 1e-4 * scale, np.abs(f_on - f_off).max() / scale
    if off["stress"] is not None:
        s_on, s_off = np.asarray(on["stress"]), np.asarray(off["stress"])
        assert np.abs(s_on - s_off).max() <= 1e-4 * max(np.abs(s_off).max(), 1e-6)
