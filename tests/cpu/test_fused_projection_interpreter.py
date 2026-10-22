"""The fused integrator stage's fixcm projections on the CPU, through Triton's interpreter (plan task 3-T22b).

``TRITON_INTERPRET=1`` runs the Triton kernels of ``fastmace.kernels.triton.fused_integrator`` on CPU
tensors. The check is in a fresh interpreter, which needs Triton installed: the GPU environments have it,
the CPU environment does not, so the test is skipped there. Run it with the GPU environment's Python and
``CUDA_VISIBLE_DEVICES=``.

Lennard-Jones argon with alternating masses (40 and 4 amu). From the same buffers and random numbers, the
fused stage and ``Langevin.first_half`` give:
- the projected noise terms ``rnd_pos`` and ``rnd_vel`` bit for bit, for both projections. The
  mass-weighted means are the PyTorch reductions and one float64 division each, as in ``first_half``;
- positions and velocities within 1e-15 relative. On the CPU PyTorch divides by a Python number exactly,
  while the kernel multiplies by its reciprocal, as PyTorch's CUDA kernel does. Bitwise on the GPU:
  tests/gpu/test_fused_projection_gpu.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

DRIVER = textwrap.dedent("""
    import json, sys
    sys.path[:0] = [{cpu!r}, {tests!r}]
    import fastmace  # noqa: F401  (before torch)
    import dataclasses
    import torch
    from test_canonical_restart import argon, lj_engine
    from fastmace.kernels.triton.fused_integrator import FusedIntegrator

    out = {{}}
    for projection in ("ase", "mass_weighted"):
        atoms = argon()
        masses = atoms.get_masses()
        masses[::2] = 4.0
        atoms.set_masses(masses)
        engine = lj_engine(atoms, mode="checked")
        integ, state = engine.integrator, engine.state
        integ.config = dataclasses.replace(integ.config, fixcm_projection=projection)
        integ.mass_weighted_com = projection == "mass_weighted"
        fused = FusedIntegrator(engine)
        for _ in range(3):
            engine.step()
        engine.flush()
        integ.prepare_step(state)
        start = {{name: getattr(state, name).clone() for name in ("positions", "velocities")}}
        integ.first_half(state)
        want = {{"positions": state.positions.clone(), "velocities": state.velocities.clone(),
                 "rnd_pos": integ.rnd_pos.clone(), "rnd_vel": integ.rnd_vel.clone()}}
        for name, value in start.items():
            getattr(state, name).copy_(value)
        fused.stage(pipelined=False)
        got = {{"positions": state.positions, "velocities": state.velocities, "rnd_pos": integ.rnd_pos,
                "rnd_vel": integ.rnd_vel}}
        out[projection] = {{
            name: {{"bitwise": bool(torch.equal(got[name], want[name])),
                    "relative": float((got[name] - want[name]).abs().max() / want[name].abs().max())}}
            for name in want}}
        momentum = (integ.rnd_vel * state.masses).sum(dim=0).abs().max()
        out[projection]["noise_momentum"] = float(momentum)
    print(json.dumps(out))
""")


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="needs Triton (the GPU environments)")
def test_the_fused_projections_match_the_pytorch_stage_in_the_interpreter():
    environment = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=str(TESTS.parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    code = DRIVER.format(cpu=str(TESTS), tests=str(TESTS.parent))
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    for projection, values in out.items():
        for name in ("rnd_pos", "rnd_vel"):
            assert values[name]["bitwise"], (projection, name, values[name])
        for name in ("positions", "velocities"):
            assert values[name]["relative"] < 1e-15, (projection, name, values[name])
    assert out["mass_weighted"]["noise_momentum"] < 1e-15
