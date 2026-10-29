"""Captured float64 relaxations under cuBLAS's fixed-mantissa float64 emulation (fp64-D1 relaxation fix, 28 Oct 2026).

Owner decision 65 D1 runs float64 MH-1 plans on the RTX 5060 Ti and RTX 4000 Ada with
``CUBLAS_EMULATE_DOUBLE_PRECISION=1``, ``CUBLAS_EMULATION_STRATEGY=eager`` and
``CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=56``. Inside a captured graph, FP64 GEMMs on small operands then came
back about 8 bits accurate; the device LBFGS over the Frechet cell filter diverged from a non-minimal start
(ResearchModule Experiments/Energy_Hull, H0-diag). cuBLAS reads the variables when it creates a handle, so the
relaxation of ``test_cell_filter_gpu`` (Lennard-Jones argon, strained and rattled) runs in a child process with them
set, and must still match ASE's LBFGS on ASE's FrechetCellFilter within that test's limits.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

D1_ENVIRONMENT = {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1", "CUBLAS_EMULATION_STRATEGY": "eager",
                  "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT": "56"}


def _child(output):
    """In the child: the captured LBFGS + FrechetCell relaxation of ``test_cell_filter_gpu``, written as JSON."""
    import fastmace  # noqa: F401  (before torch's first CUDA use, as everywhere)
    from fastmace.model.compile import fp64_emulation_active
    from fastmace.optimize import driver

    sys.path.insert(0, str(Path(__file__).parent))
    import test_cell_filter_gpu as cell_filter

    active = fp64_emulation_active()
    engine = cell_filter.build(cell_filter.RECORDS["lbfgs"])
    summary = driver.relax(engine, fmax=cell_filter.FMAX, steps=cell_filter.STEPS)
    Path(output).write_text(json.dumps({
        "emulation_active": active, "converged": summary["converged"], "steps": summary["steps"],
        "positions": engine.state.positions.cpu().numpy().tolist(), "cell": engine.state.cell.cpu().numpy().tolist()}))


def test_captured_lbfgs_frechet_matches_ase_under_fixed_mantissa_emulation(tmp_path):
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    sys.path.insert(0, str(Path(__file__).parent))
    import test_cell_filter_gpu as cell_filter

    output = tmp_path / "relaxation.json"
    environment = {**os.environ, **D1_ENVIRONMENT, "PYTHONDONTWRITEBYTECODE": "1"}
    script = f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); " \
             f"import test_fp64_emulation_relax_gpu as t; t._child({str(output)!r})"
    subprocess.run([sys.executable, "-c", script], env=environment, check=True, timeout=1200)
    result = json.loads(output.read_text())
    if not result["emulation_active"]:
        pytest.skip("cuBLAS did not run emulated float64 products on this GPU")

    reference = cell_filter.crystal()
    from ase.calculators.lj import LennardJones

    reference.calc = LennardJones(sigma=cell_filter.ENGINE.SIGMA, epsilon=cell_filter.ENGINE.EPSILON,
                                  rc=cell_filter.ENGINE.RC, smooth=False)
    optimizer = LBFGS(FrechetCellFilter(reference), logfile=None)
    assert optimizer.run(fmax=cell_filter.FMAX, steps=cell_filter.STEPS)
    assert result["converged"]
    assert abs(result["steps"] - optimizer.nsteps) <= 2, (result["steps"], optimizer.nsteps)
    np.testing.assert_allclose(np.array(result["cell"]), reference.cell.array, atol=1e-6)
    np.testing.assert_allclose(np.array(result["positions"]), reference.positions, atol=1e-6)
