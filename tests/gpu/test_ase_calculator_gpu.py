"""``fm.calc`` on the GPU (plan task 1b-T18; SPEC §7.4, §7.7 level 0).

MH-1 (``matpes_r2scan``) on rattled NaCl 2x2x2: ``ase.optimize.LBFGS`` with ``fm.calc`` (float32
forces, float64 energy, compiled tier) reaches the minimum that LBFGS reaches with MACE's own
calculator (float64). The relaxation runs in a child process: the calculator's compiled plan fixes
the process's compile family (SPEC §8.7), which must not depend on the tests run before it.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")


def _mh1_path():
    if not MODELS:
        return None
    from fastmace.model import labels

    path = Path(MODELS) / labels.LABELS["mh1-r2scan"].checkpoint.file_name
    return path if path.is_file() else None


CHILD = textwrap.dedent("""
    import json, sys, time
    import fastmace as fm  # before MACE, so the planner may choose the hybrid backend
    import numpy as np
    sys.path.insert(0, sys.argv[2])
    from _support.structures import nacl
    from ase.optimize import LBFGS

    def relax(calculator):
        atoms = nacl(repeat=2, rattle=0.08, seed=21)
        atoms.calc = calculator
        start = time.perf_counter()
        optimizer = LBFGS(atoms, logfile=None)
        converged = optimizer.run(fmax=2e-3, steps=300)
        return atoms, dict(converged=bool(converged), steps=optimizer.nsteps,
                           seconds=time.perf_counter() - start, energy=atoms.get_potential_energy(),
                           fmax=float(np.abs(atoms.get_forces()).max()))

    calc = fm.calc("mh1", head="matpes_r2scan")
    calls, calculate = [], calc.calculate

    def timed(*args, **kwargs):
        start = time.perf_counter()
        calculate(*args, **kwargs)
        calls.append(round(time.perf_counter() - start, 4))

    calc.calculate = timed
    ours, ours_info = relax(calc)
    ours_info["call_seconds"] = calls
    plan = calc.plan
    from mace.calculators import MACECalculator
    theirs, theirs_info = relax(MACECalculator(model_paths=sys.argv[1], device="cuda", default_dtype="float64",
                                               head="matpes_r2scan"))
    delta = ours.get_positions() - theirs.get_positions()  # same start, LBFGS does not wrap
    print(json.dumps(dict(ours=ours_info, theirs=theirs_info, atoms=len(ours),
                          max_displacement=float(np.abs(delta).max()), tier=plan.tier,
                          backend=plan.engine.backend, dtype=plan.precision.dtype, notes=calc.notes)))
""")


@pytest.mark.slow
@pytest.mark.skipif(_mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")
@pytest.mark.parametrize("python", ["pytest", "hybrid"])
def test_lbfgs_with_fm_calc_reaches_maces_minimum(python):
    """``pytest``: this environment's Python; ``hybrid``: ``FASTMACE_HYBRID_PYTHON`` (OEq installed)."""
    if python == "hybrid":
        executable = os.environ.get("FASTMACE_HYBRID_PYTHON")
        if not executable:
            pytest.skip("FASTMACE_HYBRID_PYTHON is not set")
    else:
        executable = sys.executable
    environ = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPOSITORY_ROOT),
                                                                         os.environ.get("PYTHONPATH")])))
    done = subprocess.run([executable, "-c", CHILD, str(_mh1_path()), str(REPOSITORY_ROOT / "tests")],
                          capture_output=True, text=True, env=environ, timeout=1200, cwd=REPOSITORY_ROOT)
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    print(json.dumps(result, indent=1))
    assert result["tier"] == "compiled" and result["dtype"] == "float32"
    if python == "hybrid":
        assert result["backend"] == "hybrid", result["notes"]
    assert result["ours"]["converged"] and result["theirs"]["converged"]
    assert abs(result["ours"]["energy"] - result["theirs"]["energy"]) / result["atoms"] < 1e-4  # eV/atom
    assert result["max_displacement"] < 5e-3  # Å
