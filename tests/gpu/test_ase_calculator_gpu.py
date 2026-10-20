"""``fm.calc`` on the GPU (plan task 1b-T18; SPEC §7.4, §7.7 level 0).

MH-1 (``matpes_r2scan``) on rattled NaCl 2x2x2: ``ase.optimize.LBFGS`` with ``fm.calc`` (float32
forces, float64 energy, compiled tier) reaches the minimum that LBFGS reaches with MACE's own
calculator (float64). The relaxation runs in a child process: the calculator's compiled plan fixes
the process's compile family (SPEC §8.7), which must not depend on the tests run before it. The
relaxation compiles once, although the edge count changes between its calls (2-T13). MACE's own
relaxation is the same for both parametrisations, so it runs once per pytest process, in a child of
this environment's Python (2-T14e: it took about 24 s in each before).
"""

import functools
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")


def _mh1_path():
    if not MODELS:
        return None
    from fastmace.model import labels

    path = Path(MODELS) / labels.LABELS["mh1-r2scan"].checkpoint.file_name
    return path if path.is_file() else None


RELAX = """
    def relax(calculator):
        atoms = nacl(repeat=2, rattle=0.08, seed=21)
        atoms.calc = calculator
        start = time.perf_counter()
        optimizer = LBFGS(atoms, logfile=None)
        converged = optimizer.run(fmax=2e-3, steps=300)
        return atoms, dict(converged=bool(converged), steps=optimizer.nsteps,
                           seconds=time.perf_counter() - start, energy=atoms.get_potential_energy(),
                           fmax=float(np.abs(atoms.get_forces()).max()),
                           positions=atoms.get_positions().tolist())
"""

CHILD = textwrap.dedent("""
    import json, sys, time
    import fastmace as fm  # before MACE, so the planner may choose the hybrid backend
    import numpy as np
    from torch._dynamo.utils import counters
    sys.path.insert(0, sys.argv[2])
    from _support.structures import nacl
    from ase.optimize import LBFGS
""") + textwrap.dedent(RELAX) + textwrap.dedent("""

    calc = fm.calc("mh1", head="matpes_r2scan")
    calls, graphs, edges, calculate = [], [], [], calc.calculate

    def timed(*args, **kwargs):
        start = time.perf_counter()
        calculate(*args, **kwargs)
        calls.append(round(time.perf_counter() - start, 4))
        graphs.append(counters["stats"]["unique_graphs"])
        edges.append(int(calc._evaluator.neighbours.count))

    calc.calculate = timed
    ours, ours_info = relax(calc)
    ours_info.update(call_seconds=calls, graphs_after_call=graphs, edges=edges)
    plan = calc.plan
    print(json.dumps(dict(ours=ours_info, atoms=len(ours), tier=plan.tier, backend=plan.engine.backend,
                          dtype=plan.precision.dtype, dynamic_edges=plan.engine.dynamic_edges, notes=calc.notes)))
""")

MACE_CHILD = textwrap.dedent("""
    import json, sys, time
    import numpy as np
    sys.path.insert(0, sys.argv[2])
    from _support.structures import nacl
    from ase.optimize import LBFGS
""") + textwrap.dedent(RELAX) + textwrap.dedent("""
    from mace.calculators import MACECalculator
    theirs, theirs_info = relax(MACECalculator(model_paths=sys.argv[1], device="cuda", default_dtype="float64",
                                               head="matpes_r2scan"))
    print(json.dumps(theirs_info))
""")


def _run(executable, code):
    environ = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPOSITORY_ROOT),
                                                                         os.environ.get("PYTHONPATH")])))
    done = subprocess.run([executable, "-c", code, str(_mh1_path()), str(REPOSITORY_ROOT / "tests")],
                          capture_output=True, text=True, env=environ, timeout=1200, cwd=REPOSITORY_ROOT)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@functools.cache
def maces_minimum():
    """LBFGS with MACE's own calculator in float64, once per pytest process (this environment's Python)."""
    return _run(sys.executable, MACE_CHILD)


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
    result = _run(executable, CHILD)
    theirs = maces_minimum()
    # Same start, LBFGS does not wrap.
    result.update(theirs=theirs, max_displacement=float(np.abs(np.array(result["ours"]["positions"])
                                                                - np.array(theirs["positions"])).max()))
    print(json.dumps({key: value for key, value in result.items() if key not in ("ours", "theirs")}, indent=1))
    assert result["tier"] == "compiled" and result["dtype"] == "float32" and result["dynamic_edges"]
    if python == "hybrid":
        assert result["backend"] == "hybrid", result["notes"]
    assert result["ours"]["converged"] and result["theirs"]["converged"]
    assert abs(result["ours"]["energy"] - result["theirs"]["energy"]) / result["atoms"] < 1e-4  # eV/atom
    assert result["max_displacement"] < 5e-3  # Å
    # One compile (2-T13): the edge count changes, no graph is added after the first call.
    graphs, edges = result["ours"]["graphs_after_call"], result["ours"]["edges"]
    assert len(set(edges)) > 1, edges
    assert graphs[0] > 0 and set(graphs) == {graphs[0]}, graphs
