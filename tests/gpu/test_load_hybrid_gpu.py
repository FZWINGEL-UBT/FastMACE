"""After a plain public ``fm.load``, the planner can still choose the hybrid (plan task 1b-T08b).

``fm.load`` imports MACE; it prepares the OEq runtime first, so the minimal
planner's backend rule gives ``hybrid``, not ``backend-cueq-mace-imported-first``.
Runs in a fresh interpreter named by ``FASTMACE_HYBRID_PYTHON`` (a hybrid
environment), with the MH-1 checkpoint from ``FASTMACE_MODELS``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastmace.model import labels as table

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS = os.environ.get("FASTMACE_MODELS")
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable


def _hybrid_has(code):
    return subprocess.run([HYBRID_PYTHON, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


pytestmark = [
    pytest.mark.skipif(not _hybrid_has("import torch, sys; sys.exit(not torch.cuda.is_available())"),
                       reason="no CUDA device in the hybrid interpreter"),
    pytest.mark.skipif(not _hybrid_has("import importlib.util, sys; "
                                       "sys.exit(importlib.util.find_spec('openequivariance') is None)"),
                       reason="OpenEquivariance is not installed in the hybrid interpreter"),
    pytest.mark.skipif(not _hybrid_has("import fastmace.planner.minimal"), reason="no MinimalPlanner (1b-T12)"),
    pytest.mark.skipif(not (MODELS and (Path(MODELS) / table.LABELS["mh1-r2scan"].checkpoint.file_name).is_file()),
                       reason="MH-1 checkpoint not in FASTMACE_MODELS"),
]


def test_fm_load_then_minimal_planner_gives_the_hybrid():
    code = textwrap.dedent("""
        import json, sys
        import fastmace as fm
        assert "mace" not in sys.modules
        model = fm.load("mh1-r2scan")
        from _support.structures import nacl
        from fastmace.planner.minimal import MinimalPlanner
        atoms = nacl(repeat=2, rattle=0.05, seed=17)
        ff = fm.ForceField(model, atoms)
        plan = MinimalPlanner().plan_evaluation(ff, atoms)
        print(json.dumps({"oeq_runtime": model.oeq_runtime, "backend": plan.engine.backend}))
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if HYBRID_PYTHON != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    done = subprocess.run([HYBRID_PYTHON, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                          text=True, timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-6000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {"oeq_runtime": "prepared", "backend": "hybrid"}
