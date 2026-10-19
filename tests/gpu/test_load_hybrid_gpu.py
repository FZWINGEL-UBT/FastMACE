"""After a plain public ``fm.load``, the planner can still choose the hybrid (plan task 1b-T08b).

``fm.load`` imports MACE; it prepares the OEq runtime first, so the minimal
planner's backend rule gives ``hybrid``, not ``backend-cueq-mace-imported-first``.
Runs in a fresh interpreter named by ``FASTMACE_HYBRID_PYTHON`` (a hybrid
environment), with the MH-1 checkpoint from ``FASTMACE_MODELS``.
"""

import importlib.util
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


needs_cuda = pytest.mark.skipif(not _hybrid_has("import torch, sys; sys.exit(not torch.cuda.is_available())"),
                                reason="no CUDA device in the hybrid interpreter")
needs_oeq = pytest.mark.skipif(not _hybrid_has("import importlib.util, sys; "
                                               "sys.exit(importlib.util.find_spec('openequivariance') is None)"),
                               reason="OpenEquivariance is not installed in the hybrid interpreter")
needs_planner = pytest.mark.skipif(not _hybrid_has("import fastmace.planner.minimal"),
                                   reason="no MinimalPlanner (1b-T12)")
needs_mh1 = pytest.mark.skipif(
    not (MODELS and (Path(MODELS) / table.LABELS["mh1-r2scan"].checkpoint.file_name).is_file()),
    reason="MH-1 checkpoint not in FASTMACE_MODELS")

# fm.load, then the evaluation plan; prints the outcome of the OEq preparation and the backend decision.
PLAN = """
    import json, sys
    {before}
    import fastmace as fm
    model = fm.load("mh1-r2scan")
    from _support.structures import nacl
    from fastmace.planner.minimal import MinimalPlanner
    atoms = nacl(repeat=2, rattle=0.05, seed=17)
    plan = MinimalPlanner().plan_evaluation(fm.ForceField(model, atoms), atoms)
    reason = next(r for key, choice, r in plan.decisions if key == "EngineConfig.backend")
    print(json.dumps({{"oeq_runtime": model.oeq_runtime, "backend": plan.engine.backend,
                      "code": getattr(reason, "code", None)}}))
"""


def run_plan(python, extensions_dir, before=""):
    code = textwrap.dedent(PLAN).format(before=before)
    # An empty TORCH_EXTENSIONS_DIR of its own: when MACE comes first, OEq 0.7 (no precompiled extension)
    # tries a JIT build in torch's default directory, where a lock left by a killed build makes it wait
    # 5 minutes (1b-T27: ~/.cache/torch_extensions/.../libtorch_tp_jit/lock from 8 October, 310 s).
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1", TORCH_EXTENSIONS_DIR=str(extensions_dir))
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    done = subprocess.run([python, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                          text=True, timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-6000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


pytestmark = [needs_cuda, needs_planner, needs_mh1]


@needs_oeq
def test_fm_load_then_minimal_planner_gives_the_hybrid(tmp_path):
    assert run_plan(HYBRID_PYTHON, tmp_path) == {"oeq_runtime": "prepared", "backend": "hybrid",
                                                 "code": "backend-hybrid"}


@needs_oeq
def test_mace_imported_first_gives_plain_cueq_with_the_reason(tmp_path):
    result = run_plan(HYBRID_PYTHON, tmp_path, before="import mace.calculators")
    assert result == {"oeq_runtime": "too late: mace was imported first", "backend": "cueq",
                      "code": "backend-cueq-mace-imported-first"}


@pytest.mark.skipif(importlib.util.find_spec("openequivariance") is not None,
                    reason="this interpreter has OpenEquivariance; the plain case needs one without it")
def test_without_oeq_fm_load_gives_plain_cueq(tmp_path):
    result = run_plan(sys.executable, tmp_path)
    assert result == {"oeq_runtime": "skipped: OpenEquivariance is not installed", "backend": "cueq",
                      "code": "backend-cueq-oeq-unavailable"}
