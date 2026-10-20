"""The converted-model template cache on the GPU (opt-1 OPT-T06): a template copy is the reload, bitwise.

Runs ``results/opt-T06-probes/gpu_copy_vs_load.py`` on the tiny MH-1-like model (``_support.tiny_models``) in a
fresh interpreter per backend (SPEC §8.7: one backend family per process); the hybrid in
``FASTMACE_HYBRID_PYTHON``. Each child compares a plan's model from a template copy with one from a fresh reload
(today's later plan):

- fast tier: the model level, energy and forces on rattled NaCl 2x2x2, for hybrid and plain cuEq in float32 and
  float64, and that ``release_templates`` gives the template's memory back;
- slow tier: the engine level, a second ``Simulation`` with the cache on against one with it off, initial forces
  and 10 MD steps.

Bitwise wherever the same plan repeated is bitwise (the child measures that noise first).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from _support.tiny_models import FAMILIES, tiny_mace_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPOSITORY_ROOT / "results" / "opt-T06-probes" / "gpu_copy_vs_load.py"
HYBRID_PYTHON = os.environ.get("FASTMACE_HYBRID_PYTHON") or sys.executable
CASES = [("hybrid", "float32"), ("hybrid", "float64"), ("cueq", "float32"), ("cueq", "float64")]


def _python(backend):
    return HYBRID_PYTHON if backend == "hybrid" else sys.executable


def _has(python, code):
    return subprocess.run([python, "-c", code], capture_output=True, timeout=120, check=False).returncode == 0


def _skip_unless_possible(backend):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if backend == "hybrid" and not _has(HYBRID_PYTHON, "import openequivariance"):
        pytest.skip("OpenEquivariance is not installed in the hybrid interpreter")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny") / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    return path


def _probe(backend, dtype, checkpoint, directory, engine_steps):
    output = directory / f"{backend}-{dtype}.json"
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = subprocess.run([_python(backend), str(PROBE), "--checkpoint", str(checkpoint), "--backend", backend,
                          "--dtype", dtype, "--engine-steps", str(engine_steps), "--json", str(output)],
                         capture_output=True, text=True, timeout=1200, env=environment, cwd=directory, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(output.read_text())


@pytest.mark.parametrize(("backend", "dtype"), CASES)
def test_a_template_copy_is_the_reload(backend, dtype, tiny_checkpoint, tmp_path):
    _skip_unless_possible(backend)
    result = _probe(backend, dtype, tiny_checkpoint, tmp_path, engine_steps=0)
    model = result["model"]
    if model["first plan rerun (noise)"]["bitwise"]:
        assert model["template copy vs reload"]["bitwise"], model
    else:  # a nondeterministic kernel: the copy is no further from the reload than a rerun is from itself
        noise = model["first plan rerun (noise)"]["max_force"]
        assert model["template copy vs reload"]["max_force"] <= 10 * noise, model
    assert result["template_bytes"] > 0
    assert result["allocated_freed_by_release"] >= 0.9 * result["template_bytes"], result


@pytest.mark.slow
@pytest.mark.parametrize(("backend", "dtype"), [("hybrid", "float32"), ("cueq", "float32")])
def test_a_second_simulation_from_a_template_copy_runs_as_from_a_reload(backend, dtype, tiny_checkpoint, tmp_path):
    _skip_unless_possible(backend)
    engine = _probe(backend, dtype, tiny_checkpoint, tmp_path, engine_steps=10)["engine"]
    assert engine["templates_kept"] == {"on": 1, "off": 0}
    first, second = engine["first_plans"], engine["second_plans_copy_vs_reload"]
    for quantity in ("forces", "positions"):
        if first[quantity]["bitwise"]:
            assert second[quantity]["bitwise"], engine
        else:
            assert second[quantity]["max"] <= 10 * first[quantity]["max"], engine
