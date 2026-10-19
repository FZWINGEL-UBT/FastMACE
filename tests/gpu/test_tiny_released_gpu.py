"""Tiny-model versions of two GPU tests that run released checkpoints, for the fast tier (plan task 1b-T26).

1b-T26 marked the MH-1 and POLAR-1-S tests of 1b-T04, T08b, T14, T16, T17 and T18 ``slow``. Each
test here runs the same child script as its MH-1 test, with a tiny random MH-1-like model
(``_support.tiny_models``) loaded from a path with ``trusted=True``, and makes the same assertions:

- ``test_registry_gpu::test_mh1_plain_cueq_float64``: capture warm-ups leave no trace, and the
  registry covers what the stages mutate (29 s);
- ``test_selfcheck_gpu::test_a_broken_captured_graph_fails_while_eager_passes``: the gate fails a
  wrong captured graph and passes the eager stages (36 s).

Tiny versions of the other tests were tried and left out (1b-T26 report): they cost almost as much as
the MH-1 tests (30 s to 5 min), because fresh processes, compiles and child processes dominate. And
a random tiny model breaks two of them (LBFGS does not converge; a 500-step restart run drifts to
2.5e-11 Å). The behaviours tested only in the slow tier are listed in ``docs/development.md``.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from _support.tiny_models import FAMILIES, tiny_mace_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _module(name, file_name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_tiny_released_engine_helpers", "test_engine_gpu.py")
SELFCHECK = _module("_tiny_released_selfcheck", "test_selfcheck_gpu.py")
REGISTRY = _module("_tiny_released_registry", "test_registry_gpu.py")


def tiny(directory):
    """A tiny MH-1-like Na/Cl checkpoint in ``directory``; returns its path."""
    path = directory / "tiny-mh1.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    return str(path)


def swap(text, old, new):
    assert old in text, old
    return text.replace(old, new)


def test_tiny_registry_plain_cueq_float64(tmp_path):
    spec = ENGINE.spec("cueq")
    spec.update(planner_backend="cueq", label=tiny(tmp_path), ensemble="nvt")
    child = swap(REGISTRY.CHILD, 'planner.load(SPEC["label"])', 'planner.load(SPEC["label"], trusted=True)')
    result = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + child)
    assert (result["backend"], result["dtype"]) == ("cueq", "float64")
    REGISTRY.check(result, "tiny-mh1-cueq-float64")


def test_tiny_broken_captured_graph_fails_while_eager_passes(tmp_path):
    child = swap(SELFCHECK.CHILD, 'loading.load_pristine(SPEC["label"])',
                 'loading.load_pristine(SPEC["label"], trusted=True)')
    python = SELFCHECK.HYBRID_PYTHON or sys.executable
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "tests")]),
               PYTHONDONTWRITEBYTECODE="1")
    if python != sys.executable:
        env.pop("LD_LIBRARY_PATH", None)  # set for this environment; the in-process OEq runtime needs none
    spec = dict(label=tiny(tmp_path), **SELFCHECK.NACL, broken_capture=True)
    done = subprocess.run([python, "-c", child, json.dumps(spec)], cwd=REPOSITORY_ROOT, env=env,
                          capture_output=True, text=True, timeout=1800, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-6000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    SELFCHECK._report("tiny-mh1-broken-capture", out)
    assert out["eager"]["status"] == "passed" and out["eager"]["mode"] == "eager stages C, B"
    gate = out["gate"]
    assert gate["status"] == "failed" and gate["mode"] == "captured replay of stages C, B"
    assert not gate["quantities"]["force_eV_per_A"]["passed"]
    assert gate["quantities"]["energy_meV_per_atom"]["passed"]
