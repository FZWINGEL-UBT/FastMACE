"""``combo_kernels`` moves no value beyond the backend's own repeatability (opt-3 OPT3-T04), on CUDA.

In one process: ``fm.ForceField.evaluate`` (the compiled evaluation plan) of MH-1 on a rattled NaCl 2×2×2 with
the record off, off again (the backend's own repeatability) and on. Where two "off" evaluations are bitwise
equal, "on" must be too; where the backend is not bitwise repeatable (atomics), "on" stays within the "off"
repeat's difference. Slow tier (two compiles).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

MODELS = os.environ.get("FASTMACE_MODELS")

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def _mh1():
    if not MODELS:
        return None
    from fastmace.model import labels

    path = Path(MODELS) / labels.LABELS["mh1-r2scan"].checkpoint.file_name
    return path if path.is_file() else None


@pytest.mark.skipif(_mh1() is None, reason="needs MH-1 in FASTMACE_MODELS")
def test_combo_kernels_change_no_value():
    import fastmace as fm
    from _support.structures import nacl
    from fastmace.config import RewriteConfig

    atoms = nacl(repeat=2, rattle=0.05, seed=4)
    model = fm.load("mh1-r2scan")
    results = {}
    for name, combo in (("off", "off"), ("on", "on")):
        ff = fm.ForceField(model, atoms, precision="float32", rewrites=RewriteConfig(combo_kernels=combo))
        results[name] = [ff.evaluate(atoms, stress=True) for _ in range(2 if name == "off" else 1)]
        plan = ff.evaluation_plan(atoms)
        assert plan.decision("RewriteConfig.combo_kernels")[1] == combo

    def diff(a, b):
        return max(abs(a[0] - b[0]), float(np.abs(a[1] - b[1]).max()), float(np.abs(a[2] - b[2]).max()))

    repeat = diff(results["off"][0], results["off"][1])
    change = diff(results["off"][0], results["on"][0])
    print(f"\ncombo_kernels: off repeat {repeat:.3g}, on vs off {change:.3g}")
    if repeat == 0.0:
        assert change == 0.0
    else:
        assert change <= 2 * repeat
