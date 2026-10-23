"""``combo_kernels`` moves no value beyond the backend's own repeatability (opt-3 OPT3-T04), on CUDA.

In one process: ``fm.ForceField.evaluate`` (the compiled evaluation plan) of MH-1 on a rattled NaCl 2×2×2 with
the record off, three times (the backend's own repeatability), and on. Where the "off" evaluations are bitwise
equal, "on" must be too; where the backend is not bitwise repeatable (atomics), "on" stays within twice the "off"
repeats' largest difference, plus 4 float32 ulps of the largest value compared. One pair of "off" evaluations was
too small a noise estimate on a Colab G4 (RTX PRO 6000, sm_120; fm-ops-colab's compatibility runs, 23 October
2026): a repeat of 4.8e-7 (1 ulp in [4, 8)) against a change of 1.2e-6. A fusion that changed the arithmetic moves
values by far more than a few ulps. Slow tier (two compiles).
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
        results[name] = [ff.evaluate(atoms, stress=True) for _ in range(3 if name == "off" else 1)]
        plan = ff.evaluation_plan(atoms)
        assert plan.decision("RewriteConfig.combo_kernels")[1] == combo

    def diff(a, b):
        return max(abs(a[0] - b[0]), float(np.abs(a[1] - b[1]).max()), float(np.abs(a[2] - b[2]).max()))

    off = results["off"]
    repeat = max(diff(off[i], off[j]) for i in range(len(off)) for j in range(i + 1, len(off)))
    change = diff(off[0], results["on"][0])
    largest = max(abs(off[0][0]), float(np.abs(off[0][1]).max()), float(np.abs(off[0][2]).max()))
    ulps = 4 * float(np.spacing(np.float32(largest)))
    print(f"\ncombo_kernels: off repeat {repeat:.3g}, on vs off {change:.3g}, 4 ulps {ulps:.3g}")
    if repeat == 0.0:
        assert change == 0.0
    else:
        assert change <= 2 * repeat + ulps
