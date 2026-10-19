"""The corrected POLAR-1-S reference against 1a-T19's gate references (plan task 1b-T13, GPU check).

1a-T19's references (run ``1a-T19-gate-leto-20261018``, ``gate-polar1-s``) ran in fresh processes
on leto: the research engine's benchmark calculator, MACE 0.3.16's cuEq conversion, the one-graph
strain and ``polar_gpu``'s corrected stress. This module's float64 reference, built from the
record in a process of the same type (no production engine), must give the same energy, forces
and stress to 1e-12 on all six NaCl 3x3x3 configurations, and pass the issue 1642
finite-difference self-test (1e-6 eV/A^3) on the first one.

``FASTMACE_1A_T19_POLAR1S`` names the run's ``gate-polar1-s`` directory with ``metadata.json``,
``configs/NaCl.npz`` and ``values/reference-NaCl-f64.npz``; the checkpoint is the one its
metadata names, checked by its SHA-256. Without it the test skips.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

RUN = os.environ.get("FASTMACE_1A_T19_POLAR1S")

pytestmark = pytest.mark.skipif(not RUN, reason="FASTMACE_1A_T19_POLAR1S is not set (1a-T19's gate-polar1-s run)")


def _configurations(path):
    from ase import Atoms

    with np.load(path) as data:
        numbers = data["numbers"]
        base = Atoms(numbers=numbers, positions=data["base_positions"], cell=data["base_cell"], pbc=True)
        keys = sorted({name.rsplit("/", 1)[0] for name in data.files if "/" in name})
        configs = {key: Atoms(numbers=numbers, positions=data[f"{key}/positions"], cell=data[f"{key}/cell"], pbc=True)
                   for key in keys}
    return base, configs


def _values(path, structure):
    with np.load(path) as data:
        keys = sorted({name.rsplit("/", 1)[0] for name in data.files})
        return {key.split("/", 1)[1]: (float(data[f"{key}/energy"]), data[f"{key}/forces"].astype(float),
                                       data[f"{key}/stress"].astype(float))
                for key in keys if key.startswith(f"{structure}/")}


def test_polar1s_reference_equals_1a_t19():
    from fastmace.reference import calculators, records

    run = Path(RUN)
    metadata = json.loads((run / "metadata.json").read_text())
    base, configs = _configurations(run / "configs" / "NaCl.npz")
    expected = _values(run / "values" / "reference-NaCl-f64.npz", "NaCl")
    assert sorted(expected) == sorted(configs) and len(configs) == 6
    reference = calculators.load(metadata["checkpoint"], base, dtype="float64", device="cuda:0",
                                 sha256=metadata["checkpoint_sha256"], family="polar")
    info = reference.info()
    assert info["corrections_active"] == ["polar_kspace_stress"] and info["cueq_conv_fusion"] is False
    assert info["record_id"] == records.record_for(records.mace_version(), "polar").id
    worst = {"energy_rel": 0.0, "forces": 0.0, "stress": 0.0}
    for key, atoms in configs.items():
        energy, forces, stress = reference.evaluate(atoms)
        e1a, f1a, s1a = expected[key]
        worst["energy_rel"] = max(worst["energy_rel"], abs(energy - e1a) / abs(e1a))
        worst["forces"] = max(worst["forces"], float(np.abs(forces - f1a).max()))
        worst["stress"] = max(worst["stress"], float(np.abs(stress - s1a).max()))
    check = reference.check_stress_fix(configs["hydrostatic-0"])
    print(json.dumps({"record_id": info["record_id"], "worst": worst, "fd_stress_check": check["max_abs"]}))
    assert worst["energy_rel"] <= 1e-12, worst
    assert worst["forces"] <= 1e-12, worst
    assert worst["stress"] <= 1e-12, worst
    assert check["passed"]
