"""NPT with D3 on CUDA against ASE (plan task 3b-T05; SPEC §11.2, §12.2).

A tiny MH-1-like model on NaCl 2x2x2 (64 atoms), a float64 plan through the minimal planner on ``cuda:0`` in a
fresh process, D3(BJ) r2scan on its Triton passes with a small D3 skin (the D3 list rebuilds during the run),
captured graphs, checked and pipelined. 100 hydrostatic NPT steps at 600 K and 0.5 GPa follow ASE's
``LangevinBAOAB`` driving MACE's own calculator plus ``torch-dftd`` (float64, ``SumCalculator``) with the same
seed: cell, positions and momenta at the end, and the barostat moved the cell.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from _support.tiny_models import FAMILIES, tiny_mace_model

pytest.importorskip("triton")

STEPS = 100


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_npt_d3_gpu_helpers", Path(__file__).parent / "test_engine_gpu.py")

CHILD = """
import json, warnings
import numpy as np
from ase import units
from ase.calculators.mixing import SumCalculator
from ase.md.langevinbaoab import LangevinBAOAB as ASELangevinBAOAB
from fastmace import integrators
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.longrange.d3 import D3
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner import minimal
from fastmace.planner.workload import Workload
from fastmace.reference.d3 import torch_dftd3_calculator

RECORD = D3(functional="r2scan")

def structure():
    from ase.build import bulk
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=3)
    return atoms

loaded = loading.load_pristine(SPEC["path"], trusted=True)

def build(pipelined):
    atoms = structure()
    made = minimal.MinimalPlanner(engine=EngineConfig(device="cuda:0", compile_mode="none", expandable_segments="off",
                                                      pipelined="on" if pipelined else "off"),
                                  neighbours=NeighbourConfig(d3_skin=0.05), environ={},
                                  process=constraints.ProcessRecord())
    field = ForceField(loaded, atoms, precision="float64", dispersion=RECORD)
    record = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.01,
                                       pressure_GPa=0.5).to_config(seed=7)
    plan = made.plan(field, atoms, record, Workload(max_temperature_K=600.0))
    return made.build(plan, field, atoms)

def ase_run(dyn):
    atoms = structure()
    atoms.calc = SumCalculator([loaded.calculator, torch_dftd3_calculator(RECORD, device="cuda")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase_dyn = ASELangevinBAOAB(atoms, timestep=1.0 * units.fs, temperature_K=600.0,
                                   externalstress=-0.5 * units.GPa, hydrostatic=True, T_tau=dyn.T_tau,
                                   P_tau=dyn.P_tau, rng=np.random.default_rng(7))
    out = {"mass_ratio": dyn.barostat_mass / ase_dyn.barostat_mass}
    ase_dyn.run(SPEC["steps"])
    return atoms, out

out = {"modes": {}}
reference = None
for pipelined in (False, True):
    engine = build(pipelined)
    dyn = engine.integrator
    if reference is None:
        reference, out["ase"] = ase_run(dyn)
    dyn.set_temperature(600.0)
    engine.run(SPEC["steps"])
    values = engine.sync()
    out["modes"]["pipelined" if pipelined else "checked"] = {
        "type": type(engine.force_field).__name__,
        "captured": bool(engine.capture),
        "cell": float(np.abs(values["cell"] - reference.cell.array).max()),
        "positions": float(np.abs(values["positions"] - reference.positions).max()),
        "momenta": float(np.abs(values["momenta"] - reference.get_momenta()).max()),
        "d3_builds": int(engine.force_field.d3.builds),
    }
out["volume_change"] = reference.get_volume() / structure().get_volume() - 1
print(json.dumps(out))
"""


def test_npt_with_d3_follows_ase(tmp_path):
    pytest.importorskip("torch_dftd")
    path = tmp_path / "tiny-mh1-11-17.model"
    torch.save(tiny_mace_model([11, 17], **FAMILIES["mh1"]), path)
    out = ENGINE.run_fresh(f"SPEC = {{'path': {str(path)!r}, 'steps': {STEPS}}}\n" + CHILD)
    assert out["ase"]["mass_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert abs(out["volume_change"]) > 1e-4  # the barostat moved the cell
    for mode, result in out["modes"].items():
        assert result["type"] == "DispersedForceField" and result["captured"], (mode, result)
        assert result["d3_builds"] > 1, (mode, result)  # the D3 list rebuilt during the run
        assert result["cell"] < 1e-8 and result["positions"] < 1e-8, (mode, result)
        assert result["momenta"] < 1e-6, (mode, result)
