"""Tiny-model simulations for the MC driver tests (plan task 3-T09; also used by 3-T12).

- ``tiny_loaded(directory, z_values, family="mh1", charge=None, spin_multiplicity=None)``: a tiny MACE
  model of ``family`` saved and loaded as ``fm.load`` does. ``family="omol"`` with a charge and spin is a
  model that reports charge and spin inputs (its force field registers ``total_charge`` and
  ``total_spin``), for the charge/spin parity refusal of SPEC §10.4.
- ``mc_simulation(loaded, atoms, superset, ...)``: a CPU float64 ``Simulation`` with that superset and a
  workload declaring species changes and MC moves.
- ``cpu_planner(monkeypatch)``: the eager CPU planner the ``Simulation`` builds with.
"""

from __future__ import annotations

import torch

from _support.tiny_models import FAMILIES, tiny_mace_model

MC_WORKLOAD = dict(max_temperature_K=5000.0, species_changes=True, min_mc_temperature_K=1.0,
                   mc_moves=("swap", "transmute"))


def tiny_loaded(directory, z_values, family="mh1", charge=None, spin_multiplicity=None, **options):
    from fastmace.model import loading

    model = tiny_mace_model(list(z_values), **{**FAMILIES[family], **options})
    path = directory / f"tiny-{family}-{'-'.join(map(str, z_values))}.model"
    if not path.exists():
        torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True, charge=charge, spin_multiplicity=spin_multiplicity)


def cpu_planner(monkeypatch):
    """Make ``Simulation`` plan eagerly on the CPU (as ``tests/cpu/test_events.py``)."""
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.planner import minimal

    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def mc_simulation(loaded, atoms, superset, *, integrator, seed=1, workload=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    workload = Workload(**MC_WORKLOAD) if workload is None else workload
    force_field = ForceField(loaded, atoms, precision="float64", species=list(superset))
    return Simulation(atoms, force_field, integrator=integrator, workload=workload, seed=seed)
