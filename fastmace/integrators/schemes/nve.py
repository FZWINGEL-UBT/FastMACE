"""Velocity Verlet NVE (ASE ``VelocityVerlet``), plan task 4-T02."""

from fastmace.integrators.registry import IntegratorRecord

RECORDS = (
    IntegratorRecord(scheme="nve", implementation="fastmace.integrators.nve:VelocityVerlet",
                     ensembles=("nve",), cell_modes=("fixed",), constraints=("fixcm",),
                     refusals={"fix_atoms": "FixAtoms comes with plan task 4-T04 (owner decision D1)"},
                     ase_class="ase.md.verlet.VelocityVerlet", conserved="E_kin + E_pot"),
)
