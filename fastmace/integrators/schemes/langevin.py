"""Langevin NVT (ASE ``Langevin``) and Langevin-Hoover NPT (ASE 3.27 ``LangevinBAOAB``), phases 1a-3."""

from fastmace.integrators.registry import IntegratorRecord

RECORDS = (
    IntegratorRecord(scheme="langevin", implementation="fastmace.integrators.langevin:Langevin",
                     ensembles=("nvt",), cell_modes=("fixed",), constraints=("fixcm", "fix_atoms"),
                     ase_class="ase.md.langevin.Langevin",
                     dof_rule="langevin"),
    IntegratorRecord(scheme="baoab", implementation="fastmace.integrators.baoab:LangevinBAOAB",
                     ensembles=("npt",), cell_modes=("hydrostatic", "full"), constraints=("fixcm",),
                     refusals={"fix_atoms": "a variable cell with fixed atoms is refused (owner decision D1)"},
                     ase_class="ase.md.langevinbaoab.LangevinBAOAB", dof_rule="langevin",
                     keywords="fastmace.planner.fixed:integrator_keywords",
                     report="experimental: not yet qualified"),
)
