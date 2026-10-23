"""Langevin NVT (ASE ``Langevin``) and Langevin-Hoover NPT (ASE 3.27 ``LangevinBAOAB``), phases 1a-3."""

from fastmace.integrators.registry import IntegratorRecord

_NO_FIX_ATOMS = "FixAtoms comes with plan task 4-T04 (owner decision D1)"

RECORDS = (
    IntegratorRecord(scheme="langevin", implementation="fastmace.integrators.langevin:Langevin",
                     ensembles=("nvt",), cell_modes=("fixed",), constraints=("fixcm",),
                     refusals={"fix_atoms": _NO_FIX_ATOMS}, ase_class="ase.md.langevin.Langevin",
                     dof_rule="langevin"),
    IntegratorRecord(scheme="baoab", implementation="fastmace.integrators.baoab:LangevinBAOAB",
                     ensembles=("npt",), cell_modes=("hydrostatic", "full"), constraints=("fixcm",),
                     refusals={"fix_atoms": "a variable cell with fixed atoms is refused (owner decision D1)"},
                     ase_class="ase.md.langevinbaoab.LangevinBAOAB", dof_rule="langevin",
                     keywords="fastmace.planner.fixed:integrator_keywords",
                     report="experimental: not yet qualified"),
)
