"""CSVR, Bussi-Donadio-Parrinello velocity rescaling (ASE 3.27 ``Bussi``), plan task 4-T02."""

from fastmace.integrators.registry import IntegratorRecord

RECORDS = (
    IntegratorRecord(scheme="csvr", implementation="fastmace.integrators.csvr:CSVR",
                     ensembles=("nvt",), cell_modes=("fixed",), constraints=("fixcm",),
                     refusals={"fix_atoms": "FixAtoms comes with plan task 4-T04 (owner decision D1)"},
                     ase_class="ase.md.bussi.Bussi", state_keys=("transferred_energy",),
                     # noise: the step's row, written by prepare_step before every step; target_kinetic:
                     # from the temperature target and ndof (the restart's integrator record)
                     buffers=(("noise", "derived", None), ("transferred_energy", "save", None),
                              ("target_kinetic", "derived", "Integrator.set_temperature")),
                     conserved="E_kin + E_pot - transferred energy"),
)
