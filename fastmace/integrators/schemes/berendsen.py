"""Berendsen thermostat and barostats (ASE 3.27 ``NVTBerendsen``, ``NPTBerendsen``,
``Inhomogeneous_NPTBerendsen``), plan task 4-T02. Non-canonical: equilibration only.

Cell mode ``"hydrostatic"`` is ``NPTBerendsen`` (the whole cell scaled), ``"full"`` is
``Inhomogeneous_NPTBerendsen`` (each cell vector scaled by its ``mask`` flag; the angles stay).
``FixAtoms`` runs with NVT Berendsen only: rule 1 refuses it with every variable cell (decision 44).
"""

from fastmace.integrators.registry import IntegratorRecord

RECORDS = (
    IntegratorRecord(scheme="berendsen", implementation="fastmace.integrators.berendsen:Berendsen",
                     ensembles=("nvt", "npt"), cell_modes=("fixed", "hydrostatic", "full"), constraints=("fixcm", "fix_atoms"),
                     ase_class="ase.md.nvtberendsen.NVTBerendsen", non_canonical=True,
                     buffers=(("target_temperature", "derived", "Integrator.set_temperature"),
                              ("berendsen_pressure", "derived", "Berendsen.restore_externalstress")),
                     report="non-canonical: equilibration only"),
)
