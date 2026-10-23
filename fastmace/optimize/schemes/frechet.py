"""ASE 3.27's ``FrechetCellFilter`` on the device (plan task 4-T07): the cell filter both optimisers run with."""

from fastmace.integrators.registry import CellFilterRecord

# FrechetCellFilter.BUFFERS: the DOF vector, log F, the gradient, the energy and the range flag, recomputed by
# ``refresh`` at a canonical point (optimisers keep no restart state, conductor's choice 5).
_BUFFERS = ("x", "log_deform", "gradient", "energy", "cell_ok")

RECORDS = (
    CellFilterRecord(scheme="frechet", implementation="fastmace.optimize.cell:FrechetCellFilter",
                     refusals={"fix_atoms": "fixed atoms in a moving cell have no ASE reference in FastMACE 1.0 "
                                            "(owner decision D1)",
                               "fixcm": "a cell filter has no centre-of-mass option (ASE's has none)"},
                     ase_class="ase.filters.FrechetCellFilter",
                     buffers=tuple((name, "reset", "FrechetCellFilter.refresh") for name in _BUFFERS)),
)
