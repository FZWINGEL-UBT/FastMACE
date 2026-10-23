"""FIRE (ASE 3.27 ``FIRE``) on the graph engine, fixed cell and with a cell filter (plan task 4-T06)."""

from fastmace.integrators.registry import OptimizerRecord

# FIRE.BUFFERS: velocity, step scalars, downhill state and the check, recomputed by ``start`` at a canonical
# point (optimisers keep no restart state, conductor's choice 5).
_BUFFERS = ("vel", "dt", "a", "nsteps", "started", "e_last", "r_last", "restarts", "fmax", "max_force_value",
            "converged_flag", "status")

RECORDS = (
    OptimizerRecord(scheme="fire", implementation="fastmace.optimize.fire:FIRE", constraints=("fix_atoms",),
                    refusals={"fixcm": "an optimiser has no centre-of-mass option (ASE's FIRE has none)"},
                    ase_class="ase.optimize.fire.FIRE", cell_filters=(None, "frechet"),
                    buffers=tuple((name, "reset", "FIRE.start") for name in _BUFFERS)),
)
