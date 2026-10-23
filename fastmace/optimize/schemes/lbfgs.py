"""LBFGS (ASE 3.27 ``LBFGS``, plan task 4-T05): fixed cell, and with a cell filter once one is registered (4-T07)."""

from fastmace.integrators.registry import OptimizerRecord

# DeviceLBFGS.BUFFERS: the history ring and the step scalars, recomputed by ``start`` at a canonical point
# (optimisers keep no restart state, conductor's choice 5).
_BUFFERS = ("s", "y", "rho", "sy", "r0", "g0", "iteration", "fmax", "max_force_now", "converged", "status")

RECORDS = (
    OptimizerRecord(scheme="lbfgs", implementation="fastmace.optimize.lbfgs:DeviceLBFGS", constraints=("fix_atoms",),
                    refusals={"fixcm": "an optimiser has no centre-of-mass option (ASE's LBFGS has none)"},
                    ase_class="ase.optimize.lbfgs.LBFGS", cell_filters=(None, "frechet"),
                    buffers=tuple((name, "reset", "DeviceLBFGS.start") for name in _BUFFERS)),
)
