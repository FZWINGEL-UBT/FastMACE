"""FIRE (ASE 3.27 ``FIRE``) on the graph engine, fixed cell and with a cell filter (plan task 4-T06)."""

from fastmace.integrators.registry import OptimizerRecord

# No optimiser restart files (conductor's choice 5): a relaxation starts afresh with a new optimiser, so its
# history resets; the threshold and the convergence check follow from set_fmax and update.
_BUFFERS = (
    *((name, "reset", "FIRE.__init__") for name in ("vel", "dt", "a", "nsteps", "started", "e_last", "r_last")),
    ("fmax", "derived", "FIRE.set_fmax"),
    *((name, "derived", "FIRE.update") for name in ("max_force_value", "converged_flag")),
)

RECORDS = (
    OptimizerRecord(scheme="fire", implementation="fastmace.optimize.fire:FIRE", cell_filters=(None, "frechet"),
                    constraints=("fix_atoms",), ase_class="ase.optimize.fire.FIRE", buffers=_BUFFERS),
)
