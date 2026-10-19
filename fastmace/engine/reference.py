"""The reference tier for MD (SPEC §8.4, §9.3 rule 1, §9.4 terminal sequence).

Plan task 2-T01 fixes the contract; plan task 2-T03 implements it. ``ReferenceForceField``
evaluates forces, energy and stress through the versioned reference calculator
(``fastmace.reference.calculators``: MACE's ``MACECalculator`` with the record's corrections) and
implements the force-field protocol of ``fastmace.model.forces.ForceField``, so FastMACE's own
integrators, noise stream, output and restart drive it: only the forces come from the reference.
It runs on ``GraphEngine`` with ``capture=False`` and exact edge counts, with no rewrites.

The planner builds it for ``mode="reference"`` (``allow_reference=True`` where rule 1 needs it) and
as the last candidate of the fallback's terminal sequence when it fits (§9.4 step 5).
"""

from __future__ import annotations

from typing import Any

TASK = "plan task 2-T03"


class ReferenceForceField:
    """A ``fastmace.model.forces.ForceField`` whose evaluations are the reference calculator's.

    ``force_field``: the ``fm.ForceField`` (pristine model, superset, identity); ``atoms``: the run's
    structure; ``device``: where the reference runs (default: the planner's device); ``dtype``:
    ``"float64"`` (default) or ``"float32"``.
    """

    def __init__(self, force_field: Any, atoms: Any, *, device: str | None = None, dtype: str = "float64"):
        raise NotImplementedError(f"the reference tier for MD is {TASK}")
