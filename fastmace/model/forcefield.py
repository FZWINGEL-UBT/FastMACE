"""The public, plan-free force field ``fm.ForceField`` (SPEC §7.1, §7.6 "ForceField and plans").

Plan task 1b-T01 fixes the signature; plan task 1b-T12 implements it. SPEC §5's ``forces.py`` holds
phase 1a's ``MACEForceField``, the execution copy a plan builds; this module holds the object
the user holds. It keeps the pristine loaded model, the force precision, the species superset and
the force-field identity (§10.6, 1b-T05), and builds nothing: every plan, including the one-off
plan of ``evaluate``, makes its own execution copy through the planner.
"""

from __future__ import annotations

from typing import Any, Literal

_TASK = "fastmace.ForceField is a stub until plan task 1b-T12"


class ForceField:
    """``ForceField(model, atoms=None, *, precision="float32", rewrites="auto", species=None)``.

    - ``model``: an ``fm.load`` result (or what ``fm.load`` accepts);
    - ``atoms``: optional; its species are the default superset (§7.1 passes it). Without it the
      superset is the species of the first structure the force field is used with;
    - ``precision``: the force precision, ``"float32"`` or ``"float64"`` (§7.3);
    - ``rewrites``: ``"auto"`` or a ``RewriteConfig``; resolved per plan;
    - ``species``: the declared superset (§8.5); a species without an embedding raises
      ``ModelElementError``. In 1b it must equal the structure's species (owner decision D3).
    """

    def __init__(self, model: Any, atoms: Any = None, *, precision: Literal["float32", "float64"] = "float32",
                 rewrites: Any = "auto", species: Any = None):
        raise NotImplementedError(_TASK)

    def evaluate(self, atoms: Any, stress: bool = False) -> tuple[float, Any, Any]:
        """``(energy, forces, stress)`` of ``atoms`` as host copies, through a one-off minimal plan;
        ``stress`` is ``None`` unless asked for. The energy is accumulated in float64 (§7.3)."""
        raise NotImplementedError(_TASK)
