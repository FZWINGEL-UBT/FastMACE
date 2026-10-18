"""Simulation: one structure, one force field, one integrator, run in segments (SPEC §7.2, §7.6).

Plan task 1b-T01 fixes the signature; plan task 1b-T17 implements it. SPEC §5 lists no module for
``Simulation``; it lives here, above ``planner`` and ``engine``, and gets its engine from the
minimal planner (1b-T12), never by building one (SPEC §5).

What the constructor will do, in this order (``docs/design/contracts-1b.md``):

1. with a model instead of a ``ForceField``, build ``ForceField(model, atoms, precision="float32",
   species=species)`` (§7.6 "Simulation inputs");
2. ``workload=None`` becomes ``Workload(max_temperature_K=integrator.temperature_K)`` (owner
   decision D7), recorded in the plan;
3. plan through the minimal planner; ``PlanRefused`` here carries every reason;
4. with ``check="gate"`` or ``"allow_unavailable"``: the gate's references (§12.3, 1b-T16),
   before any production engine exists in the process;
5. build the engine through the planner;
6. the gate's production half through that engine (captured replay, registry snapshot and
   restore), then ``GateFailed`` or ``GateUnavailable`` if it says so; the record is ``sim.gate``.
   Without a requested gate its status is ``skipped`` (owner decision D1).

So a constructed ``Simulation`` is ready to step, and every refusal comes before the first step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastmace.io.output import Output
    from fastmace.planner.workload import Workload

Check = Literal["skip", "gate", "allow_unavailable"]
# check="skip": the gate does not run, status "skipped" (the default, owner decision D1).
# check="gate": run the gate before the first step; failed -> GateFailed, unavailable -> GateUnavailable.
# check="allow_unavailable": as "gate", but an unavailable gate is recorded and the run goes on.
CHECK_MODES = ("skip", "gate", "allow_unavailable")

_TASK = "fastmace.Simulation is a stub until plan task 1b-T17"


def check_mode(check: Any) -> str:
    """``check`` if it is one of ``CHECK_MODES``; else ``ValueError`` naming them."""
    if check not in CHECK_MODES:
        raise ValueError(f"check={check!r} is not one of {CHECK_MODES}; the gate is opt-in: check='gate' (SPEC §12.3)")
    return check


class Simulation:
    """MD on the graph engine with a plan, an optional gate, schedules and complete restarts.

    ``Simulation(atoms, force_field, *, species=None, integrator, workload=None, output=None, seed,
    check="skip", allow_naive_cueq=False)``:

    - ``atoms``: ASE ``Atoms``, full 3D PBC (§4.4);
    - ``force_field``: a ``fm.ForceField`` or a model (``fm.load`` result, label or path); a model
      is wrapped in a float32 ``ForceField`` with ``species=`` (default: the structure's species);
      ``species`` with a ``ForceField`` is refused (the ``ForceField`` owns the superset);
    - ``integrator``: ``fm.integrators.Langevin`` or ``fm.integrators.LangevinBAOAB``;
    - ``workload``: ``fm.Workload``; default ``Workload(max_temperature_K=integrator.temperature_K)``;
    - ``output``: ``fm.Output``; ``None`` writes nothing but what the caller asks for;
    - ``seed``: the integrator's noise seed, recorded;
    - ``check``: one of ``CHECK_MODES``;
    - ``allow_naive_cueq``: accept cuEquivariance's naive kernels (§9.2), recorded.
    """

    def __init__(self, atoms: Any, force_field: Any, *, species: Any = None, integrator: Any,
                 workload: Workload | None = None, output: Output | None = None, seed: int,
                 check: Check = "skip", allow_naive_cueq: bool = False):
        check_mode(check)
        raise NotImplementedError(_TASK)

    @property
    def plan(self) -> Any:
        """The active ``Plan`` revision (§9.5)."""
        raise NotImplementedError(_TASK)

    @property
    def gate(self) -> Any:
        """The gate record (§12.3): status ``passed``, ``failed``, ``unavailable`` or ``skipped``."""
        raise NotImplementedError(_TASK)

    def run(self, steps: int | None = None, schedule: Any = None, mc: Any = ...) -> None:
        """Advance ``steps`` steps, or along ``schedule`` to its end (§7.2 "Schedules and step counts").

        ``schedule=None`` continues the attached schedule, or keeps the target without one.
        ``mc=...`` (the default) keeps the attached move set, ``mc=None`` detaches it, a move set
        attaches it (§7.6). Every check runs before the first step: a schedule above
        ``Workload.max_temperature_K`` raises ``PlanRefused``, ``steps`` beyond the schedule
        ``ValueError``, and in 1b any ``mc`` is checked against the workload, then refused ("MC:
        phase 3").
        """
        raise NotImplementedError(_TASK)

    def save_restart(self, path: Any) -> None:
        """Write the complete §10.6 state at the current committed step (a canonical point)."""
        raise NotImplementedError(_TASK)

    @classmethod
    def resume(cls, directory: Any, *, allow_plan_change: bool = False, check: Check = "skip") -> Simulation:
        """Continue from the restart in ``directory``. A different force-field identity raises
        ``RestartMismatch``; a different plan needs ``allow_plan_change=True`` and makes a new
        revision (§9.5, §10.6). ``check`` as in the constructor."""
        check_mode(check)
        raise NotImplementedError(_TASK)

    def set_positions(self, positions: Any) -> None:
        """Host write of positions at a canonical point: quiesce, write, rebuild, re-evaluate (owner
        decision D4)."""
        raise NotImplementedError(_TASK)

    def set_momenta(self, momenta: Any) -> None:
        """Host write of momenta at a canonical point (owner decision D4)."""
        raise NotImplementedError(_TASK)
