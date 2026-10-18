"""The Plan record and the Planner protocol (SPEC §9.1, §9.5).

The record and protocol are plan task 1a-T02's; the fixed-plan planner (``fixed.py``, 1a-T15)
reproduces what the research engine's ``main.py`` runs today. The planner is the only code
that builds force fields, integrators and engines from a plan (SPEC §5).

The decision rules and the fallback algorithm of §9.3 and §9.4 also belong here; they come
after 1a.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastmace.config import (
        EngineConfig,
        IntegratorConfig,
        NeighbourConfig,
        PolarConfig,
        PrecisionConfig,
        RewriteConfig,
    )
    from fastmace.engine.base import Engine


@dataclass(frozen=True)
class Hardware:
    """What the planner knows about the GPU: name, compute capability, memory (bytes)."""

    name: str
    compute_capability: tuple[int, int]
    memory_bytes: int


@dataclass(frozen=True)
class Plan:
    """A frozen execution plan (SPEC §9.5).

    - ``tier``: ``"graph"`` in 1a (later ``"compiled"``, ``"eager"``, ``"reference"``);
    - the resolved config objects (no ``auto`` left): ``rewrites``, ``engine``, ``precision``,
      ``neighbours``, ``polar`` (``None`` for non-polar models), ``integrator``;
    - ``decisions``: ``(item, choice, reason)`` for every decision, enabled and disabled
      rewrites included;
    - ``requested``: ``(item, value)`` of every option the caller set to something other than
      ``"auto"`` (an explicit ``"on"`` is refused, never downgraded, when it cannot be honoured);
    - ``auto_options_off``: ``(option, reason)`` for every option that ``auto`` switched off because
      the model does not support it (the research engine's ``model.auto_options_off``);
    - ``fallbacks``: the fallback candidates, in order (empty in 1a);
    - ``predicted_memory_bytes``: ``None`` until the memory model exists;
    - ``revision``, ``predecessor``, ``reason``: a plan never changes in place; a runtime switch
      or fallback makes a new revision that points at the one before.
    """

    tier: str
    rewrites: RewriteConfig | None = None
    engine: EngineConfig | None = None
    precision: PrecisionConfig | None = None
    neighbours: NeighbourConfig | None = None
    polar: PolarConfig | None = None
    integrator: IntegratorConfig | None = None
    decisions: tuple[tuple[str, str, str], ...] = ()
    requested: tuple[tuple[str, str], ...] = ()
    auto_options_off: tuple[tuple[str, str], ...] = ()
    fallbacks: tuple[Plan, ...] = ()
    predicted_memory_bytes: int | None = None
    revision: int = 0
    predecessor: Plan | None = None
    reason: str = "initial plan"

    def describe(self) -> str:
        """The human-readable plan (``str(plan)``), written into provenance (SPEC §13)."""
        lines = [f"Plan revision {self.revision}, tier {self.tier}: {self.reason}"]
        lines += [f"  {item} = {choice}  ({reason})" for item, choice, reason in self.decisions]
        if self.requested:
            lines.append("  requested: " + ", ".join(f"{item}={value}" for item, value in self.requested))
        if self.predecessor is not None:
            lines.append(f"  predecessor: revision {self.predecessor.revision}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


@runtime_checkable
class Planner(Protocol):
    """Turns a workload into a Plan and a Plan into a running Engine."""

    def plan(self, traits: Any, atoms: Any, integrator: IntegratorConfig, hardware: Hardware) -> Plan:
        """Decide the plan for model ``traits`` (``fastmace.model.capabilities``), an ASE ``Atoms``,
        the integrator settings and the hardware. Refuses unsupported inputs (non-3D-PBC,
        constraints, inexact padding for the graph tier) with reasons, before any compilation."""
        raise NotImplementedError

    def build(self, plan: Plan, model: Any, atoms: Any) -> Engine:
        """Build the ForceField, SimulationState, Integrator and Engine that ``plan`` describes.

        The only place in the library that instantiates an engine.
        """
        raise NotImplementedError
