"""In-place state changes as transactions, and the derived-state registry (SPEC §8.5, §8.1 "Graph-safe events").

Contracts: plan task 3-T01; filled by 3-T03 (owner). Every state change outside an integrator
step is an ``Event`` run by a ``Transaction`` through the lifecycle ``LIFECYCLE``:

1. quiesce the pipeline, asynchronous output and speculative steps (a committed step);
2. snapshot the affected state, the derived buffers it feeds and the random-number cursors;
3. apply the mutation;
4. refresh every registered derived buffer that depends on it;
5. evaluate what the decision needs (a hook: 3-T07's evaluator, 3-T09's and 3-T10's moves), with
   the overflow check of §8.1 before any decision;
6. decide: accept, or restore the snapshot exactly (restored forces are restored, never recomputed);
   an overflow is no outcome but a redo with the same random numbers;
7. resume in ``RESUME_ORDER``: species, masses, charges and momenta; then forces and stress; then the
   integrator caches derived from them (BAOAB's ``force_eps``).

Schedule-driven targets (temperature, pressure) are device-scalar fills, never events. Events never
change a shape or an address inside a captured graph; an event that changes a constant baked into a
graph (a friction change: Langevin's ``c1``/``c2``) drops the affected graphs, a recorded re-capture.

``DerivedStateRegistry.register`` is complete here, so 3-T02 can register its buffers before 3-T03's
refresh lands; ``refresh``, ``check_against`` and ``Transaction`` are 3-T03's. This module imports
neither torch nor the engine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# The in-place changes of §8.5: host mutations (species, masses, charges, friction, barostat τ or mass)
# and MC moves (fastmace.mc).
EVENT_KINDS = ("species", "masses", "charges", "friction", "barostat", "swap", "transmute", "displace")
LIFECYCLE = ("quiesce", "snapshot", "apply", "refresh", "evaluate", "decide", "resume")
# "redo": a capacity overflow during evaluation; the snapshot is restored, the capacity grows and the
# same attempt runs again with the same random numbers (§8.5 step 4).
OUTCOMES = ("accepted", "rejected", "impossible", "redo")
# What a derived buffer may depend on. The first five are changed by events, "positions" and "cell" by
# displacements (and steps), the last three are the resume's.
DEPENDENCIES = ("species", "masses", "charges", "friction", "barostat", "positions", "cell", "momenta", "forces",
                "stress")
RESUME_ORDER = (("species", "masses", "charges", "momenta"), ("forces", "stress"), ("integrator caches",))


@dataclass(frozen=True)
class Event:
    """One in-place change: ``kind`` (``EVENT_KINDS``), the touched ``atoms`` (indices, distinct; ``None``
    for every atom or none, as for a friction change) and the new ``values`` (as the kind needs)."""

    kind: str
    atoms: Any = None
    values: Any = None

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Event.kind must be one of {EVENT_KINDS}, got {self.kind!r}")
        if self.atoms is not None:
            atoms = tuple(self.atoms)
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in atoms):
                raise ValueError(f"Event.atoms must be nonnegative atom indices, got {self.atoms!r}")
            if len(set(atoms)) != len(atoms):
                raise ValueError(f"Event.atoms must be distinct, got {self.atoms!r}")
            object.__setattr__(self, "atoms", atoms)


@dataclass(frozen=True)
class DerivedBuffer:
    """A registered derived buffer: its ``refresh()`` recomputes ``buffer`` in place from what it
    ``depends_on`` (``DEPENDENCIES``)."""

    name: str
    buffer: Any
    refresh: Callable[[], Any]
    depends_on: tuple[str, ...]


class DerivedStateRegistry:
    """Every buffer derived from species, masses, charges or other event-changed state (§8.5): one
    entry per name, in registration order. A CI test compares the whole registry with a fresh
    construction, so a forgotten buffer fails (3-T03)."""

    def __init__(self):
        self._entries: dict[str, DerivedBuffer] = {}

    def register(self, name: str, buffer: Any, refresh: Callable[[], Any], depends_on: Iterable[str]) -> DerivedBuffer:
        """Register ``buffer`` under ``name``, refreshed by ``refresh()`` when any of ``depends_on`` changes."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"a derived buffer needs a name, got {name!r}")
        if name in self._entries:
            raise ValueError(f"derived buffer {name!r} is registered already")
        if not callable(refresh):
            raise TypeError(f"derived buffer {name!r}: refresh must be callable, got {refresh!r}")
        depends = tuple(depends_on)
        unknown = [item for item in depends if item not in DEPENDENCIES]
        if not depends or unknown:
            raise ValueError(f"derived buffer {name!r}: depends_on must name some of {DEPENDENCIES}, got {depends!r}")
        entry = DerivedBuffer(name, buffer, refresh, depends)
        self._entries[name] = entry
        return entry

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def affected(self, changed: Iterable[str]) -> tuple[str, ...]:
        """The names of the buffers that depend on any of ``changed``, in registration order."""
        changed = set(changed)
        return tuple(name for name, entry in self._entries.items() if changed & set(entry.depends_on))

    def __getitem__(self, name: str) -> DerivedBuffer:
        return self._entries[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def refresh(self, changed: Iterable[str]) -> tuple[str, ...]:
        """Refresh every buffer affected by ``changed``, dependencies first; returns their names."""
        raise NotImplementedError("DerivedStateRegistry.refresh: plan task 3-T03")

    def check_against(self, fresh: Any) -> list[str]:
        """The names whose buffers differ from a fresh construction's (the CI registry check)."""
        raise NotImplementedError("DerivedStateRegistry.check_against: plan task 3-T03")


class Transaction:
    """``Transaction(engine, events, *, evaluate=None, registry=None)``: run ``events`` on ``engine``
    as one transaction (``LIFECYCLE``). ``evaluate(engine)`` returns the decision (``OUTCOMES``);
    ``None`` accepts (host mutations)."""

    def __init__(self, engine: Any, events: Sequence[Event], *, evaluate: Callable[[Any], str] | None = None,
                 registry: DerivedStateRegistry | None = None):
        raise NotImplementedError("events.Transaction: plan task 3-T03")

    def run(self) -> str:
        raise NotImplementedError("Transaction.run: plan task 3-T03")

    def quiesce(self) -> None:
        raise NotImplementedError("Transaction.quiesce: plan task 3-T03")

    def snapshot(self) -> None:
        raise NotImplementedError("Transaction.snapshot: plan task 3-T03")

    def apply(self) -> None:
        raise NotImplementedError("Transaction.apply: plan task 3-T03")

    def refresh(self) -> None:
        raise NotImplementedError("Transaction.refresh: plan task 3-T03")

    def accept(self) -> None:
        raise NotImplementedError("Transaction.accept: plan task 3-T03")

    def reject(self) -> None:
        raise NotImplementedError("Transaction.reject: plan task 3-T03")

    def resume(self) -> None:
        raise NotImplementedError("Transaction.resume: plan task 3-T03")
