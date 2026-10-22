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

The registry lives on the engine (``GraphEngine.derived``). The force field and the integrator
register their buffers through ``register_derived(registry, state)``, and every derived buffer is in
the engine's runtime-state registry too (``derived.<name>``, or its owner's name). Refresh functions
take no arguments, read the committed inputs from the state (``state.species``, ``state.masses``,
``state.charges``, ``state.total_charge``) and write in place at fixed addresses. Those that depend on
species, masses or charges are capturable (no host sync, no new buffer), so that 3-T09 can replay them
inside an attempt graph; friction refreshes may copy host floats, because a friction event is a host
event that drops the graphs anyway.

``Transaction`` implements the host kinds (species, masses, charges, friction); the MC kinds are
3-T09's and 3-T10's. An accepted host event ends at a canonical point
(``GraphEngine.canonicalise(reevaluate=True)``: a rebuilt neighbour list, one force evaluation,
``integrator.start``), so a fresh construction from the committed state is a plain one. A rejected
event restores the snapshot bit for bit. ``check_against(engine, fresh)`` is the CI registry check of
SPEC §12.2. This module imports neither torch nor the engine at import time.
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
    construction, so a forgotten buffer fails (``check_against``)."""

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
        """Refresh every buffer affected by ``changed``, in registration order (an owner registers a buffer
        after the buffers it reads); returns their names."""
        names = self.affected(changed)
        for name in names:
            self._entries[name].refresh()
        return names

    def check_against(self, fresh: Any) -> list[str]:
        """The names whose buffers differ, bit for bit, from a fresh construction's, or that only one side
        registers. ``fresh`` is a ``DerivedStateRegistry`` or an engine (its ``derived``)."""
        other = fresh if isinstance(fresh, DerivedStateRegistry) else fresh.derived
        names = [*self._entries, *(name for name in other._entries if name not in self._entries)]
        return [name for name in names if name not in self._entries or name not in other._entries
                or not _same(self._entries[name].buffer, other._entries[name].buffer)]


def _bits(tensor):
    """``tensor`` on the host, floating point reinterpreted as integers of its width (bitwise comparison)."""
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        integer = {8: torch.int64, 4: torch.int32, 2: torch.int16}.get(tensor.element_size())
        if integer is not None:
            return tensor.view(integer)
    return tensor


def _same(a: Any, b: Any) -> bool:
    """Bit for bit for tensors (``-0.0`` differs from ``0.0``; a NaN equals the same NaN), ``==`` otherwise."""
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a is b or bool(a == b)


# --------------------------------------------------------------------------- the CI registry check (SPEC §12.2)


def _host_constants(obj) -> dict[str, Any]:
    """The plain host values an object holds (numbers, strings, bools, config records): what a stage bakes in."""
    import dataclasses

    return {name: value for name, value in vars(obj).items()
            if isinstance(value, (bool, int, float, str))
            or (dataclasses.is_dataclass(value) and not isinstance(value, type))}


def _edges(neighbours) -> set | None:
    """The compacted edges of a skin list as a set of (sender, receiver, shift bits); ``None`` without one."""
    if neighbours is None or not hasattr(neighbours, "edges") or getattr(neighbours, "count", None) is None:
        return None
    index, shifts, _ = neighbours.edges(int(neighbours.count))
    index = index.detach().cpu()
    return set(zip(index[0].tolist(), index[1].tolist(), map(tuple, _bits(shifts).tolist()), strict=True))


def _neighbour_list(field):
    """The force field's skin list: ``field.neighbours``, else the one attribute with ``edges`` and ``count``."""
    found = getattr(field, "neighbours", None)
    if found is not None:
        return found
    lists = [value for value in vars(field).values() if hasattr(value, "edges") and hasattr(value, "count")]
    return lists[0] if len(lists) == 1 else None


def _storage(tensor):
    try:
        return tensor.device, tensor.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return None


def check_against(engine: Any, fresh: Any) -> list[str]:
    """The CI registry check (SPEC §8.5, §12.2): what in ``engine`` differs, bit for bit, from ``fresh``, a
    construction from the committed state; empty when every derived buffer was refreshed.

    Compares:
    - the derived registries (``DerivedStateRegistry.check_against``);
    - the composition in the state (species, masses, charges, total charge, spin);
    - the neighbour list as the set of compacted edges, not as candidates or reference positions;
    - the integrator's host constants (the floats a stage bakes in, its config);
    - every other tensor reachable from the force field and the integrator that the runtime-state
      registry does not hold, so that a derived buffer left out of the registry fails here.
    """
    from fastmace.engine.registry import reachable_tensors

    out = list(engine.derived.check_against(fresh.derived))
    for name in ("species", "masses", "charges", "total_charge", "spin_multiplicity"):
        if not _same(getattr(engine.state, name), getattr(fresh.state, name)):
            out.append(f"state.{name}")
    lists = [_neighbour_list(one.force_field) for one in (engine, fresh)]
    if _edges(lists[0]) != _edges(lists[1]):
        out.append("force_field.neighbours (compacted edges)")
    constants, expected = _host_constants(engine.integrator), _host_constants(fresh.integrator)
    for name in sorted(set(constants) | set(expected)):
        if name not in constants or name not in expected or not _same(constants[name], expected[name]):
            out.append(f"integrator.{name}")
    covered = {_storage(tensor) for tensor in engine.registry.tensors().values()}
    # The neighbour list is compared as its compacted edges above; its candidates, reference positions and
    # build workspaces (the GPU pair search's) depend on the build history, not on the committed state.
    inside = {id(tensor) for one in lists if one is not None for _, tensor in reachable_tensors(one).values()}
    for root in ("force_field", "integrator"):
        got = dict(reachable_tensors(getattr(engine, root), root).values())
        want = dict(reachable_tensors(getattr(fresh, root), root).values())
        for path in sorted(set(got) | set(want)):
            if path in got and _storage(got[path]) in covered:
                continue
            if id(got.get(path)) in inside or id(want.get(path)) in inside:
                continue
            if path not in got or path not in want or not _same(got[path], want[path]):
                out.append(path)
    return out


# --------------------------------------------------------------------------- transactions

# Kinds that other tasks apply through the same lifecycle.
_LATER = {"swap": "plan task 3-T09", "transmute": "plan task 3-T09", "displace": "plan task 3-T10",
          "barostat": "barostat events (τ, mass) have no host API in phase 3; a friction event moves the default "
                      "barostat τ with it"}


class Transaction:
    """``Transaction(engine, events, *, evaluate=None, registry=None)``: run ``events`` on ``engine``
    as one transaction (``LIFECYCLE``). ``evaluate(engine)`` returns the decision (``OUTCOMES``);
    ``None`` accepts (host mutations).

    Every event is validated when the transaction is made, before anything is written: a wrong shape, a
    non-finite or non-positive mass, a friction that is not a positive number (inverse ASE time units) or an
    atomic number outside 1–118 raise ``ValueError``; a species outside the force field's superset raises
    its ``SpeciesNotInSuperset`` (``force_field.check_species``, 3-T02). ``host_writes`` holds callables
    ``write(state)`` that ``apply`` runs after the events, under ``torch.no_grad()``: ``Simulation.update``'s
    positions and momenta, so that the whole update is one transaction with one force evaluation.

    Event values: ``masses`` amu and ``species`` atomic numbers (or symbols), one per atom of ``atoms``;
    ``charges`` per-atom charges, or ``{"charges": ..., "total_charge": ...}``; ``friction`` a number.
    A mass change keeps the momenta (ASE): v' = v m_old / m_new, for the atoms whose mass changes. A
    species change sets the new species' standard masses the same way; a masses event after it wins.

    ``run()`` returns the outcome:
    - ``"accepted"``: the change stays, ``engine.transactions_committed`` counts it, and the resume ends
      at a canonical point;
    - ``"rejected"`` and ``"impossible"``: the snapshot is restored bit for bit (forces restored, never
      recomputed);
    - ``"redo"`` (an overflow): restored as well; the caller runs the attempt again.
    """

    def __init__(self, engine: Any, events: Sequence[Event], *, evaluate: Callable[[Any], str] | None = None,
                 registry: DerivedStateRegistry | None = None):
        events = tuple(events)
        for event in events:
            if not isinstance(event, Event):
                raise TypeError(f"a transaction takes fastmace.events.Event records, got {type(event).__name__}")
            if event.kind in _LATER:
                raise NotImplementedError(f"{event.kind} events: {_LATER[event.kind]}")
        self.engine = engine
        self.events = events
        self.evaluate = evaluate
        self.registry = engine.derived if registry is None else registry
        self.host_writes: list[Callable[[Any], Any]] = []
        self.outcome: str | None = None
        self.changed: set[str] = set()
        self._values = [self._validated(event) for event in events]
        self._snapshot = None
        self._dropped_graphs = False

    # ------------------------------------------------------------------ validation

    def _count(self, event) -> int:
        natoms = self.engine.state.natoms
        if event.atoms is None:
            return natoms
        if any(index >= natoms for index in event.atoms):
            raise ValueError(f"{event.kind} event: atom indices must be below {natoms}, got {event.atoms!r}")
        return len(event.atoms)

    @staticmethod
    def _array(values, name, count):
        import numpy as np

        array = np.asarray(values, dtype=np.float64)
        if array.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def _validated(self, event):
        import numpy as np

        count, state = self._count(event), self.engine.state
        if event.kind == "masses":
            masses = self._array(event.values, "masses", count)
            if (masses <= 0).any():
                raise ValueError("masses must be positive (amu)")
            return masses
        if event.kind == "friction":
            friction = event.values
            if isinstance(friction, bool) or not isinstance(friction, (int, float, np.integer, np.floating)):
                raise TypeError(f"the friction must be a number (inverse ASE time units), got {friction!r}")
            friction = float(friction)
            if not (friction > 0 and np.isfinite(friction)):
                raise ValueError(f"the friction must be positive and finite, got {friction!r}")
            return friction
        if event.kind == "species":
            numbers = species_numbers(event.values, count)
            check = getattr(self.engine.force_field, "check_species", None)
            if check is None:
                raise NotImplementedError(
                    f"{type(self.engine.force_field).__name__} has no superset check (check_species, plan task "
                    "3-T02): a species event would leave its species tables as they are")
            full = state.species.detach().cpu().numpy().copy()
            full[slice(None) if event.atoms is None else list(event.atoms)] = numbers
            check(full)
            return numbers
        if event.kind == "charges":
            values = event.values if isinstance(event.values, dict) else {"charges": event.values}
            unknown = set(values) - {"charges", "total_charge"}
            if unknown:
                raise ValueError(f"a charges event takes charges and total_charge, got {sorted(unknown)}")
            charges, total = values.get("charges"), values.get("total_charge")
            if charges is not None:
                if state.charges is None:
                    raise ValueError("this model takes no per-atom charges")
                charges = self._array(charges, "charges", count)
            if total is not None:
                field = self.engine.force_field
                buffers = getattr(field, "derived_buffers", dict)()
                if getattr(field, "set_total_charge", None) is None or "total_charge" not in buffers:
                    raise ValueError("this model takes no total charge")
                total = float(total)
                if not np.isfinite(total):
                    raise ValueError("the total charge must be finite")
            if (charges is not None or total is not None) and not self.registry.affected(("charges",)):
                raise NotImplementedError("no derived buffer reads the charges (the force field registers its charge "
                                          "inputs: plan task 3-T02)")
            return charges, total
        raise ValueError(f"unknown event kind {event.kind!r}")

    # ------------------------------------------------------------------ the lifecycle

    def run(self) -> str:
        """quiesce, snapshot, apply, refresh, evaluate, decide, resume; returns the outcome."""
        self.quiesce()
        self.snapshot()
        try:
            self.apply()
            self.refresh()
            outcome = "accepted" if self.evaluate is None else self.evaluate(self.engine)
            if outcome not in OUTCOMES:
                raise ValueError(f"evaluate returned {outcome!r}, not one of the outcomes {OUTCOMES}")
        except BaseException:
            self.reject()
            self.outcome = None
            raise
        if outcome == "accepted":
            self.accept()
        else:
            self.reject()
            self.outcome = outcome
        self.resume()
        return self.outcome

    def quiesce(self) -> None:
        """Finish the pipelined and speculative work, so that the state is a committed step."""
        self.engine.quiesce()

    def snapshot(self) -> None:
        """Every captured buffer of the runtime-state registry (state, integrator, neighbour list, flags,
        derived buffers), the noise stream's generator and cursor, and the host values an event changes."""
        import copy

        engine, state = self.engine, self.engine.state
        self._snapshot = {
            "buffers": engine.registry.snapshot(),
            "rng": (copy.deepcopy(state.rng.generator), state.rng.cursor),
            "state": {name: getattr(state, name) for name in ("total_charge", "spin_multiplicity")},
            "integrator": dict(vars(engine.integrator)),
        }

    def apply(self) -> None:
        """Write every event in order, then the host writes; ``changed`` collects what changed."""
        import torch

        with torch.no_grad():
            for event, values in zip(self.events, self._values, strict=True):
                getattr(self, f"_apply_{event.kind}")(event, values)
            for write in self.host_writes:
                write(self.engine.state)

    def _rows(self, event):
        import torch

        if event.atoms is None:
            return slice(None)
        return torch.tensor(event.atoms, dtype=torch.int64, device=self.engine.state.positions.device)

    def _set_masses(self, rows, masses) -> None:
        """New masses with the momenta kept (ASE): v' = v m_old / m_new, only where the mass changes."""
        import torch

        state = self.engine.state
        new = torch.as_tensor(masses, dtype=state.masses.dtype, device=state.masses.device)[:, None]
        old = state.masses[rows]
        moved = new != old
        if not bool(moved.any()):  # host sync: a host event
            return
        velocities = state.velocities[rows]
        state.velocities[rows] = torch.where(moved, velocities * old / new, velocities)
        state.masses[rows] = new
        self.changed |= {"masses", "momenta"}

    def _apply_masses(self, event, masses) -> None:
        self._set_masses(self._rows(event), masses)

    def _apply_species(self, event, numbers) -> None:
        import torch
        from ase.data import atomic_masses

        state, rows = self.engine.state, self._rows(event)
        state.species[rows] = torch.as_tensor(numbers, dtype=state.species.dtype, device=state.species.device)
        self.changed.add("species")
        # The new species' standard masses (ASE's), unless a masses event follows (contracts-3, decision 9).
        self._set_masses(rows, atomic_masses[numbers])

    def _apply_charges(self, event, values) -> None:
        import torch

        charges, total = values
        state = self.engine.state
        if charges is not None:
            state.charges[self._rows(event)] = torch.as_tensor(charges, dtype=state.charges.dtype,
                                                               device=state.charges.device)
        if total is not None:
            # The force field's device total is base + (sum of charges - their sum at the base), so that 3-T09's
            # transmutations change it inside a graph through state.charges; a host total moves the base (3-T02).
            field = self.engine.force_field
            previous = field.derived_buffers()["total_charge"]
            self._snapshot.setdefault("total_charge", float(previous.reshape(-1)[0]))  # host sync: a host event
            field.set_total_charge(total)
            state.total_charge = total
        self.changed.add("charges")

    def _apply_friction(self, event, friction) -> None:
        integrator = self.engine.integrator
        integrator.set_friction(friction)
        self.changed.add("friction")
        if getattr(integrator.config, "ensemble", None) == "npt" and integrator.config.barostat_tau_fs is None:
            self.changed.add("barostat")  # P_tau and the barostat mass follow the friction
        self._drop_graphs("friction event: c1, c2 and the thermostat constants are baked into the captured stages")

    def _drop_graphs(self, reason: str) -> None:
        engine = self.engine
        invalidate = getattr(engine, "invalidate_graphs", None)  # 3-T04: logged; the next capture is a re-capture
        if invalidate is not None:
            invalidate(reason)
        else:
            engine._clear_graphs()
        self._dropped_graphs = True

    def refresh(self) -> None:
        """Every derived buffer that depends on what changed."""
        self.registry.refresh(self.changed)

    def accept(self) -> None:
        """Keep the new state; the engine counts the committed transaction (its ``state_version``)."""
        if self.changed or self.host_writes:
            self.engine.transactions_committed += 1
        self.outcome = "accepted"

    def reject(self) -> None:
        """Restore the snapshot exactly: buffers in place, the noise stream, the host values."""
        import torch

        if self._snapshot is None:
            return
        engine, state = self.engine, self.engine.state
        if "total_charge" in self._snapshot:  # the force field's host base, before its buffers come back
            engine.force_field.set_total_charge(self._snapshot["total_charge"])
        engine.registry.restore(self._snapshot["buffers"])
        state.rng.generator, state.rng.cursor = self._snapshot["rng"]
        for name, value in self._snapshot["state"].items():
            setattr(state, name, value)
        integrator = engine.integrator
        for name, value in self._snapshot["integrator"].items():
            if not isinstance(value, torch.Tensor):
                setattr(integrator, name, value)
        if self._dropped_graphs and engine._graphs:  # captured during the evaluation, with the trial constants
            self._drop_graphs("rejected friction event")
        self.outcome = "rejected"

    def resume(self) -> None:
        """After an acceptance, in ``RESUME_ORDER``: species, masses and momenta are written; the canonical
        point evaluates forces and stress once at a rebuilt neighbour list and calls ``integrator.start``;
        then the integrator caches derived from forces and stress are refreshed. Nothing after a rejection."""
        if self.outcome != "accepted" or not (self.changed or self.host_writes):
            return
        self.engine.canonicalise(reevaluate=True)
        self.registry.refresh(("forces", "stress"))


def species_numbers(values: Any, count: int):
    """Atomic numbers (int64, shape ``(count,)``) from atomic numbers or chemical symbols; ``ValueError`` for
    a wrong shape, an unknown symbol or a number outside 1–118."""
    import numpy as np
    from ase.data import atomic_numbers, chemical_symbols

    items = np.asarray(values, dtype=object)
    if items.shape != (count,):
        raise ValueError(f"species must have shape ({count},), got {items.shape}")
    numbers = []
    for item in items.tolist():
        if isinstance(item, str):
            if item not in atomic_numbers or atomic_numbers[item] == 0:
                raise ValueError(f"species: unknown chemical symbol {item!r}")
            numbers.append(atomic_numbers[item])
        elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"species must be atomic numbers or chemical symbols, got {item!r}")
        else:
            numbers.append(int(item))
    if any(not 1 <= number < len(chemical_symbols) for number in numbers):
        raise ValueError(f"species: atomic numbers must be in 1..{len(chemical_symbols) - 1}, got {numbers}")
    return np.asarray(numbers, dtype=np.int64)


__all__ = ["DEPENDENCIES", "EVENT_KINDS", "LIFECYCLE", "OUTCOMES", "RESUME_ORDER", "DerivedBuffer",
           "DerivedStateRegistry", "Event", "Transaction", "check_against", "species_numbers"]
