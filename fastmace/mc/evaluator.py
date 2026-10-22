"""The MC evaluator: energies of Monte Carlo states (plan task 3-T07; SPEC §10.4 "Evaluation", "Baseline
energy", "Energy precision"; §8.1 "Energy-only graph"; §7.3).

- **One evaluator per graph and precision.** ``MCEvaluator(engine, precision)`` evaluates the engine's
  committed state with a forward pass only (``MACEForceField.energy_only``: ``torch.no_grad()``, no
  forces, float64 accumulation). In ``"float32"`` (or the force precision) it is the engine's own
  force field; in ``"float64"`` under a float32 force precision it is a float64 copy of the same plan
  that follows the engine's force field (``float64_copy``), while MD forces keep their precision.
  ``"auto"`` is resolved by the planner's move-energy gate (3-T11), never here.
- **Baseline energy.** ``baseline()`` always evaluates the current state with this evaluator; it is
  never taken from the MD energy, which comes from the force graph and possibly another precision.
- **Energy cache.** One entry, tagged ``(evaluator, precision, state version)``. An accepted trial's
  energy becomes the cached baseline (``accept``); a rejection keeps it (``reject``).

The module imports neither torch nor MACE at import time (``fm.mc`` stays light).
"""

from __future__ import annotations

import itertools
from typing import Any

PRECISIONS = ("float32", "float64")
_IDS = itertools.count()


def precision_of(dtype) -> str:
    """``"float32"`` or ``"float64"`` of a torch dtype."""
    return str(dtype).removeprefix("torch.")


def float64_copy(force_field, model, *, neighbours=None, polar=None, check=True):
    """A float64 ``MACEForceField`` of ``model`` that follows the float32 ``force_field`` (SPEC §10.4).

    ``model`` is the same checkpoint in float64 (a ``LoadedModel`` or calculator; the planner loads it
    with the plan's backend if that backend runs and passes the gate in float64, else cuEq, 3-T07
    conductor answer 1). The copy is eager (no ``torch.compile``: no second compiled backend in one
    process, §9.4 step 3) and may be captured. It takes the force field's rewrites, superset, charge,
    spin and exact E0 table, and reads its compacted edges, cell and species sources
    (``MACEForceField.follow``). ``neighbours`` and ``polar`` default to the force field's skin and
    order and to PolarMACE's default options.
    """
    import ase
    import numpy as np
    import torch

    from fastmace.config import EngineConfig, NeighbourConfig, PolarConfig
    from fastmace.model.forces import MACEForceField

    if force_field.dtype != torch.float32:
        raise ValueError(f"a float64 copy is for a float32 force field; this one is {precision_of(force_field.dtype)} "
                         "and evaluates float64 energies itself")
    leader = force_field.neighbours
    if neighbours is None:
        neighbours = NeighbourConfig(skin=leader.skin, sort_receivers="on" if leader.sort_receivers else "off",
                                     gpu_neighbour_list="off")
    atoms = ase.Atoms(numbers=force_field.atomic_numbers.detach().cpu().numpy(),
                      positions=force_field.positions.detach().to(torch.float64).cpu().numpy(),
                      cell=np.asarray(leader.cell, dtype=np.float64), pbc=True)
    atoms.set_initial_charges(force_field.charges.detach().cpu().numpy())
    copy = MACEForceField(
        model, atoms, rewrites=force_field.resolution, engine=EngineConfig(compile_mode="none"), neighbours=neighbours,
        polar=polar or PolarConfig(), compute_stress=force_field.compute_stress, charge=force_field.charge,
        spin_multiplicity=force_field.spin_multiplicity, check=check, float64_energy=force_field.float64_energy,
        atomic_energies=force_field.atomic_energies, species=force_field.superset)
    if copy.dtype != torch.float64:
        raise ValueError(f"the MC energy copy needs a float64 model; this one is {precision_of(copy.dtype)}")
    copy.follow(force_field)
    return copy


class EnergyCache:
    """The MC energy cache: one float64 device energy, tagged ``(evaluator, precision, state version)``.

    The stored tensor has a fixed address once allocated, so a captured attempt graph can read it.
    """

    def __init__(self):
        self.tag = None
        self.energy = None

    def get(self, tag):
        """The cached energy for ``tag``, or None when the cache holds another evaluator, precision or state."""
        return self.energy if self.tag is not None and self.tag == tag else None

    def store(self, tag, energy):
        """Cache a float64 ``energy`` [1] for ``tag`` (copied into the cache's own buffer)."""
        import torch

        if energy.dtype != torch.float64:
            raise ValueError(f"MC energies are float64 (SPEC §7.3), got {energy.dtype}")
        if self.energy is None or self.energy.shape != energy.shape or self.energy.device != energy.device:
            self.energy = torch.empty_like(energy)
        self.energy.copy_(energy.detach())
        self.tag = tag
        return self.energy

    def clear(self):
        self.tag = None


class MCEvaluator:
    """Energies of the engine's committed state for MC moves: one graph, one precision (SPEC §10.4).

    ``precision``: ``"float32"`` or ``"float64"``; None is the force precision. A float64 evaluator of
    a float32 engine needs ``model`` (the float64 checkpoint, see ``float64_copy``) or a ready ``copy``.
    MC energies are never evaluated below the force precision.
    """

    def __init__(self, engine, precision: str | None = None, *, model=None, copy=None, polar=None):
        force_field = engine.force_field
        self.force_precision = precision_of(force_field.dtype)
        precision = self.force_precision if precision is None else precision
        if precision not in PRECISIONS:
            raise ValueError(f"MC energy precision must be one of {PRECISIONS}, got {precision!r} (the planner "
                             "resolves 'auto' through the move-energy gate, plan task 3-T11)")
        if PRECISIONS.index(precision) < PRECISIONS.index(self.force_precision):
            raise ValueError(f"MC energies in {precision} below the {self.force_precision} force precision are not "
                             "offered: evaluate them in float64")
        self.engine, self.precision = engine, precision
        if precision == self.force_precision:
            if copy is not None or model is not None:
                raise ValueError(f"a {precision} evaluator of a {self.force_precision} engine uses the engine's own "
                                 "force field: pass no model or copy")
            self.field = force_field
        elif copy is not None:
            if precision_of(copy.dtype) != precision or copy.info().get("follows") is None:
                raise ValueError(f"the copy must be a {precision} force field that follows the engine's (float64_copy)")
            self.field = copy
        elif model is not None:
            self.field = float64_copy(force_field, model, polar=polar)
        else:
            raise ValueError(f"a {precision} evaluator of a {self.force_precision} engine needs the float64 model "
                             "(float64_copy)")
        self.name = f"mc-evaluator-{next(_IDS)}"
        self.cache = EnergyCache()
        self.evaluations = 0

    # ------------------------------------------------------------------ identity

    @property
    def own_field(self) -> bool:
        """True when the evaluator runs the engine's own force field (no copy)."""
        return self.field is self.engine.force_field

    def state_version(self):
        """The version of the engine's committed state: ``GraphEngine.state_version`` (3-T03), which every
        committed step and event advances; the committed step count until then."""
        version = getattr(self.engine, "state_version", None)
        return ("step", int(self.engine.state.step)) if version is None else version

    def tag(self):
        """The cache tag of the current state: ``(evaluator, precision, state version)``."""
        return (self.name, self.precision, self.state_version())

    # ------------------------------------------------------------------ evaluation

    def evaluate(self):
        """The float64 energy [1] of the engine's committed state on its current edges, forward only.

        A device tensor; no forces, no ``grad_fn``. Uncaptured here (any engine mode, CPU or CUDA).
        """
        engine = self.engine
        if not self.own_field:
            self.field.use_edges(int(engine.force_field.data["edge_index"].shape[1]))
        self.evaluations += 1
        return self.field.energy_only(engine.state.positions)

    def baseline(self):
        """Evaluate the current state with this evaluator and cache it as the block's baseline (never the MD
        energy; SPEC §10.4 "Baseline energy"). Returns the cached device energy."""
        return self.cache.store(self.tag(), self.evaluate())

    def current(self):
        """The cached baseline of the current state, or None if the state moved on since it was cached."""
        return self.cache.get(self.tag())

    def accept(self, energy):
        """An accepted trial: its energy becomes the cached baseline of the committed state."""
        return self.cache.store(self.tag(), energy)

    def reject(self):
        """A rejected trial: the state is restored and the cached baseline stays."""
        return self.current()

    # ------------------------------------------------------------------ provenance

    def memory_bytes(self) -> int:
        """Device bytes of the float64 copy's weights and inputs (0 for the engine's own force field)."""
        if self.own_field:
            return 0
        import torch

        tensors = list(self.field.model.parameters()) + list(self.field.model.buffers())
        tensors += [value for value in self.field.data.values() if torch.is_tensor(value)]
        tensors += [value for value in (self.field._shifts, self.field._shifts64, self.field._unit_shifts)
                    if value is not None]
        seen, total = set(), 0
        for tensor in tensors:
            key = (tensor.untyped_storage().data_ptr(), tensor.device)
            if key not in seen:
                seen.add(key)
                total += tensor.untyped_storage().nbytes()
        return total

    def info(self) -> dict[str, Any]:
        """The evaluator's choices for provenance: precision, backend, whether it is captured, its memory."""
        return {
            "precision": self.precision, "force_precision": self.force_precision, "backend": self.field.backend,
            "copy": not self.own_field, "compile_mode": self.field.compile_mode, "captured": False,
            "memory_bytes": self.memory_bytes(), "evaluations": self.evaluations,
        }


__all__ = ["PRECISIONS", "EnergyCache", "MCEvaluator", "float64_copy", "precision_of"]
