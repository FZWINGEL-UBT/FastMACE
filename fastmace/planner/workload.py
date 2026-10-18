"""Workload: what the user's run will do, as far as no other object knows it (SPEC §9.2).

Plan task 1b-T01 fixes the record and its validation; the minimal planner (1b-T12) reads it.
SPEC §5 calls the module's record ``WorkloadSpec``; §7.2 and §9.2 call it ``Workload``, which is
the public name (``fm.Workload``).

``Workload`` holds only facts with no other owner. Periodicity and constraints come from the
``Atoms``; ensemble, cell mode and whether stress is needed from the integrator record; force
precision and the species superset from the ``ForceField`` (§7.1, §7.6). Construction validates
the record (``ValueError``); ``PlanRefused`` stays the planner's aggregate of §9.3, so
``atom_count_changes=True`` is accepted here and refused by the planner (rule 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fastmace.config import _check_choices

# The MC move types of SPEC §10.4 a workload can declare; insertion and deletion are reserved.
MC_MOVES = ("swap", "transmute", "displace")
# Moves that change a site's species and therefore need species_changes=True.
SPECIES_MOVES = ("swap", "transmute")


@dataclass(frozen=True)
class Workload:
    """What the run will do (SPEC §9.2). Only ``max_temperature_K`` is required.

    - ``max_temperature_K``: the highest thermostat target of the run, K; sizes the skin and the
      bucket headroom (rule 11). ``Simulation.run`` refuses a schedule above it (§7.6);
    - ``species_changes``: swaps, transmutations or species events (§8.5);
    - ``atom_count_changes``: reserved; the planner refuses it in 1.0 (§9.3 rule 7, §8.6);
    - ``min_mc_temperature_K``: the lowest MC target temperature, K; the ΔE gate threshold
      0.01 k_B T_min (§12.3). Required when ``mc_moves`` is non-empty, and positive;
    - ``host_callbacks_every``: user Python every n steps (rule 9); ``None`` = no callbacks;
    - ``needs_per_atom``: per-atom outputs the run needs (``"energies"``, ``"charges"``, ...);
    - ``mc_moves``: declared move types (``"swap"``, ``"transmute"``, ``"displace"``);
    - ``reproducible``: bitwise within the scope of §9.3; the planner refuses what it cannot
      guarantee (in 1b: everything, "no family qualified");
    - ``mc_energy_precision``: §10.4; ``"auto"`` = the force precision if the ΔE gate passes.
    """

    max_temperature_K: float
    species_changes: bool = False
    atom_count_changes: bool = False
    min_mc_temperature_K: float | None = None
    host_callbacks_every: int | None = None
    needs_per_atom: frozenset[str] = frozenset()
    mc_moves: tuple[str, ...] = ()
    reproducible: bool = False
    mc_energy_precision: Literal["auto", "float32", "float64"] = "auto"

    def __post_init__(self):
        _check_choices(self)
        _temperature("max_temperature_K", self.max_temperature_K, positive=False)
        if self.min_mc_temperature_K is not None:
            # The ΔE floor 0.01 k_B T_min would be zero (§12.3, owner decision 8).
            _temperature("min_mc_temperature_K", self.min_mc_temperature_K, positive=True)
        every = self.host_callbacks_every
        if every is not None and (isinstance(every, bool) or not isinstance(every, int) or every <= 0):
            raise ValueError(f"Workload.host_callbacks_every must be a positive int or None, got {every!r}")
        if isinstance(self.needs_per_atom, str) or not all(isinstance(n, str) for n in self.needs_per_atom):
            raise ValueError(f"Workload.needs_per_atom must be a collection of names, got {self.needs_per_atom!r}")
        object.__setattr__(self, "needs_per_atom", frozenset(self.needs_per_atom))
        if isinstance(self.mc_moves, str):
            raise ValueError(f"Workload.mc_moves must be a tuple of move names, got the string {self.mc_moves!r}")
        moves = tuple(self.mc_moves)
        unknown = [move for move in moves if move not in MC_MOVES]
        if unknown:
            raise ValueError(f"Workload.mc_moves: unknown move types {unknown}; known: {MC_MOVES}")
        if len(set(moves)) != len(moves):
            raise ValueError(f"Workload.mc_moves lists a move type twice: {moves}")
        object.__setattr__(self, "mc_moves", moves)
        if moves and self.min_mc_temperature_K is None:
            raise ValueError("Workload.min_mc_temperature_K is required when mc_moves is non-empty: it is the "
                             "lowest MC target temperature and sets the ΔE gate threshold (SPEC §9.2, §12.3)")
        needs_species = [move for move in moves if move in SPECIES_MOVES]
        if needs_species and not self.species_changes:
            raise ValueError(f"Workload.mc_moves {needs_species} change species and need species_changes=True "
                             "(SPEC §9.2)")


def _temperature(name, value, positive):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"Workload.{name} must be a finite, {bound} temperature in K, got {value!r}")
