"""MC move records and their attempts (SPEC §10.4; contracts: plan task 3-T01).

``Swap``, ``Transmute`` and ``Displace`` are frozen records a user builds before any model is
loaded, so this module imports neither torch nor MACE nor ASE. They are checked against the
workload and the superset by ``Simulation.run(mc=...)`` (§7.6). Species stay as given (symbols
or atomic numbers); the driver maps them to atomic numbers. ``build_attempt`` builds the captured
attempt of a move: 3-T09 (swaps, transmutations), 3-T10 (displacements).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from fastmace.config import SWAP_VELOCITIES, TRANSMUTE_VELOCITIES

MOVE_TYPES = ("swap", "transmute", "displace")
# Boltzmann constant in eV/K (CODATA 2018, as ase.units.kB).
K_B_EV_PER_K = 8.617333262e-5


def _species(value: Any, where: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value in ("", 0) or (
            isinstance(value, int) and value < 0):
        raise ValueError(f"{where}: a species is a symbol or an atomic number >= 1, got {value!r}")
    return value


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be an int of at least 1, got {value!r}")
    return value


def _per_species(value: Any, species: tuple, where: str, kind: type) -> tuple:
    """A mapping species -> number as sorted ``(species, value)`` pairs (hashable), keys within ``species``."""
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    try:
        pairs = [(key, number) for key, number in items]
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a mapping species -> value, got {value!r}") from None
    outside = [key for key, _ in pairs if key not in species]
    if outside:
        raise ValueError(f"{where} names {outside}, which are not in the transmutable set {species}")
    for key, number in pairs:
        if isinstance(number, bool) or not isinstance(number, (int, float) if kind is float else int) or (
                kind is float and not math.isfinite(number)):
            raise ValueError(f"{where}[{key!r}] must be a finite {kind.__name__}, got {number!r}")
    return tuple(sorted(((key, kind(number)) for key, number in pairs), key=lambda pair: str(pair[0])))


@dataclass(frozen=True)
class Swap:
    """Swap the species of two atoms (§10.4): a pair type uniformly from ``pairs``, then one atom of
    each species uniformly. ``every`` MD steps, then ``attempts`` attempts. ``velocities``:
    ``exchange`` (default; the atoms swap velocities), ``rescale``, ``redraw`` or ``redraw_all``."""

    move: ClassVar[str] = "swap"

    pairs: Any
    every: int
    attempts: int
    velocities: str = "exchange"

    def __post_init__(self):
        if isinstance(self.pairs, (str, bytes)):
            raise ValueError(f"Swap.pairs must be pairs of species, got {self.pairs!r}")
        try:
            pairs = tuple(tuple(pair) for pair in self.pairs)
        except TypeError:
            raise ValueError(f"Swap.pairs must be pairs of species, got {self.pairs!r}") from None
        if not pairs:
            raise ValueError("Swap.pairs needs at least one pair")
        for pair in pairs:
            if len(pair) != 2 or isinstance(pair, str):
                raise ValueError(f"Swap.pairs: each pair names two species, got {pair!r}")
            first, second = (_species(item, "Swap.pairs") for item in pair)
            if first == second:
                raise ValueError(f"Swap.pairs: a pair needs two different species, got {pair!r}")
        object.__setattr__(self, "pairs", pairs)
        _count(self.every, "Swap.every")
        _count(self.attempts, "Swap.attempts")
        if self.velocities not in SWAP_VELOCITIES:
            raise ValueError(f"Swap.velocities must be one of {SWAP_VELOCITIES}, got {self.velocities!r}")


@dataclass(frozen=True)
class Transmute:
    """Change one atom's species within the transmutable set ``species`` (§10.4, semi-grand canonical).

    ``velocities``: ``rescale``, ``redraw`` or ``redraw_all``; ``None`` takes the default (``rescale``,
    or ``redraw_all`` when the integrator fixes the centre of mass). ``mu``: configurational chemical
    potential differences in eV against a reference species (``configurational_mu`` converts full
    ones); absent species have 0. ``formal_charges``: q_s per species (0 if absent), for the parity
    rule ΔZ − Δq even (§10.4 "Charge and spin"). Both are stored as sorted pairs.
    """

    move: ClassVar[str] = "transmute"

    species: Any
    every: int
    attempts: int
    velocities: str | None = None
    mu: Any = None
    formal_charges: Any = None

    def __post_init__(self):
        if isinstance(self.species, (str, bytes)):
            raise ValueError(f"Transmute.species must be a set of species, got {self.species!r}")
        species = tuple(_species(item, "Transmute.species") for item in self.species)
        if len(species) < 2 or len(set(species)) != len(species):
            raise ValueError(f"Transmute.species needs at least two different species, got {self.species!r}")
        object.__setattr__(self, "species", species)
        _count(self.every, "Transmute.every")
        _count(self.attempts, "Transmute.attempts")
        if self.velocities is not None and self.velocities not in TRANSMUTE_VELOCITIES:
            raise ValueError(f"Transmute.velocities must be None or one of {TRANSMUTE_VELOCITIES}, "
                             f"got {self.velocities!r}")
        object.__setattr__(self, "mu", _per_species(self.mu, species, "Transmute.mu", float))
        object.__setattr__(self, "formal_charges",
                           _per_species(self.formal_charges, species, "Transmute.formal_charges", int))


@dataclass(frozen=True)
class Displace:
    """Displace one atom uniformly within a sphere of radius ``delta_A`` (§10.4). ``adaptive=True``
    tunes δ during equilibration only, and the value is recorded."""

    move: ClassVar[str] = "displace"

    delta_A: float
    every: int
    attempts: int
    adaptive: bool = False

    def __post_init__(self):
        if isinstance(self.delta_A, bool) or not isinstance(self.delta_A, (int, float)) or not (
                math.isfinite(self.delta_A) and self.delta_A > 0):
            raise ValueError(f"Displace.delta_A must be positive and finite, got {self.delta_A!r}")
        object.__setattr__(self, "delta_A", float(self.delta_A))
        _count(self.every, "Displace.every")
        _count(self.attempts, "Displace.attempts")
        if not isinstance(self.adaptive, bool):
            raise ValueError(f"Displace.adaptive must be a bool, got {self.adaptive!r}")


def _mass(species: Any) -> float:
    from ase.data import atomic_masses, atomic_numbers

    number = species if isinstance(species, int) else atomic_numbers[species]
    return float(atomic_masses[number])


def configurational_mu(mu_full: Mapping[Any, float], temperature_K: float, reference: Any,
                       masses: Mapping[Any, float] | None = None) -> dict[Any, float]:
    """Configurational chemical-potential differences from full ones (§10.4):
    Δμ_s^conf = Δμ_s^full + (3/2) k_B T ln(m_s / m_ref).

    ``mu_full`` maps species to Δμ_s^full in eV relative to ``reference`` (whose own entry, if
    given, must be 0). ``masses`` in amu, by species; ``None`` takes ASE's standard atomic masses.
    The result has the keys of ``mu_full``, for ``fm.mc.Transmute(mu=...)``.
    """
    if isinstance(temperature_K, bool) or not isinstance(temperature_K, (int, float)) or not (
            math.isfinite(temperature_K) and temperature_K > 0):
        raise ValueError(f"configurational_mu: temperature_K must be positive and finite, got {temperature_K!r}")
    if reference in mu_full and mu_full[reference] != 0:
        raise ValueError(f"configurational_mu: the reference {reference!r} has Δμ = 0 by definition, "
                         f"got {mu_full[reference]!r}")

    def mass(species):
        if masses is None:
            return _mass(species)
        if species not in masses:
            raise ValueError(f"configurational_mu: no mass for {species!r}")
        return float(masses[species])

    reference_mass = mass(reference)
    k_t = K_B_EV_PER_K * float(temperature_K)
    return {species: 0.0 if species == reference
            else float(value) + 1.5 * k_t * math.log(mass(species) / reference_mass)
            for species, value in mu_full.items()}


def build_attempt(move: Swap | Transmute | Displace, engine: Any, evaluator: Any, streams: Any) -> Any:
    """The captured attempt of ``move`` (propose → apply → refresh → energy → accept mask → restore,
    §10.4 "Graph capture") on ``engine``, with the MC ``evaluator`` (3-T07) and the device ``streams``."""
    task = "3-T10" if isinstance(move, Displace) else "3-T09"
    raise NotImplementedError(f"MC attempts of {type(move).__name__}: plan task {task}")
