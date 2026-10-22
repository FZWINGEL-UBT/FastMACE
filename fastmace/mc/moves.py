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


def _species(value: Any, where: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value in ("", 0) or (
            isinstance(value, int) and value < 0):
        raise ValueError(f"{where}: a species is a symbol or an atomic number >= 1, got {value!r}")
    return value


def _key(species: Any) -> Any:
    """A species as compared between records: its atomic number. Symbols are looked up in ASE only when a
    record mixes symbols and numbers, so building a record of symbols (or of numbers) imports nothing."""
    if isinstance(species, int):
        return species
    from ase.data import atomic_numbers

    return atomic_numbers.get(species, species)


def _distinct(items: tuple) -> bool:
    """True when no species repeats, symbols and atomic numbers compared as atomic numbers ("Na" is 11)."""
    if len(set(items)) != len(items):
        return False
    if all(isinstance(item, str) for item in items) or all(isinstance(item, int) for item in items):
        return True
    keys = [_key(item) for item in items]
    return len(set(keys)) == len(keys)


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
            if not _distinct((first, second)):
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
        if len(species) < 2 or not _distinct(species):
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

    from ase.units import kB  # the thermostat's constant (the integrators' set_temperature)

    reference_mass = mass(reference)
    k_t = kB * float(temperature_K)
    return {species: 0.0 if species == reference
            else float(value) + 1.5 * k_t * math.log(mass(species) / reference_mass)
            for species, value in mu_full.items()}


def species_number(species: Any) -> int:
    """An atomic number from a symbol or an atomic number."""
    from ase.data import atomic_numbers

    if isinstance(species, str):
        if species not in atomic_numbers:
            raise ValueError(f"unknown chemical symbol {species!r}")
        return int(atomic_numbers[species])
    return int(species)


def resolved_velocities(move: Swap | Transmute, fixcm: bool) -> str:
    """The momentum treatment a move uses: its own, or for ``Transmute(velocities=None)`` the default
    (``rescale``, or ``redraw_all`` when the integrator fixes the centre of mass; SPEC §10.4)."""
    if move.velocities is not None:
        return move.velocities
    return "redraw_all" if fixcm else "rescale"


def metropolis(u, excess, kT):
    """The accept mask: ``ln u < -excess / kT`` with ``excess`` = ΔE − Σ Δμ ΔN and ``kT`` in eV; a NaN
    excess rejects. Looked up when an attempt is built, so a test can replace it."""
    import torch

    return torch.log(u) < -excess / kT


def charge_inputs(field: Any) -> bool:
    """Whether the force field's model or head takes charge or spin inputs (SPEC §10.4 "Charge and spin"):
    PolarMACE, or a joint embedding of the total charge or spin (MACE-OMOL-0)."""
    if getattr(field, "is_polar", False):
        return True
    model = getattr(field, "model", None)
    model = getattr(model, "_orig_mod", model)
    specs = getattr(model, "embedding_specs", None) or {}
    return any(name in specs for name in ("total_charge", "total_spin"))


# Tables by atomic number (0 .. 118), ASE's element range.
TABLE_SIZE = 119


class Attempt:
    """The attempt of a swap or transmutation move on a built engine. ``run()`` is one attempt: propose →
    apply → refresh → energy → accept mask → restore, as device operations with fixed shapes and
    addresses, so the driver captures it (``GraphEngine.captured``) and replays it once per attempt.

    ``counters`` (int64 ``[3]``: attempted, impossible, accepted) count on the device. The block's
    ``streams`` (``streams.BlockStreams``) give the random numbers, kT and the baseline energy.
    """

    def __init__(self, move: Swap | Transmute, engine: Any, evaluator: Any, streams: Any, *, fixcm: bool):
        import torch
        from ase.data import atomic_masses

        from fastmace.mc import momenta
        from fastmace.mc import streams as mc_streams

        self.move, self.engine, self.evaluator, self.streams = move, engine, evaluator, streams
        state = engine.state
        device = state.positions.device
        self.natoms = state.natoms
        self.velocities = resolved_velocities(move, fixcm)
        self.counts = mc_streams.attempt_counts(move.move, self.velocities, self.natoms)
        self.block_velocity_count = mc_streams.block_velocity_count(self.velocities, self.natoms)
        self.counters = torch.zeros(3, dtype=torch.int64, device=device)
        self.standard_masses = torch.as_tensor(atomic_masses[:TABLE_SIZE], dtype=torch.float64).to(device)
        field = engine.force_field
        self.charge_inputs = charge_inputs(field)
        # The per-atom charge source of the force field (it aliases state.charges when the model reads them).
        self.charges = getattr(field, "charges", None)
        if isinstance(move, Swap):
            pairs = [(species_number(a), species_number(b)) for a, b in move.pairs]
            self.pairs = torch.tensor(pairs, dtype=torch.int64, device=device)
            # Per-atom charges travel with the species where the model reads them.
            self.write_charges = state.charges is not None and self.charges is not None
        else:
            numbers = [species_number(item) for item in move.species]
            self.set = torch.tensor(numbers, dtype=torch.int64, device=device)
            self.members = torch.zeros(TABLE_SIZE, dtype=torch.bool, device=device)
            self.members[self.set] = True
            self.position = torch.zeros(TABLE_SIZE, dtype=torch.int64, device=device)
            self.position[self.set] = torch.arange(len(numbers), dtype=torch.int64, device=device)
            self.mu = torch.zeros(TABLE_SIZE, dtype=torch.float64, device=device)
            for species, value in move.mu:
                self.mu[species_number(species)] = float(value)
            self.formal = torch.zeros(TABLE_SIZE, dtype=torch.float64, device=device)
            for species, value in move.formal_charges:
                self.formal[species_number(species)] = float(value)
            changes_charge = any(value for _, value in move.formal_charges)
            self.write_charges = (changes_charge and self.charges is not None
                                  and (state.charges is not None or self.charge_inputs))
        self.refreshed = ("species", "charges") if self.write_charges else ("species",)
        self.treatment = None if self.velocities == "redraw_all" else getattr(momenta, self.velocities)
        self.accept = metropolis
        self.stage, self.trial = None, None  # the energy stage of the edges the replay reads (``bound``)
        self.accepted = torch.zeros(1, dtype=torch.bool, device=device)  # the last attempt's decision

    def energy(self):
        """The MC evaluator's float64 energy ``[1]`` of the current (trial) state."""
        if self.stage is not None:
            self.stage()
            return self.trial
        return self.evaluator.evaluate()

    def bound(self, stage, trial):
        """The function to capture for one edge size: ``run`` with that size's energy stage and output buffer
        (``MCEvaluator.stage``; ``None`` on engines without one, which evaluate directly)."""

        def run() -> None:
            self.stage, self.trial = stage, trial
            self.run()

        return run

    def run(self) -> None:
        import torch

        with torch.no_grad():
            self._run(torch)

    def _run(self, torch) -> None:
        state, streams = self.engine.state, self.streams
        species, masses, velocities = state.species, state.masses, state.velocities
        u = streams.uniforms("proposal", self.counts["proposal"])
        acceptance = streams.uniforms("acceptance", self.counts["acceptance"])
        normals = (streams.normals("velocity", self.counts["velocity"]).view(-1, 3) if self.counts["velocity"]
                   else None)
        if isinstance(self.move, Swap):
            from fastmace.mc.streams import decode_swap

            atoms, possible = decode_swap(u, species, self.pairs)
            old_species = species[atoms]
            new_species = torch.where(possible, old_species.flip(0), old_species)
            old_masses = masses[atoms]
            new_masses = old_masses.flip(0)
            excess_mu = None
        else:
            from fastmace.mc.streams import decode_transmute

            atoms, proposed, possible = decode_transmute(u, species, self.set, self.members, self.position)
            old_species = species[atoms]
            new_species = torch.where(possible, proposed, old_species)
            old_masses = masses[atoms]
            new_masses = self.standard_masses[new_species][:, None]
            excess_mu = self.mu[new_species] - self.mu[old_species]
        new_masses = torch.where(possible[:, None], new_masses, old_masses)
        if self.write_charges:
            old_charges = self.charges[atoms]
            if excess_mu is None:
                new_charges = old_charges.flip(0)
            else:
                new_charges = old_charges + (self.formal[new_species] - self.formal[old_species])
            new_charges = torch.where(possible, new_charges, old_charges)
            self.charges[atoms] = new_charges
        # Apply the trial species, refresh what the energy reads, evaluate.
        species[atoms] = new_species
        self.engine.derived.refresh(self.refreshed)
        trial = self.energy()
        excess = trial - streams.baseline
        if excess_mu is not None:
            excess = excess - excess_mu
        accepted = possible & self.accept(acceptance, excess, streams.kT)
        self.accepted.copy_(accepted)
        streams.baseline.copy_(torch.where(accepted, trial, streams.baseline))
        self.counters.add_(torch.cat((torch.ones_like(possible), ~possible, accepted)).to(torch.int64))
        # Keep the trial where accepted, restore the snapshot rows elsewhere (copies, so bit for bit).
        species[atoms] = torch.where(accepted, new_species, old_species)
        masses[atoms] = torch.where(accepted[:, None], new_masses, old_masses)
        if self.write_charges:
            self.charges[atoms] = torch.where(accepted, new_charges, old_charges)
        if self.treatment is not None:
            old_velocities = velocities[atoms]
            new_velocities = self.treatment(old_velocities, old_masses, new_masses, normals, streams.kT)
            velocities[atoms] = torch.where(accepted[:, None], new_velocities, old_velocities)
        self.engine.derived.refresh(self.refreshed)
        streams.index.add_(1)

    def tensors(self):
        """The attempt's own mutable device tensors (snapshots around a capture)."""
        out = [self.counters, self.accepted]
        if self.write_charges:
            out.append(self.charges)
        return tuple(out)


def build_attempt(move: Swap | Transmute | Displace, engine: Any, evaluator: Any, streams: Any, *,
                  fixcm: bool = False) -> Attempt:
    """The attempt of ``move`` (propose → apply → refresh → energy → accept mask → restore, §10.4 "Graph
    capture") on ``engine``, with the MC ``evaluator`` (3-T07) and the block's device ``streams``
    (``streams.BlockStreams``); ``Attempt.run`` is what the driver captures. ``fixcm``: the integrator
    fixes the centre of mass, which sets a transmutation's default treatment. Imports torch."""
    if isinstance(move, Displace):
        raise NotImplementedError("MC attempts of Displace: plan task 3-T10")
    if not isinstance(move, (Swap, Transmute)):
        raise TypeError(f"build_attempt takes a Swap or a Transmute, got {type(move).__name__}")
    return Attempt(move, engine, evaluator, streams, fixcm=fixcm)
