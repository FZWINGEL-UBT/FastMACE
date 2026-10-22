"""SimulationState: the device-resident physical state of one structure (SPEC §10.6, docs/design/interfaces.md).

Phase 1a (plan tasks 1a-T02 and 1a-T12). The state holds tensors at fixed device
addresses: every update is in place (``copy_``, ``add_``), never a rebinding, because the CUDA
graphs of the engine capture those addresses. Integrators read and write it during the half
steps; the engine writes the force results of each step into it; nothing else changes it
during a run (from phase 3, changes between steps go through ``fastmace.events``).

Plan task 1b-T05 completed the §10.6 state except MC (``FIELDS_ADDED_IN_1B``, phase 3) and added
``ForceFieldIdentity``, the record a restart is resumed with.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

# SPEC §10.6 state that SimulationState does not carry yet, with where it is instead. Plan task
# 1b-T05 added charges, total charge, spin, PBC, constraints, barostat, protocol, force-field identity
# and plan revision; plan task 3-T08 the MC state (``mc``). Nothing is missing.
FIELDS_ADDED_IN_1B: dict[str, str] = {}

IDENTITY_SCHEMA = "fastmace.identity/1"
FORCE_PRECISIONS = ("float32", "float64")
# The identity fields a resume takes with a warning instead of refusing (SPEC §10.6, owner decision 23).
VERSION_FIELDS = ("mace_version", "fastmace_version")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json(value: Any) -> str:
    """JSON with sorted keys, no spaces and no NaN or infinity: equal values give equal text.

    ``ValueError`` names what is not JSON-safe (only dicts with string keys, lists and tuples,
    strings, numbers, booleans and ``None`` are).
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"not JSON-safe: {error}") from None


@dataclass(frozen=True)
class ForceFieldIdentity:
    """What a restart must be resumed with (SPEC §10.6 "Force-field identity"): same physics, same numbers.

    - ``checkpoint_sha256``: SHA-256 of the checkpoint file, 64 lowercase hex digits;
    - ``head``: the model head, ``None`` for single-head models;
    - ``species``: the species superset, atomic numbers (stored sorted and unique);
    - ``force_precision``: ``"float32"`` or ``"float64"``;
    - ``physics``: the physics options as a JSON-safe mapping: charge and spin inputs, D3 with all
      its parameters (§11.2), polar options (such as the PR 1762 settings), the reference
      corrections of §12.1. Stored as canonical JSON text, so the record is hashable;
    - ``reference_record_id``: the gate's reference record (§12.1), ``None`` when no gate ran;
    - ``mace_version``, ``fastmace_version``.

    ``digest`` is the SHA-256 of the canonical JSON of ``to_dict()``, schema name included: equal
    records have equal digests in every process and on every machine. Every field is part of it,
    the versions too. A resume takes another MACE or FastMACE version with a warning
    (``version_changes``, owner decision 23) and refuses every other difference (SPEC §10.6).
    Plan task 1b-T12 fills the record (``fm.ForceField``); 1b-T17 compares it on resume.
    """

    checkpoint_sha256: str
    head: str | None
    species: tuple[int, ...]
    force_precision: str
    physics: str = "{}"
    reference_record_id: str | None = None
    mace_version: str = ""
    fastmace_version: str = ""

    def __post_init__(self):
        if not isinstance(self.checkpoint_sha256, str) or not _SHA256.fullmatch(self.checkpoint_sha256):
            raise ValueError("ForceFieldIdentity.checkpoint_sha256 must be 64 lowercase hex digits, "
                             f"got {self.checkpoint_sha256!r}")
        for name in ("head", "reference_record_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"ForceFieldIdentity.{name} must be a non-empty string or None, got {value!r}")
        try:
            species = tuple(sorted(int(z) for z in self.species))
        except (TypeError, ValueError):
            raise ValueError(f"ForceFieldIdentity.species must be atomic numbers, got {self.species!r}") from None
        if (not species or len(set(species)) != len(species) or species[0] < 1
                or any(isinstance(z, bool) for z in self.species)):
            raise ValueError("ForceFieldIdentity.species must be distinct atomic numbers >= 1, "
                             f"got {self.species!r}")
        object.__setattr__(self, "species", species)
        if self.force_precision not in FORCE_PRECISIONS:
            raise ValueError(f"ForceFieldIdentity.force_precision must be one of {FORCE_PRECISIONS}, "
                             f"got {self.force_precision!r}")
        physics = json.loads(self.physics) if isinstance(self.physics, str) else self.physics
        if not isinstance(physics, Mapping):
            raise ValueError(f"ForceFieldIdentity.physics must be a mapping, got {self.physics!r}")
        try:
            object.__setattr__(self, "physics", canonical_json(dict(physics)))
        except ValueError as error:
            raise ValueError(f"ForceFieldIdentity.physics: {error}") from None
        for name in ("mace_version", "fastmace_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ForceFieldIdentity.{name} must be a non-empty string, got {value!r}")

    @property
    def physics_options(self) -> dict[str, Any]:
        """``physics`` as a new dict."""
        return json.loads(self.physics)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict with the schema name; ``from_dict`` reads it back."""
        return {"schema": IDENTITY_SCHEMA, "checkpoint_sha256": self.checkpoint_sha256, "head": self.head,
                "species": list(self.species), "force_precision": self.force_precision,
                "physics": self.physics_options, "reference_record_id": self.reference_record_id,
                "mace_version": self.mace_version, "fastmace_version": self.fastmace_version}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> ForceFieldIdentity:
        record = dict(record)
        schema = record.pop("schema", None)
        if schema != IDENTITY_SCHEMA:
            raise ValueError(f"force-field identity schema {schema!r}, this FastMACE reads {IDENTITY_SCHEMA!r}")
        return cls(**record)

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical JSON of ``to_dict()``."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def differences(self, other: ForceFieldIdentity) -> tuple[str, ...]:
        """The names of the fields in which ``other`` differs, in field order."""
        return tuple(field.name for field in fields(self) if getattr(self, field.name) != getattr(other, field.name))

    def version_changes(self, other: ForceFieldIdentity,
                        same_reference: Callable[[str, str], bool] | None = None) -> dict[str, tuple[Any, Any]]:
        """The version fields in which ``other`` differs: ``{name: (this record's, other's)}``.

        The reference record's id carries the MACE version, so ``reference_record_id`` counts as a
        version change too when both records exist and ``same_reference(this id, other id)`` says
        their content is the same (owner decision 23). Without ``same_reference`` it never does.
        """
        names = list(VERSION_FIELDS)
        mine, theirs = self.reference_record_id, other.reference_record_id
        if (same_reference is not None and mine is not None and theirs is not None and mine != theirs
                and same_reference(mine, theirs)):
            names.append("reference_record_id")
        return {name: (getattr(self, name), getattr(other, name)) for name in names
                if getattr(self, name) != getattr(other, name)}

    def with_versions_of(self, other: ForceFieldIdentity,
                         names: tuple[str, ...] = VERSION_FIELDS) -> ForceFieldIdentity:
        """A copy with ``other``'s values of ``names`` (the version fields): the rest is compared as it is."""
        return replace(self, **{name: getattr(other, name) for name in names})


@dataclass
class RandomStream:
    """The integrator's random numbers: generator plus the batch already drawn ahead (SPEC §10.6).

    The research engine draws ``noise_batch`` steps of normal numbers at once on the host and
    copies them to the device in one transfer; the restart file must hold the drawn batch and its
    cursor as well as the generator, or a resumed run would skip or repeat numbers.

    ``generator`` is a ``numpy.random.Generator`` (``default_rng(seed)``), the stream ASE uses, so
    a seed gives ASE's numbers. ``batch`` is allocated by the integrator, whose noise layout it is
    (Langevin: ``(noise_batch, 2, N, 3)``; BAOAB: ``(noise_batch, 3N + cell numbers)``).
    ``cursor`` is the index of the next unused step; ``cursor == len(batch)`` means "refill".
    """

    generator: np.random.Generator
    batch: torch.Tensor | None = None
    cursor: int = 0


@dataclass
class SimulationState:
    """Positions, velocities, masses, cell, species, the committed force results and the noise stream.

    All tensors are float64 on the device except ``species`` (int64). Shapes:

    - ``positions`` ``(N, 3)``, Å;
    - ``velocities`` ``(N, 3)``, ASE units. The stored quantity is the velocity, not the
      momentum, because the research engine integrates velocities; storing momenta would change
      the rounding and break bitwise equality with the baseline. ``momenta`` derives them;
    - ``masses`` ``(N, 1)``, amu (column vector so it broadcasts against ``(N, 3)``);
    - ``cell`` ``(3, 3)``, Å, rows are the lattice vectors. Constant for fixed-cell integrators,
      updated in place by variable-cell ones (BAOAB);
    - ``species`` ``(N,)`` atomic numbers;
    - ``forces`` ``(N, 3)`` eV/Å, ``energy`` ``(1,)`` eV, ``stress`` ``(3, 3)`` eV/Å³ (ASE sign
      convention, no kinetic part; ``None`` when the force field computes no stress): the results
      at the current positions, written by the engine's force stage;
    - ``rng``: the noise stream;
    - ``step``: number of committed MD steps (host integer).

    The rest of SPEC §10.6 (plan task 1b-T05; ``docs/design/contracts-1b.md``):

    - ``charges`` ``(N,)`` float64 per-atom charges for models that take them, else ``None``;
      ``total_charge`` (float) and ``spin_multiplicity`` (int), host, ``None`` when the model
      takes none;
    - ``pbc``: three bools, host; ``constraints``: a tuple, always empty until phase 4 (§4.4);
    - ``p_eps``, ``force_eps``: the barostat momentum and cell force, float64, shape ``()``
      (hydrostatic) or ``(3, 3)`` (full cell); ``None`` for fixed-cell integrators. The
      variable-cell integrator allocates them, as it allocates ``rng.batch``;
    - ``protocol``: the schedule with its position and target, an object with ``to_restart()``
      (a JSON-safe dict) and a ``from_restart(dict)`` class method (plan task 1b-T07), or ``None``;
    - ``plan_revision``: the active plan revision (§9.5);
    - ``force_field_identity``: the ``ForceFieldIdentity`` of the force field, or ``None``;
    - ``mc``: the attached move set's state, a JSON-safe dict with ``io.restart.MC_RECORD_KEYS``
      (``MCDriver.state_dict()``, plan tasks 3-T09 and 3-T10), or ``None`` without MC (plan task 3-T08).
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    masses: torch.Tensor
    cell: torch.Tensor
    species: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor | None
    rng: RandomStream
    step: int = 0
    charges: torch.Tensor | None = None
    total_charge: float | None = None
    spin_multiplicity: int | None = None
    pbc: tuple[bool, bool, bool] = (True, True, True)
    constraints: tuple = ()
    p_eps: torch.Tensor | None = None
    force_eps: torch.Tensor | None = None
    protocol: Any = None
    plan_revision: int = 0
    force_field_identity: ForceFieldIdentity | None = None
    mc: dict[str, Any] | None = None

    @classmethod
    def from_atoms(cls, atoms: Any, device: Any, seed: int, compute_stress: bool = False) -> SimulationState:
        """Build the state from an ASE ``Atoms`` (positions, velocities, masses, cell, numbers, PBC).

        Forces and energy start as zeros; the engine fills them with the initial evaluation.
        ``compute_stress`` allocates ``stress``. Refuses constraints (SPEC §4.4). The noise batch is
        left to the integrator, whose layout it is; charges, total charge and spin to the planner,
        because whether they are inputs depends on the model.
        """
        if atoms.constraints:
            raise ValueError("Constraints are not supported (SPEC §4.4)")
        import numpy as np  # here, so that importing the interfaces stays cheap (no torch)
        import torch

        f64 = dict(dtype=torch.float64, device=device)
        positions = torch.tensor(atoms.get_positions(), **f64)
        return cls(
            positions=positions,
            velocities=torch.tensor(atoms.get_velocities(), **f64),
            masses=torch.tensor(atoms.get_masses()[:, None], **f64),
            cell=torch.tensor(atoms.cell.array, **f64),
            species=torch.tensor(atoms.numbers, dtype=torch.int64, device=device),
            forces=torch.zeros_like(positions),
            energy=torch.zeros(1, **f64),
            stress=torch.zeros((3, 3), **f64) if compute_stress else None,
            rng=RandomStream(np.random.default_rng(seed)),
            pbc=tuple(bool(periodic) for periodic in atoms.pbc),
        )

    @property
    def natoms(self) -> int:
        """Number of atoms N."""
        return self.positions.shape[0]

    @property
    def momenta(self) -> torch.Tensor:
        """``velocities * masses``: a new tensor, the same expression as the baseline's output path."""
        return self.velocities * self.masses
