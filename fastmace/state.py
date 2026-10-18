"""SimulationState: the device-resident physical state of one structure (SPEC §10.6, docs/design/interfaces.md).

Phase 1a (plan tasks 1a-T02 and 1a-T12). The state holds tensors at fixed device
addresses: every update is in place (``copy_``, ``add_``), never a rebinding, because the CUDA
graphs of the engine capture those addresses. Integrators read and write it during the half
steps; the engine writes the force results of each step into it; nothing else changes it
during a run (from phase 3, changes between steps go through ``fastmace.events``).

The complete §10.6 state comes in 1b. ``FIELDS_ADDED_IN_1B`` lists what it adds, so that no
task adds those fields to the 1a state on the side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

# SPEC §10.6 state that phase 1a does not carry, with where it is in 1a instead. The keys are the
# SimulationState field names that plan task 1b-T05 adds (decided in 1b-T01,
# docs/design/contracts-1b.md); 1b-T05 empties this table except "mc".
FIELDS_ADDED_IN_1B = {
    "charges": "torch.Tensor (N,) float64 or None: per-atom charges for models that take them "
               "(1a: in the force field's input batch)",
    "total_charge": "float or None, host: total charge (1a: in the force field's input batch, OMOL)",
    "spin_multiplicity": "int or None, host: spin (1a: in the force field's input batch, OMOL)",
    "pbc": "tuple of 3 bools, host (1a: always 3D-periodic; the planner refuses anything else)",
    "constraints": "tuple, host, always empty in 1b (1a: refused at plan time, SPEC §4.4; phase 4)",
    "p_eps": "torch.Tensor () hydrostatic or (3, 3) full cell, float64, or None for fixed-cell "
             "integrators: barostat momentum (1a: LangevinBAOAB.p_eps, Integrator.state_dict)",
    "force_eps": "as p_eps: the cell force (1a: LangevinBAOAB.force_eps)",
    "protocol": "host record or None: schedule, position in it and thermostat target, fastmace.protocols "
                "(1b-T07) (1a: the caller's loop)",
    "mc": "MC move set, cycle position, counters and step sizes (phase 3)",
    "force_field_identity": "frozen record with a stable digest (1b-T05): checkpoint hash, head, species "
                            "superset, precision, physics options, reference record id, versions",
    "plan_revision": "int, host: the active plan revision (SPEC §9.5)",
}


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

    @classmethod
    def from_atoms(cls, atoms: Any, device: Any, seed: int, compute_stress: bool = False) -> SimulationState:
        """Build the state from an ASE ``Atoms`` (positions, velocities, masses, cell, numbers).

        Forces and energy start as zeros; the engine fills them with the initial evaluation.
        ``compute_stress`` allocates ``stress``. Refuses constraints (SPEC §4.4). The noise batch is
        left to the integrator, whose layout it is.
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
        )

    @property
    def natoms(self) -> int:
        """Number of atoms N."""
        return self.positions.shape[0]

    @property
    def momenta(self) -> torch.Tensor:
        """``velocities * masses``: a new tensor, the same expression as the baseline's output path."""
        return self.velocities * self.masses
