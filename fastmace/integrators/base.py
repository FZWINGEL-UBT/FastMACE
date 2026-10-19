"""The integrator contract: graph-capturable half steps over a SimulationState (SPEC §10.1).

Phase 1a stub (plan task 1a-T02); 1a-T12 writes ``langevin.py`` (ASE ``Langevin``) and
``baoab.py`` (ASE ``LangevinBAOAB`` with barostat) against it. docs/design/interfaces.md has
the step sequence.

The cut: an integrator is the tensor maths of one MD step, split at the force evaluation.
It never calls the force field, never touches the neighbour list, never captures or replays a
graph and never synchronises with the host inside a half step. The engine composes

    prepare_step                       host, before the step's graphs (noise row)
    first_half                         stage A: kick, drift (and the cell), in the graph
    [engine: set_cell, neighbour check, compaction]
    [engine: force call, results copied into state.forces/energy/stress]
    second_half                        stage B: the second kick, in the graph

so the captured kernels are the research engine's, in the same order. Every tensor that a half
step reads or writes lives at a fixed address, allocated when the integrator is built; targets
such as the temperature are device scalars filled between replays.

Constructors are not part of the contract: the planner builds each integrator from its
``IntegratorConfig``, the timestep and the ``SimulationState`` (whose masses fix the
coefficients and whose ``rng.batch`` it allocates).
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from fastmace.state import SimulationState


class Integrator(abc.ABC):
    """One integrator instance per run, bound to one SimulationState.

    Constraint support (SPEC §4.4): none in 1.0; the planner refuses constraints before an
    integrator exists.
    """

    @property
    @abc.abstractmethod
    def variable_cell(self) -> bool:
        """True if ``first_half`` changes ``state.cell`` (BAOAB NPT); the engine then needs stress
        and passes the new cell to the force field after ``first_half``."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_temperature(self, temperature_K: float) -> None:
        """Fill the device target scalar (``sqrt(kB T)``) for the next step.

        Graph-safe between replays (a ``fill_``, no reallocation), so schedules change the target
        without quiescing the pipeline.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_step(self, state: SimulationState) -> None:
        """Host-side work before the step's graphs: copy the next noise row into the step buffers.

        Refills ``state.rng.batch`` from ``state.rng.generator`` when it is used up (one host draw
        and one upload per ``noise_batch`` steps), in ASE's order. Never captured; the engine calls
        it once per launched step, also for steps a pipelined redo repeats only partly.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def first_half(self, state: SimulationState) -> None:
        """Stage A: everything from the start of the step up to the new positions (and cell).

        Langevin: random displacements (centre of mass fixed if configured), half kick, drift,
        velocity from the displacement. BAOAB: barostat and atom half kicks, cell and position
        half drifts, the Ornstein-Uhlenbeck steps of atoms and cell, the second drifts. Reads
        ``state.forces`` of the previous step. Graph-capturable.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def second_half(self, state: SimulationState) -> None:
        """Stage B after the force call: the second half kick (BAOAB: the cell force first, then
        the atom and barostat kicks), with ``state.forces``, ``energy`` and ``stress`` already
        holding the new step's values. Graph-capturable."""
        raise NotImplementedError

    def start(self, state: SimulationState) -> None:
        """Called once by the engine after the initial force evaluation, before the first step.

        BAOAB computes the starting cell force here (``LangevinBAOAB.__init__``); the default
        does nothing.
        """

    @abc.abstractmethod
    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        """Tensors that ``first_half`` produces and ``second_half`` changes, in a fixed order.

        Langevin: ``(state.velocities,)``; BAOAB: ``(state.velocities, p_eps)``. The engine copies
        them to its own buffers inside the pipelined and speculative graphs and restores them
        before redoing stage B alone.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        """Every tensor a half step writes, by name: the state fields it changes and its own
        scratch (old positions, random displacements, barostat state). The engine saves and
        restores them around capture warm-ups."""
        raise NotImplementedError

    @abc.abstractmethod
    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        """Device tensors for one output row: positions, momenta, forces, energy, kinetic energy;
        variable-cell integrators add cell and stress. Fresh tensors or state views; the engine
        copies them to pinned host memory asynchronously (``Engine.snapshot``)."""
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Internal state for a restart (host arrays) that ``SimulationState`` does not hold.

        Empty for both integrators since 1b: the noise stream is in ``SimulationState.rng`` and
        BAOAB's barostat state (``p_eps``, ``force_eps``) in ``SimulationState`` (SPEC §10.6).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore ``state_dict()`` in place (fixed addresses kept); BAOAB also takes its 1a state."""
        raise NotImplementedError
