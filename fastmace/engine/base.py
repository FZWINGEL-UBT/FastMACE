"""The Engine contract: runs MD steps of one integrator on one force field and state (SPEC §8).

Phase 1a stub (plan task 1a-T02); 1a-T13 writes the CUDA-graph engine (``graph.py``) from the
capture, replay and pipelining half of the research engine's ``GPULangevin``. Only the planner
instantiates engines (SPEC §5); docs/design/interfaces.md has the step sequence.

An engine owns the execution: the stage graphs (``AC``, ``ACP``, ``("B", edges)``, ``C`` and
the speculative ``("ACB", edges, slot)``), the graph pool and side stream, the edge bucket and
padded sizes, the host-read flags (rebuild, count, early rebuild) with their pinned mirrors,
the mid-step copies of the integrator's ``mid_step_tensors``, the pipelined or speculative step
in flight, and the counters (redos, rollbacks, rebuilds, pool resets, capture memory). It
composes the integrator's half steps with the force field's calls; it holds no MD maths.

Reading the state (``snapshot``, ``sync``, ``potential_energy``, ``flush``) first validates the
step in flight, exactly what the next step would do.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from fastmace.config import EngineConfig
    from fastmace.integrators.base import Integrator
    from fastmace.model.forces import ForceField
    from fastmace.state import SimulationState


@dataclass
class Snapshot:
    """Output values of one committed step, being copied to pinned host memory (async output).

    ``values`` maps names (positions, momenta, forces, energy, kinetic; cell and stress for
    variable-cell runs) to host tensors; ``event`` is the CUDA event recorded behind the copies
    (``None`` on the CPU). ``step`` is the committed step the values belong to.
    """

    step: int
    values: Mapping[str, torch.Tensor] = field(default_factory=dict)
    event: Any = None

    def wait(self) -> Mapping[str, torch.Tensor]:
        """Block until the copies are done and return ``values``."""
        raise NotImplementedError("1a-T13")


class Engine(abc.ABC):
    """Steps a SimulationState with an Integrator and a ForceField.

    Concrete engines take ``(config: EngineConfig, force_field: ForceField, integrator:
    Integrator, state: SimulationState)``. Building one performs the initial compaction and
    force evaluation (writing ``state.forces``, ``energy``, ``stress``) and calls
    ``integrator.start(state)``; graphs are captured lazily on first use.
    """

    config: EngineConfig
    force_field: ForceField
    integrator: Integrator
    state: SimulationState

    @abc.abstractmethod
    def step(self) -> None:
        """Launch one MD step. Pipelined and speculative modes return before it is validated."""
        raise NotImplementedError

    def run(self, steps: int) -> None:
        """``steps`` calls of ``step``, then ``flush``."""
        raise NotImplementedError("1a-T13")

    @abc.abstractmethod
    def flush(self) -> None:
        """Validate every step still in flight (redo its force stage if needed); no-op otherwise."""
        raise NotImplementedError

    @abc.abstractmethod
    def snapshot(self) -> Snapshot:
        """Flush, then queue asynchronous copies of the integrator's ``output_values`` behind the
        current step without draining the GPU; the MD state is unchanged."""
        raise NotImplementedError

    @abc.abstractmethod
    def sync(self) -> Mapping[str, Any]:
        """Flush and return the output values as host arrays (synchronising; for frames and restarts)."""
        raise NotImplementedError

    @abc.abstractmethod
    def potential_energy(self) -> float:
        """Flush and return the committed potential energy in eV."""
        raise NotImplementedError

    @abc.abstractmethod
    def info(self) -> Mapping[str, Any]:
        """Run counters for provenance: edge bucket, edge sizes used, neighbour rebuilds, redos,
        rollbacks, pool resets, capture memory records."""
        raise NotImplementedError
