"""The ForceField protocol: energy, forces and stress from device positions (SPEC §7.1, §8.1).

Phase 1a stub (plan task 1a-T02); 1a-T11 implements it from the research engine's
``MACEForces``. See docs/design/interfaces.md for who calls what in a step.

A force field owns the model (compiled and rewritten), its input batch, the skin neighbour list
with its edge buffers and device edge counter (so the streaming counter is per instance and
``bind_streaming_count`` disappears), and the fixed-address cell, volume and reciprocal cell
the model reads. It knows nothing about integrators or engines.

Two levels:

- ``evaluate(atoms, stress)``: the public one-off evaluation of §7.1, host in, host out;
- the graph-level methods the engine calls inside and between captured stages:
  ``__call__``, ``set_cell``, ``needs_rebuild`` and ``compact`` are graph-capturable (no host
  synchronisation, fixed addresses); ``rebuild`` and ``use_edges`` are host-side and run only
  between replays.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import torch


class ForceOutput(NamedTuple):
    """One evaluation: ``energy`` ``(1,)`` eV, ``forces`` ``(N, 3)`` eV/Å, ``stress`` ``(3, 3)`` eV/Å³ or ``None``.

    Tensors in the force field's precision, owned by the model's output (copy them out before
    the next call). ``stress`` is MACE's: ASE sign convention, no kinetic part, ``None`` unless
    the force field was built with ``compute_stress``.
    """

    energy: torch.Tensor
    forces: torch.Tensor
    stress: torch.Tensor | None


@runtime_checkable
class ForceField(Protocol):
    """Energy, forces and stress of one structure, with its own neighbour list.

    Built by the planner for one structure, species superset, precision and rewrite set; one
    instance per engine (two force fields from one compiled model step independently).
    """

    @property
    def device(self) -> torch.device:
        """Device of the model and every buffer."""
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """Model precision (float32 or float64); positions and cell come in as float64."""
        raise NotImplementedError

    @property
    def compute_stress(self) -> bool:
        """Whether ``__call__`` returns a stress (variable-cell integrators need it)."""
        raise NotImplementedError

    @property
    def padding_exact(self) -> bool:
        """Whether padded edges beyond the cutoff contribute exactly zero (graph tier needs True)."""
        raise NotImplementedError

    def evaluate(self, atoms: Any, stress: bool = False) -> tuple[float, np.ndarray, np.ndarray | None]:
        """One-off evaluation of an ASE ``Atoms`` (SPEC §7.1): energy (eV), forces, 3x3 stress or None.

        Not for the MD loop: it synchronises, rebuilds the neighbour list and returns host values.
        """
        raise NotImplementedError

    def __call__(self, positions: torch.Tensor) -> ForceOutput:
        """Energy, forces (and stress) at float64 ``positions`` ``(N, 3)`` on the current edges.

        Graph-capturable: copies the positions into the model's input leaf and runs the model;
        reads the edges that the last ``use_edges`` selected and the cell of the last ``set_cell``.
        This is stage B's force call.
        """
        raise NotImplementedError

    def set_cell(self, cell: torch.Tensor) -> None:
        """Copy a new float64 cell into the neighbour list and the model inputs (volume, reciprocal cell).

        Graph-capturable; only for ``compute_stress`` force fields (variable-cell runs).
        """
        raise NotImplementedError

    def needs_rebuild(self, positions: torch.Tensor, fraction: float = 1.0) -> torch.Tensor:
        """Device bool: has any atom moved more than ``fraction`` of half the skin since the last build?

        Deformation-aware in variable-cell runs. Graph-capturable. ``fraction < 1`` is the
        pipelined early-rebuild test.
        """
        raise NotImplementedError

    def compact(self, positions: torch.Tensor) -> None:
        """Stage C: write the candidate pairs within the cutoff to the front of the edge buffers.

        Graph-capturable; updates the device edge count (``edge_count``).
        """
        raise NotImplementedError

    def rebuild(self, positions: torch.Tensor) -> bool:
        """Rebuild the candidate pairs from float64 positions (host-side, between replays).

        Returns True if the edge capacity grew: buffers moved, so every captured graph is stale.
        """
        raise NotImplementedError

    def use_edges(self, num_edges: int) -> None:
        """Let the model read the first ``num_edges`` compacted edges (views, no copy; host-side)."""
        raise NotImplementedError

    @property
    def edge_count(self) -> torch.Tensor:
        """Device int64 scalar: edges within the cutoff after the last ``compact``."""
        raise NotImplementedError

    @property
    def edge_capacity(self) -> int:
        """Size of the edge buffers (host integer); padded edge sizes never exceed it."""
        raise NotImplementedError

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Persistent device buffers that the graph-level methods write, by name.

        Neighbour-list state (edge index, shifts, count, build cell, reference positions) and the
        model inputs the cell feeds (cell, volume, reciprocal cell). The engine saves and restores
        them around capture warm-ups; the runtime-state registry of SPEC §8.1 (1b) starts here.
        """
        raise NotImplementedError

    def info(self) -> Mapping[str, Any]:
        """Counters for provenance: neighbour rebuilds, edge capacity, applied rewrites."""
        raise NotImplementedError
