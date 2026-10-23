"""The optimiser contract: graph-capturable relaxation steps over a DOF view (SPEC §10.3; plan task 4-T01).

An optimiser never sees atoms or a cell. It works on a ``DOFView``: a flat device vector ``x`` of the
degrees of freedom, the energy and gradient at ``x``, ``set_x`` to move, and an optional mask of the free
components. Positions (``PositionsDOF``, fixed cell) and 4-T07's device ``FrechetCellFilter`` (positions and
the log of the deformation gradient, the (N + 3) x 3 vector of ASE's filter) both implement it, so FIRE and
LBFGS run unchanged at fixed and at variable cell.

The driver (4-T05, ``optimize/driver.py``) composes one step as

    propose                         the step from the gradient and the history; dof.set_x, in the graph
    [engine: set_cell, neighbour check, compaction]
    [engine: force call, results copied into state.forces/energy/stress]
    dof.refresh                     derived quantities at the new x (the cell filter's virial), in the graph
    update                          history, fmax and the device ``converged`` flag, in the graph

and reads ``converged`` and ``max_force`` on the host once per step (conductor's choice 5), as ASE checks
``fmax`` before each step. ``start`` runs once after the first force call. Optimiser state is float64.

Rows: ``x`` has ``3 * rows`` entries; ``x.view(rows, 3)`` are the per-atom (and per-cell-vector) rows, the
unit of ASE's ``fmax`` (largest row norm of the forces) and of LBFGS's ``maxstep`` scaling.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from fastmace.config import OptimizerConfig
    from fastmace.state import SimulationState


class DOFView(abc.ABC):
    """The degrees of freedom an optimiser moves, at fixed device addresses.

    Every tensor is float64 on the state's device. Graph-safe: ``set_x`` and ``refresh`` are in-place
    tensor operations without host syncs.
    """

    @property
    @abc.abstractmethod
    def rows(self) -> int:
        """Number of 3-vectors: N for positions, N + 3 with a cell filter."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def variable_cell(self) -> bool:
        """True if ``set_x`` changes the cell; the engine then needs stress and passes the new cell to the force
        field after ``propose``."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x(self) -> torch.Tensor:
        """The current values, shape ``(3 * rows,)``. A view or a buffer the view keeps current; treat it as
        read-only and move through ``set_x``."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_x(self, x: torch.Tensor) -> None:
        """Move to ``x`` (shape ``(3 * rows,)``) in place. Masked (fixed) components keep their values."""
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self) -> torch.Tensor:
        """The gradient of ``energy()`` at ``x``, shape ``(3 * rows,)``, with masked components zero: minus ASE's
        ``optimizable.get_gradient()`` forces after the constraints (``atoms.get_forces()``)."""
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self) -> torch.Tensor:
        """The objective at ``x``, shape ``(1,)``: E_pot, plus p V with a cell filter's scalar pressure (ASE's
        ``get_potential_energy`` of the filter, whose force consistency is ``energy`` not ``free_energy``)."""
        raise NotImplementedError

    @property
    def mask(self) -> torch.Tensor | None:
        """Boolean ``(3 * rows,)``, True for free components; ``None`` when every component is free."""
        return None

    def refresh(self) -> None:
        """Derived quantities at the new ``x`` after the force call (the cell filter's virial gradient). Default:
        none."""

    def max_force(self) -> torch.Tensor:
        """ASE's ``fmax`` measure: the largest row norm of ``-gradient()``, a device scalar (shape ``()``)."""
        import torch

        return torch.linalg.vector_norm(self.gradient().view(self.rows, 3), dim=1).max()

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Every tensor ``set_x`` and ``refresh`` write, by name; the engine saves and restores them around capture
        warm-ups. Default: none of its own (``PositionsDOF`` writes only the state's positions)."""
        return {}


class PositionsDOF(DOFView):
    """Positions at a fixed cell: ``x`` is ``state.positions`` flattened (a view), the gradient minus
    ``state.forces``. ``FixAtoms`` (``state.free_mask``) zeroes the fixed atoms' gradient and keeps their
    positions, as ASE's ``FixAtoms.adjust_positions`` and ``adjust_forces`` do."""

    def __init__(self, state: SimulationState):
        self.state = state
        free = state.free_mask
        self._mask = None if free is None else free.expand(state.natoms, 3).reshape(-1).bool().clone()

    @property
    def rows(self) -> int:
        return self.state.natoms

    @property
    def variable_cell(self) -> bool:
        return False

    @property
    def x(self) -> torch.Tensor:
        return self.state.positions.view(-1)

    def set_x(self, x: torch.Tensor) -> None:
        import torch

        target = self.state.positions.view(-1)
        target.copy_(x if self._mask is None else torch.where(self._mask, x, target))

    def gradient(self) -> torch.Tensor:
        gradient = -self.state.forces.reshape(-1)
        return gradient if self._mask is None else gradient * self._mask

    def energy(self) -> torch.Tensor:
        return self.state.energy

    @property
    def mask(self) -> torch.Tensor | None:
        return self._mask


class Optimizer(abc.ABC):
    """One optimiser instance per relaxation, bound to one ``DOFView`` at construction.

    Constructors: ``Class(dof, config)`` with an ``OptimizerConfig`` (``registry.build_optimizer``); only the
    planner builds optimisers (SPEC §5). Every tensor a step reads or writes is allocated there, at a fixed
    address; ``fmax`` is a device scalar filled between replays.
    """

    def __init__(self, dof: DOFView, config: OptimizerConfig):
        self.dof = dof
        self.config = config

    @property
    def variable_cell(self) -> bool:
        """The DOF view's: True with a cell filter."""
        return self.dof.variable_cell

    @abc.abstractmethod
    def set_fmax(self, fmax: float) -> None:
        """Fill the device convergence threshold (eV/Angstrom) for the next ``update``; a ``fill_``."""
        raise NotImplementedError

    @abc.abstractmethod
    def start(self) -> None:
        """Once after the first force call: the starting history, ``max_force`` and ``converged`` at the start
        (ASE checks convergence before its first step)."""
        raise NotImplementedError

    @abc.abstractmethod
    def propose(self) -> None:
        """The step from the gradient and the history, then ``dof.set_x``. Graph-capturable; never called once
        ``converged`` is set."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self) -> None:
        """After the force call and ``dof.refresh``: the history, ``max_force`` and ``converged``.
        Graph-capturable."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def converged(self) -> torch.Tensor:
        """Device bool scalar: ``max_force() < fmax`` at the current ``x`` (ASE's ``converged``)."""
        raise NotImplementedError

    def max_force(self) -> torch.Tensor:
        """The largest constraint-adjusted row force at the current ``x``, a device scalar (``dof.max_force``)."""
        return self.dof.max_force()

    @abc.abstractmethod
    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Every tensor ``propose`` and ``update`` write (history, step, scalars), by name, with the DOF view's;
        the engine saves and restores them around capture warm-ups."""
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """The optimiser's state as host arrays, for tests (no optimiser restart files: conductor's choice 5)."""
        raise NotImplementedError

    def output_values(self) -> Mapping[str, torch.Tensor]:
        """Device tensors for one relaxation row: energy, max force, the positions (and cell). Default: the DOF
        view's energy and max force with the state's positions and cell."""
        state = getattr(self.dof, "state", None)
        values = {"energy": self.dof.energy(), "fmax": self.max_force()}
        if state is not None:
            values["positions"] = state.positions
            if self.variable_cell:
                values["cell"] = state.cell
        return values
