# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/optimize/lbfgs.py (LBFGS), LGPL-2.1-or-later; see NOTICE.md.
"""ASE 3.27's ``LBFGS`` on the device (SPEC §10.3; plan task 4-T05).

ASE's step, at iteration ``k`` with positions ``x``, gradient ``g`` (minus the constraint-adjusted forces) and
the previous ``r0``, ``g0``:

- the history gains ``s = x - r0``, ``y = g - g0``, ``rho = 1 / (y . s)`` (from ``k = 1``), and loses its oldest
  pair once ``k > memory``;
- the two-loop recursion gives ``z ~ H g`` with ``H0 = 1 / alpha``, the direction ``p = -z``;
- ``determine_step``: if the largest per-atom (per-row) length of ``p`` is ``>= maxstep``, the whole step is
  scaled to it; then times ``damping``;
- ``x + dr``, ``k + 1``, ``r0, g0 = x, g``.

Here the history update runs after the force call (``update``, stage B) and the rest before it (``propose``,
stage A), which is the same sequence. The history is a ring of ``memory`` slots: the pair of iteration ``k``
goes to slot ``(k - 1) mod memory``, so the slot of the oldest pair is ``k mod memory``. Unfilled slots hold
zeros, which drop out of the recursion exactly.

**The two-loop recursion in matrix form.** ASE's loops, written out on the chronological history,

    a_i = rho_i s_i . (q0 - sum_{j > i} a_j y_j)                  (newest to oldest)
    b_i = rho_i y_i . (H0 q + sum_{j < i} s_j c_j),  c_j = a_j - b_j   (oldest to newest)

are two unit-triangular systems in the Gram matrix ``G_ij = s_i . y_j``:
``(I + diag(rho) triu(G)) a = diag(rho) S q0`` and ``(I + diag(rho) tril(G^T)) c = a - diag(rho) Y H0 q``, with
``q = q0 - Y^T a`` and ``z = H0 q + S^T c``. A captured loop over ``memory = 100`` pairs would be some 600
small kernels per step, the order of a force call; this is six mat-vecs and two triangular solves. ``G`` is
kept current with two mat-vecs per new pair. The result equals ASE's to float64 rounding (the summation
order differs, as NumPy's ``dot`` differs from torch's anyway).

**The converged no-op.** ``propose`` leaves ``x`` bitwise unchanged and the iteration count as it was when
``converged`` is already set, so the engine can run stage A before the host has read the flag and stop
there (``GraphEngine._step_checked``): one host transfer per step carries the flag with the neighbour flags.
``status`` holds ``[converged, max force, energy]`` for that transfer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fastmace.optimize.base import Optimizer

# ASE's limit on ``maxstep`` (``LBFGS.__init__``), Angstrom.
MAX_MAXSTEP_A = 1.0


class DeviceLBFGS(Optimizer):
    """ASE 3.27 ``LBFGS`` (``use_line_search=False``) over a ``DOFView``, float64, graph-capturable."""

    # Every tensor ``propose`` and ``update`` write (the record's buffers, ``optimizer.<name>``).
    BUFFERS = ("s", "y", "rho", "sy", "r0", "g0", "iteration", "fmax", "max_force_now", "converged", "status")

    def __init__(self, dof, config):
        super().__init__(dof, config)
        self.maxstep = float(config.parameter("maxstep_A", 0.2))
        if self.maxstep > MAX_MAXSTEP_A:
            raise ValueError(f"You are using a much too large value for the maximum step size: {self.maxstep:.1f} "
                             "Angstrom (ASE's LBFGS refuses maxstep above 1.0 Angstrom)")
        self.memory = int(config.parameter("memory", 100))
        self.damping = float(config.parameter("damping", 1.0))
        self.alpha = float(config.parameter("alpha", 70.0))
        self.H0 = 1.0 / self.alpha
        x = dof.x
        f64 = dict(dtype=torch.float64, device=x.device)
        n = x.numel()
        self.s = torch.zeros((self.memory, n), **f64)
        self.y = torch.zeros((self.memory, n), **f64)
        self.rho = torch.zeros(self.memory, **f64)
        self.sy = torch.zeros((self.memory, self.memory), **f64)  # sy[i, j] = s_i . y_j, ring slots
        self.r0 = torch.zeros(n, **f64)
        self.g0 = torch.zeros(n, **f64)
        self.iteration = torch.zeros((), dtype=torch.int64, device=x.device)
        self.fmax = torch.full((), float(config.fmax), **f64)
        self.max_force_now = torch.zeros((), **f64)
        self._converged = torch.zeros((), dtype=torch.bool, device=x.device)
        self.status = torch.zeros(3, **f64)

    # ------------------------------------------------------------------ the contract

    @property
    def converged(self) -> torch.Tensor:
        return self._converged

    def set_fmax(self, fmax: float) -> None:
        self.fmax.fill_(float(fmax))

    def start(self) -> None:
        """No history (ASE's ``initialize``); ``max_force`` and ``converged`` at the current ``x``."""
        with torch.no_grad():
            for tensor in (self.s, self.y, self.rho, self.sy, self.r0, self.g0, self.iteration):
                tensor.zero_()
            self._check()

    def recheck(self) -> None:
        """``max_force`` and ``converged`` at the current ``x`` against the current ``fmax``, history kept: what
        ASE's ``run`` checks first when it is called again with another ``fmax``."""
        with torch.no_grad():
            self._check()

    def propose(self) -> None:
        dof = self.dof
        with torch.no_grad():
            g = dof.gradient()
            x = dof.x
            p = self.two_loop(g, self.s, self.y, self.rho, self.sy, self.iteration, self.H0)
            longest = torch.linalg.vector_norm(p.view(dof.rows, 3), dim=1).max()
            scale = torch.where(longest >= self.maxstep, self.maxstep / longest, torch.ones_like(longest))
            dr = p * scale * self.damping
            moved = torch.where(self._converged, x, x + dr)
            self.r0.copy_(x)
            self.g0.copy_(g)
            dof.set_x(moved)
            self.iteration.add_((~self._converged).to(torch.int64))

    def update(self) -> None:
        with torch.no_grad():
            g = self.dof.gradient()
            s = self.dof.x - self.r0
            y = g - self.g0
            rho = 1.0 / torch.dot(y, s)
            slot = torch.remainder(self.iteration - 1, self.memory).reshape(1)
            self.s.index_copy_(0, slot, s.unsqueeze(0))
            self.y.index_copy_(0, slot, y.unsqueeze(0))
            self.rho.index_copy_(0, slot, rho.reshape(1))
            self.sy.index_copy_(0, slot, (self.y @ s).unsqueeze(0))   # s_slot . y_j
            self.sy.index_copy_(1, slot, (self.s @ y).unsqueeze(1))   # s_i . y_slot
            self._check()

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {"s": self.s, "y": self.y, "rho": self.rho, "sy": self.sy, "r0": self.r0, "g0": self.g0,
                "iteration": self.iteration, "fmax": self.fmax, "max_force_now": self.max_force_now,
                "converged": self._converged, "status": self.status}

    def state_dict(self) -> dict[str, Any]:
        """The history oldest first (ASE's ``s``, ``y``, ``rho`` lists), the iteration and the step scalars."""
        iteration = int(self.iteration)
        count = min(iteration, self.memory)
        order = [(iteration - count + k) % self.memory for k in range(count)]
        return {"iteration": iteration, "s": self.s[order].cpu().numpy(), "y": self.y[order].cpu().numpy(),
                "rho": self.rho[order].cpu().numpy(), "r0": self.r0.cpu().numpy(), "g0": self.g0.cpu().numpy(),
                "fmax": float(self.fmax), "max_force": float(self.max_force_now), "converged": bool(self._converged)}

    def max_force(self) -> torch.Tensor:
        return self.max_force_now

    # ------------------------------------------------------------------ the maths

    def _check(self) -> None:
        self.max_force_now.copy_(self.dof.max_force())
        self._converged.copy_(self.max_force_now < self.fmax)
        self.status.copy_(torch.stack((self._converged.to(torch.float64), self.max_force_now,
                                       self.dof.energy().reshape(()).to(torch.float64))))

    @staticmethod
    def two_loop(q0: torch.Tensor, s: torch.Tensor, y: torch.Tensor, rho: torch.Tensor, sy: torch.Tensor,
                 iteration: torch.Tensor, H0: float) -> torch.Tensor:
        """ASE's direction ``p = -z`` for the gradient ``q0`` from the ring ``s``, ``y``, ``rho`` (slot-indexed,
        with ``sy[i, j] = s_i . y_j``) after ``iteration`` pairs. Fixed shapes, no host sync."""
        memory = s.shape[0]
        order = torch.remainder(iteration - memory + torch.arange(memory, device=s.device), memory)
        S, Y, R = s.index_select(0, order), y.index_select(0, order), rho.index_select(0, order)
        G = sy.index_select(0, order).index_select(1, order)  # chronological, oldest first
        a = torch.linalg.solve_triangular(R.unsqueeze(1) * G, (R * (S @ q0)).unsqueeze(1), upper=True,
                                          unitriangular=True).squeeze(1)
        z0 = H0 * (q0 - Y.T @ a)
        c = torch.linalg.solve_triangular(R.unsqueeze(1) * G.T, (a - R * (Y @ z0)).unsqueeze(1), upper=False,
                                          unitriangular=True).squeeze(1)
        return -(z0 + S.T @ c)


__all__ = ["DeviceLBFGS", "MAX_MAXSTEP_A"]
