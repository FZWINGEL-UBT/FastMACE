"""ASE 3.27's ``FIRE`` on a ``DOFView``, graph-capturable (SPEC §10.3; plan task 4-T06).

ASE's ``FIRE.step`` branches on host floats: whether ``v.f > 0``, whether ``Nsteps > Nmin``, whether the step
exceeds ``maxstep``, and with ``downhill_check`` whether the energy rose. Here every branch is computed and
selected with ``torch.where`` on device booleans, so a step has no host sync and replays from a CUDA graph.
ASE's operation order is kept, so on float64 the only difference from ASE is the summation order of the dot
products.

ASE's state maps as: ``vel`` (``None`` before the first step: ``started`` False and ``vel`` zero), ``dt``,
``a``, ``Nsteps`` (``nsteps``), and with ``downhill_check`` ``e_last`` and ``r_last``. ASE's ``vel_last`` is
written but never read, so it is not kept. All of it is float64 (conductor's choice 5). ``restarts`` counts the
steps that took ASE's restart branch (``v.f <= 0`` or uphill: ``v = 0``, ``dt *= fdec``), a diagnostic of how
often FIRE's switches fire; it is not ASE state.

**The converged no-op** (as ``optimize.lbfgs``): when ``converged`` is already set, ``propose`` leaves ``x``
and every piece of FIRE's state bitwise as they were, so the engine can run stage A before the host has read
the flag (``GraphEngine._step_checked``). ``status`` holds ``[converged, max force, energy]`` for that read.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from fastmace.optimize.base import Optimizer

_REFUSED = {
    "position_reset_callback": "FIRE's position_reset_callback is a host callback inside the step, which a "
                               "captured step cannot make (plan task 4-T06)",
    "maxmove": "FIRE's maxmove is ASE's deprecated name; use maxstep_A",
}


def _defaults() -> dict[str, Any]:
    """``fm.optimize.FIRE``'s fields and defaults: the parameters an ``OptimizerConfig`` for "fire" may carry."""
    from fastmace.optimize import FIRE as record

    return {field.name: field.default for field in dataclasses.fields(record) if field.name != "cell_filter"}


class FIRE(Optimizer):
    """ASE 3.27 ``FIRE(dt, maxstep, dtmax, Nmin, finc, fdec, astart, fa, a, downhill_check)`` over ``dof``."""

    # Every tensor ``propose``, ``update`` and ``recheck`` write (the record's buffers, ``optimizer.<name>``).
    BUFFERS = ("vel", "dt", "a", "nsteps", "started", "e_last", "r_last", "restarts", "fmax", "max_force_value",
               "converged_flag", "status")

    def __init__(self, dof, config):
        super().__init__(dof, config)
        parameters = _defaults()
        for name, value in config.parameters:
            if name in _REFUSED:
                raise ValueError(_REFUSED[name])
            if name not in parameters:
                raise ValueError(f"FIRE has no parameter {name!r}; its parameters are {sorted(parameters)}")
            parameters[name] = value
        self.parameters = parameters
        self.downhill_check = bool(parameters["downhill_check"])
        x = dof.x
        f64 = dict(dtype=torch.float64, device=x.device)
        self.vel = torch.zeros_like(x)
        self.dt = torch.tensor(float(parameters["dt"]), **f64)
        self.a = torch.tensor(float(parameters["a"]), **f64)
        self.nsteps = torch.zeros((), dtype=torch.int64, device=x.device)
        self.started = torch.zeros((), dtype=torch.bool, device=x.device)
        self.restarts = torch.zeros((), dtype=torch.int64, device=x.device)
        self.e_last = torch.zeros(1, **f64)
        self.r_last = torch.zeros_like(x)
        self.fmax = torch.tensor(float(config.fmax), **f64)
        self.max_force_value = torch.zeros((), **f64)
        self.converged_flag = torch.zeros((), dtype=torch.bool, device=x.device)
        self.status = torch.zeros(3, **f64)

    # ------------------------------------------------------------------ the contract

    def set_fmax(self, fmax: float) -> None:
        self.fmax.fill_(float(fmax))

    def start(self) -> None:
        """ASE's state at construction (``vel`` None, the initial ``dt`` and ``a``); ``max_force`` and
        ``converged`` at the current ``x``."""
        with torch.no_grad():
            for tensor in (self.vel, self.nsteps, self.started, self.e_last, self.r_last, self.restarts):
                tensor.zero_()
            self.dt.fill_(float(self.parameters["dt"]))
            self.a.fill_(float(self.parameters["a"]))
            self._check()

    def recheck(self) -> None:
        """``max_force`` and ``converged`` at the current ``x`` against the current ``fmax``, the state kept: what
        ASE's ``run`` checks first when it is called again with another ``fmax``."""
        with torch.no_grad():
            self._check()

    def update(self) -> None:
        with torch.no_grad():
            self._check()

    @property
    def converged(self) -> torch.Tensor:
        return self.converged_flag

    def max_force(self) -> torch.Tensor:
        return self.max_force_value

    def propose(self) -> None:
        p = self.parameters
        with torch.no_grad():
            done = self.converged_flag
            gradient = -self.dof.gradient()  # ASE's "gradient": the forces
            x = self.dof.x
            vel, dt, a, nsteps, started = self.vel, self.dt, self.a, self.nsteps, self.started
            here = x
            if self.downhill_check:
                # ASE resets to r_last when the energy rose; its recomputed energy there is e_last. The step from
                # r_last still uses the gradient at the uphill positions, as ASE's.
                energy = self.dof.energy()
                uphill = started & (energy > self.e_last)[0]
                here = torch.where(uphill, self.r_last, x)
                self.e_last.copy_(torch.where(done | uphill, self.e_last, energy))
                self.r_last.copy_(torch.where(done, self.r_last, here))
            else:
                uphill = torch.zeros_like(started)
            vf = torch.dot(gradient, vel)
            grad2 = torch.dot(gradient, gradient)
            mix = started & (vf > 0.0) & ~uphill
            reset = started & ~mix
            grow = mix & (nsteps > p["Nmin"])
            safe_grad2 = torch.where(grad2 > 0.0, grad2, torch.ones_like(grad2))  # the unselected branch stays finite
            mixed = (1.0 - a) * vel + a * gradient / torch.sqrt(safe_grad2) * torch.sqrt(torch.dot(vel, vel))
            new_vel = torch.where(mix, mixed, torch.where(reset, vel * 0.0, vel))
            new_dt = torch.where(grow, torch.clamp(dt * p["finc"], max=p["dtmax"]),
                                 torch.where(reset, dt * p["fdec"], dt))
            new_a = torch.where(grow, a * p["fa"], torch.where(reset, torch.full_like(a, p["astart"]), a))
            new_nsteps = torch.where(mix, nsteps + 1, torch.where(reset, torch.zeros_like(nsteps), nsteps))
            new_vel = new_vel + new_dt * gradient
            dr = new_dt * new_vel
            normdr = torch.sqrt(torch.dot(dr, dr))
            maxstep = p["maxstep_A"]
            clip = normdr > maxstep
            dr = torch.where(clip, maxstep * dr / torch.where(clip, normdr, torch.ones_like(normdr)), dr)
            moved = torch.where(done, x, here + dr)
            self.vel.copy_(torch.where(done, vel, new_vel))
            self.dt.copy_(torch.where(done, dt, new_dt))
            self.a.copy_(torch.where(done, a, new_a))
            self.nsteps.copy_(torch.where(done, nsteps, new_nsteps))
            self.started.copy_(started | ~done)
            self.restarts.add_((reset & ~done).to(torch.int64))
            self.dof.set_x(moved)

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.BUFFERS}

    def state_dict(self) -> dict[str, Any]:
        record = {"vel": self.vel.detach().cpu().numpy().copy(), "dt": float(self.dt), "a": float(self.a),
                  "Nsteps": int(self.nsteps), "started": bool(self.started), "restarts": int(self.restarts),
                  "fmax": float(self.fmax),
                  "max_force": float(self.max_force_value), "converged": bool(self.converged_flag)}
        if self.downhill_check:
            record.update(e_last=float(self.e_last[0]), r_last=self.r_last.detach().cpu().numpy().copy())
        return record

    # ------------------------------------------------------------------ the convergence check

    def _check(self) -> None:
        # ASE's converged: the largest row norm of the constraint-adjusted forces below fmax
        self.max_force_value.copy_(self.dof.max_force())
        self.converged_flag.copy_(self.max_force_value < self.fmax)
        self.status.copy_(torch.stack((self.converged_flag.to(torch.float64), self.max_force_value,
                                       self.dof.energy().reshape(()).to(torch.float64))))


__all__ = ["FIRE"]
