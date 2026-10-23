"""ASE 3.27's ``FIRE`` on a ``DOFView``, graph-capturable (SPEC §10.3; plan task 4-T06).

ASE's ``FIRE.step`` branches on host floats: whether ``v.f > 0``, whether ``Nsteps > Nmin``, whether the step
exceeds ``maxstep``, and with ``downhill_check`` whether the energy rose. Here every branch is computed and
selected with ``torch.where`` on device booleans, so a step has no host sync and replays from a CUDA graph.
ASE's operation order is kept, so on float64 the only difference from ASE is the summation order of the dot
products.

ASE's state maps as: ``vel`` (``None`` before the first step: ``started`` False and ``vel`` zero), ``dt``,
``a``, ``Nsteps`` (``nsteps``), and with ``downhill_check`` ``e_last`` and ``r_last``. ASE's ``vel_last`` is
written but never read, so it is not kept. All of it is float64 (conductor's choice 5).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from fastmace.optimize.base import Optimizer

# fm.optimize.FIRE's fields, the parameters an OptimizerConfig for "fire" may carry.
_REFUSED = {
    "position_reset_callback": "FIRE's position_reset_callback is a host callback inside the step, which a "
                               "captured step cannot make (plan task 4-T06)",
    "maxmove": "FIRE's maxmove is ASE's deprecated name; use maxstep_A",
}


def _defaults() -> dict[str, Any]:
    from fastmace.optimize import FIRE as record

    return {field.name: field.default for field in dataclasses.fields(record) if field.name != "cell_filter"}


class FIRE(Optimizer):
    """ASE 3.27 ``FIRE(dt, maxstep, dtmax, Nmin, finc, fdec, astart, fa, a, downhill_check)`` over ``dof``."""

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
        self.e_last = torch.zeros(1, **f64)
        self.r_last = torch.zeros_like(x)
        self.fmax = torch.tensor(float(config.fmax), **f64)
        self.max_force_value = torch.zeros((), **f64)
        self.converged_flag = torch.zeros((), dtype=torch.bool, device=x.device)

    def set_fmax(self, fmax: float) -> None:
        self.fmax.fill_(float(fmax))

    def start(self) -> None:
        self._check()

    def update(self) -> None:
        self._check()

    def _check(self) -> None:
        # ASE's converged: the largest row norm of the constraint-adjusted forces below fmax
        self.max_force_value.copy_(self.dof.max_force())
        torch.lt(self.max_force_value, self.fmax, out=self.converged_flag)

    @property
    def converged(self) -> torch.Tensor:
        return self.converged_flag

    def max_force(self) -> torch.Tensor:
        return self.max_force_value

    def propose(self) -> None:
        p = self.parameters
        gradient = -self.dof.gradient()  # ASE's "gradient": the forces
        x = self.dof.x
        vel, dt, a, started = self.vel, self.dt, self.a, self.started
        if self.downhill_check:
            # ASE resets to r_last when the energy rose; its recomputed energy there is e_last. The step from
            # r_last still uses the gradient at the uphill positions, as ASE's.
            energy = self.dof.energy()
            uphill = started & (energy > self.e_last)[0]
            x = torch.where(uphill, self.r_last, x)
            self.e_last.copy_(torch.where(uphill, self.e_last, energy))
            self.r_last.copy_(x)
        else:
            uphill = torch.zeros_like(started)
        vf = torch.dot(gradient, vel)
        grad2 = torch.dot(gradient, gradient)
        mix = started & (vf > 0.0) & ~uphill
        reset = started & ~mix
        grow = mix & (self.nsteps > p["Nmin"])
        safe_grad2 = torch.where(grad2 > 0.0, grad2, torch.ones_like(grad2))  # the unselected branch stays finite
        mixed = (1.0 - a) * vel + a * gradient / torch.sqrt(safe_grad2) * torch.sqrt(torch.dot(vel, vel))
        vel = torch.where(mix, mixed, torch.where(reset, vel * 0.0, vel))
        new_dt = torch.where(grow, torch.clamp(dt * p["finc"], max=p["dtmax"]),
                             torch.where(reset, dt * p["fdec"], dt))
        self.a.copy_(torch.where(grow, a * p["fa"], torch.where(reset, torch.full_like(a, p["astart"]), a)))
        self.nsteps.copy_(torch.where(mix, self.nsteps + 1, torch.where(reset, torch.zeros_like(self.nsteps),
                                                                            self.nsteps)))
        self.dt.copy_(new_dt)
        vel = vel + new_dt * gradient
        self.vel.copy_(vel)
        dr = new_dt * vel
        normdr = torch.sqrt(torch.dot(dr, dr))
        maxstep = p["maxstep_A"]
        clip = normdr > maxstep
        dr = torch.where(clip, maxstep * dr / torch.where(clip, normdr, torch.ones_like(normdr)), dr)
        self.started.fill_(True)
        self.dof.set_x(x + dr)

    def buffers(self):
        names = ("vel", "dt", "a", "nsteps", "started", "e_last", "r_last", "fmax", "max_force_value",
                 "converged_flag")
        return {**{name: getattr(self, name) for name in names}, **self.dof.buffers()}

    def state_dict(self) -> dict[str, Any]:
        record = {"vel": self.vel.detach().cpu().numpy().copy(), "dt": float(self.dt), "a": float(self.a),
                  "Nsteps": int(self.nsteps), "started": bool(self.started)}
        if self.downhill_check:
            record.update(e_last=float(self.e_last[0]), r_last=self.r_last.detach().cpu().numpy().copy())
        return record
