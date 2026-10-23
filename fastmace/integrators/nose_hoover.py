"""ASE 3.27's ``NoseHooverChainNVT`` as graph-capturable half steps on a SimulationState (SPEC §10.2).

ASE's step (``ase/md/nose_hoover_chain.py``) is a Trotter splitting of the Nose-Hoover chain equations
(Martyna, Klein and Tuckerman 1992; Tuckerman et al. 2006): the chain for dt/2, a half kick, a drift, a half
kick, the chain for dt/2. Each chain half is ``tloop`` times the three fourth-order Suzuki-Yoshida
sub-steps, and each sub-step the chain loop down and up. The cut at the force evaluation falls between the
drift and the second kick, so ``first_half`` is the chain, kick and drift, ``second_half`` the kick and the
chain. ``NoseHooverChain`` is ASE's ``NoseHooverChainThermostat`` and ``IsotropicMTKBarostat`` /
``MTKBarostat``'s chain in one class, which ``mtk.py`` uses for the barostat too.

The chain half (plan task 4-T03b). ASE recomputes the chain force ``sum(p**2 / m)`` after every momentum
scaling and scales the momenta three times per half. Here the force is reduced once per half and carried
through each scaling s as ``F s**2``, and the momenta are scaled once by the product of the s: the same
algebra, rounding-level from ASE. Every constant (the sub-step deltas over the chain masses) is a host float
in a float64 table, so ``fastmace.kernels.triton.nose_hoover_chain`` runs the half as one Triton program
with bitwise the PyTorch stage's results. ``fused`` picks it: by default on CUDA with Triton. Without it, the
PyTorch stage runs the same arithmetic as Python loops over device scalars, unrolled at capture. No host
sync either way.

ASE integrates momenta; this port integrates the state's velocities, the same algebra with per-atom masses
divided out (ULP-level differences).

Degrees of freedom (owner decision D2): ASE counts 3N in the first thermostat mass, its force and its energy.
``N_f = degrees_of_freedom(state)`` replaces 3N there: 3N - 3 under the ``fixcm`` default (the total
momentum is removed at the start and projected out after every step against float drift), 3N with
``dof="ase"`` or ``fixcm=False``. The thermostat masses come from the construction temperature, as ASE's;
``set_temperature`` moves only the target kT (conductor's choice 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units

from fastmace.config import IntegratorConfig

from .base import CONSERVED_KEY, Integrator
from .langevin import base_output_values, check_ensemble

if TYPE_CHECKING:
    from fastmace.state import SimulationState

# The fourth-order Suzuki-Yoshida coefficients, ASE's expressions (Yoshida 1990).
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
]


def require(config: IntegratorConfig, name: str, kind: type = float) -> Any:
    """The scheme parameter ``name``: a number (``float``) or a positive int (``int``); ValueError if missing."""
    value = config.parameter(name)
    if value is None:
        raise ValueError(f"the {config.scheme} integrator needs the parameter {name!r} (no default, as in ASE)")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and not isinstance(value, int)):
        raise ValueError(f"{name} must be {'an int' if kind is int else 'a number'}, got {value!r}")
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return kind(value)


def mass_temperature(config: IntegratorConfig, temperature_K: float | None) -> float:
    """The temperature of the thermostat and barostat masses: the config's, else the starting target."""
    value = config.temperature_K if config.temperature_K is not None else temperature_K
    if value is None or not value > 0:
        raise ValueError(f"the {config.scheme} integrator needs a positive construction temperature for its "
                         f"thermostat masses (ASE's temperature_K), got {value!r}")
    return float(value)


def remove_momentum(state: SimulationState) -> None:
    """Subtract the centre-of-mass velocity: zero total momentum (D2's ``fixcm``)."""
    v, masses = state.velocities, state.masses
    v.sub_((masses * v).sum(dim=0) / masses.sum())


def twice_kinetic(state: SimulationState) -> torch.Tensor:
    """sum(p**2 / m), ASE's chain force on the first thermostat."""
    return (state.masses * state.velocities ** 2).sum()


class NoseHooverChain:
    """One Nose-Hoover chain of ``length`` thermostats on the device (ASE's chain loops, 4-T03b's algorithm).

    ``masses`` are host floats (Q or R), ``loops`` ASE's ``tloop`` or ``ploop``, ``delta`` the half step (dt / 2).
    ``first_dof`` is the count in the first link's force, ``F - first_dof kT`` with F the coupled kinetic term
    (``sum(p**2/m)`` for a thermostat), and ``energy_dof`` the count in its energy term (ASE's isotropic
    barostat counts 1 there, though its mass counts 9). ``integrate(kT, force)`` takes F as a device scalar
    and returns the device scalar by which the owner scales its momenta.
    """

    def __init__(self, length: int, loops: int, masses: list[float], *, delta: float, first_dof: float,
                 energy_dof: float, device: Any, fused: bool = False):
        f64 = dict(dtype=torch.float64, device=device)
        self.length, self.loops, self.delta = length, loops, delta
        self.first_dof, self.energy_dof = first_dof, energy_dof
        self.fused = fused
        self.eta = torch.zeros(length, **f64)
        self.p_eta = torch.zeros(length, **f64)
        self.mass_tensor = torch.zeros(length, **f64)
        self.inverse = torch.zeros(length, **f64)
        # Per Suzuki-Yoshida coefficient: delta/2, the scaling's -delta/Q0, delta/Q_j (eta), -delta/4 / Q_(j+1)
        self.table = torch.zeros((3, 2 + 2 * length), **f64)
        self.dof_tensor = torch.tensor(float(first_dof), **f64)
        self.scale = torch.zeros((), **f64)   # the fused kernel's output
        self.set_masses(masses)
        self._p = [self.p_eta[j] for j in range(length)]

    def set_masses(self, masses: list[float]) -> None:
        """Q (host floats) and every constant that follows from them, in place."""
        self.masses = [float(mass) for mass in masses]
        self.inv = [1.0 / mass for mass in self.masses]
        self.rows = []
        for coeff in FOURTH_ORDER_COEFFS:
            delta = coeff * self.delta / self.loops
            row = ([delta / 2, -delta * self.inv[0]] + [delta * inv for inv in self.inv]
                   + [-(delta / 4) * inv for inv in self.inv[1:]] + [0.0])
            self.rows.append(row)
        self.mass_tensor.copy_(torch.tensor(self.masses, dtype=torch.float64))
        self.inverse.copy_(torch.tensor(self.inv, dtype=torch.float64))
        self.table.copy_(torch.tensor(self.rows, dtype=torch.float64))

    def integrate(self, kT: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """exp(i L_NHC delta): ``loops`` times the three Suzuki-Yoshida sub-steps. Returns the momentum scale."""
        if self.fused:
            from fastmace.kernels.triton.nose_hoover_chain import chain_half

            chain_half(self.p_eta, self.eta, self.table, self.inverse, self.dof_tensor, kT, force, self.scale,
                       length=self.length, loops=self.loops)
            return self.scale
        F, S = force, torch.ones_like(force)
        for _ in range(self.loops):
            for k, row in enumerate(self.rows):
                d2, scale_c = row[0], row[1]
                damp_c = row[2 + self.length:]
                for j in range(self.length):
                    self._link(self.length - j - 1, d2, damp_c, F, kT)
                self.eta.add_(self.p_eta * self.table[k, 2:2 + self.length])
                s = torch.exp(self._p[0] * scale_c)
                F = F * s * s
                S = S * s
                for j in range(self.length):
                    self._link(j, d2, damp_c, F, kT)
        return S

    def _link(self, j, d2, damp_c, F, kT):
        """ASE's _integrate_p_eta_j: p[j] damped by p[j + 1], kicked by its force, damped again (ASE evaluates
        the same factor twice: p[j + 1] does not change in between)."""
        p = self._p
        damp = torch.exp(p[j + 1] * damp_c[j]) if j < self.length - 1 else None
        if damp is not None:
            p[j].mul_(damp)
        if j == 0:
            g = F - self.first_dof * kT
        else:
            g = p[j - 1] * p[j - 1] * self.inv[j - 1] - kT
        p[j].add_(g * d2)
        if damp is not None:
            p[j].mul_(damp)

    def energy(self, kT: torch.Tensor) -> torch.Tensor:
        """ASE's get_thermostat_energy / get_barostat_energy."""
        return (self.energy_dof * kT * self.eta[0] + kT * self.eta[1:].sum()
                + (0.5 * self.p_eta ** 2 / self.mass_tensor).sum())

    def state_dict(self) -> dict[str, np.ndarray]:
        return dict(eta=self.eta.detach().cpu().numpy().copy(), p_eta=self.p_eta.detach().cpu().numpy().copy())

    def load(self, eta: Any, p_eta: Any) -> None:
        for target, value, name in ((self.eta, eta, "eta"), (self.p_eta, p_eta, "p_eta")):
            source = torch.from_numpy(np.array(value, dtype=np.float64))
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"{name} has shape {tuple(source.shape)}, this chain has length {self.length}")
            target.copy_(source)


def use_fused(fused: bool | None, device: Any) -> bool:
    """The chain stage: ``fused=None`` picks the Triton kernels where they run (CUDA with Triton); ``True``
    requires them."""
    from fastmace.kernels.triton.nose_hoover_chain import available

    reason = available(torch.device(device))
    if fused is None:
        return reason is None
    if fused and reason is not None:
        raise ValueError(f"fused=True: {reason}")
    return bool(fused)


def thermostat_masses(ndof: int, kT: float, tdamp: float, tchain: int) -> list[float]:
    """NoseHooverChainThermostat's Q: N_f kT tdamp**2, then kT tdamp**2 (ASE: 3N for N_f)."""
    return [ndof * kT * tdamp ** 2] + [kT * tdamp ** 2] * (tchain - 1)


class NoseHooverChainNVT(Integrator):
    """``ase.md.nose_hoover_chain.NoseHooverChainNVT`` with the state on the device.

    ``config.parameters``: ``tdamp_fs`` (ASE's ``tdamp``), ``tchain`` and ``tloop`` (ASE's defaults 3 and 1).
    The chain masses Q come from ``config.temperature_K`` (else the ``temperature_K`` keyword), the
    construction temperature, and are restart state (``mass_temperature_K``), so a resumed run keeps them.

    Port of ASE 3.27, including where it is inconsistent: the conserved energy uses the target kT, which ASE
    never changes; with a schedule it is not conserved (ASE has no schedule).
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float | None = None, fused: bool | None = None):
        if config is None:
            raise ValueError("NoseHooverChainNVT needs an IntegratorConfig (scheme 'nhc') with its parameters")
        check_ensemble(config, "nvt", "NoseHooverChainNVT")
        self.config = config
        self.dt = config.timestep_fs * units.fs
        self.tdamp = require(config, "tdamp_fs") * units.fs
        tchain = int(config.parameter("tchain", 3))
        tloop = int(config.parameter("tloop", 1))
        if tchain < 1 or tloop < 1:
            raise ValueError(f"tchain and tloop must be at least 1, got {tchain}, {tloop}")
        self.fixcm = config.fixcm
        self.ndof = self.degrees_of_freedom(state)
        self.mass_temperature_K = mass_temperature(config, temperature_K)
        f64 = dict(dtype=torch.float64, device=state.positions.device)
        self.kT = torch.zeros((), **f64)
        self.kT_step = torch.zeros((), **f64)  # the target of the step in flight: stage A copies, stage B reads
        self.tchain = tchain
        self.fused = use_fused(fused, state.positions.device)
        self.thermostat = NoseHooverChain(tchain, tloop, self._thermostat_masses(), delta=self.dt / 2,
                                          first_dof=self.ndof, energy_dof=self.ndof, device=state.positions.device,
                                          fused=self.fused)
        self.set_temperature(self.mass_temperature_K if temperature_K is None else temperature_K)
        self.kT_step.copy_(self.kT)

    def _thermostat_masses(self) -> list[float]:
        return thermostat_masses(self.ndof, units.kB * self.mass_temperature_K, self.tdamp, self.tchain)

    @property
    def variable_cell(self) -> bool:
        return False

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = float(temperature_K)
        self.kT.fill_(units.kB * temperature_K)

    def prepare_step(self, state: SimulationState) -> None:
        """Nothing: the chain is deterministic."""

    def start(self, state: SimulationState) -> None:
        """D2: remove the total momentum at the start of a run (not on a resume, where the restart replaces the
        velocities, nor again after steps, where ``second_half`` keeps it at zero)."""
        if self.fixcm and state.step == 0:
            remove_momentum(state)

    def _thermostat_half(self, state):
        state.velocities.mul_(self.thermostat.integrate(self.kT_step, twice_kinetic(state)))

    def first_half(self, state: SimulationState) -> None:
        self.kT_step.copy_(self.kT)                      # this step's target
        self._thermostat_half(state)
        state.velocities.add_(self.dt / 2 * state.forces / state.masses)
        state.positions.add_(self.dt * state.velocities)

    def second_half(self, state: SimulationState) -> None:
        state.velocities.add_(self.dt / 2 * state.forces / state.masses)
        self._thermostat_half(state)
        if self.fixcm:
            remove_momentum(state)

    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        return (state.velocities, self.thermostat.eta, self.thermostat.p_eta)

    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        return dict(positions=state.positions, velocities=state.velocities, thermostat_eta=self.thermostat.eta,
                    thermostat_p_eta=self.thermostat.p_eta, thermostat_scale=self.thermostat.scale, kT=self.kT,
                    kT_step=self.kT_step)

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        values = base_output_values(state)
        values[CONSERVED_KEY] = (state.energy.reshape(-1)[0] + values["kinetic"][0]
                                 + self.thermostat.energy(self.kT_step)).reshape(1)
        return values

    def state_dict(self) -> dict[str, Any]:
        return {**self.thermostat.state_dict(), "mass_temperature_K": np.asarray(self.mass_temperature_K)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = {"eta", "p_eta", "mass_temperature_K"}
        if set(state_dict) != expected:
            raise ValueError(f"NoseHooverChainNVT's state is {sorted(expected)}, got {sorted(state_dict)}")
        self.mass_temperature_K = float(np.asarray(state_dict["mass_temperature_K"]))
        self.thermostat.set_masses(self._thermostat_masses())
        self.thermostat.load(state_dict["eta"], state_dict["p_eta"])
        self.kT_step.copy_(self.kT)


__all__ = ["FOURTH_ORDER_COEFFS", "NoseHooverChain", "NoseHooverChainNVT"]
