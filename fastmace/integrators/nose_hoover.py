"""ASE 3.27's ``NoseHooverChainNVT`` as graph-capturable half steps on a SimulationState (SPEC §10.2).

ASE's step (``ase/md/nose_hoover_chain.py``) is a Trotter splitting of the Nose-Hoover chain equations
(Martyna, Klein and Tuckerman 1992; Tuckerman et al. 2006): the chain for dt/2, a half kick, a drift, a half
kick, the chain for dt/2. Each chain half is ``tloop`` times the three fourth-order Suzuki-Yoshida
sub-steps, and each sub-step the chain loop down and up. The cut at the force evaluation falls between the
drift and the second kick, so ``first_half`` is the chain, kick and drift, ``second_half`` the kick and the
chain. The chain loops are Python loops over device scalars: unrolled when the stages are captured, no host
sync. ``NoseHooverChain`` is ASE's ``NoseHooverChainThermostat`` and ``IsotropicMTKBarostat`` /
``MTKBarostat``'s chain in one class, which ``mtk.py`` uses for the barostat too.

ASE integrates momenta; this port integrates the state's velocities, the same algebra with per-atom masses
divided out (ULP-level differences). ``sum(p**2 / m)`` is recomputed wherever ASE recomputes it.

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
    """One Nose-Hoover chain of ``length`` thermostats on the device (ASE's chain loops).

    ``masses`` are host floats (Q or R), ``loops`` ASE's ``tloop`` or ``ploop``. ``first_dof`` is the count in
    the first link's force, ``sum(p**2/m) - first_dof kT`` for a thermostat, and in its energy term
    (``energy_dof``; ASE's isotropic barostat counts 1 there, though its mass counts 9). The coupled variable
    is read by ``force`` (its kinetic term, ``sum(p**2/m)``) and scaled by ``scale``, callables of the owner.
    """

    def __init__(self, length: int, loops: int, masses: list[float], *, first_dof: float, energy_dof: float,
                 device: Any):
        f64 = dict(dtype=torch.float64, device=device)
        self.length, self.loops = length, loops
        self.first_dof, self.energy_dof = first_dof, energy_dof
        self.eta = torch.zeros(length, **f64)
        self.p_eta = torch.zeros(length, **f64)
        self.mass_tensor = torch.zeros(length, **f64)
        self.set_masses(masses)
        self._p = [self.p_eta[j] for j in range(length)]

    def set_masses(self, masses: list[float]) -> None:
        self.masses = [float(mass) for mass in masses]
        self.mass_tensor.copy_(torch.tensor(self.masses, dtype=torch.float64))

    def integrate(self, delta: float, kT: torch.Tensor, force, scale) -> None:
        """exp(i L_NHC delta): ``loops`` times the three Suzuki-Yoshida sub-steps."""
        for _ in range(self.loops):
            for coeff in FOURTH_ORDER_COEFFS:
                self._loop(coeff * delta / self.loops, kT, force, scale)

    def _loop(self, delta, kT, force, scale):
        delta2 = delta / 2
        delta4 = delta / 4
        for j in range(self.length):
            self._integrate_p_eta_j(self.length - j - 1, delta2, delta4, kT, force)
        self.eta.add_(delta * self.p_eta / self.mass_tensor)
        scale(torch.exp(-delta * self._p[0] / self.masses[0]))
        for j in range(self.length):
            self._integrate_p_eta_j(j, delta2, delta4, kT, force)

    def _integrate_p_eta_j(self, j, delta2, delta4, kT, force):
        p, Q = self._p, self.masses
        # ASE evaluates the same factor before and after the kick (p[j + 1] does not change in between)
        damping = torch.exp(-delta4 * p[j + 1] / Q[j + 1]) if j < self.length - 1 else None
        if damping is not None:
            p[j].mul_(damping)
        if j == 0:
            g_j = force() - self.first_dof * kT
        else:
            g_j = p[j - 1] ** 2 / Q[j - 1] - kT
        p[j].add_(delta2 * g_j)
        if damping is not None:
            p[j].mul_(damping)

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
                 temperature_K: float | None = None):
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
        self.thermostat = NoseHooverChain(tchain, tloop, self._thermostat_masses(), first_dof=self.ndof,
                                          energy_dof=self.ndof, device=state.positions.device)
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
        v = state.velocities
        self.thermostat.integrate(self.dt / 2, self.kT_step, lambda: twice_kinetic(state), v.mul_)

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
                    thermostat_p_eta=self.thermostat.p_eta, kT=self.kT, kT_step=self.kT_step)

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
