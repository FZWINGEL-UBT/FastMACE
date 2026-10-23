"""ASE 3.27's ``IsotropicMTKNPT`` and ``MTKNPT`` as graph-capturable half steps on a SimulationState (SPEC §10.2).

Martyna-Tobias-Klein NPT (Martyna, Tobias and Klein 1994) with Nose-Hoover chains on the atoms and on the
cell, as ASE integrates it (``ase/md/nose_hoover_chain.py``). ASE's step

    barostat chain, thermostat chain, cell kick, atom kick (dt/2 each); atom drift, cell drift (dt);
    atom kick, cell kick, thermostat chain, barostat chain (dt/2 each)

is cut after the cell drift, where the force field evaluates forces and stress at the new positions and
cell: ``first_half`` up to the drifts, ``second_half`` from the second kicks. The first cell kick reads
the stress of the previous force call (ASE's calculator cache gives exactly that) plus the ideal-gas term
of the current momenta, which ASE adds through ``get_stress(include_ideal_gas=True)``; this port computes
that term on the device. The force field needs ``compute_stress``, and the engine sets the new cell after
``first_half`` (``variable_cell``), as for BAOAB.

``cell_mode="hydrostatic"`` is ``IsotropicMTKNPT``: the strain ``eps`` with ``cell = cell0 exp(eps)`` and
``V = V0 exp(3 eps)``, positions not scaled with the cell (they evolve by ASE's exp/exprel drift).
``"full"`` is ``MTKNPT``: the 3 x 3 cell momentum ``p_g``. ASE diagonalises ``p_g`` (``np.linalg.eigh``)
and applies exp and exprel to its eigenvalues; every use is a matrix function U f(L) U^T of the symmetric
``p_g``, so this port evaluates exp and exprel of the matrix directly (``exp_and_exprel``: scaling and
squaring of a 6 x 6 block Taylor series, no LAPACK, CUDA-graph safe, no eigenvector ambiguity at the
degenerate ``p_g = 0`` of the start). The difference from ASE is rounding.

Degrees of freedom (owner decision D2, conductor's answer of 23 October 2026): ASE counts 3N (as N) in
every coupling term: the thermostat mass, force and energy, the drag ``1 + 1/N`` (``1 + d/N_f``), the
kinetic term of the cell force, the barostat mass W and, in the full cell, the trace term of the drag.
``N_f = degrees_of_freedom(state)`` replaces 3N in all of them consistently: 3N - 3 under the ``fixcm``
default, 3N with ``dof="ase"`` or ``fixcm=False``, bitwise ASE's arithmetic then. The barostat chain's own
counts (9 and 1) are ASE's.

ASE 3.27's inconsistencies, ported as they are (decision 6: agreement with the reference is the
acceptance):
- ``IsotropicMTKBarostat`` gives the first barostat-chain mass 9 kT pdamp**2 (nine cell degrees of
  freedom) but its chain force targets 1 kT and its energy counts 1 kT per link: one degree of freedom;
- ``MTKBarostat`` counts 9 cell degrees of freedom however the cell is constrained (its own TODO);
- the conserved energies use the target kT, which ASE never changes; with a schedule they are not
  conserved (ASE has no schedule).
Masses Q, R and W come from the construction temperature (conductor's choice 4); a schedule moves only the
target kT, and the pressure target only the pressure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import units

from fastmace.config import IntegratorConfig
from fastmace.neighbors.skin import det_3x3, times_3x3

from .base import CONSERVED_KEY, Integrator
from .langevin import base_output_values, check_ensemble
from .nose_hoover import NoseHooverChain, mass_temperature, remove_momentum, require, thermostat_masses, twice_kinetic

if TYPE_CHECKING:
    from fastmace.state import SimulationState

EXPM_SCALING = 4   # exp(M) = exp(M / 2^s)^(2^s) on the 6 x 6 block matrix; with the Taylor degree below
EXPM_DEGREE = 16   # exact in float64 for |M| <= 16, while a barostat step moves |p_g dt / W| ~ 1e-4


def exp_and_exprel(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """exp(A) and exprel(A) = (exp(A) - I) A^-1 = sum_k A^k / (k + 1)! of a 3 x 3 tensor.

    exp([[A, I], [0, 0]]) = [[exp(A), exprel(A)], [0, I]]: one scaling-and-squaring Taylor series of the block
    matrix gives both. For a symmetric A they equal SciPy's ``U f(L) U^T`` (ASE's) to rounding.
    """
    n = a.shape[0]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    block = torch.zeros((2 * n, 2 * n), dtype=a.dtype, device=a.device)
    block[:n, :n] = a
    block[:n, n:] = eye
    block = block / 2 ** EXPM_SCALING
    big = torch.eye(2 * n, dtype=a.dtype, device=a.device)
    result, term = big.clone(), big
    for k in range(1, EXPM_DEGREE + 1):
        term = term @ block / k
        result = result + term
    for _ in range(EXPM_SCALING):
        result = result @ result
    return result[:n, :n], result[:n, n:]


def exprel(x: torch.Tensor) -> torch.Tensor:
    """scipy.special.exprel of a device scalar: 1 for |x| < 1e-16, else expm1(x) / x (SciPy's branches)."""
    return torch.where(x.abs() < 1e-16, torch.ones_like(x), torch.expm1(x) / x)


def kinetic_tensor(state: SimulationState) -> torch.Tensor:
    """sum_i m_i v_i v_i^T: -V times ASE's ideal-gas stress (``Atoms.get_kinetic_stress``)."""
    v = state.velocities
    mv = state.masses * v
    return torch.stack([(v[:, a:a + 1] * mv).sum(dim=0) for a in range(3)])


def symmetric_upper(stress: torch.Tensor) -> torch.Tensor:
    """The upper triangle mirrored: what ASE's Voigt round trip (``full_3x3_to_voigt_6_stress``) makes of a stress."""
    upper = torch.triu(stress)
    return upper + torch.triu(stress, diagonal=1).T


class MTK(Integrator):
    """``IsotropicMTKNPT`` (``cell_mode="hydrostatic"``) and ``MTKNPT`` (``"full"``) with the state on the device.

    ``config.parameters``: ``tdamp_fs``, ``pdamp_fs`` (ASE's ``tdamp``, ``pdamp``), ``tchain``, ``pchain``,
    ``tloop``, ``ploop`` (ASE's defaults 3, 3, 1, 1). The pressure is ``config.pressure_GPa`` (ASE's
    ``pressure_au`` in GPa): MTK takes a scalar pressure, so ``external_stress_GPa`` is refused. Q, R and W
    come from ``config.temperature_K`` (else the ``temperature_K`` keyword) and are restart state.
    """

    def __init__(self, state: SimulationState, config: IntegratorConfig | None = None, *,
                 temperature_K: float | None = None):
        if config is None:
            raise ValueError("MTK needs an IntegratorConfig (scheme 'mtk') with its parameters")
        check_ensemble(config, "npt", "MTK")
        if state.stress is None:
            raise ValueError("NPT needs a state with stress (a force field with compute_stress=True)")
        if config.external_stress_GPa is not None:
            raise ValueError("MTK takes a scalar pressure (ASE's pressure_au, pressure_GPa here), not an external "
                             f"stress; got external_stress_GPa={config.external_stress_GPa!r}")
        self.config = config
        self.hydrostatic = config.cell_mode == "hydrostatic"
        self.dt = config.timestep_fs * units.fs
        self.tdamp = require(config, "tdamp_fs") * units.fs
        self.pdamp = require(config, "pdamp_fs") * units.fs
        self.tchain, self.pchain, tloop, ploop = (int(config.parameter(name, default)) for name, default in (
            ("tchain", 3), ("pchain", 3), ("tloop", 1), ("ploop", 1)))
        if min(self.tchain, self.pchain, tloop, ploop) < 1:
            raise ValueError("tchain, pchain, tloop and ploop must be at least 1")
        self.fixcm = config.fixcm
        self.ndof = self.degrees_of_freedom(state)
        self.n_eff = self.ndof / 3  # ASE's N in 1 + 1/N and sum(p**2/m) / N: exactly N when ndof = 3N
        self.mass_temperature_K = mass_temperature(config, temperature_K)
        device = state.positions.device
        f64 = dict(dtype=torch.float64, device=device)

        self.kT = torch.zeros((), **f64)
        self.kT_step = torch.zeros((), **f64)        # this step's targets: stage A copies, stage B reads
        self.pressure = torch.zeros((), **f64)       # eV / A^3
        self.pressure_step = torch.zeros((), **f64)
        self.thermostat = NoseHooverChain(self.tchain, tloop, [0.0] * self.tchain, first_dof=self.ndof,
                                          energy_dof=self.ndof, device=device)
        cell_dof = 1 if self.hydrostatic else 9
        self.barostat = NoseHooverChain(self.pchain, ploop, [0.0] * self.pchain, first_dof=cell_dof,
                                        energy_dof=cell_dof, device=device)
        self._masses()
        if self.hydrostatic:
            self.p_cell = torch.zeros((), **f64)
            self.eps = torch.zeros((), **f64)
            self.cell0 = state.cell.detach().clone()
            self.volume0 = torch.zeros((), **f64)
            self._volume0()
        else:
            self.p_cell = torch.zeros((3, 3), **f64)
        self.eye = torch.eye(3, **f64)
        self.set_temperature(self.mass_temperature_K if temperature_K is None else temperature_K)
        self.set_pressure(config.pressure_GPa)
        self.kT_step.copy_(self.kT)
        self.pressure_step.copy_(self.pressure)

    def _masses(self) -> None:
        """Q, R and W from the construction temperature (ASE's thermostat and barostat constructors)."""
        kT = units.kB * self.mass_temperature_K
        self.thermostat.set_masses(thermostat_masses(self.ndof, kT, self.tdamp, self.tchain))
        cell_dof = 9  # ASE: 9 in both barostats' first mass (IsotropicMTKBarostat's chain force counts 1)
        self.barostat.set_masses([cell_dof * kT * self.pdamp ** 2] + [kT * self.pdamp ** 2] * (self.pchain - 1))
        if self.hydrostatic:
            self.W = (self.ndof + 3) * kT * self.pdamp ** 2             # ASE: (3 N + 3) kT pdamp**2
        else:
            self.W = (self.n_eff + 1) * kT * self.pdamp ** 2            # ASE: (N + 1) kT pdamp**2

    def _volume0(self) -> None:
        cell0 = self.cell0.detach().cpu().numpy()
        self.volume0.fill_(float(np.abs(np.linalg.det(cell0))))  # ASE's Atoms.get_volume

    @property
    def variable_cell(self) -> bool:
        return True

    def set_temperature(self, temperature_K: float) -> None:
        self.temperature_K = float(temperature_K)
        self.kT.fill_(units.kB * temperature_K)

    def set_pressure(self, pressure_GPa: float) -> None:
        """Fill the pressure target (GPa) for the next step: ASE's ``pressure_au`` replaced between steps."""
        if (isinstance(pressure_GPa, bool) or not isinstance(pressure_GPa, (int, float))
                or not np.isfinite(pressure_GPa)):
            raise ValueError(f"pressure_GPa must be a finite number of GPa, got {pressure_GPa!r}")
        self.pressure_GPa = float(pressure_GPa)
        self.pressure.fill_(pressure_GPa * units.GPa)

    def set_external_stress(self, stress_GPa: Any) -> None:
        """A scalar external stress (GPa, ASE sign) is the pressure ``-stress_GPa``; MTK takes no other."""
        if isinstance(stress_GPa, bool) or not isinstance(stress_GPa, (int, float)):
            raise ValueError(f"MTK takes a scalar pressure, not the external stress {stress_GPa!r}")
        self.set_pressure(-stress_GPa)

    def prepare_step(self, state: SimulationState) -> None:
        """Nothing: the chains are deterministic."""

    def start(self, state: SimulationState) -> None:
        """D2: remove the total momentum at the start of a run (see ``NoseHooverChainNVT.start``)."""
        if self.fixcm and state.step == 0:
            remove_momentum(state)

    # -- ASE's operators --------------------------------------------------------------------------------

    def _barostat_half(self):
        if self.hydrostatic:
            force = lambda: self.p_cell ** 2 / self.W                      # noqa: E731
        else:
            force = lambda: (self.p_cell * self.p_cell).sum() / self.W     # noqa: E731  trace(p_g^T p_g) / W
        self.barostat.integrate(self.dt / 2, self.kT_step, force, self.p_cell.mul_)

    def _thermostat_half(self, state):
        self.thermostat.integrate(self.dt / 2, self.kT_step, lambda: twice_kinetic(state), state.velocities.mul_)

    def _volume(self, state):
        """ASE's ``_get_volume``: V0 exp(3 eps) (isotropic), |det h| (full)."""
        if self.hydrostatic:
            return self.volume0 * torch.exp(3 * self.eps)
        return det_3x3(state.cell).abs()

    def _kick_cell(self, state, delta):
        """exp(i L_(eps,2) delta): the cell momentum from the stress with the ideal-gas term."""
        kinetic = kinetic_tensor(state)
        cell_volume = det_3x3(state.cell).abs()          # Atoms.get_volume in the ideal-gas term
        volume = self._volume(state)
        if self.hydrostatic:
            pressure = -(torch.diagonal(state.stress).sum() - torch.diagonal(kinetic).sum() / cell_volume) / 3
            G = 3 * volume * (pressure - self.pressure_step) + twice_kinetic(state) / self.n_eff
        else:
            stress = -(symmetric_upper(state.stress) - kinetic / cell_volume)   # MTKNPT._get_stress
            G = (volume * (stress - self.pressure_step * self.eye)
                 + twice_kinetic(state) / self.ndof * self.eye)
        self.p_cell.add_(delta * G)

    def _kick(self, state, delta):
        """exp(i L_2 delta): the forces with the cell momentum's drag."""
        v, force = state.velocities, delta * state.forces / state.masses
        if self.hydrostatic:
            x = (1 + 1 / self.n_eff) * self.p_cell * delta / self.W
            v.copy_(v * torch.exp(-x) + force * exprel(-x))
        else:
            kappa = self.p_cell + torch.diagonal(self.p_cell).sum() / self.ndof * self.eye
            decay, rel = exp_and_exprel(-kappa * delta / self.W)
            v.copy_(times_3x3(v, decay) + times_3x3(force, rel))

    def _drift(self, state, delta):
        """exp(i L_1 delta) then exp(i L_(eps,1) delta): positions, then the cell, with the same momentum."""
        x, v = state.positions, state.velocities
        if self.hydrostatic:
            rate = delta * self.p_cell / self.W
            x.copy_(x * torch.exp(rate) + v * delta * exprel(rate))
            self.eps.add_(rate)
            state.cell.copy_(self.cell0 * torch.exp(self.eps))
        else:
            grow, rel = exp_and_exprel(self.p_cell * delta / self.W)
            x.copy_(times_3x3(x, grow) + times_3x3(v * delta, rel))
            state.cell.copy_(state.cell @ grow)

    def first_half(self, state: SimulationState) -> None:
        dt2 = self.dt / 2
        self.kT_step.copy_(self.kT)
        self.pressure_step.copy_(self.pressure)
        self._barostat_half()
        self._thermostat_half(state)
        self._kick_cell(state, dt2)
        self._kick(state, dt2)
        self._drift(state, self.dt)

    def second_half(self, state: SimulationState) -> None:
        dt2 = self.dt / 2
        self._kick(state, dt2)
        self._kick_cell(state, dt2)
        self._thermostat_half(state)
        self._barostat_half()
        if self.fixcm:
            remove_momentum(state)

    # -- the contract's bookkeeping ---------------------------------------------------------------------

    def mid_step_tensors(self, state: SimulationState) -> tuple[torch.Tensor, ...]:
        return (state.velocities, self.thermostat.eta, self.thermostat.p_eta, self.barostat.eta,
                self.barostat.p_eta, self.p_cell)

    def buffers(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        named = dict(positions=state.positions, velocities=state.velocities, cell=state.cell,
                     thermostat_eta=self.thermostat.eta, thermostat_p_eta=self.thermostat.p_eta,
                     barostat_xi=self.barostat.eta, barostat_p_xi=self.barostat.p_eta, p_cell=self.p_cell,
                     kT=self.kT, kT_step=self.kT_step, pressure=self.pressure, pressure_step=self.pressure_step)
        if self.hydrostatic:
            named.update(eps=self.eps, cell0=self.cell0, volume0=self.volume0)
        return named

    def volume(self, state: SimulationState) -> torch.Tensor:
        return det_3x3(state.cell)

    def output_values(self, state: SimulationState) -> Mapping[str, torch.Tensor]:
        values = base_output_values(state)
        values.update(cell=state.cell, stress=state.stress)
        if self.hydrostatic:
            cell_kinetic = self.p_cell * self.p_cell / (2 * self.W)
        else:
            cell_kinetic = (self.p_cell * self.p_cell).sum() / (2 * self.W)
        values[CONSERVED_KEY] = (state.energy.reshape(-1)[0] + values["kinetic"][0]
                                 + self.thermostat.energy(self.kT_step) + self.barostat.energy(self.kT_step)
                                 + cell_kinetic + self.pressure_step * self._volume(state)).reshape(1)
        return values

    def _keys(self) -> set[str]:
        keys = {"eta", "p_eta", "xi", "p_xi", "p_cell", "pressure_au", "mass_temperature_K"}
        return keys | {"eps", "cell0"} if self.hydrostatic else keys

    def state_dict(self) -> dict[str, Any]:
        """The chains, the cell momentum, the pressure target and the masses' temperature (isotropic: the strain
        and the reference cell too): the record's ``state_keys`` of this cell mode."""
        host = lambda tensor: tensor.detach().cpu().numpy().copy()  # noqa: E731
        thermostat, barostat = self.thermostat.state_dict(), self.barostat.state_dict()
        saved = dict(eta=thermostat["eta"], p_eta=thermostat["p_eta"], xi=barostat["eta"], p_xi=barostat["p_eta"],
                     p_cell=host(self.p_cell), pressure_au=host(self.pressure),
                     mass_temperature_K=np.asarray(self.mass_temperature_K))
        if self.hydrostatic:
            saved.update(eps=host(self.eps), cell0=host(self.cell0))
        return saved

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if set(state_dict) != self._keys():
            raise ValueError(f"MTK's {self.config.cell_mode} state is {sorted(self._keys())}, got {sorted(state_dict)}")
        self.mass_temperature_K = float(np.asarray(state_dict["mass_temperature_K"]))
        self._masses()
        self.thermostat.load(state_dict["eta"], state_dict["p_eta"])
        self.barostat.load(state_dict["xi"], state_dict["p_xi"])
        tensors = [("p_cell", self.p_cell), ("pressure_au", self.pressure)]
        if self.hydrostatic:
            tensors += [("eps", self.eps), ("cell0", self.cell0)]
        for name, target in tensors:
            source = torch.from_numpy(np.array(state_dict[name], dtype=np.float64))
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"{name} has shape {tuple(source.shape)}, this cell mode needs {tuple(target.shape)}")
            target.copy_(source)
        self.pressure_GPa = float(self.pressure) / units.GPa
        if self.hydrostatic:
            self._volume0()
        self.kT_step.copy_(self.kT)
        self.pressure_step.copy_(self.pressure)


__all__ = ["MTK", "exp_and_exprel", "exprel"]
