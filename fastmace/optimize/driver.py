"""The optimiser driver: relaxations on the graph engine (SPEC §10.3; plan task 4-T05, conductor's choice 5).

``OptimizerStepper`` presents an optimiser to ``GraphEngine`` in the place of an integrator, split at the force
call as the integrators are: stage A is ``propose`` (the step, the new positions), then the engine's neighbour
check and compaction, stage B the force call, ``dof.refresh`` and ``update`` (history, max force, the device
``converged`` flag). Every engine tier runs it, since the compiled and reference tiers are uncaptured
``GraphEngine``\\ s.

One optimiser step is one ``engine.step()``, in checked mode only. Its single host transfer carries the
optimiser's ``status`` (converged, max force, energy of the committed state) with the neighbour flags. A
step that finds ``converged`` set stops there (``engine.halted``): ``propose`` left the positions unchanged
(the converged no-op of ``optimize.lbfgs``), and no force call runs. So a relaxation costs one force call
per step plus the start, as ASE's, and no extra synchronisation.

``relax`` is ASE's ``Optimizer.run(fmax, steps)``: convergence is checked first, then at most ``steps``
steps, each followed by the check. It returns the summary ``protocols.quench.relax`` records, plus the
per-step energies and max forces and the engine's rebuild and re-capture counts.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from fastmace.integrators.base import Integrator


class OptimizerStepper(Integrator):
    """An optimiser in the integrator's place in ``GraphEngine`` (``first_half`` = ``propose``, ``second_half`` =
    ``dof.refresh`` and ``update``). No noise, no thermostat, nothing saved mid-step."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.dof = optimizer.dof
        self.config = optimizer.config

    @property
    def variable_cell(self) -> bool:
        return self.optimizer.variable_cell

    @property
    def status(self):
        """The optimiser's device ``[converged, max force, energy]`` (float64), copied to the host each step."""
        return self.optimizer.status

    def set_temperature(self, temperature_K: float) -> None:
        raise NotImplementedError("an optimiser has no temperature target")

    def prepare_step(self, state) -> None:
        """No noise to load."""

    def first_half(self, state) -> None:
        self.optimizer.propose()

    def second_half(self, state) -> None:
        self.dof.refresh()
        self.optimizer.update()

    def start(self, state) -> None:
        # A fresh relaxation at a canonical point (a build or a host write): the cell filter's reference is the
        # current cell, as a new engine's would be (relax-throughput A-T01).
        reset = getattr(self.dof, "reset_reference", None)
        if reset is not None:
            reset()
        self.dof.refresh()  # the cell filter's x and gradient at the first (or re-evaluated) forces, 4-T07
        self.optimizer.start()

    def mid_step_tensors(self, state) -> tuple:
        return ()

    def buffers(self, state) -> Mapping[str, Any]:
        return {}  # the engine registers the optimiser's (``optimizer.``) and the DOF view's (``dof.``)

    def output_values(self, state) -> Mapping[str, Any]:
        return {**self.optimizer.output_values(), "forces": state.forces, "momenta": state.momenta}

    def state_dict(self) -> dict[str, Any]:
        return {}  # no optimiser restart files (conductor's choice 5)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict:
            raise ValueError("optimisers keep no restart state (conductor's choice 5)")


def relax(engine, *, fmax: float, steps: int) -> dict[str, Any]:
    """Relax ``engine``'s state with its optimiser to ``fmax`` (eV/Angstrom) in at most ``steps`` steps.

    ASE's ``run``: a converged start takes no step; otherwise a step at a time until the largest
    constraint-adjusted force is below ``fmax`` or ``steps`` steps are done. Called again (another ``fmax``,
    more steps), it continues with the history, as a second ``run`` of one ASE optimiser does.
    """
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None:
        raise TypeError("relax needs an engine built with an optimiser (OptimizerStepper)")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError(f"steps must be a nonnegative int, got {steps!r}")
    natoms = engine.state.natoms
    started = time.perf_counter()
    rebuilds, recaptures = engine.rebuilds, engine.recaptures
    optimizer.set_fmax(fmax)
    optimizer.recheck()
    converged, force, energy = engine.read_optimizer_status()  # host sync
    _check_finite(optimizer, force)
    energies, forces = [energy], [force]
    taken = 0
    while not converged and taken < steps:
        engine.step()
        if taken:  # the status this step read is the state before it; the start's is recorded already
            _, force, energy = engine.optimizer_status()
            _check_finite(optimizer, force)
            energies.append(energy)
            forces.append(force)
        if engine.halted:
            converged = True
            break
        taken += 1
    else:
        if taken:  # stopped by the step limit (or a converged last step): its state has not been read
            converged, force, energy = engine.read_optimizer_status()
            _check_finite(optimizer, force)
            energies.append(energy)
            forces.append(force)
    record = engine.integrator.config
    summary = {"optimizer": record.scheme, "fixed_cell": not optimizer.variable_cell, "fmax_target_eV_A": float(fmax),
               "max_steps": int(steps), "converged": bool(converged), "steps": taken,
               "seconds": time.perf_counter() - started, "initial_eV_atom": energies[0] / natoms,
               "energy_eV_atom": energies[-1] / natoms, "fmax_eV_A": forces[-1], "energies_eV": energies,
               "fmax_history_eV_A": forces, "neighbour_rebuilds": engine.rebuilds - rebuilds,
               "recaptures": engine.recaptures - recaptures}
    if optimizer.variable_cell:  # 4-T07: the relaxed cell; with a scalar pressure the energies are E + p V (ASE's)
        import numpy as np
        from ase import units

        state = engine.state
        cell = state.cell.detach().cpu().numpy()
        summary.update({"cell_A": cell.tolist(), "volume_A3": abs(float(np.linalg.det(cell))),
                        "stress_GPa": (state.stress.detach().cpu().numpy() / units.GPa).tolist(),
                        "scalar_pressure_GPa": float(record.cell_filter_parameter("scalar_pressure_GPa", 0.0))})
    return summary


def _check_finite(optimizer, force: float) -> None:
    """A NaN max force stops the relaxation with the DOF view's reason (the cell filter's range guard)."""
    if not math.isfinite(force):
        reason = getattr(optimizer.dof, "failure_reason", lambda: "a non-finite force or gradient")()
        raise FloatingPointError(f"the relaxation stopped: {reason}")


__all__ = ["OptimizerStepper", "relax"]
