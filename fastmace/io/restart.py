"""The phase-1a restart file: today's ``restart.npz`` of the research engine (SPEC §10.6).

One ``.npz`` per run directory, written atomically (temporary file, then rename) at committed steps
after the output queue has been drained and the engine flushed. Its keys are the research engine's
(``workflow.write_restart`` with ``GPUEngine.state``), so either side reads the other's files:

- ``step``, ``engine`` (the engine name, checked on resume);
- ``positions``, ``momenta``, ``cell`` (the cell of the restart step);
- ``rng_state`` (the NumPy bit generator's state as JSON), ``noise`` (the batch of random numbers
  already drawn ahead) and ``noise_next`` (its cursor): a resumed run uses exactly the numbers of an
  uninterrupted one;
- ``forces``, ``energy``; variable-cell runs add ``cell_state``, ``stress`` and the integrator's
  ``state_dict`` (BAOAB: ``p_eps``, ``force_eps``).

FastMACE adds ``velocities`` (the integrated quantity: rebuilding it as momenta / masses, as the
research engine does on resume, rounds), ``masses`` and ``species`` (checked on load),
``integrator_keys`` and ``fastmace_restart`` (the format version).

Phase 1a stores only this state. The complete §10.6 state (force-field identity, plan, protocol,
MC state) and canonical restart points (pipeline quiesced, neighbour lists rebuilt) come in 1b, which
raises ``FORMAT_VERSION``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from fastmace.state import SimulationState

RESTART_NAME = "restart.npz"
FORMAT_VERSION = 1
# The integrator keys of research-engine files, which have no "integrator_keys" entry (GPUEngine.state).
_LEGACY_INTEGRATOR_KEYS = ("p_eps", "force_eps")
_STATE_KEYS = frozenset({
    "step", "engine", "positions", "momenta", "velocities", "masses", "species", "cell", "cell_state",
    "rng_state", "noise", "noise_next", "forces", "energy", "stress", "integrator_keys", "fastmace_restart",
})


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def state_arrays(state: SimulationState, engine: str, integrator_state: Mapping[str, Any] | None = None,
                 variable_cell: bool = False) -> dict[str, Any]:
    """The restart file's entries for ``state`` (host copies; the caller has flushed the engine)."""
    integrator_state = dict(integrator_state or {})
    clash = _STATE_KEYS & integrator_state.keys()
    if clash:
        raise ValueError(f"integrator state keys {sorted(clash)} clash with the restart file's state keys")
    arrays: dict[str, Any] = dict(
        step=state.step,
        positions=_numpy(state.positions),
        momenta=_numpy(state.velocities * state.masses),
        velocities=_numpy(state.velocities),
        masses=_numpy(state.masses),
        species=_numpy(state.species),
        cell=_numpy(state.cell),
        engine=engine,
        rng_state=json.dumps(state.rng.generator.bit_generator.state),
        noise_next=state.rng.cursor,
        forces=_numpy(state.forces),
        energy=_numpy(state.energy),
        fastmace_restart=FORMAT_VERSION,
    )
    if state.rng.batch is not None:
        arrays["noise"] = _numpy(state.rng.batch)
    if state.stress is not None:
        arrays["stress"] = _numpy(state.stress)
    if variable_cell:
        arrays["cell_state"] = _numpy(state.cell)
    arrays.update({key: np.asarray(value) for key, value in integrator_state.items()})
    arrays["integrator_keys"] = np.array(sorted(integrator_state), dtype=str)
    return arrays


def write_restart(path: str | Path, state: SimulationState, *, engine: str,
                  integrator_state: Mapping[str, Any] | None = None, variable_cell: bool = False) -> None:
    """Atomic restart file: write a temporary file, then rename it over the old one.

    Call at a committed step, after draining the output queue and flushing the engine.
    ``integrator_state`` is ``Integrator.state_dict()`` (host arrays); ``variable_cell`` stores the
    cell again as ``cell_state``, the research engine's NPT key.
    """
    path = Path(path)
    arrays = state_arrays(state, engine, integrator_state, variable_cell)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_restart(path: str | Path, engine: str | None = None) -> dict[str, np.ndarray]:
    """Every entry of a restart file, read into memory with the file closed.

    The next restart file replaces this one, which fails for a file still open on Windows drives
    mounted in WSL. ``engine`` refuses a file written by another engine.
    """
    with np.load(path, allow_pickle=False) as stored:
        restart = {key: stored[key] for key in stored.files}
    if engine is not None and str(restart["engine"]) != engine:
        raise ValueError(f"{path} was written by the {restart['engine']} engine")
    return restart


def integrator_state(restart: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The integrator's ``state_dict`` stored in a restart file (empty for fixed-cell runs)."""
    if "integrator_keys" in restart:
        keys = [str(key) for key in restart["integrator_keys"]]
    else:
        keys = [key for key in _LEGACY_INTEGRATOR_KEYS if key in restart]
    return {key: restart[key] for key in keys}


def _copy(target: torch.Tensor, source: Any, name: str) -> None:
    source = torch.from_numpy(np.ascontiguousarray(source))
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(f"restart file {name} has shape {tuple(source.shape)}, the state {tuple(target.shape)}")
    target.copy_(source)


def load_state(restart: Mapping[str, Any], state: SimulationState, integrator: Any = None) -> None:
    """Restore a restart file's state into ``state`` in place (fixed device addresses kept).

    Positions, velocities (momenta / masses for research-engine files), cell, committed forces,
    energy and stress, the noise generator with its drawn batch and cursor, and the step; then
    ``integrator.load_state_dict`` with the stored integrator state, if an integrator is given.
    Refuses another structure (species, masses or atom count). As in the research engine's resume,
    call it after the engine is built: building evaluates the forces, and the stored ones replace
    them. A variable-cell force field then needs ``set_cell(state.cell)``.
    """
    if "species" in restart and not np.array_equal(restart["species"], _numpy(state.species)):
        raise ValueError("the restart file is for other species than the state")
    if "masses" in restart and not np.array_equal(restart["masses"], _numpy(state.masses)):
        raise ValueError("the restart file has other masses than the state")
    _copy(state.positions, restart["positions"], "positions")
    if "velocities" in restart:
        _copy(state.velocities, restart["velocities"], "velocities")
    else:  # ASE's Atoms.get_velocities after set_momenta, the research engine's resume path
        masses = _numpy(state.masses).reshape(-1)
        _copy(state.velocities, restart["momenta"] / masses[:, np.newaxis], "momenta")
    _copy(state.cell, restart["cell_state"] if "cell_state" in restart else restart["cell"], "cell")
    _copy(state.forces, restart["forces"], "forces")
    _copy(state.energy, restart["energy"], "energy")
    if state.stress is not None and "stress" in restart:
        _copy(state.stress, restart["stress"], "stress")
    rng = state.rng
    rng.generator.bit_generator.state = json.loads(str(restart["rng_state"]))
    if "noise" in restart:
        if rng.batch is None:
            rng.batch = torch.from_numpy(np.array(restart["noise"])).to(state.positions.device)
        else:
            _copy(rng.batch, restart["noise"], "noise")
    rng.cursor = int(restart["noise_next"])
    state.step = int(restart["step"])
    if integrator is not None:
        integrator.load_state_dict(integrator_state(restart))
