"""The restart file: the complete simulation state, format version 2 (SPEC §10.6, §9.5).

One ``.npz`` per run directory (plain arrays and JSON text, no pickles), written atomically
(temporary file, then rename) at committed steps after the output queue has been drained and the
engine flushed. Version 2 (plan task 1b-T05) holds every §10.6 item except the MC state (phase 3):

- the research engine's keys (``workflow.write_restart`` with ``GPUEngine.state``), so either side
  reads the other's files: ``step``, ``engine`` (checked on resume), ``positions``, ``momenta``,
  ``cell``, ``rng_state`` (the NumPy bit generator's state as JSON), ``noise`` (the batch of random
  numbers already drawn ahead) and ``noise_next`` (its cursor), ``forces``, ``energy``; variable-cell
  runs add ``cell_state``, ``stress``, ``p_eps`` and ``force_eps`` (the barostat, which is in
  ``SimulationState`` since 1b);
- phase 1a's additions: ``velocities`` (the integrated quantity: rebuilding it as momenta / masses,
  as the research engine does on resume, rounds), ``masses`` and ``species`` (checked on load),
  ``integrator_keys`` (the integrator's own ``state_dict`` entries, none since 1b) and
  ``fastmace_restart`` (the format version);
- version 2's: ``charges``; ``plan_revision``; the JSON records ``physical`` (total charge, spin,
  PBC, constraints), ``integrator`` (class, ``IntegratorConfig``, thermostat target), ``protocol``
  (``state.protocol.to_restart()``), ``plan`` (a JSON mapping until plan task 1b-T12's
  ``Plan.to_restart()``; it includes ``edge_bucket`` and ``neighbour_capacity``) and ``identity``
  (the ``ForceFieldIdentity``) with ``identity_digest``; ``output_chunk``, ``output_rows`` and
  ``output_frames`` (the output chunk indices at the restart point, plan task 1b-T06).

``main.py``'s adapter writes ``state_arrays`` without identity and plan: such files are version 2
without ``identity``. ``check_identity`` (``Simulation.resume``) refuses them, as it refuses 1a
(version 1) and research-engine (no version) files, which ``load_state`` still reads.

The writer makes the same bytes from the same state: entries in name order, a fixed zip time stamp
and canonical JSON (``np.savez`` stamps the time of writing), so save → load → save gives an
identical file. Canonical restart points (pipeline quiesced, neighbour lists rebuilt) are the
engine's, plan task 1b-T14.
"""

from __future__ import annotations

import dataclasses
import json
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastmace.errors import RestartMismatch
from fastmace.state import ForceFieldIdentity, canonical_json

if TYPE_CHECKING:
    from fastmace.state import SimulationState

RESTART_NAME = "restart.npz"
FORMAT_VERSION = 2
# The barostat's keys: SimulationState fields since 1b; integrator state in 1a files and in
# research-engine files (which have no "integrator_keys" entry, GPUEngine.state).
BAROSTAT_KEYS = ("p_eps", "force_eps")
_LEGACY_INTEGRATOR_KEYS = BAROSTAT_KEYS
# The output chunk indices at the restart point (plan task 1b-T06).
OUTPUT_KEYS = ("output_chunk", "output_rows", "output_frames")
# What the plan record holds besides the plan itself, for resuming the engine (plan task 1b-T14).
PLAN_BUCKET_KEYS = ("edge_bucket", "neighbour_capacity")
_RECORDS = ("physical", "integrator", "protocol", "plan", "identity")
_STATE_KEYS = frozenset({
    "step", "engine", "positions", "momenta", "velocities", "masses", "species", "cell", "cell_state",
    "rng_state", "noise", "noise_next", "forces", "energy", "stress", "integrator_keys", "fastmace_restart",
    "charges", "plan_revision", "identity_digest", *_RECORDS, *OUTPUT_KEYS,
})
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # the earliest zip time stamp: the bytes do not depend on when they are written


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _json(value: Any, name: str) -> str:
    try:
        return canonical_json(value)
    except ValueError as error:
        raise ValueError(f"restart {name} record is not JSON-safe: {error}") from None


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"restart {name} must be a nonnegative int, got {value!r}")
    return int(value)


def _plan_json(plan: Mapping[str, Any]) -> str:
    if not isinstance(plan, Mapping):
        raise ValueError(f"restart plan must be a JSON mapping, got {type(plan).__name__}")
    for key in PLAN_BUCKET_KEYS:
        if key not in plan:
            raise ValueError(f"restart plan record needs {key!r} (SPEC §10.6: edge and capacity buckets)")
        _count(plan[key], f"plan {key}")
    return _json(dict(plan), "plan")


def _integrator_json(integrator: Any) -> str:
    return _json({"class": type(integrator).__name__, "config": dataclasses.asdict(integrator.config),
                  "temperature_K": getattr(integrator, "temperature_K", None)}, "integrator")


def state_arrays(state: SimulationState, engine: str, integrator_state: Mapping[str, Any] | None = None,
                 variable_cell: bool = False, *, integrator: Any = None, plan: Mapping[str, Any] | None = None,
                 outputs: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The restart file's entries for ``state`` (host copies; the caller has flushed the engine).

    ``integrator_state`` is ``Integrator.state_dict()``, taken from ``integrator`` if only that is
    given; ``integrator`` adds its class, config and target (the §10.6 integrator parameters).
    ``plan`` is the plan record (a JSON mapping with ``PLAN_BUCKET_KEYS``), ``outputs`` the output
    chunk indices (exactly ``OUTPUT_KEYS``). The identity, protocol and plan revision come from the
    state. ``main.py``'s adapter passes only the first four arguments.
    """
    if integrator_state is None and integrator is not None:
        integrator_state = integrator.state_dict()
    integrator_state = dict(integrator_state or {})
    reserved = _STATE_KEYS | (set(BAROSTAT_KEYS) if state.p_eps is not None else set())
    clash = reserved & integrator_state.keys()
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
        plan_revision=_count(state.plan_revision, "plan_revision"),
        physical=_json({"total_charge": state.total_charge, "spin_multiplicity": state.spin_multiplicity,
                        "pbc": [bool(periodic) for periodic in state.pbc],
                        "constraints": list(state.constraints)}, "physical"),
    )
    if state.rng.batch is not None:
        arrays["noise"] = _numpy(state.rng.batch)
    if state.stress is not None:
        arrays["stress"] = _numpy(state.stress)
    if variable_cell:
        arrays["cell_state"] = _numpy(state.cell)
    if state.p_eps is not None:
        arrays.update(p_eps=_numpy(state.p_eps), force_eps=_numpy(state.force_eps))
    if state.charges is not None:
        arrays["charges"] = _numpy(state.charges)
    if integrator is not None:
        arrays["integrator"] = _integrator_json(integrator)
    if state.protocol is not None:
        arrays["protocol"] = _json(state.protocol.to_restart(), "protocol")
    if plan is not None:
        arrays["plan"] = _plan_json(plan)
    if state.force_field_identity is not None:
        arrays["identity"] = _json(state.force_field_identity.to_dict(), "identity")
        arrays["identity_digest"] = state.force_field_identity.digest
    if outputs is not None:
        if set(outputs) != set(OUTPUT_KEYS):
            raise ValueError(f"restart outputs must hold exactly {OUTPUT_KEYS}, got {sorted(outputs)}")
        arrays.update({key: _count(outputs[key], key) for key in OUTPUT_KEYS})
    arrays.update({key: np.asarray(value) for key, value in integrator_state.items()})
    arrays["integrator_keys"] = np.array(sorted(integrator_state), dtype=str)
    return arrays


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """``np.savez`` with entries in name order and a fixed time stamp: the same arrays give the same bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(key + ".npy", date_time=_ZIP_TIME)
            info.external_attr = 0o600 << 16  # as np.savez
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(arrays[key]), allow_pickle=False)


def write_restart(path: str | Path, state: SimulationState, *, engine: str,
                  integrator_state: Mapping[str, Any] | None = None, variable_cell: bool = False,
                  integrator: Any = None, plan: Mapping[str, Any] | None = None,
                  outputs: Mapping[str, int] | None = None) -> None:
    """Atomic restart file: write a temporary file, then rename it over the old one.

    Call at a committed step, after draining the output queue and flushing the engine. The other
    arguments are ``state_arrays``'s; ``variable_cell`` stores the cell again as ``cell_state``, the
    research engine's NPT key. Every record is checked before anything is written.
    """
    path = Path(path)
    arrays = state_arrays(state, engine, integrator_state, variable_cell, integrator=integrator, plan=plan,
                          outputs=outputs)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        _write_npz(temporary, arrays)
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


# -- the records of a file read by read_restart --

def format_version(restart: Mapping[str, Any]) -> int:
    """The FastMACE format version: 2, 1 (phase 1a) or 0 (a research-engine file)."""
    return int(restart["fastmace_restart"]) if "fastmace_restart" in restart else 0


def _record(restart: Mapping[str, Any], name: str) -> Any:
    return json.loads(str(restart[name])) if name in restart else None


def physical_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """Total charge, spin, PBC and constraints, or ``None`` before version 2."""
    return _record(restart, "physical")


def integrator_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """The integrator's class, ``IntegratorConfig`` fields and thermostat target, if stored."""
    return _record(restart, "integrator")


def protocol_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """``state.protocol.to_restart()`` at the restart step, if the run had a protocol object."""
    return _record(restart, "protocol")


def plan_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """The plan record (with ``edge_bucket`` and ``neighbour_capacity``), if stored."""
    return _record(restart, "plan")


def output_indices(restart: Mapping[str, Any]) -> dict[str, int] | None:
    """``output_chunk``, ``output_rows`` and ``output_frames`` at the restart step, if stored."""
    if not all(key in restart for key in OUTPUT_KEYS):
        return None
    return {key: int(restart[key]) for key in OUTPUT_KEYS}


def stored_identity(restart: Mapping[str, Any]) -> ForceFieldIdentity | None:
    """The force-field identity of the run, or ``None`` (``main.py``, 1a and research-engine files).

    A record that does not match its stored digest is refused (``ValueError``).
    """
    record = _record(restart, "identity")
    if record is None:
        return None
    identity = ForceFieldIdentity.from_dict(record)
    if "identity_digest" in restart and str(restart["identity_digest"]) != identity.digest:
        raise ValueError("the restart file's force-field identity does not match its digest "
                         f"({restart['identity_digest']} stored, {identity.digest} computed)")
    return identity


def check_identity(restart: Mapping[str, Any], identity: ForceFieldIdentity) -> None:
    """``RestartMismatch`` unless the file was written with ``identity`` (SPEC §10.6, ``Simulation.resume``)."""
    stored = stored_identity(restart)
    if stored is None:
        raise RestartMismatch(
            "the restart file has no force-field identity",
            why=f"it was written by phase 1a, main.py or the research engine (format version "
                f"{format_version(restart)}), which do not record the force field, so a resume cannot "
                "check that it continues the same physics.",
            next_step="Resume it with the program that wrote it, or start a new run from its positions.")
    different = stored.differences(identity)
    if different:
        changes = "; ".join(f"{name}: {getattr(stored, name)!r} in the file, {getattr(identity, name)!r} now"
                            for name in different)
        raise RestartMismatch(f"the force field differs from the restart's in {', '.join(different)}",
                              why=f"a resume continues the same physics only (SPEC §10.6). {changes}.")


def integrator_state(restart: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The integrator's ``state_dict`` stored in a restart file (empty since 1b).

    1a and research-engine files list BAOAB's ``p_eps`` and ``force_eps`` here; ``load_state`` puts
    them into the state when it has a barostat.
    """
    if "integrator_keys" in restart:
        keys = [str(key) for key in restart["integrator_keys"]]
    else:
        keys = [key for key in _LEGACY_INTEGRATOR_KEYS if key in restart]
    return {key: restart[key] for key in keys}


def _copy(target: torch.Tensor, source: Any, name: str) -> None:
    source = torch.from_numpy(np.array(source, order="C"))  # 0-d stays 0-d (ascontiguousarray makes it 1-d)
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(f"restart file {name} has shape {tuple(source.shape)}, the state {tuple(target.shape)}")
    target.copy_(source)


def _check_physical(restart: Mapping[str, Any], state: SimulationState) -> None:
    physical = physical_record(restart)
    if physical is None:
        return
    current = {"total_charge": state.total_charge, "spin_multiplicity": state.spin_multiplicity,
               "pbc": [bool(periodic) for periodic in state.pbc], "constraints": list(state.constraints)}
    for key, value in current.items():
        if physical.get(key) != value:
            raise ValueError(f"the restart file has {key}={physical.get(key)!r}, the state {value!r}")


def load_state(restart: Mapping[str, Any], state: SimulationState, integrator: Any = None) -> None:
    """Restore a restart file's state into ``state`` in place (fixed device addresses kept).

    Positions, velocities (momenta / masses for research-engine files), cell, charges, committed
    forces, energy and stress, the barostat (``p_eps``, ``force_eps``), the noise generator with its
    drawn batch and cursor, the step and the plan revision; the protocol through
    ``type(state.protocol).from_restart`` if the state has one (else the caller builds it from
    ``protocol_record``); then ``integrator.load_state_dict`` with what is left of the stored
    integrator state, if an integrator is given. Refuses another structure (species, masses, atom
    count, PBC, total charge, spin, constraints) and a newer format. The force-field identity is not
    restored: it is the force field's, and ``check_identity`` compares it. As in the research
    engine's resume, call it after the engine is built: building evaluates the forces, and the
    stored ones replace them. A variable-cell force field then needs ``set_cell(state.cell)``.
    """
    version = format_version(restart)
    if version > FORMAT_VERSION:
        raise ValueError(f"the restart file has format version {version}; this FastMACE reads up to "
                         f"{FORMAT_VERSION}")
    if "species" in restart and not np.array_equal(restart["species"], _numpy(state.species)):
        raise ValueError("the restart file is for other species than the state")
    if "masses" in restart and not np.array_equal(restart["masses"], _numpy(state.masses)):
        raise ValueError("the restart file has other masses than the state")
    _check_physical(restart, state)
    stored_integrator = integrator_state(restart)
    for name in BAROSTAT_KEYS:  # into the state; 1a files list them as integrator state
        if name not in restart:
            continue
        if getattr(state, name) is not None:
            stored_integrator.pop(name, None)
        elif name not in stored_integrator:
            raise ValueError(f"the restart file has a barostat ({name}), the state none (a fixed-cell run)")
    _copy(state.positions, restart["positions"], "positions")
    if "velocities" in restart:
        _copy(state.velocities, restart["velocities"], "velocities")
    else:  # ASE's Atoms.get_velocities after set_momenta, the research engine's resume path
        masses = _numpy(state.masses).reshape(-1)
        _copy(state.velocities, restart["momenta"] / masses[:, np.newaxis], "momenta")
    _copy(state.cell, restart["cell_state"] if "cell_state" in restart else restart["cell"], "cell")
    if "charges" in restart:
        if state.charges is None:
            raise ValueError("the restart file has per-atom charges, the state none")
        _copy(state.charges, restart["charges"], "charges")
    _copy(state.forces, restart["forces"], "forces")
    _copy(state.energy, restart["energy"], "energy")
    if state.stress is not None and "stress" in restart:
        _copy(state.stress, restart["stress"], "stress")
    for name in BAROSTAT_KEYS:
        if name in restart and getattr(state, name) is not None:
            _copy(getattr(state, name), restart[name], name)
    rng = state.rng
    rng.generator.bit_generator.state = json.loads(str(restart["rng_state"]))
    if "noise" in restart:
        if rng.batch is None:
            rng.batch = torch.from_numpy(np.array(restart["noise"])).to(state.positions.device)
        else:
            _copy(rng.batch, restart["noise"], "noise")
    rng.cursor = int(restart["noise_next"])
    state.step = int(restart["step"])
    if "plan_revision" in restart:
        state.plan_revision = int(restart["plan_revision"])
    protocol = protocol_record(restart)
    if protocol is not None and state.protocol is not None:
        state.protocol = type(state.protocol).from_restart(protocol)
    if integrator is not None:
        integrator.load_state_dict(stored_integrator)
