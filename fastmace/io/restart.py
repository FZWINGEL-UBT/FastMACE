"""The restart file: the complete simulation state, format version 4 (SPEC §10.6, §9.5).

One ``.npz`` per run directory (plain arrays and JSON text, no pickles), written durably and
atomically (``io.durable``: fsynced temporary file, rename, directory fsync) at committed steps
after the output queue has been drained, its chunks fsynced, and the engine flushed.
Version 2 (plan task 1b-T05) holds every §10.6 item except the MC state; version 3 (plan task
3-T08) adds it and makes the composition the resumed state:

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
  ``output_frames`` (the output chunk indices at the restart point, plan task 1b-T06);
- version 3's: ``mc``, the JSON record of the attached move set (``MC_RECORD_KEYS``; only with one).
  The composition keys (``COMPOSITION_KEYS``: per-atom ``species`` and ``masses``, ``charges``, and
  total charge and spin in ``physical``) are the run's *current* composition, after any accepted
  swap, transmutation or host change: ``Simulation.resume`` builds the engine from them
  (``composition``), not from the initial structure in ``simulation.json``. Version 2 files hold
  the same keys and resume the same way.
- version 4's (plan task 4-T01): the phase 4 schemes. The integrator record's config holds the
  ``scheme`` (and its ``dof`` and ``parameters``); each scheme's own device state is its
  ``state_dict`` under ``integrator_keys``, exactly the keys its registry record declares
  (``IntegratorRecord.state_keys``: NHC ``eta``, ``p_eta``; MTK the cell momenta and the barostat
  chain; CSVR its transferred energy, its noise rows and cursor being ``noise``/``noise_next``; NVE
  and Berendsen none). The physical record adds ``fixed_atoms``, the ``FixAtoms`` indices
  (``fixed_atoms``). Versions 0 to 3 load as before: their scheme is the ensemble's default
  (Langevin or BAOAB) and they have no fixed atoms.

``main.py``'s adapter writes ``state_arrays`` without identity and plan: such files have no
``identity``. ``check_identity`` (``Simulation.resume``) refuses them, as it refuses 1a
(version 1) and research-engine (no version) files, which ``load_state`` still reads.

The writer makes the same bytes from the same state: entries in name order, a fixed zip time stamp
and canonical JSON (``np.savez`` stamps the time of writing), so save → load → save gives an
identical file.

Every restart write of a run is a canonical restart point (plan task 1b-T14, SPEC §10.6):
``write_engine_restart`` calls ``GraphEngine.canonicalise`` (pipeline quiesced, neighbour lists
rebuilt, floors reset) and stores the engine's edge bucket and neighbour capacity in the plan
record; ``resume_engine`` loads a file into a freshly built engine and gives it those buckets, so
the resumed engine is in the state of the uninterrupted one.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import torch

from fastmace.errors import RestartMismatch
from fastmace.io import durable
from fastmace.state import ForceFieldIdentity, canonical_json

if TYPE_CHECKING:
    from fastmace.state import SimulationState

RESTART_NAME = "restart.npz"
# Version 3 (SPEC §10.6; contracts: plan task 3-T01, writer and resume: 3-T08). The composition keys,
# already written since versions 1 and 2, are the resumed state (before 3-T08 resume rebuilt the Atoms
# from the initial structure); the JSON record "mc" holds the attached move set's state (3-T09, 3-T10).
# Version 4 (plan task 4-T01): the scheme's state keys checked against its record, and the fixed atoms in
# the physical record.
NEXT_FORMAT_VERSION = 4
FORMAT_VERSION = NEXT_FORMAT_VERSION
V3_KEYS = ("mc",)
COMPOSITION_KEYS = ("species", "masses", "charges", "physical")
# The "mc" record: the move records, the step within the MD/MC cycle, counters per move type, each
# stream's seed and offset (config.MC_STREAMS), and adaptive displacement step sizes in Angstrom.
MC_RECORD_KEYS = ("moves", "cycle_position", "counters", "streams", "delta_A")
MC_COUNTER_KEYS = ("attempted", "impossible", "accepted")
MC_STREAM_KEYS = ("seed", "offset")
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
    "charges", "plan_revision", "identity_digest", *_RECORDS, *OUTPUT_KEYS, *V3_KEYS,
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


def _mc_checked(record: Any) -> dict[str, Any]:
    """The MC record, checked against ``MC_RECORD_KEYS`` and its parts: a JSON-safe copy, or ``ValueError``."""
    from fastmace.config import MC_STREAMS
    from fastmace.planner.workload import MC_MOVES

    def mapping(value: Any, name: str, keys: tuple[str, ...] | None = None) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"restart mc {name} must be a mapping, got {type(value).__name__}")
        if keys is not None and set(value) != set(keys):
            raise ValueError(f"restart mc {name} must have exactly the keys {keys}, got {sorted(value)}")
        return value

    mapping(record, "record", MC_RECORD_KEYS)
    moves = record["moves"]
    if not isinstance(moves, (list, tuple)) or not moves:
        raise ValueError(f"restart mc moves must be a nonempty list of move records, got {moves!r}")
    for move in moves:
        if mapping(move, "moves entry").get("move") not in MC_MOVES:
            raise ValueError(f"restart mc moves entry needs a 'move' in {MC_MOVES}, got {move!r}")
    _count(record["cycle_position"], "mc cycle_position")
    for move, counters in mapping(record["counters"], "counters").items():
        if move not in MC_MOVES:
            raise ValueError(f"restart mc counters: unknown move type {move!r}; known: {MC_MOVES}")
        for key, value in mapping(counters, f"counters[{move!r}]", MC_COUNTER_KEYS).items():
            _count(value, f"mc counters[{move!r}] {key}")
    for stream, position in mapping(record["streams"], "streams", MC_STREAMS).items():
        for key, value in mapping(position, f"streams[{stream!r}]", MC_STREAM_KEYS).items():
            _count(value, f"mc streams[{stream!r}] {key}")
    for move, delta in mapping(record["delta_A"], "delta_A").items():
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not 0.0 < delta < float("inf"):
            raise ValueError(f"restart mc delta_A[{move!r}] must be a positive finite step in Å, "
                             f"got {delta!r}")
    return json.loads(_json(dict(record), "mc"))


def _integrator_json(integrator: Any) -> str:
    record = {"class": type(integrator).__name__, "config": dataclasses.asdict(integrator.config),
              "temperature_K": getattr(integrator, "temperature_K", None)}
    stress = getattr(integrator, "externalstress_host", None)
    if stress is not None:  # the current pressure target in ASE units (3b-T05), exact
        record["externalstress"] = np.asarray(stress).tolist()
    return _json(record, "integrator")


def _physical(state: SimulationState) -> dict[str, Any]:
    """The physical record of ``state``: total charge, spin, PBC, other constraints and the fixed atoms."""
    return {"total_charge": state.total_charge, "spin_multiplicity": state.spin_multiplicity,
            "pbc": [bool(periodic) for periodic in state.pbc], "constraints": list(state.constraints),
            "fixed_atoms": [int(index) for index in state.fixed_indices]}


def _check_scheme_keys(integrator: Any, integrator_state: Mapping[str, Any]) -> None:
    """Format 4: a registered scheme's ``state_dict`` holds exactly its record's ``state_keys``."""
    config = getattr(integrator, "config", None)
    scheme = getattr(config, "scheme", None)
    if scheme is None:
        return
    from fastmace.integrators.registry import registry

    record = registry().get("integrator", scheme)
    if record is not None and set(integrator_state) != set(record.state_keys):
        raise ValueError(f"the {scheme} integrator's state_dict has keys {sorted(integrator_state)}, its scheme "
                         f"record declares {sorted(record.state_keys)} (restart format 4)")


def state_arrays(state: SimulationState, engine: str, integrator_state: Mapping[str, Any] | None = None,
                 variable_cell: bool = False, *, integrator: Any = None, plan: Mapping[str, Any] | None = None,
                 outputs: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The restart file's entries for ``state`` (host copies; the caller has flushed the engine).

    ``integrator_state`` is ``Integrator.state_dict()``, taken from ``integrator`` if only that is
    given; ``integrator`` adds its class, config and target (the §10.6 integrator parameters).
    ``plan`` is the plan record (a JSON mapping with ``PLAN_BUCKET_KEYS``), ``outputs`` the output
    chunk indices (exactly ``OUTPUT_KEYS``). The identity, protocol, plan revision and MC record
    (``state.mc``, checked by ``_mc_checked``) come from the state. ``main.py``'s adapter passes
    only the first four arguments.
    """
    if integrator_state is None and integrator is not None:
        integrator_state = integrator.state_dict()
    integrator_state = dict(integrator_state or {})
    if integrator is not None:
        _check_scheme_keys(integrator, integrator_state)
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
        physical=_json(_physical(state), "physical"),
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
    if state.mc is not None:
        arrays["mc"] = _json(_mc_checked(state.mc), "mc")
    if outputs is not None:
        if set(outputs) != set(OUTPUT_KEYS):
            raise ValueError(f"restart outputs must hold exactly {OUTPUT_KEYS}, got {sorted(outputs)}")
        arrays.update({key: _count(outputs[key], key) for key in OUTPUT_KEYS})
    arrays.update({key: np.asarray(value) for key, value in integrator_state.items()})
    arrays["integrator_keys"] = np.array(sorted(integrator_state), dtype=str)
    return arrays


def _write_npz(handle: IO[bytes], arrays: Mapping[str, Any]) -> None:
    """``np.savez`` with entries in name order and a fixed time stamp: the same arrays give the same bytes."""
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(key + ".npy", date_time=_ZIP_TIME)
            info.external_attr = 0o600 << 16  # as np.savez
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(arrays[key]), allow_pickle=False)


def write_restart(path: str | Path, state: SimulationState, *, engine: str,
                  integrator_state: Mapping[str, Any] | None = None, variable_cell: bool = False,
                  integrator: Any = None, plan: Mapping[str, Any] | None = None,
                  outputs: Mapping[str, int] | None = None) -> None:
    """Durable atomic restart file: write and fsync a temporary file, rename it over the old one, then
    fsync the directory (``io.durable``).

    Call at a committed step, after draining the output queue and flushing the engine. The other
    arguments are ``state_arrays``'s; ``variable_cell`` stores the cell again as ``cell_state``, the
    research engine's NPT key. Every record is checked before anything is written.
    """
    path = Path(path)
    arrays = state_arrays(state, engine, integrator_state, variable_cell, integrator=integrator, plan=plan,
                          outputs=outputs)
    durable.write_atomically(path, lambda handle: _write_npz(handle, arrays),
                             temporary=path.with_name(path.name + ".tmp.npz"))


def write_engine_restart(path: str | Path, engine: Any, *, name: str, plan: Any = None,
                         outputs: Mapping[str, int] | None = None) -> None:
    """A canonical restart point of ``engine``, then its restart file (SPEC §10.6).

    ``engine.canonicalise()`` first, then ``write_restart`` with the engine's state and integrator.
    ``name`` is the restart file's ``engine`` entry. ``plan`` is the active ``Plan``; its record is
    ``plan.to_restart`` with the engine's edge bucket and neighbour capacity, or those two alone
    without a plan. ``outputs`` is ``ChunkedOutput.restart_point()`` (the output queue drained first).
    """
    engine.canonicalise()
    buckets = engine.restart_buckets()
    record = buckets if plan is None else plan.to_restart(**buckets)
    write_restart(path, engine.state, engine=name, variable_cell=engine.variable_cell, integrator=engine.integrator,
                  plan=record, outputs=outputs)


def resume_engine(restart: Mapping[str, Any], engine: Any) -> None:
    """Load a restart file's state into a freshly built engine and give it the file's buckets.

    Build the engine from the restart's positions and cell first. ``load_state`` replaces the build's
    forces with the stored ones; then the force field takes the stored cell and the engine the plan
    record's edge bucket and neighbour capacity, and canonicalises (``restore_restart_point``).

    A total charge the file records and the state does not have (an accepted transmutation with formal
    charges, or a host total-charge event, moved it) is restored first through the force field's
    ``set_total_charge`` and the ``charges`` refresh (3-T09). The spin multiplicity is still checked:
    no move changes it (the parity rule, SPEC §10.4).
    """
    physical = physical_record(restart) or {}
    total = physical.get("total_charge")
    state, field = engine.state, engine.force_field
    if total is not None and total != state.total_charge and callable(getattr(field, "set_total_charge", None)):
        field.set_total_charge(total)
        engine.derived.refresh(("charges",))
        state.total_charge = total
    load_state(restart, engine.state, engine.integrator)
    if engine.variable_cell:
        engine.force_field.set_cell(engine.state.cell)
    record = plan_record(restart) or {}
    engine.restore_restart_point(**{key: record[key] for key in PLAN_BUCKET_KEYS if key in record})


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
    """The FastMACE format version: 4, 3 (fastmace-3), 2 (fastmace-2), 1 (phase 1a) or 0 (a research-engine file)."""
    return int(restart["fastmace_restart"]) if "fastmace_restart" in restart else 0


def _record(restart: Mapping[str, Any], name: str) -> Any:
    return json.loads(str(restart[name])) if name in restart else None


def mc_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """The MC record (``MC_RECORD_KEYS``) of a version 3 file, or ``None`` without an attached move set
    (and in older files). A malformed record is refused (``ValueError``)."""
    record = _record(restart, "mc")
    return None if record is None else _mc_checked(record)


def composition(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """The run's composition at the restart step (``COMPOSITION_KEYS``), or ``None`` without ``species``
    (research-engine files).

    ``species`` (atomic numbers) and ``masses`` (amu) per atom; ``charges`` per atom or ``None``;
    ``total_charge`` and ``spin_multiplicity`` from the physical record (``None`` before version 2 or
    when the model takes none). ``Simulation.resume`` builds its ``Atoms`` from it: after an accepted
    swap or transmutation it differs from the initial structure.
    """
    if "species" not in restart:
        return None
    physical = physical_record(restart) or {}
    masses = np.asarray(restart["masses"]).reshape(-1) if "masses" in restart else None
    charges = np.asarray(restart["charges"]).reshape(-1) if "charges" in restart else None
    return {"species": [int(z) for z in np.asarray(restart["species"]).reshape(-1)],
            "masses": None if masses is None else [float(m) for m in masses],
            "charges": None if charges is None else [float(q) for q in charges],
            "total_charge": physical.get("total_charge"), "spin_multiplicity": physical.get("spin_multiplicity")}


def physical_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """Total charge, spin, PBC, constraints and (version 4) the fixed atoms, or ``None`` before version 2."""
    return _record(restart, "physical")


def fixed_atoms(restart: Mapping[str, Any]) -> tuple[int, ...]:
    """The ``FixAtoms`` indices of the run (version 4), ``()`` without fixed atoms or in older files."""
    return tuple(int(index) for index in (physical_record(restart) or {}).get("fixed_atoms", ()))


def scheme(restart: Mapping[str, Any]) -> str | None:
    """The integrator scheme of the run: the stored config's (version 4), else its ensemble's default
    (``config.DEFAULT_SCHEMES``: versions 2 and 3 ran Langevin or BAOAB); ``None`` without an integrator
    record."""
    from fastmace.config import DEFAULT_SCHEMES

    record = integrator_record(restart)
    if record is None:
        return None
    config = record.get("config") or {}
    return config.get("scheme") or DEFAULT_SCHEMES.get(config.get("ensemble", "nvt"))


def integrator_record(restart: Mapping[str, Any]) -> dict[str, Any] | None:
    """The integrator's class, ``IntegratorConfig`` fields, thermostat target and (variable cell, since
    3b-T05) the current ``externalstress`` in ASE units, if stored."""
    return _record(restart, "integrator")


def restore_targets(restart: Mapping[str, Any], integrator: Any) -> None:
    """Fill the integrator's targets with the restart's: temperature, and the pressure target if stored.

    Files without ``externalstress`` (before 3b-T05) keep the configured target. The fills do not
    touch the stored cell force, so the resumed run continues as the uninterrupted one.
    """
    record = integrator_record(restart) or {}
    if record.get("temperature_K") is not None:
        integrator.set_temperature(record["temperature_K"])
    if record.get("externalstress") is not None:
        integrator.restore_externalstress(record["externalstress"])


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
    physical = {"fixed_atoms": [], **physical}  # versions 2 and 3 have no fixed atoms
    for key, value in _physical(state).items():
        if physical.get(key) != value:
            raise ValueError(f"the restart file has {key}={physical.get(key)!r}, the state {value!r}")


def load_state(restart: Mapping[str, Any], state: SimulationState, integrator: Any = None) -> None:
    """Restore a restart file's state into ``state`` in place (fixed device addresses kept).

    Positions, velocities (momenta / masses for research-engine files), cell, charges, committed
    forces, energy and stress, the barostat (``p_eps``, ``force_eps``), the noise generator with its
    drawn batch and cursor, the step and the plan revision; the protocol through
    ``type(state.protocol).from_restart`` if the state has one (else the caller builds it from
    ``protocol_record``); the MC record (``state.mc``, ``None`` without one); then
    ``integrator.load_state_dict`` with what is left of the stored integrator state, if an integrator
    is given. Refuses another structure (species, masses, atom count, PBC, total charge, spin,
    constraints) and a newer format: the state must be built with the file's composition
    (``composition``), because the engine's species-dependent tables are built with it. The force-field identity is not
    restored: it is the force field's, and ``check_identity`` compares it. As in the research
    engine's resume, call it after the engine is built: building evaluates the forces, and the
    stored ones replace them. A variable-cell force field then needs ``set_cell(state.cell)``.
    """
    version = format_version(restart)
    if version > FORMAT_VERSION:
        raise ValueError(f"the restart file has format version {version}; this FastMACE reads up to "
                         f"{FORMAT_VERSION}")
    if "species" in restart and not np.array_equal(restart["species"], _numpy(state.species)):
        raise ValueError("the restart file is for other species than the state: build the state from the "
                         "file's composition (io.restart.composition), as Simulation.resume does")
    if "masses" in restart and not np.array_equal(restart["masses"], _numpy(state.masses)):
        raise ValueError("the restart file has other masses than the state: build the state from the "
                         "file's composition (io.restart.composition), as Simulation.resume does")
    _check_physical(restart, state)
    mc = mc_record(restart)
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
    state.mc = mc
    protocol = protocol_record(restart)
    if protocol is not None and state.protocol is not None:
        state.protocol = type(state.protocol).from_restart(protocol)
    if integrator is not None:
        integrator.load_state_dict(stored_integrator)
