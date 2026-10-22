"""Protocol runs: relaxation, MD in resumable pieces, quench (SPEC §10.5; plan task 3-T14).

A protocol is ``fm.md``'s arguments plus two optional stage tables, as ``fastmace run`` reads them
from TOML: ``[relax]`` (``RELAX_KEYS``: an initial relaxation at fixed cell) and ``[quench]``
(``profiles.QUENCH_KEYS``: frames picked per target bin after the MD, relaxed to 0 K). Protocols
shipped with FastMACE are the ``.toml`` files of ``DATA``, named by their stem
(``protocol_names``); a shipped protocol's relative structure path is a file of ``DATA``.

The stages, in the run's output directory:

1. ``start``: the relaxation with ``fm.calc``'s calculator on the model and in the force field's
   precision (``relaxation.log``, ``relaxed.extxyz``), then ``fm.md`` on the relaxed structure,
   for at most ``steps`` steps of the schedule;
2. ``resume``: ``Simulation.resume`` and the next at most ``steps`` steps. The move set is attached
   over the steps whose target is at or above ``min_mc_temperature_K`` (``quick.mc_ranges``), as
   ``fm.md`` attaches it; an attached set stays attached across a resume (§7.6);
3. after the last MD step, the quench (``quench.quench_run``) with the model of the run.

Every piece ends with a restart at its last step, so a long protocol runs as pieces of bounded
wall time. ``protocol.json`` records the stages, the pieces, the relaxation and the quench summary,
and the engine counters of ``PIECE_COUNTERS`` per piece (each piece builds a new engine) with their
sums over the run in ``counters``; ``metadata.json`` is the run's, with the plan, gate and
``counters.mc``.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastmace.protocols import quench as quench_stage
from fastmace.protocols.profiles import OPTIMIZERS, _quench

DATA = Path(__file__).resolve().parent / "data"
PROTOCOL_RECORD = "protocol.json"
PROTOCOL_RECORD_VERSION = 1
RELAX_KEYS = ("fmax", "optimizer", "max_steps")
STAGES = ("relax", "quench")
RELAXATION_LOG, RELAXED = "relaxation.log", "relaxed.extxyz"
_LOAD_KEYS = ("head", "sha256", "trusted", "charge", "spin_multiplicity")
# The engine counters (``Engine.info()``) a protocol run sums over its pieces: the MACE neighbour list's
# rebuilds (3b-T06's cost cross-check reads them from the full ramp).
PIECE_COUNTERS = ("neighbour_rebuilds",)


def protocol_names() -> tuple[str, ...]:
    """The shipped protocols, by name (the stems of ``DATA``'s ``.toml`` files)."""
    return tuple(sorted(path.stem for path in DATA.glob("*.toml")))


def protocol_file(name_or_path: str | Path) -> tuple[Path, bool]:
    """``(file, shipped)``: an existing file as it is, else the shipped protocol of that name."""
    path = Path(name_or_path)
    if path.is_file():
        return path, False
    if str(name_or_path) in protocol_names():
        return DATA / f"{name_or_path}.toml", True
    raise ValueError(f"{name_or_path} is neither a file nor a shipped protocol; shipped: {', '.join(protocol_names())}")


def _relax(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping) or "fmax" not in data or set(data) - set(RELAX_KEYS):
        raise ValueError(f"relax needs fmax and takes {', '.join(RELAX_KEYS)}, got {data!r}")
    fmax, optimizer = data["fmax"], data.get("optimizer", "LBFGS")
    max_steps = data.get("max_steps", quench_stage.QUENCH_MAX_STEPS)
    if isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or not fmax > 0:
        raise ValueError(f"relax fmax must be a positive number (eV/Å), got {fmax!r}")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"relax optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"relax max_steps must be a positive int, got {max_steps!r}")
    return {"fmax": float(fmax), "optimizer": optimizer, "max_steps": max_steps}


def split_stages(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``({"relax": settings or None, "quench": settings or None}, the other keys)`` of a protocol mapping."""
    rest = {key: value for key, value in data.items() if key not in STAGES}
    stages = {"relax": None, "quench": None}
    if "relax" in data:
        stages["relax"] = _relax(data["relax"])
    if "quench" in data:
        try:
            stages["quench"] = _quench(data["quench"])
        except ValueError as error:
            raise ValueError(f"quench: {error}") from None
    return stages, rest


def _write(directory: Path, record: Mapping[str, Any]) -> None:
    path = directory / PROTOCOL_RECORD
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_record(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / PROTOCOL_RECORD
    if not path.is_file():
        raise ValueError(f"{directory} holds no {PROTOCOL_RECORD}: fastmace run --resume continues protocol runs "
                         "that fastmace run started")
    return json.loads(path.read_text(encoding="utf-8"))


def _release_device_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _remaining(schedule, timestep_fs, steps):
    """The steps of this piece: ``steps`` clamped to the schedule's remaining steps (``None``: all)."""
    if schedule is None:
        return steps
    from fastmace.protocols import ScheduleState

    remaining = ScheduleState(schedule, timestep_fs).remaining
    return remaining if steps is None else min(steps, remaining)


def start(structure: Any, options: Mapping[str, Any], *, relax: Mapping[str, Any] | None = None,
          quench: Mapping[str, Any] | None = None, steps: int | None = None, source: str | None = None,
          log: Callable[[str], Any] = print) -> dict[str, Any]:
    """Start a protocol run: the relaxation, then ``fm.md`` for at most ``steps`` steps (``None``: the whole
    schedule), then, if the schedule is done, the quench. Returns ``protocol.json``'s record.

    ``structure`` and ``options`` are ``fm.md``'s (``options["model"]`` a label, path or ``fm.load``
    result); ``relax`` and ``quench`` the stage settings (``split_stages``).
    """
    from fastmace import quick
    from fastmace.ase.calculator import FastMACECalculator
    from fastmace.model.loading import LoadedModel, load_pristine
    from fastmace.simulation import mc_moves, move_record

    options = dict(options)
    output = quick._output(options.pop("output", None), structure)
    directory = Path(output.directory)
    if (directory / PROTOCOL_RECORD).is_file():
        raise ValueError(f"{directory} already holds a protocol run; continue it with fastmace run --resume "
                         f"{directory}")
    schedule = options.get("schedule")
    if quench is not None and (schedule is None or output.trajectory_every is None):
        raise ValueError("a quench picks trajectory frames by the schedule's target: it needs schedule= and "
                         "trajectory frames (output trajectory_every)")
    mc = mc_moves(options["mc"]) if options.get("mc") is not None else None
    if mc is not None and schedule is not None and options.get("min_mc_temperature_K") is None:
        raise ValueError("mc= with a schedule needs min_mc_temperature_K= (SPEC §7.7)")
    repeat = quick._repeat(options.pop("repeat", (1, 1, 1)))
    timestep_fs = options.get("timestep_fs", 0.5)
    steps = _remaining(schedule, timestep_fs, steps)

    atoms = quick._structure(structure)
    if repeat != (1, 1, 1):
        atoms = atoms.repeat(repeat)
    model = options.pop("model")
    load = {key: options.pop(key) for key in _LOAD_KEYS if key in options}
    if not isinstance(model, LoadedModel):
        model = load_pristine(str(model), load.pop("head", None), **load)
    elif load:
        raise ValueError(f"model is already loaded; {', '.join(sorted(load))} belong to fm.load")

    directory.mkdir(parents=True, exist_ok=True)
    record = {"protocol_record_version": PROTOCOL_RECORD_VERSION, "source": source, "structure": str(structure),
              "repeat": list(repeat), "atoms": len(atoms), "timestep_fs": timestep_fs, "relax": relax,
              "relaxation": None, "quench": quench, "quench_result": None,
              "mc": None if mc is None else [move_record(move) for move in mc],
              "min_mc_temperature_K": options.get("min_mc_temperature_K"), "pieces": [], "counters": None,
              "status": "relax"}
    _write(directory, record)
    if relax is not None:
        calc = FastMACECalculator(model)
        atoms, summary = quench_stage.relax(atoms, calc, relax["fmax"], relax["max_steps"],
                                            logfile=directory / RELAXATION_LOG, optimizer=relax["optimizer"])
        del calc
        _release_device_memory()
        atoms.write(directory / RELAXED)
        record["relaxation"] = summary
        log(f"relaxation: {summary['optimizer']} at fixed cell, {summary['initial_eV_atom']:.6f} -> "
            f"{summary['energy_eV_atom']:.6f} eV/atom in {summary['steps']} steps ({summary['seconds']:.1f} s), "
            f"fmax {summary['fmax_eV_A']:.4f} eV/Å")
        if not summary["converged"]:
            record["status"] = "relaxation not converged"
            _write(directory, record)
            raise RuntimeError(f"the initial relaxation did not reach fmax {relax['fmax']} eV/Å in "
                               f"{relax['max_steps']} steps ({directory / RELAXATION_LOG})")
    record["status"] = "md"
    _write(directory, record)

    started = time.perf_counter()
    result = quick.md(atoms, model=model, output=output, steps=steps, **options)
    log(result.summary)
    sim = result.simulation
    sim.report.repeat = repeat
    return _end_piece(sim, directory, record, 0, started, log)


def resume(directory: str | Path, *, steps: int | None = None, log: Callable[[str], Any] = print) -> dict[str, Any]:
    """Continue the protocol run in ``directory``: at most ``steps`` more MD steps (``None``: the rest), then
    the quench once the schedule is done. Returns ``protocol.json``'s record."""
    from fastmace.quick import mc_ranges
    from fastmace.simulation import Simulation, move_from_record

    directory = Path(directory)
    record = read_record(directory)
    if record["status"] == "complete":
        log(f"{directory}: the protocol run is complete")
        return record
    if record["status"] not in ("md", "quench"):
        raise ValueError(f"{directory}: the protocol run stopped at {record['status']!r} and cannot be resumed")
    started = time.perf_counter()
    sim = Simulation.resume(directory)
    sim.report.repeat = tuple(record["repeat"])
    first = int(sim.committed().step)
    position = sim.schedule
    if record["status"] == "md" and position is not None and not position.finished:
        count = position.remaining if steps is None else min(steps, position.remaining)
        moves = None if record["mc"] is None else tuple(move_from_record(move) for move in record["mc"])
        if moves is None:
            sim.run(steps=count)
        else:
            low = record["min_mc_temperature_K"]
            # The last step ran with the move set iff its target is at or above the minimum (fm.md's rule),
            # and an attached set is in the restart: keep it (mc=...) rather than attach a new driver.
            attached = position.step > 0 and position.resolved.target(position.step) >= low
            for piece, wanted in mc_ranges(position, count, low):
                sim.run(steps=piece, mc=... if wanted == attached else moves if wanted else None)
                attached = wanted
        sim.flush_output()
        log(sim.report.text())
    return _end_piece(sim, directory, record, first, started, log)


def _end_piece(sim, directory, record, first, started, log):
    """A restart at the piece's last step; the quench once the schedule is done."""
    step = int(sim.committed().step)
    if step > first or not record["pieces"]:
        sim.save_restart(directory)
        record["pieces"].append({"first_step": first, "last_step": step, "wall_s": time.perf_counter() - started,
                                 "counters": _piece_counters(sim)})
        record["counters"] = _sum_counters(record["pieces"])
    position = sim.schedule
    if position is not None and not position.finished:
        sim.flush_output()
        _write(directory, record)
        log(f"paused at step {step} of {position.total_steps}; continue with fastmace run --resume {directory}")
        return record
    resolved = None if position is None else position.resolved
    frame_every = sim.output.trajectory_every
    model, precision = sim.force_field.model, sim.force_field.precision
    sim.close()
    _log_counters(directory, log)
    if record["quench"] is not None:
        record["status"] = "quench"
        _write(directory, record)
        del sim
        _release_device_memory()
        from fastmace.ase.calculator import FastMACECalculator

        calc = FastMACECalculator(model, precision=precision)
        result = quench_stage.quench_run(directory, calc, resolved, frame_every, record["timestep_fs"],
                                         record["quench"], log=log)
        record["quench_result"] = result
        log(f"quench: {result['frames']} frames, final quenched {result['final_quenched_eV_atom']:.6f} eV/atom, "
            f"{result['not_converged']} not converged ({directory / quench_stage.QUENCH_DIRECTORY})")
    record["status"] = "complete"
    _write(directory, record)
    return record


def _piece_counters(sim):
    """``PIECE_COUNTERS`` of this piece's engine (``None`` where the engine does not report one)."""
    info = (sim.metadata().get("counters") or {}).get("engine") or {}
    return {key: info.get(key) for key in PIECE_COUNTERS}


def _sum_counters(pieces):
    """The sums of the pieces' counters; ``None`` for a counter some piece lacks."""
    counts = [piece.get("counters") or {} for piece in pieces]
    return {key: None if any(c.get(key) is None for c in counts) else sum(c[key] for c in counts)
            for key in PIECE_COUNTERS}


def _log_counters(directory, log):
    metadata = json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
    for move, counts in ((metadata.get("counters") or {}).get("mc") or {}).items():
        log(f"mc {move}: {counts.get('attempted')} attempted, {counts.get('impossible')} impossible, "
            f"{counts.get('accepted')} accepted")
