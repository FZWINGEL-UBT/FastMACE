"""``fm.md`` and ``RunResult``: MD in one call (SPEC §7.7 level 2).

Plan task 1b-T01 fixed the names and the signature; plan task 1b-T20 implements ``md``. No engine
code here: ``md`` checks its arguments, builds the integrator record, ``Workload`` and ``Output``,
and hands them with the loaded model to a ``Simulation`` (1b-T17), which plans, gates, builds and
runs. So ``fm.md`` and the equivalent ``Simulation`` get the same plan and gate (§7.7 acceptance).

What ``md`` decides itself, before anything is loaded:

- the integrator: ``ensemble="NVT"`` is ``fm.integrators.Langevin``, ``"NPT"`` is
  ``LangevinBAOAB``, ``"NVE"`` is ``NVE`` (4-T10). An explicit ``integrator=`` (any
  ``fm.integrators`` record) wins; a conflicting non-default ``ensemble`` or argument raises
  ``ValueError``;
- NVE has no target: the momenta are the structure's (``fm.md`` draws no velocities),
  ``temperature_K=``, ``schedule=`` and ``mc=`` raise ``ValueError``, and the workload's
  ``max_temperature_K`` is ``nve_max_temperature_K`` of the structure (conductor's answer to 4-T10);
- what it can infer from the whole run (an explicit argument always wins): the start target and
  ``max_temperature_K`` from ``temperature_K`` or the schedule, the MC moves, species changes,
  ``min_mc_temperature_K`` and the species superset (the structure's plus every species ``mc=``
  names) from ``mc=``;
- the dispersion record: ``"d3bj"`` expands from the head's training functional in the label table
  (``longrange.d3.expand``);
- the defaults: friction 0.02 per ASE time unit (``main.py``'s ``--friction``, about
  0.001965/fs), a drawn and recorded seed, the output directory ``./<structure stem>.fastmace/``
  (``./fastmace-run/`` for an in-memory ``Atoms``), and ``repeat=(1, 1, 1)``.

A D3 record goes into the ``ForceField`` (``dispersion=``; 3b-T04), and the report and ``metadata.json``
show it.

MC (3-T15): with a schedule, the move set is attached only over the step ranges whose target is at
or above ``min_mc_temperature_K`` (``mc_ranges``), through ``Simulation.run(mc=…)`` and
``run(mc=None)``; the ranges are in the report and its ``metadata.json`` block.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# main.py's --friction 0.02, per ASE time unit (conductor answer to 1b-T01 point 5; SPEC §7.7).
DEFAULT_FRICTION_ASE = 0.02
DEFAULT_OUTPUT = "fastmace-run"  # the output directory for an in-memory Atoms
OUTPUT_SUFFIX = ".fastmace"  # ./<structure stem>.fastmace/ for a structure file
ENSEMBLES = ("NVE", "NVT", "NPT")
_SEED_BOUND = 2**31
NVE_MIN_TEMPERATURE_K = 1.0  # the floor of an NVE run's max_temperature_K (a start from rest)


def nve_max_temperature_K(atoms: Any) -> float:
    """An NVE run's ``Workload.max_temperature_K``: twice the structure's kinetic temperature (equipartition:
    a start at a minimum shares the kinetic energy with the potential), at least ``NVE_MIN_TEMPERATURE_K``
    (conductor's answer to 4-T10). ASE's ``Atoms.get_temperature`` count; ``metadata.json`` records the value
    with the workload."""
    temperature = float(atoms.get_temperature()) if len(atoms) else 0.0
    return max(2.0 * temperature, NVE_MIN_TEMPERATURE_K)


def default_friction_per_fs() -> float:
    """``fm.md``'s Langevin friction, per fs: 0.02 per ASE time unit (``main.py``), about 0.001965/fs."""
    from ase import units

    return DEFAULT_FRICTION_ASE * units.fs


@dataclass(frozen=True)
class RunResult:
    """What ``fm.md`` returns: the report and the expert objects it built.

    - ``summary``: the one-screen report (§7.7, 1b-T15), ``simulation.report.text()`` after the run;
    - ``plan``, ``gate``: the ``Plan`` and gate records (§9.5, §12.3);
    - ``simulation``: the ``Simulation``, for restarts and the expert API; its output is flushed
      and it can run on (``simulation.run``);
    - ``metadata``: the dict written to ``metadata.json`` (§13);
    - ``output``: the output directory.
    """

    summary: str
    plan: Any
    gate: Any
    simulation: Any
    metadata: dict
    output: str


def md(structure: Any, *, model: Any, head: str | None = None, charge: int | None = None,
       spin_multiplicity: int | None = None, sha256: str | None = None, trusted: bool = False,
       ensemble: str = "NVT", temperature_K: float | None = None, schedule: Any = None,
       pressure_GPa: float = 0.0, cell_mode: str = "hydrostatic", timestep_fs: float = 0.5,
       steps: int | None = None, mc: Any = None, min_mc_temperature_K: float | None = None,
       dispersion: Any = None, repeat: tuple[int, int, int] = (1, 1, 1), output: Any = None,
       seed: int | None = None, check: str = "skip", integrator: Any = None, mode: str | None = None,
       allow_reference: bool = False, fallback: Any = None, gate: Any = None, runner: Any = None,
       engine: Any = None, neighbours: Any = None, calibrate: bool = False) -> RunResult:
    """Run MD on ``structure`` (path or ``Atoms``) with ``model`` and return a ``RunResult``.

    Arguments as SPEC §7.7 level 2:

    - ``model``: a label, path or URL, with ``head``, ``charge``, ``spin_multiplicity`` and
      ``sha256``/``trusted`` as ``fm.load``; or an ``fm.load`` result (then without those);
    - ``ensemble``: ``"NVT"`` (Langevin), ``"NPT"`` (BAOAB with ``pressure_GPa`` and ``cell_mode``)
      or ``"NVE"`` (velocity Verlet from the structure's momenta, no ``temperature_K``).
      ``integrator=`` chooses any ``fm.integrators`` record explicitly;
    - ``temperature_K`` or ``schedule`` (an ``fm.protocols`` schedule); ``steps`` (optional with a
      schedule: the whole schedule);
    - ``mc`` and ``min_mc_temperature_K``: one ``fm.mc`` record or a sequence of them, checked as §7.7
      says, attached where the target is at or above ``min_mc_temperature_K``;
    - ``dispersion``: ``None``, ``"d3bj"`` or an ``fm.D3``: the complete record goes into the force field
      (MACE plus D3, SPEC §11.2) and is shown in the report; a functional other than the training one
      is a report warning;
    - ``repeat``: the supercell, ``(1, 1, 1)`` by default (never expanded silently);
    - ``output``: a directory or an ``fm.Output``; default ``./<structure stem>.fastmace/``, or
      ``./fastmace-run/`` for an ``Atoms``;
    - ``seed``: the noise seed; ``None`` draws one, recorded in ``metadata.json``;
    - ``check``: as ``Simulation`` (the gate is opt-in, ``check="gate"``);
    - ``mode``, ``allow_reference``, ``fallback``, ``gate``, ``runner``, ``engine``, ``neighbours``:
      the planner inputs, passed to ``Simulation`` as they are (``simulation.PlannerOptions``);
    - ``calibrate``: ``True`` measures the backends the rules rank equal (hybrid against plain cuEq) on this
      structure before the run and takes plain cuEq only if it is faster by more than 5 % in every pair (SPEC
      §9.1, decision 55; ``CalibrationConfig``, ``planner.calibrate``). Off by default: it costs four child
      builds. Refused with MC moves.

    Every ``ValueError`` and refusal comes before the model is loaded.
    """
    from fastmace import integrators
    from fastmace.longrange import d3
    from fastmace.model.loading import LoadedModel
    from fastmace.planner.workload import SPECIES_MOVES, Workload
    from fastmace.protocols import Schedule, ScheduleState
    from fastmace.simulation import Simulation, _steps, check_mode, mc_moves, planner_options

    check_mode(check)
    if not isinstance(calibrate, bool):
        raise TypeError(f"calibrate must be a bool, got {calibrate!r}")
    from fastmace.config import CalibrationConfig

    planner_inputs = planner_options(mode=mode, allow_reference=allow_reference, fallback=fallback, gate=gate,
                                     runner=runner, engine=engine, neighbours=neighbours,
                                     calibration=CalibrationConfig(calibrate=True) if calibrate else None)
    repeat = _repeat(repeat)
    if not isinstance(ensemble, str) or ensemble.upper() not in ENSEMBLES:
        raise ValueError(f"ensemble={ensemble!r} is not one of {ENSEMBLES} (SPEC §7.7)")
    ensemble = ensemble.upper()
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError(f"seed must be a nonnegative int or None, got {seed!r}")

    # The model's name and training functional, from the label table where it is a label: the
    # multi-head ValueError and the dispersion checks need no checkpoint.
    if isinstance(model, LoadedModel):
        options = {"head": head, "charge": charge, "spin_multiplicity": spin_multiplicity, "sha256": sha256}
        given = [name for name, value in options.items() if value is not None] + (["trusted"] if trusted else [])
        if given:
            raise ValueError(f"model is already loaded; {', '.join(given)} belong to fm.load (SPEC §7.7)")
        name, functional = model.label or str(model.path), model.functional
    elif isinstance(model, (str, os.PathLike)):
        from fastmace.model import labels

        name = str(model)
        entry = labels.find(name, head)
        functional = entry.functional if entry is not None else None
    else:
        raise TypeError(f"model must be a label, path, URL or fm.load result, got {type(model).__name__}")

    # The integrator record: explicit, or from ensemble and the physics arguments.
    if integrator is not None:
        if type(integrator) not in integrators.RECORDS.values():
            raise TypeError(f"integrator must be an fm.integrators record ({', '.join(integrators.RECORDS)}), got "
                            f"{type(integrator).__name__}")
        record_ensemble = integrator.to_config(seed=0).ensemble.upper()
        if ensemble != "NVT" and ensemble != record_ensemble:
            raise ValueError(f"ensemble={ensemble!r} conflicts with integrator={type(integrator).__name__}; the "
                             "explicit integrator decides, so drop ensemble= (SPEC §7.7)")
        ensemble = record_ensemble
        conflicts = []
        if temperature_K is not None and temperature_K != getattr(integrator, "temperature_K", None):
            conflicts.append(f"temperature_K={temperature_K!r}")
        if timestep_fs != 0.5 and timestep_fs != integrator.timestep_fs:
            conflicts.append(f"timestep_fs={timestep_fs!r}")
        if pressure_GPa != 0.0 and pressure_GPa != getattr(integrator, "pressure_GPa", None):
            conflicts.append(f"pressure_GPa={pressure_GPa!r}")
        if cell_mode != "hydrostatic" and cell_mode != getattr(integrator, "cell_mode", None):
            conflicts.append(f"cell_mode={cell_mode!r}")
        if conflicts:
            raise ValueError(f"{', '.join(conflicts)} conflicts with the explicit integrator's fields; set them in "
                             f"the integrator record only (SPEC §7.7)")
        timestep_fs = integrator.timestep_fs
    elif ensemble != "NPT" and (pressure_GPa != 0.0 or cell_mode != "hydrostatic"):
        raise ValueError("pressure_GPa and cell_mode are for ensemble='NPT' (SPEC §7.7)")
    nve = ensemble == "NVE"
    if nve:
        given = [name for name, value in (("temperature_K", temperature_K), ("schedule", schedule), ("mc", mc))
                 if value is not None]
        if given:
            raise ValueError(f"ensemble='NVE' has no target temperature, so {', '.join(given)} cannot apply; the "
                             "momenta are the structure's (set them on the Atoms, for example with ASE's "
                             "MaxwellBoltzmannDistribution) (SPEC §7.7)")

    # The temperature: the start target, the highest and (for MC) the lowest of the run.
    if schedule is not None:
        if not isinstance(schedule, Schedule):
            raise TypeError(f"schedule must be an fm.protocols schedule (ramp, hold, +), got {type(schedule).__name__}")
        if temperature_K is not None:
            raise ValueError("temperature_K and schedule= exclude each other: the schedule gives every target "
                             "(SPEC §7.7)")
        position = ScheduleState(schedule, timestep_fs)
        count = position.steps_to_run(steps)
        start = float(integrator.temperature_K) if integrator is not None else float(position.target_K)
        # The whole schedule's maximum (§7.7), also when steps= runs only part of it: a run continued
        # later (run() or a resume, 3-T14) keeps the plan its skin and buckets were sized for.
        highest = max(start, position.max_target())
    else:
        if temperature_K is None and integrator is None and not nve:
            raise ValueError("fm.md needs temperature_K= or schedule= (SPEC §7.7)")
        if steps is None:
            raise ValueError("steps= is required without a schedule: nothing else fixes the run's length (SPEC §7.7)")
        _steps(steps)
        start = None if nve else float(integrator.temperature_K if integrator is not None else temperature_K)
        highest = start

    atoms = _structure(structure)
    if repeat != (1, 1, 1):
        atoms = atoms.repeat(repeat)
    if nve:
        highest = nve_max_temperature_K(atoms)

    # The workload: the highest target, and what mc= declares.
    workload_fields = {"max_temperature_K": highest, "min_mc_temperature_K": min_mc_temperature_K}
    if mc is not None:
        mc = mc_moves(mc)
        moves = _mc_moves(mc)
        if min_mc_temperature_K is None:
            if schedule is not None:
                raise ValueError("mc= with a schedule needs min_mc_temperature_K=: no value can be inferred safely "
                                 "(an anneal from 0 K would make the ΔE floor zero) (SPEC §7.7, §12.3)")
            if start <= 0:
                raise ValueError("temperature_K=0 with mc=: the MC temperature would be 0 K, which makes the ΔE "
                                 "floor 0.01 k_B T_min zero (SPEC §7.7, §12.3)")
            workload_fields["min_mc_temperature_K"] = start
        workload_fields.update(mc_moves=moves, species_changes=any(move in SPECIES_MOVES for move in moves))
    workload = Workload(**workload_fields)

    record, d3_warnings = d3.expand(dispersion, functional=functional, model=name)

    if integrator is None:
        friction = default_friction_per_fs()
        if nve:
            integrator = integrators.NVE(timestep_fs=timestep_fs)
        elif ensemble == "NPT":
            integrator = integrators.LangevinBAOAB(timestep_fs=timestep_fs, temperature_K=start,
                                                   friction_per_fs=friction, pressure_GPa=pressure_GPa,
                                                   cell_mode=cell_mode)
        else:
            integrator = integrators.Langevin(timestep_fs=timestep_fs, temperature_K=start, friction_per_fs=friction)

    output = _output(output, structure)
    if seed is None:
        import secrets

        seed = secrets.randbelow(_SEED_BOUND)

    if not isinstance(model, LoadedModel):
        from fastmace.model.loading import load_pristine

        model = load_pristine(name, head, sha256=sha256, trusted=trusted, charge=charge,
                              spin_multiplicity=spin_multiplicity)
    species = None
    if mc is not None:  # the superset: the structure's species and every species the move set names
        species = sorted({int(z) for z in atoms.numbers} | mc_species(mc))
    force_field = model
    if record is not None:  # MACE plus D3 (3b-T04): the force field Simulation would build, with the record
        from fastmace.model.forcefield import ForceField

        force_field = ForceField(model, atoms, precision="float32", species=species, dispersion=record)
        species = None
    sim = Simulation(atoms, force_field, species=species, integrator=integrator, workload=workload, output=output,
                     seed=seed, check=check,
                     **{field.name: getattr(planner_inputs, field.name) for field in fields(planner_inputs)})
    sim.report.repeat = repeat
    if mc is None:
        sim.run(steps=steps, schedule=schedule)
    elif schedule is None:
        sim.run(steps=steps, mc=mc)
    else:
        given = schedule
        for piece, attached in mc_ranges(ScheduleState(schedule, timestep_fs), count,
                                         workload.min_mc_temperature_K):
            sim.run(steps=piece, schedule=given, mc=mc if attached else None)
            given = None
    sim.flush_output()
    import json

    # The record as written at the end of the run (sim.metadata() would stamp a new time).
    metadata = json.loads((Path(output.directory) / "metadata.json").read_text(encoding="utf-8"))
    return RunResult(summary=sim.report.text(), plan=sim.plan, gate=sim.gate, simulation=sim, metadata=metadata,
                     output=str(output.directory))


def _repeat(repeat):
    if isinstance(repeat, (str, bytes)) or not hasattr(repeat, "__len__") or len(repeat) != 3 \
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in repeat):
        raise ValueError(f"repeat must be three positive ints, got {repeat!r}")
    return tuple(repeat)


def _mc_moves(mc):
    """The move types a move set declares, in order, each once."""
    moves = []
    for block in mc:
        if block.move not in moves:
            moves.append(block.move)
    return tuple(moves)


def mc_species(mc: Any) -> set[int]:
    """The atomic numbers a move set names (swap pairs, transmutable sets)."""
    from fastmace.events import species_numbers
    from fastmace.simulation import mc_moves

    named = set()
    for block in mc_moves(mc):
        for attribute in ("pairs", "species"):
            for item in getattr(block, attribute, None) or ():
                items = list(item) if isinstance(item, (list, tuple)) else [item]
                named.update(int(z) for z in species_numbers(items, len(items)))
    return named


def mc_ranges(position: Any, count: int, min_mc_temperature_K: float) -> list[tuple[int, bool]]:
    """The next ``count`` steps of a schedule position as runs ``(steps, attached)``: attached where the step's
    target is at or above ``min_mc_temperature_K`` (SPEC §7.7), in order, adjacent runs merged."""
    resolved, first = position.resolved, position.step + 1
    runs = []
    for step in range(first, first + count):
        attached = resolved.target(step) >= min_mc_temperature_K
        if runs and runs[-1][1] == attached:
            runs[-1][0] += 1
        else:
            runs.append([1, attached])
    return [(steps, attached) for steps, attached in runs]


def _structure(structure):
    from ase import Atoms

    if isinstance(structure, Atoms):
        return structure.copy()
    if isinstance(structure, (str, os.PathLike)):
        from ase.io import read

        return read(structure)
    raise TypeError(f"structure must be a path or an ase.Atoms, got {type(structure).__name__}")


def _output(output, structure):
    from fastmace.io.output import Output

    if isinstance(output, Output):
        return output
    if output is None:
        if isinstance(structure, (str, os.PathLike)):
            return Output(Path(structure).stem + OUTPUT_SUFFIX)
        return Output(DEFAULT_OUTPUT)
    if isinstance(output, (str, os.PathLike)):
        return Output(output)
    raise TypeError(f"output must be a directory, an fm.Output or None, got {type(output).__name__}")
