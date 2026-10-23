"""The initial relaxation, the quench-frame selection and the quench of an anneal (SPEC §10.3, §10.5).

``docs/assets/Simulated_Annealing.txt`` of the research engine: relax the starting supercell tightly
(LBFGS, fixed cell, fmax 0.01 eV/Å, the MD precision), run the MD to the end, then pick trajectory
frames per target-temperature bin (separately on the heating and the cooling ramp), quench each to
0 K with the same optimizer settings, and always the final frame too.

The optimizer (``profiles.OPTIMIZERS``, plan task 4-T09) is ASE's LBFGS or BFGS on an ASE calculator,
``fm.calc``'s ``FastMACECalculator`` (``relax``, ``ASERelaxer``), or ``fm.optimize``'s LBFGS or FIRE on
the graph engine (``GPURelaxer``). A GPU relaxer builds one engine on its first structure and relaxes
every later one on it (``Simulation.update(cell=, positions=)``, a canonical point: no compile or capture
per frame), each as a fresh relaxation would; a quench's frames keep their own (NPT) cells.

Files of a quench, in ``<run>/quench/`` (``<run>/quench-NAME/`` for another quench of the same run,
``quench_directory``): ``settings.json`` (the resolved settings, written first: an interrupted quench
continues only with them), ``captured.traj`` (the picked MD frames), ``quenched.traj`` (their minima),
``quench.csv`` (one row per frame, written after every frame, so an interrupted quench continues) and
``quench.json`` (settings and summary, written last: a quench with it is complete). Imports neither
torch nor MACE at module level.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fastmace.protocols.profiles import GPU_OPTIMIZERS, resolve_optimizer

QUENCH_MAX_STEPS = 5000
QUENCH_DIRECTORY = "quench"
CAPTURED, QUENCHED, ROWS, SUMMARY = "captured.traj", "quenched.traj", "quench.csv", "quench.json"
SETTINGS = "settings.json"
ROW_FIELDS = ("phase", "bin", "frame", "step", "time_ps", "target_K", "md_temperature_K", "md_potential_eV_atom",
              "singlepoint_eV_atom", "quenched_eV_atom", "converged", "optimizer_steps", "seconds", "fmax_eV_A",
              "volume_A3_atom")
_INTEGER_FIELDS = ("frame", "step", "optimizer_steps")
SELECTION = ("thermostat target temperature; each ramp separately; evenly spaced within each bin; plus the final "
             "frame")


def _max_force(forces) -> float:
    import numpy as np

    return float(np.sqrt((forces ** 2).sum(axis=1)).max())


def single_point(atoms, calc) -> tuple[float, float]:
    """(energy eV/atom, largest force eV/Å) of a copy of ``atoms``."""
    copy = atoms.copy()
    copy.calc = calc
    energy = float(copy.get_potential_energy()) / len(copy)
    return energy, _max_force(copy.get_forces())


def relax(atoms, calc, fmax: float, max_steps: int = QUENCH_MAX_STEPS, logfile=None,
          optimizer: str = "ase-LBFGS", cell_filter=None) -> tuple[Any, dict[str, Any]]:
    """ASE LBFGS (default) or BFGS on ``calc``; returns (relaxed copy, summary).

    At fixed cell, or with ``cell_filter`` (an ``fm.optimize.FrechetCell``) through ASE's
    ``FrechetCellFilter``, whose energies are E + p V and whose fmax covers the cell rows, as ASE's.
    Both optimizers take one force evaluation per step. BFGS also decomposes a dense 3N × 3N matrix at
    every step, which dominates for large cells; on the 210 annealing frames of the research engine LBFGS
    found the same minima (within 0.025 meV/atom) about 25× faster (13 October 2026).
    """
    import numpy as np
    from ase.optimize import BFGS, LBFGS

    optimizer = resolve_optimizer(optimizer)
    if optimizer in GPU_OPTIMIZERS:
        raise ValueError(f"{optimizer} runs on the graph engine: relax with relaxer({optimizer!r}, model), not on "
                         "an ASE calculator")
    relaxed = atoms.copy()
    relaxed.set_momenta(np.zeros((len(relaxed), 3)))
    relaxed.calc = calc
    target = relaxed if cell_filter is None else _ase_filter(relaxed, cell_filter)
    started = time.perf_counter()
    initial = float(target.get_potential_energy())
    run = {"ase-LBFGS": LBFGS, "ase-BFGS": BFGS}[optimizer](target, logfile=None if logfile is None else str(logfile))
    converged = bool(run.run(fmax=fmax, steps=max_steps))
    energy = float(target.get_potential_energy())
    summary = {"optimizer": optimizer, "fixed_cell": cell_filter is None, "fmax_target_eV_A": float(fmax),
               "max_steps": int(max_steps), "converged": converged, "steps": int(run.get_number_of_steps()),
               "seconds": time.perf_counter() - started, "initial_eV_atom": initial / len(relaxed),
               "energy_eV_atom": energy / len(relaxed), "fmax_eV_A": _max_force(target.get_forces())}
    if cell_filter is not None:  # the keys of fastmace.optimize.driver.relax's variable-cell summary
        from ase import units

        summary.update({"cell_A": relaxed.cell.array.tolist(), "volume_A3": float(relaxed.get_volume()),
                        "stress_GPa": (relaxed.get_stress(voigt=False) / units.GPa).tolist(),
                        "scalar_pressure_GPa": float(cell_filter.scalar_pressure_GPa)})
    relaxed.calc = None
    return relaxed, summary


def _ase_filter(atoms, cell):
    """ASE's ``FrechetCellFilter`` with the arguments of the ``fm.optimize.FrechetCell`` record ``cell``."""
    from ase import units
    from ase.filters import FrechetCellFilter

    options = {"scalar_pressure": cell.scalar_pressure_GPa * units.GPa, "hydrostatic_strain": cell.hydrostatic_strain,
               "constant_volume": cell.constant_volume}
    if cell.mask is not None:
        options["mask"] = cell.mask
    if cell.exp_cell_factor is not None:
        options["exp_cell_factor"] = cell.exp_cell_factor
    return FrechetCellFilter(atoms, **options)


class ASERelaxer:
    """``relax`` with one calculator and optimizer, as a relaxer: ``relaxer(atoms, fmax, max_steps, logfile=None)``
    returns (relaxed copy, summary); ``close`` drops the calculator. ``builds`` is 0 (no engine)."""

    builds = 0

    def __init__(self, calc, optimizer: str = "ase-LBFGS", cell_filter=None):
        self.calc, self.optimizer, self.cell_filter = calc, resolve_optimizer(optimizer), cell_filter

    def __call__(self, atoms, fmax: float, max_steps: int = QUENCH_MAX_STEPS, logfile=None):
        return relax(atoms, self.calc, fmax, max_steps, logfile=logfile, optimizer=self.optimizer,
                     cell_filter=self.cell_filter)

    def close(self) -> None:
        self.calc = None


class GPURelaxer:
    """``fm.optimize``'s LBFGS or FIRE on the graph engine (4-T05, 4-T07), as a relaxer (``ASERelaxer``'s calls).

    The first call builds a ``Simulation(optimizer=)`` on its structure; every later one moves that engine to the
    new structure (``update(cell=, positions=)``: the same atoms, any cell, at fixed cell) and relaxes from a fresh
    optimiser state, so a quench compiles and captures once. Another composition builds a new engine. ``builds``
    counts the engines. ``engine`` is the run's ``EngineConfig`` (backend, device) or ``None``. The per-step
    energies and max forces go to ``logfile``, not into the summary.
    """

    def __init__(self, model, optimizer: str = "gpu-LBFGS", *, precision: str = "float32", dispersion=None,
                 engine=None, cell_filter=None, seed: int = 0):
        from fastmace import optimize

        self.optimizer = resolve_optimizer(optimizer)
        if self.optimizer not in GPU_OPTIMIZERS:
            raise ValueError(f"GPURelaxer runs {GPU_OPTIMIZERS}, got {self.optimizer!r}")
        record = {"gpu-LBFGS": optimize.LBFGS, "gpu-FIRE": optimize.FIRE}[self.optimizer]
        self.record = record(cell_filter=cell_filter)
        self.model, self.precision, self.dispersion, self.engine, self.seed = model, precision, dispersion, engine, seed
        self.builds = 0
        self._sim = None
        self._numbers = None

    def __call__(self, atoms, fmax: float, max_steps: int = QUENCH_MAX_STEPS, logfile=None):
        import numpy as np

        if self._sim is not None and not np.array_equal(atoms.numbers, self._numbers):
            self.close()
        if self._sim is None:
            from fastmace.model.forcefield import ForceField
            from fastmace.simulation import Simulation

            field = ForceField(self.model, atoms, precision=self.precision, dispersion=self.dispersion)
            self._sim = Simulation(atoms, field, optimizer=self.record, seed=self.seed, engine=self.engine)
            self._numbers = atoms.numbers.copy()
            self.builds += 1
        else:
            self._sim.update(cell=atoms.cell.array, positions=atoms.positions)
        relaxed, summary = self._sim.relax(fmax=fmax, steps=max_steps)
        energies, forces = summary.pop("energies_eV"), summary.pop("fmax_history_eV_A")
        if logfile is not None:
            lines = [f"{'Step':>6} {'Energy':>18} {'fmax':>12}"]
            lines += [f"{step:6d} {energy:18.6f} {force:12.6f}"
                      for step, (energy, force) in enumerate(zip(energies, forces, strict=True))]
            Path(logfile).write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["optimizer"] = self.optimizer
        return relaxed, summary

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None


def relaxer(optimizer: str, model, *, precision: str = "float32", dispersion=None, engine=None, cell_filter=None):
    """The relaxer of ``optimizer`` (``profiles.OPTIMIZERS``) on ``model``: a ``GPURelaxer`` for ``gpu-*``, else an
    ``ASERelaxer`` on ``FastMACECalculator(model, precision=, dispersion=)``. ``engine`` (an ``EngineConfig``)
    is the GPU relaxer's only."""
    name = resolve_optimizer(optimizer)
    if name in GPU_OPTIMIZERS:
        return GPURelaxer(model, name, precision=precision, dispersion=dispersion, engine=engine,
                          cell_filter=cell_filter)
    from fastmace.ase.calculator import FastMACECalculator

    return ASERelaxer(FastMACECalculator(model, precision=precision, dispersion=dispersion), name, cell_filter)


def select_frames(frames: Sequence[Mapping[str, Any]], min_K: float, bin_K: float, per_bin: int,
                  peak_K: float) -> list[dict[str, Any]]:
    """Pick ``per_bin`` frames per ``bin_K`` target-temperature bin from ``min_K`` up, on each ramp.

    ``frames``: dicts with ``step``, ``target_K`` and ``phase`` (``heating``/``cooling``), in step
    order. Within a bin the picks sit at evenly spaced fractions (j + 1/2) / per_bin of the bin's
    step range (nearest saved frame). Returns the chosen frame dicts with a ``bin`` label, in step
    order.
    """
    chosen = {}
    bins = max(1, math.ceil((peak_K - min_K) / bin_K - 1e-9))
    for phase in ("heating", "cooling"):
        for k in range(bins):
            low, high = min_K + k * bin_K, min(min_K + (k + 1) * bin_K, peak_K)
            members = [f for f in frames if f["phase"] == phase and low - 1e-9 <= f["target_K"]
                       and (f["target_K"] < high - 1e-9 or (k == bins - 1 and f["target_K"] <= high + 1e-9))]
            if not members:
                continue
            first, last = members[0]["step"], members[-1]["step"]
            for j in range(per_bin):
                goal = first + (j + 0.5) / per_bin * (last - first)
                free = [f for f in members if f["step"] not in chosen]
                if not free:
                    break
                pick = min(free, key=lambda f: abs(f["step"] - goal))
                chosen[pick["step"]] = dict(pick, bin=f"{low:g}-{high:g} K")
    return [chosen[s] for s in sorted(chosen)]


def frame_table(resolved, frame_every: int, last_step: int | None = None) -> list[dict[str, Any]]:
    """The trajectory frames of a run along ``resolved`` (a ``ResolvedSchedule``): one at step 0 and one
    every ``frame_every`` steps up to ``last_step`` (default: the schedule's end), with their index,
    step, phase and target."""
    last_step = resolved.total_steps if last_step is None else last_step
    return [{"index": index, "step": step, "phase": resolved.phase(step), "target_K": resolved.target(step)}
            for index, step in enumerate(range(0, last_step + 1, frame_every))]


def picks(resolved, frame_every: int, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The frames a quench relaxes: ``select_frames`` over the run's frames, then the final frame (always,
    as the research engine's ``--quench``; it can repeat a frame a bin picked)."""
    frames = frame_table(resolved, frame_every)
    peak = resolved.max_target(0, resolved.total_steps)
    chosen = select_frames(frames, settings["min_K"], settings["bin_K"], settings["per_bin"], peak)
    return chosen + [dict(frames[-1], phase="final", bin="final")]


def _chunk_lengths(directory: Path) -> list[tuple[Path, int]]:
    from ase.io.trajectory import Trajectory

    from fastmace.io.output import chunk_indices, chunk_paths

    lengths = []
    for index in chunk_indices(directory):
        path = chunk_paths(directory, index)[1]
        if path.is_file():
            with Trajectory(str(path)) as trajectory:
                lengths.append((path, len(trajectory)))
    return lengths


def read_frames_at(directory: str | Path, indices: Sequence[int]) -> tuple[dict[int, Any], int]:
    """The trajectory frames ``indices`` over all chunks of a run (``{index: Atoms}``) and the frame count.

    Only those frames are read: a 270 ps anneal has 27,000 frames.
    """
    from ase.io.trajectory import Trajectory

    lengths = _chunk_lengths(Path(directory))
    total = sum(length for _, length in lengths)
    wanted = sorted(set(int(i) for i in indices))
    found, offset = {}, 0
    for path, length in lengths:
        inside = [i for i in wanted if offset <= i < offset + length]
        if inside:
            with Trajectory(str(path)) as trajectory:
                for i in inside:
                    found[i] = trajectory[i - offset]
        offset += length
    missing = [i for i in wanted if i not in found]
    if missing:
        raise ValueError(f"{directory}: frames {missing[:5]} are beyond the trajectory's {total} frames")
    return found, total


def _thermo_by_step(directory: Path) -> dict[int, dict[str, float]]:
    from fastmace.io.output import read_thermo

    header, rows = read_thermo(directory)
    if not header:
        return {}
    column = {name: header.index(name) for name in ("step", "temperature_K", "potential_eV_atom") if name in header}
    return {int(row[column["step"]]): {"temperature_K": float(row[column["temperature_K"]]),
                                       "potential_eV_atom": float(row[column["potential_eV_atom"]])}
            for row in rows if row}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """``quench.csv`` rows with their numeric fields converted."""
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in ("phase", "bin"):
                    row[key] = value
                elif key == "converged":
                    row[key] = value == "True"
                else:
                    row[key] = int(value) if key in _INTEGER_FIELDS else float(value)
            rows.append(row)
    return rows


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def quench_summary(rows: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any] | None:
    values = [r["quenched_eV_atom"] for r in rows if r["phase"] == phase]
    if not values:
        return None
    return {"n": len(values), "mean_eV_atom": statistics.fmean(values),
            "sd_eV_atom": statistics.stdev(values) if len(values) > 1 else 0.0}


def _keep_frames(path: Path, count: int) -> None:
    """Cut a trajectory to its first ``count`` frames (a quench stopped between a row and its frames)."""
    from ase.io.trajectory import Trajectory

    if not path.is_file():
        if count:
            raise ValueError(f"{path} is missing but {count} quench rows exist")
        return
    with Trajectory(str(path)) as trajectory:
        length = len(trajectory)
        frames = [trajectory[i] for i in range(min(count, length))]
    if length < count:
        raise ValueError(f"{path} has {length} frames for {count} quench rows")
    if length > count:
        with Trajectory(str(path), "w") as trajectory:
            for frame in frames:
                trajectory.write(frame)


def quench_directory(name: str | None) -> str:
    """The directory of a quench: ``quench`` (``name`` ``None``), else ``quench-NAME`` (letters, digits, ``._-``)."""
    if name is None:
        return QUENCH_DIRECTORY
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or set(name) == {"."}:
        raise ValueError(f"a quench name is letters, digits, '.', '_' and '-', got {name!r}")
    return f"{QUENCH_DIRECTORY}-{name}"


def _check_settings(out: Path, settings: Mapping[str, Any]) -> None:
    """Write ``settings.json``, or refuse to continue a quench that ran with other settings. A quench from before
    4-T09 has no ``settings.json``; it continues, and gets one."""
    path = out / SETTINGS
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(key for key in set(recorded) | set(settings) if recorded.get(key) != settings.get(key))
        if changed:
            what = ", ".join(f"{key} {recorded.get(key)!r} (now {settings.get(key)!r})" for key in changed)
            raise ValueError(f"{out} ran with {what}: continue it with its own settings, or quench again into "
                             "another directory (fastmace quench RUN_DIR --into NAME)")
        return
    path.write_text(json.dumps(dict(settings), indent=2) + "\n", encoding="utf-8")


def quench_run(directory: str | Path, relaxer, resolved, frame_every: int, timestep_fs: float,
               settings: Mapping[str, Any], *, name: str = QUENCH_DIRECTORY,
               log: Callable[[str], Any] = print) -> dict[str, Any]:
    """Quench the picked frames of the finished run in ``directory`` with ``relaxer``; returns the summary.

    ``relaxer``: an ``ASERelaxer`` or ``GPURelaxer`` (``relaxer(...)``; not closed here), or an ASE calculator
    for the ASE optimizers. ``resolved``: the run's ``ResolvedSchedule``; ``frame_every``: its trajectory
    interval in steps; ``settings``: ``profiles.QUENCH_KEYS``; ``name``: the quench's directory in
    ``directory`` (``quench_directory``). A complete quench (``quench.json``) is returned as it is; an
    interrupted one continues after its last row, with the same settings only.
    """
    from ase.io.trajectory import Trajectory

    directory = Path(directory)
    settings = {**dict(settings), "optimizer": resolve_optimizer(settings["optimizer"])}
    if not isinstance(relaxer, (ASERelaxer, GPURelaxer)):
        if settings["optimizer"] in GPU_OPTIMIZERS:
            raise ValueError(f"a {settings['optimizer']} quench needs relaxer({settings['optimizer']!r}, model), "
                             "not an ASE calculator")
        relaxer = ASERelaxer(relaxer, settings["optimizer"])
    elif relaxer.optimizer != settings["optimizer"]:
        raise ValueError(f"the relaxer runs {relaxer.optimizer}, the quench settings name {settings['optimizer']}")
    out = directory / name
    if (out / SUMMARY).is_file():
        return json.loads((out / SUMMARY).read_text(encoding="utf-8"))
    out.mkdir(exist_ok=True)
    chosen = picks(resolved, frame_every, settings)
    total = sum(length for _, length in _chunk_lengths(directory))
    expected = resolved.total_steps // frame_every + 1
    if total != expected:
        raise ValueError(f"{directory}: {total} trajectory frames, expected {expected} (every {frame_every} steps "
                         f"over {resolved.total_steps})")
    frames, _ = read_frames_at(directory, [pick["index"] for pick in chosen])
    thermo = _thermo_by_step(directory)
    rows = read_rows(out / ROWS) if (out / ROWS).is_file() else []
    if len(rows) > len(chosen) or any(row["frame"] != pick["index"] for row, pick in zip(rows, chosen)):
        raise ValueError(f"{out / ROWS} does not belong to this run's picks")
    _check_settings(out, settings)
    for trajectory in (CAPTURED, QUENCHED):
        _keep_frames(out / trajectory, len(rows))
    if rows:
        log(f"quench: continuing after {len(rows)} of {len(chosen)} frames")
    started = time.perf_counter()
    builds = relaxer.builds
    for number, pick in enumerate(chosen[len(rows):], len(rows) + 1):
        atoms = frames[pick["index"]].copy()
        atoms.calc = None
        relaxed, summary = relaxer(atoms, settings["fmax"], settings["max_steps"])
        for trajectory, frame in ((CAPTURED, atoms), (QUENCHED, relaxed)):
            path = out / trajectory
            with Trajectory(str(path), "a" if path.is_file() else "w") as handle:
                handle.write(frame)
        sample = thermo.get(pick["step"])
        rows.append({"phase": pick["phase"], "bin": pick["bin"], "frame": pick["index"], "step": pick["step"],
                     "time_ps": pick["step"] * timestep_fs / 1000, "target_K": pick["target_K"],
                     "md_temperature_K": sample["temperature_K"] if sample else math.nan,
                     "md_potential_eV_atom": sample["potential_eV_atom"] if sample else summary["initial_eV_atom"],
                     "singlepoint_eV_atom": summary["initial_eV_atom"], "quenched_eV_atom": summary["energy_eV_atom"],
                     "converged": summary["converged"], "optimizer_steps": summary["steps"],
                     "seconds": summary["seconds"], "fmax_eV_A": summary["fmax_eV_A"],
                     "volume_A3_atom": atoms.get_volume() / len(atoms)})
        write_rows(out / ROWS, rows)
        log(f"quench {number}/{len(chosen)}: {pick['phase']} {pick['target_K']:.0f} K -> "
            f"{summary['energy_eV_atom']:.6f} eV/atom in {summary['steps']} steps ({summary['seconds']:.1f} s)"
            f"{'' if summary['converged'] else ' NOT CONVERGED'}")
    result = {"settings": {**dict(settings), "fixed_cell": True, "cell": "frame", "selection": SELECTION},
              "frames": len(rows), "frame_every": int(frame_every), "trajectory_frames": total,
              "quench_wall_s": time.perf_counter() - started, "engines": relaxer.builds - builds,
              "quench_seconds_sum": sum(r["seconds"] for r in rows),
              "not_converged": sum(not r["converged"] for r in rows),
              "heating": quench_summary(rows, "heating"), "cooling": quench_summary(rows, "cooling"),
              "final_quenched_eV_atom": rows[-1]["quenched_eV_atom"]}
    (out / SUMMARY).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
