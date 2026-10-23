"""ASE relaxation scripts on the graph engine: ``fastmace.ase.LBFGS`` and ``FIRE`` (SPEC §7.7; decision 48).

An ASE script is ported by changing the import and adding ``model=``. Each class takes the argument names,
units and defaults of its ASE 3.27 class (``SUPPORTED``) and refuses the rest with the reason
(``REFUSED``): ``restart`` other than ``None`` (a relaxation writes no restart files, conductor's choice 5),
LBFGS's ``use_line_search=True``, FIRE's deprecated ``maxmove`` and ``position_reset_callback``, and a
non-default ``comm``/``master``. The FastMACE additions are keyword-only (``ADDITIONS``).

``atoms`` is an ``Atoms`` (fixed cell) or ``ase.filters.FrechetCellFilter(atoms, ...)`` (variable cell, read
into ``fm.optimize.FrechetCell``); every other filter (``UnitCellFilter``, ``ExpCellFilter``,
``StrainFilter``, ...) is refused with the reason (decision 48).

Nothing here builds an engine: the classes build a ``fm.Simulation(optimizer=...)`` at the first ``run``,
``irun`` or ``step``, and ``run(fmax, steps)`` is ``Simulation.relax`` in chunks that end where an observer
or a trajectory frame is due (ASE's ``attach`` rule). ``logfile`` is ASE's optimiser log (ASE's ``Log``,
``"-"`` by default as ASE's), one line per step from the relaxation's per-step energies and max forces;
``trajectory=`` is one ASE ``Trajectory`` with a frame every ``loginterval`` steps. Before the observers of
a step the relaxed state is written into ``atoms`` (positions and cell) with a single-point results cache
as ``atoms.calc``. An observer that changes ``atoms`` raises: the optimiser's history would no longer
describe the structure (conductor's answer to 4-T10).
"""

from __future__ import annotations

import sys
from typing import Any

from fastmace.ase.md import _REFUSALS, ASE_OPTIMIZER_ARGUMENTS, PLAN_INFO_KEYS, _dispersion, _model, _refuse

# The FastMACE additions, keyword-only. A relaxation is deterministic and writes no output files
# (Simulation.relax), so seed= and output= of the MD classes have no meaning here.
ADDITIONS = ("model", "force_field", "head", "check", "dispersion")
MAPPED = ("LBFGS", "FIRE")
DEFAULT_MAX_STEPS = 100_000_000  # ase.optimize.optimize.DEFAULT_MAX_STEPS
FILTER_REASON = ("only ase.filters.FrechetCellFilter is supported for a variable cell (ASE's recommended filter, "
                 "decision 48); relax with FrechetCellFilter(atoms, ...)")


def _arguments(name, refused):
    arguments = ASE_OPTIMIZER_ARGUMENTS[name]
    return (tuple(argument for argument in arguments if argument not in refused),
            tuple(argument for argument in arguments if argument in refused))


def _structure(atoms):
    """``(Atoms, fm.optimize.FrechetCell or None)`` of the ``atoms`` argument: an ``Atoms`` or a
    ``FrechetCellFilter``; any other optimisable is refused with the reason."""
    from ase import Atoms
    from ase.filters import FrechetCellFilter

    if isinstance(atoms, Atoms):
        return atoms, None
    if type(atoms) is not FrechetCellFilter:
        raise _refuse(f"{type(atoms).__name__} is refused: {FILTER_REASON}", code="ase-cell-filter")
    import numpy as np
    from ase import units

    from fastmace import optimize

    inner = atoms.atoms
    if not np.array_equal(np.asarray(atoms.orig_cell), inner.cell.array):
        raise _refuse("FrechetCellFilter(orig_cell=...) other than the atoms' cell is refused: the device filter "
                      "deforms from the cell the relaxation starts in", code="ase-cell-filter-orig-cell")
    mask = np.asarray(atoms.mask)
    exp_cell_factor = float(atoms.exp_cell_factor)
    cell = optimize.FrechetCell(
        scalar_pressure_GPa=float(atoms.scalar_pressure) / units.GPa,
        hydrostatic_strain=bool(atoms.hydrostatic_strain), constant_volume=bool(atoms.constant_volume),
        mask=None if (mask == 1).all() else tuple(tuple(int(flag) for flag in row) for row in mask),
        exp_cell_factor=None if exp_cell_factor == float(len(inner)) else exp_cell_factor)
    return inner, cell


class _Optimizer:
    """What both optimiser classes share: arguments, the lazy ``Simulation``, the log, trajectory and observers."""

    SUPPORTED: tuple[str, ...] = ()
    REFUSED: tuple[str, ...] = ()

    def _refuse_extra(self, extra, refused):
        unknown = sorted(set(extra) - set(self.REFUSED) - {"communicator"})
        if unknown:
            raise TypeError(f"{type(self).__name__} got unexpected argument(s) {', '.join(unknown)}; it takes ASE's "
                            f"{', '.join(self.SUPPORTED)} and FastMACE's keyword-only {', '.join(ADDITIONS)} "
                            "(SPEC §7.7)")
        for name, value in extra.items():
            accepts, reason = _REFUSALS[name]
            if not accepts(value):
                raise _refuse(f"{type(self).__name__}({name}=...) is refused: {reason}", code=f"ase-{name}")
        for name, (value, reason) in refused.items():
            if value:
                raise _refuse(f"{type(self).__name__}({name}=...) is refused: {reason}", code=f"ase-{name}")

    def _setup(self, atoms, record_fields, *, restart, logfile, trajectory, append_trajectory, loginterval,
               model, force_field, head, check, dispersion):
        from pathlib import Path

        from ase.optimize.optimize import Log

        from fastmace.simulation import check_mode

        check_mode(check)
        if restart is not None:
            raise _refuse(f"{type(self).__name__}(restart=...) is refused: a relaxation writes no restart files "
                          "(conductor's choice 5); relax again from the structure", code="ase-restart")
        if isinstance(loginterval, bool) or not isinstance(loginterval, int) or loginterval < 1:
            raise ValueError(f"loginterval must be a positive int, got {loginterval!r}")
        inner, cell = _structure(atoms)
        self._model = _dispersion(dispersion, _model(model, force_field, head), inner)
        self.atoms = atoms  # ASE's attribute: what was passed (the filter, if any)
        self._atoms = inner
        self.record = self.RECORD(**record_fields, cell_filter=cell)
        self.check = check
        self.restart = restart
        self.logfile = Log(logfile, comm=_world())
        self.observers: list[tuple[Any, int, tuple, dict]] = []
        self.nsteps = 0
        self.max_steps = 0
        self.loginterval = loginterval
        self._trajectory = None
        if trajectory is not None and not hasattr(trajectory, "write"):
            trajectory = Path(trajectory)
            if not append_trajectory:
                trajectory.unlink(missing_ok=True)  # as ASE's BaseDynamics
        self._trajectory = trajectory
        self._sim = None
        self.relaxations: list[dict] = []  # each Simulation.relax summary, in order

    # ------------------------------------------------------------------ ASE members

    def todict(self) -> dict[str, Any]:
        description = {"type": "optimization", "optimizer": type(self).__name__, "restart": str(self.restart)}
        for attr in ("maxstep", "alpha", "max_steps", "fmax"):  # ASE's Optimizer.todict
            if hasattr(self, attr):
                description[attr] = getattr(self, attr)
        return description

    def get_number_of_steps(self) -> int:
        return self.nsteps

    def attach(self, function, interval=1, *args, **kwargs) -> None:
        """Attach ``function(*args, **kwargs)``, called every ``interval`` steps (or once at step ``|interval|``
        for ``interval <= 0``), as ASE's ``attach``; an object without ``__call__`` has its ``write`` attached."""
        if not callable(function):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def run(self, fmax: float = 0.05, steps: int = DEFAULT_MAX_STEPS) -> bool:
        """Relax until the largest force is below ``fmax`` (eV/Å) or ``steps`` steps are done; whether it converged."""
        converged = False
        for converged in self._iterate(fmax, steps, chunked=True):
            pass
        return converged

    def irun(self, fmax: float = 0.05, steps: int = DEFAULT_MAX_STEPS):
        """ASE's ``irun``: yields whether the structure is converged, at the start and after every step."""
        return self._iterate(fmax, steps, chunked=False)

    def step(self, gradient=None) -> None:
        """One optimiser step, whether or not the structure is converged (ASE's ``step``), without the log or
        the observers."""
        if gradient is not None:
            raise TypeError("step(gradient=...) is not supported: the engine evaluates the forces")
        self._relax(sys.float_info.min, 1)  # an fmax no force reaches: exactly one step
        self.nsteps += 1
        self._write_atoms()

    def converged(self, gradient=None) -> bool:
        """Whether the largest force of the current structure is below ``fmax`` (ASE's ``converged``)."""
        if gradient is not None:
            raise TypeError("converged(gradient=...) is not supported: the engine evaluates the forces")
        return bool(self._relax(getattr(self, "fmax", 0.05), 0)["converged"])

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ FastMACE members

    @property
    def simulation(self):
        """The underlying ``fm.Simulation`` (``None`` before the first ``run``, ``irun`` or ``step``)."""
        return self._sim

    @property
    def plan(self):
        return None if self._sim is None else self._sim.plan

    @property
    def gate(self):
        return None if self._sim is None else self._sim.gate

    @property
    def summary(self) -> dict | None:
        """The last ``Simulation.relax`` summary (steps, converged, energies, fmax, ...)."""
        return self.relaxations[-1] if self.relaxations else None

    # ------------------------------------------------------------------ the relaxation

    def _simulation(self):
        if self._sim is None:
            from fastmace import simulation

            self._sim = simulation.Simulation(self._atoms, self._model, optimizer=self.record, seed=0,
                                              check=self.check)
        return self._sim

    def _relax(self, fmax, steps):
        _, summary = self._simulation().relax(fmax=fmax, steps=steps)
        self.relaxations.append(summary)
        return summary

    def _iterate(self, fmax, steps, *, chunked):
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or not fmax > 0:
            raise ValueError(f"fmax must be a positive number (eV/Angstrom), got {fmax!r}")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(f"steps must be a nonnegative int, got {steps!r}")
        self.fmax = fmax
        self.max_steps = self.nsteps + steps
        summary = self._relax(fmax, 0)  # the start: its energy, max force and convergence
        if self.nsteps == 0:
            self._log(summary["energies_eV"][0], summary["fmax_history_eV_A"][0])
            self._at(0, first=True)
        converged = bool(summary["converged"])
        yield converged
        while not converged and self.nsteps < self.max_steps:
            stop = self._next_stop() if chunked else self.nsteps + 1
            summary = self._relax(fmax, stop - self.nsteps)
            for energy, force in zip(summary["energies_eV"][1:], summary["fmax_history_eV_A"][1:]):
                self.nsteps += 1
                self._log(energy, force)
            converged = bool(summary["converged"])
            self._at(self.nsteps)
            yield converged
        self._write_atoms()

    def _next_stop(self):
        """The next step after ``nsteps`` where an observer or a trajectory frame is due, at most ``max_steps``."""
        now = self.nsteps
        stops = [self.max_steps]
        intervals = [interval for _, interval, _, _ in self.observers]
        if self._trajectory is not None:
            intervals.append(self.loginterval)
        for interval in intervals:
            if interval > 0:
                stops.append((now // interval + 1) * interval)
            elif abs(interval) > now:
                stops.append(abs(interval))
        return min(stops)

    def _log(self, energy, force):
        """ASE's ``Optimizer.log`` line: the energy is the optimisable's (E + p V with a cell filter)."""
        import time

        name = type(self).__name__
        if self.nsteps == 0:
            self.logfile.write("%s  %4s %8s %15s  %12s\n" % (" " * len(name), "Step", "Time", "Energy", "fmax"))
        now = time.localtime()
        self.logfile.write("%s:  %3d %02d:%02d:%02d %15.6f %15.6f\n" % (name, self.nsteps, now[3], now[4], now[5],
                                                                       energy, force))

    def _due(self, step):
        return [(function, args, kwargs) for function, interval, args, kwargs in self.observers
                if (interval > 0 and step % interval == 0) or (interval <= 0 and step == abs(interval))]

    def _at(self, step, first=False):
        """Step ``step`` is relaxed: its trajectory frame, then the observers due, with the state in ``atoms``."""
        frame = self._trajectory is not None and step % self.loginterval == 0
        if first and self._trajectory is not None and not self._trajectory_is_empty():
            frame = False  # as ASE: a restarted trajectory gets no second copy of the start
        due = self._due(step)
        if not frame and not due:
            return
        self._write_atoms()
        if frame:
            self._write_frame()
        if due:
            before = _Snapshot(self._atoms)
            for function, args, kwargs in due:
                function(*args, **kwargs)
            before.check(self._atoms)

    def _write_atoms(self):
        """The relaxed positions and cell into ``atoms`` with a results cache as ``atoms.calc``."""
        if self._sim is None:
            return
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.stress import full_3x3_to_voigt_6_stress

        committed = self._sim.committed()
        atoms = self._atoms
        atoms.set_cell(committed.cell)
        atoms.set_positions(committed.positions, apply_constraint=False)
        extra = {} if committed.stress is None else {"stress": full_3x3_to_voigt_6_stress(committed.stress)}
        atoms.calc = SinglePointCalculator(atoms, energy=committed.energy, free_energy=committed.energy,
                                           forces=committed.forces.copy(), **extra)

    def _trajectory_is_empty(self):
        from ase.io.trajectory import Trajectory

        if hasattr(self._trajectory, "write"):
            return len(self._trajectory) == 0 if hasattr(self._trajectory, "__len__") else True
        if not self._trajectory.exists():
            return True
        with Trajectory(str(self._trajectory)) as reader:
            return len(reader) == 0

    def _write_frame(self):
        from ase.io.trajectory import Trajectory

        description = self.todict() | {"interval": self.loginterval}
        if hasattr(self._trajectory, "write"):
            self._trajectory.set_description(description)
            self._trajectory.write(self._atoms)
            return
        with Trajectory(str(self._trajectory), mode="a") as writer:
            writer.set_description(description)
            writer.write(self._atoms)


class LBFGS(_Optimizer):
    """ASE 3.27's ``ase.optimize.LBFGS`` on FastMACE's graph engine (``fm.optimize.LBFGS``, SPEC §10.3).

    ``LBFGS(atoms, restart=None, logfile="-", trajectory=None, maxstep=None, memory=100, damping=1.0,
    alpha=70.0, use_line_search=False, *, append_trajectory=False, loginterval=1, model=None, force_field=None,
    head=None, check="skip", dispersion=None)``: ``maxstep`` in Å (``None`` is ASE's 0.2), ``alpha`` in eV/Å²
    as ASE's. ``atoms`` an ``Atoms`` or a ``FrechetCellFilter``.
    """

    SUPPORTED, REFUSED = _arguments("LBFGS", ("use_line_search", "master", "comm"))

    @property
    def RECORD(self):  # noqa: N802
        from fastmace import optimize

        return optimize.LBFGS

    def __init__(self, atoms, restart=None, logfile="-", trajectory=None, maxstep=None, memory=100, damping=1.0,
                 alpha=70.0, use_line_search=False, *, append_trajectory=False, loginterval=1, model=None,
                 force_field=None, head=None, check="skip", dispersion=None, **kwargs):
        self._refuse_extra(kwargs, {"use_line_search": (use_line_search, "FastMACE's LBFGS has no line search "
                                                                          "(ASE's default use_line_search=False)")})
        self.maxstep = 0.2 if maxstep is None else maxstep  # ASE's LBFGS.defaults["maxstep"]
        self.memory, self.damping, self.alpha = memory, damping, alpha
        self._setup(atoms, {"maxstep_A": float(self.maxstep), "memory": memory, "damping": float(damping),
                            "alpha": float(alpha)}, restart=restart, logfile=logfile, trajectory=trajectory,
                    append_trajectory=append_trajectory, loginterval=loginterval, model=model,
                    force_field=force_field, head=head, check=check, dispersion=dispersion)


class FIRE(_Optimizer):
    """ASE 3.27's ``ase.optimize.FIRE`` on FastMACE's graph engine (``fm.optimize.FIRE``, SPEC §10.3).

    ``FIRE(atoms, restart=None, logfile="-", trajectory=None, dt=0.1, maxstep=None, maxmove=None, dtmax=1.0,
    Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99, a=0.1, downhill_check=False, position_reset_callback=None,
    *, append_trajectory=False, loginterval=1, model=None, ...)``: ASE's arguments and defaults (``dt`` and
    ``dtmax`` in ASE time units, ``maxstep`` in Å, ``None`` is ASE's 0.2). Refused: the deprecated ``maxmove``
    (use ``maxstep``) and ``position_reset_callback`` (decision 48).
    """

    SUPPORTED, REFUSED = _arguments("FIRE", ("maxmove", "position_reset_callback", "master", "comm"))

    @property
    def RECORD(self):  # noqa: N802
        from fastmace import optimize

        return optimize.FIRE

    def __init__(self, atoms, restart=None, logfile="-", trajectory=None, dt=0.1, maxstep=None, maxmove=None,
                 dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99, a=0.1, downhill_check=False,  # noqa: N803
                 position_reset_callback=None, *, append_trajectory=False, loginterval=1, model=None,
                 force_field=None, head=None, check="skip", dispersion=None, **kwargs):
        self._refuse_extra(kwargs, {
            "maxmove": (maxmove is not None, "ASE deprecates it; pass maxstep= (decision 48)"),
            "position_reset_callback": (position_reset_callback is not None,
                                        "the downhill check restores positions on the device, without a host "
                                        "callback (decision 48)")})
        self.maxstep = 0.2 if maxstep is None else maxstep  # ASE's FIRE.defaults["maxstep"]
        self.dt, self.dtmax, self.Nmin, self.finc, self.fdec = dt, dtmax, Nmin, finc, fdec
        self.astart, self.fa, self.a, self.downhill_check = astart, fa, a, downhill_check
        self._setup(atoms, {"dt": float(dt), "maxstep_A": float(self.maxstep), "dtmax": float(dtmax), "Nmin": Nmin,
                            "finc": float(finc), "fdec": float(fdec), "astart": float(astart), "fa": float(fa),
                            "a": float(a), "downhill_check": bool(downhill_check)},
                    restart=restart, logfile=logfile, trajectory=trajectory, append_trajectory=append_trajectory,
                    loginterval=loginterval, model=model, force_field=force_field, head=head, check=check,
                    dispersion=dispersion)


# --------------------------------------------------------------------------- helpers


def _world():
    from ase.parallel import world

    return world


class _Snapshot:
    """``atoms`` before the observers: any change after them is refused (conductor's answer to 4-T10)."""

    def __init__(self, atoms):
        self.calc = atoms.calc
        self.state = self._state(atoms)

    @staticmethod
    def _state(atoms):
        return (len(atoms), tuple(bool(p) for p in atoms.pbc), atoms.cell.array.tobytes(), atoms.positions.tobytes(),
                atoms.get_momenta().tobytes(), atoms.numbers.tobytes(), atoms.get_masses().tobytes(),
                atoms.get_initial_charges().tobytes(), repr(atoms.constraints),
                tuple((key, repr(atoms.info.get(key))) for key in PLAN_INFO_KEYS))

    def check(self, atoms):
        if atoms.calc is not self.calc or self._state(atoms) != self.state:
            raise _refuse("an observer changed atoms during a relaxation: the optimiser's history would no longer "
                          "describe the structure, so the change is refused (SPEC §7.7); relax the changed "
                          "structure with a new optimiser", code="observer-relaxation-change")


__all__ = ("ADDITIONS", "FIRE", "LBFGS", "MAPPED")
