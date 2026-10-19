"""ASE MD scripts on the graph engine: ``fastmace.ase.Langevin`` and ``LangevinBAOAB`` (SPEC §7.7 level 1).

An ASE script is ported by changing the import and adding ``model=``. Each class takes the argument
names and units of its ASE 3.27 class (``SUPPORTED``) and refuses the rest with the reason
(``REFUSED``); any other argument is a ``TypeError`` naming it. The FastMACE additions are
keyword-only (``ADDITIONS``). The shim never reinterprets an ASE argument: ``friction`` and the
time constants are in ASE units, and the conversion to the public integrator record is the only
arithmetic (the record's ``to_config`` converts back, so the friction may differ by one ulp).

Nothing here builds an engine: the classes build a ``fm.Simulation`` at the first ``run`` or
``step``, which plans, runs the optional gate (owner decision D1: ``check="gate"``) and builds
through the minimal planner (SPEC §5). The setup report is printed then.

``run(n)`` advances the ``Simulation`` in chunks that end where an observer is due (ASE's
``attach`` rule: every ``interval`` steps from step 0, or once at step ``|interval|`` for
``interval <= 0``). Before the observers of a step the committed state is written into ``atoms``
with a single-point results cache as ``atoms.calc``; after them, changed positions or momenta
become host writes through ``Simulation.set_positions``/``set_momenta`` (owner decision D4), and
any other change raises ``PlanRefused`` with its reason. ``trajectory=`` and ``logfile=`` keep
their ASE meaning (one ASE ``Trajectory``, ASE's ``MDLogger`` format) and are written from the
``Simulation``'s asynchronous snapshots; every other file goes to ``output``.
"""

from __future__ import annotations

import math
import os
import secrets
import warnings
from pathlib import Path
from typing import Any

# The FastMACE additions, keyword-only in every class.
ADDITIONS = ("model", "force_field", "head", "seed", "output", "check", "dispersion")

# Mapped ASE classes (phase 1b), and the ones that get a mapping with their integrators (§10.2).
MAPPED = ("Langevin", "LangevinBAOAB")
LATER = {"VelocityVerlet": "NVE", "NoseHooverChainNVT": "NHC", "Bussi": "CSVR", "IsotropicMTKNPT": "MTK",
         "MTKNPT": "MTK", "NVTBerendsen": "Berendsen", "NPTBerendsen": "Berendsen",
         "Inhomogeneous_NPTBerendsen": "Berendsen"}
UNPLANNED = ("Andersen", "NPT")

DEFAULT_OUTPUT = "fastmace-run"
OUTPUT_SUFFIX = ".fastmace"
REPORT_NAME = "report.txt"
METADATA_NAME = "metadata.json"
# atoms.info keys the force field reads (MACE's calculator: total charge and spin).
PLAN_INFO_KEYS = ("charge", "spin", "total_charge", "total_spin", "spin_multiplicity")
OBSERVER_WARNING = ("observers: the run stops at every observer call for a host sync (rule 9 kept on the graph "
                    "tier, owner decision D6); trajectory= and logfile= do not stop it")


def _refuse(text, code):
    from fastmace.errors import PlanRefused, Reason

    return PlanRefused.from_reasons([Reason(text, code=code)])


def unmapped(name: str):
    """``fastmace.ase.<name>`` for an ASE MD class without a mapping: ``PlanRefused`` listing the mapped ones."""
    mapped = ", ".join(MAPPED)
    if name in LATER:
        raise _refuse(f"ASE's {name} has no fastmace.ase mapping in this version; mapped: {mapped}. {name} maps "
                      f"to FastMACE's {LATER[name]} integrator in phase 4 (SPEC §7.7, §10.2); FastMACE never falls "
                      "back to ASE's own loop", code="ase-class-phase-4")
    if name in UNPLANNED:
        raise _refuse(f"ASE's {name} has no fastmace.ase mapping; mapped: {mapped} (SPEC §7.7); FastMACE never "
                      "falls back to ASE's own loop", code="ase-class-unmapped")
    raise AttributeError(f"module 'fastmace.ase' has no attribute {name!r}")


def _is_default_comm(value):
    from ase.parallel import world

    return value is None or value is world


# Refused ASE arguments: (accepts the value, reason). The value ASE uses by default is accepted.
_REFUSALS = {
    "rng": (lambda value: value is None,
            "the noise comes from FastMACE's device generator; pass seed= instead (SPEC §7.7)"),
    "comm": (_is_default_comm, "FastMACE runs one process per structure; only ASE's default comm is accepted"),
    "communicator": (_is_default_comm,
                     "FastMACE runs one process per structure; only ASE's default communicator is accepted"),
    "master": (lambda value: value is None,
               "FastMACE runs one process per structure, which writes every file; only master=None is accepted"),
}


class _Dynamics:
    """What both shims share: arguments, the lazy ``Simulation``, observers, files and the report."""

    SUPPORTED: tuple[str, ...] = ()
    REFUSED: tuple[str, ...] = ()

    # ------------------------------------------------------------------ construction

    def _refuse_extra(self, extra):
        accepted = set(self.REFUSED) | {"communicator"}
        unknown = sorted(set(extra) - accepted)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unexpected argument(s) {', '.join(unknown)}; it takes ASE's "
                            f"{', '.join(self.SUPPORTED)} and FastMACE's keyword-only {', '.join(ADDITIONS)} "
                            "(SPEC §7.7)")
        for name, value in extra.items():
            accepts, reason = _REFUSALS[name]
            if not accepts(value):
                raise _refuse(f"{type(self).__name__}({name}=...) is refused: {reason}", code=f"ase-{name}")

    def _setup(self, atoms, timestep, record, *, trajectory, logfile, loginterval, append_trajectory, model,
               force_field, head, seed, output, check, dispersion):
        import numpy as np

        from fastmace.simulation import check_mode

        check_mode(check)
        if isinstance(loginterval, bool) or not isinstance(loginterval, int) or loginterval < 1:
            raise ValueError(f"loginterval must be a positive int, got {loginterval!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError(f"seed must be an int, got {seed!r}")
        self._model = _model(model, force_field, head)
        _dispersion(dispersion, self._model)
        _check_constraints(atoms)

        self.atoms = atoms
        self.dt = timestep
        self.integrator = record
        self.check = check
        self.seed = secrets.randbelow(2**31) if seed is None else seed  # drawn and recorded, as fm.md
        self.observers: list[tuple[Any, int, tuple, dict]] = []
        self.nsteps = 0
        self.max_steps = 0
        if not atoms.has("momenta"):  # as ASE's MolecularDynamics
            atoms.set_momenta(np.zeros((len(atoms), 3)))

        # Files: ASE's trajectory and log; everything else in output.
        self._trajectory = None
        self._trajectory_writer = None
        self._own_trajectory = False
        if trajectory is not None:
            if hasattr(trajectory, "write"):
                self._trajectory_writer = trajectory  # an open Trajectory: the caller closes it
            else:
                self._trajectory = Path(trajectory).absolute()
                self._own_trajectory = True
                if not append_trajectory:
                    self._trajectory.unlink(missing_ok=True)  # as ASE's BaseDynamics
        if output is not None:
            self.output = os.fspath(output)
        elif self._trajectory is not None:
            self.output = str(self._trajectory.parent / (self._trajectory.stem + OUTPUT_SUFFIX))
        else:
            self.output = str(Path.cwd() / DEFAULT_OUTPUT)
        self.loginterval = loginterval
        self._logfile = logfile
        self._frame = atoms.copy()  # what the trajectory writer and the logger see
        self._frame.set_constraint()
        self._frame.calc = None
        self._clock = _Clock(timestep)
        self._logger = None
        if logfile:
            from ase.md.logger import MDLogger

            self._logger = MDLogger(dyn=self._clock, atoms=self._frame, logfile=logfile)
        self._last_frame_step = None
        self._sim = None
        self._cache = None
        self._speed_printed = False

    # ------------------------------------------------------------------ ASE members

    def todict(self) -> dict[str, Any]:
        return {"type": "molecular-dynamics", "md-type": type(self).__name__, "timestep": self.dt}

    def get_time(self) -> float:
        return self.nsteps * self.dt

    def get_number_of_steps(self) -> int:
        return self.nsteps

    def attach(self, function, interval=1, *args, **kwargs) -> None:
        """Attach ``function(*args, **kwargs)``, called every ``interval`` steps (or once at step
        ``|interval|`` for ``interval <= 0``), as ASE's ``attach``; an object without ``__call__`` has
        its ``write`` attached."""
        if not callable(function):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def run(self, steps: int = 50) -> bool:
        """Advance ``steps`` MD steps, calling the observers as ASE does (step 0 included on the first run)."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(f"steps must be a non-negative int, got {steps!r}")
        sim = self._simulation()
        self.max_steps = self.nsteps + steps
        try:
            if self.nsteps == 0:
                self._at(0)
            while self.nsteps < self.max_steps:
                stop = self._next_stop()
                sim.run(steps=stop - self.nsteps)
                self.nsteps = stop
                self._print_speed()
                self._at(stop)
        except BaseException:
            self._finish_run(write_atoms=False)  # atoms stay as the observer left them
            raise
        self._finish_run()
        return True

    def step(self, forces=None):
        """One MD step without observers (ASE's ``step``); returns the new forces."""
        if forces is not None:
            raise TypeError("step(forces=...) is not supported: the engine evaluates the forces")
        sim = self._simulation()
        sim.run(steps=1)
        self.nsteps += 1
        committed = self._write_atoms()
        self._finish_run()
        return committed.forces.copy()

    def close(self) -> None:
        """Write what is queued, close the files and write the final ``metadata.json``."""
        if self._sim is not None:
            self._sim.close()
            self._write_records()
        if self._own_trajectory and self._trajectory_writer is not None:
            self._trajectory_writer.close()
            self._trajectory_writer = None
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ FastMACE members

    @property
    def simulation(self):
        """The underlying ``fm.Simulation`` (``None`` before the first ``run`` or ``step``)."""
        return self._sim

    @property
    def plan(self):
        return None if self._sim is None else self._sim.plan

    @property
    def gate(self):
        return None if self._sim is None else self._sim.gate

    @property
    def summary(self) -> str | None:
        """The run report (SPEC §7.7), the same as ``fm.md``'s; ``None`` before the first ``run``."""
        return None if self._sim is None else self._sim.report.text()

    @property
    def metadata(self) -> dict[str, Any] | None:
        """The ``metadata.json`` record: the ``Simulation``'s, with this shim's ``"ase"`` block (as written
        at the end of the last ``run``, like ``fm.md``'s ``RunResult.metadata``)."""
        if self._sim is None:
            return None
        path = Path(self.output) / METADATA_NAME
        if path.is_file():
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        return {**self._sim.metadata(), "ase": self._ase_record()}

    # ------------------------------------------------------------------ the Simulation

    def _simulation(self):
        if self._sim is not None:
            return self._sim
        from fastmace import simulation
        from fastmace.io.output import Output

        # Thermo rows and restart files at the production cadences; ASE's trajectory is written here.
        output = Output(self.output, trajectory_every=None)
        self._sim = sim = simulation.Simulation(self.atoms, self._model, integrator=self.integrator, seed=self.seed,
                                                output=output, check=self.check)
        if self._has_files():
            sim.add_sink(self._write_frame, self.loginterval)  # in the step loop, behind the GPU
        report = sim.report
        files = [str(path) for path in (self._trajectory, self._logfile) if path not in (None, "-")
                 and isinstance(path, (str, os.PathLike))]
        report.outputs = (*files, self.output)
        if self.observers:
            report.add_warning(OBSERVER_WARNING)
        print(report.setup_text())
        return sim

    def _print_speed(self):
        report = self._sim.report
        if report.measured is not None and not self._speed_printed:
            self._speed_printed = True
            print(next(line for line in report.text().splitlines() if line.startswith("speed:")))

    def _finish_run(self, write_atoms=True):
        sim = self._sim
        sim.flush_output()
        if self._trajectory_writer is not None and self._own_trajectory:
            self._trajectory_writer.close()  # complete on disk after every run; reopened in append mode
            self._trajectory_writer = None
        if write_atoms:
            self._write_atoms()
        self._write_records()

    def _write_records(self):
        """Add the ``"ase"`` block to the ``metadata.json`` the ``Simulation`` has just written, and write
        the report. The record is not built again: its software block (``provenance.fastmace_block``)
        costs about 0.25 s, which made each ``run()`` that much slower than ``Simulation.run`` (1b-T21b)."""
        import json

        from fastmace import provenance

        directory = Path(self.output)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / METADATA_NAME
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = self._sim.metadata()
        record["ase"] = self._ase_record()
        provenance.write_metadata(path, record)
        (directory / REPORT_NAME).write_text(self.summary + "\n", encoding="utf-8")

    def _ase_record(self):
        return {"class": type(self).__name__, "todict": _json_safe(self.todict()),
                "trajectory": None if self._trajectory is None else str(self._trajectory),
                "logfile": self._logfile if isinstance(self._logfile, str) else None,
                "loginterval": self.loginterval, "last_frame_step": self._last_frame_step, "nsteps": self.nsteps,
                "observers": len(self.observers)}

    # ------------------------------------------------------------------ observers

    def _due(self, step):
        """The observers due at ``step``, in order (ASE's ``call_observers``)."""
        return [(function, args, kwargs) for function, interval, args, kwargs in self.observers
                if (interval > 0 and step % interval == 0) or (interval <= 0 and step == abs(interval))]

    def _has_files(self):
        return self._trajectory is not None or self._trajectory_writer is not None or self._logger is not None

    def _next_stop(self):
        """The next step after ``nsteps`` where an observer is due, at most ``max_steps`` (the files are
        written by the ``Simulation``'s sink and never stop the run)."""
        now = self.nsteps
        stops = [self.max_steps]
        for _, interval, _, _ in self.observers:
            if interval > 0:
                stops.append((now // interval + 1) * interval)
            elif abs(interval) > now:
                stops.append(abs(interval))
        return min(stops)

    def _at(self, step):
        """Step ``step`` is committed: queue the step-0 files (the sink has the others), then call the
        observers due with the state in ``atoms``."""
        if step == 0 and self._has_files():
            self._sim.submit_frame(self._write_frame)
        due = self._due(step)
        if not due:
            return
        committed = self._write_atoms()
        before = _Written(self.atoms, committed, self._cache)
        for function, args, kwargs in due:
            function(*args, **kwargs)
        self._apply(before)

    def _write_atoms(self):
        """The committed state into ``atoms`` with a results cache as ``atoms.calc``; returns it."""
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.stress import full_3x3_to_voigt_6_stress

        committed = self._sim.committed()
        atoms = self.atoms
        atoms.set_cell(committed.cell)
        atoms.set_positions(committed.positions, apply_constraint=False)
        atoms.set_momenta(committed.momenta, apply_constraint=False)
        extra = {} if committed.stress is None else {"stress": full_3x3_to_voigt_6_stress(committed.stress)}
        self._cache = SinglePointCalculator(atoms, energy=committed.energy, free_energy=committed.energy,
                                            forces=committed.forces.copy(), **extra)
        atoms.calc = self._cache
        return committed

    def _apply(self, before):
        """Compare ``atoms`` with what was written; apply positions and momenta (D4), refuse the rest."""
        import numpy as np

        atoms = self.atoms
        if atoms.calc is not before.calc:
            raise _refuse("an observer replaced atoms.calc: the engine is the force field of this run, and a "
                          "different calculator would invalidate the plan (SPEC §7.7)", code="observer-calc")
        if len(atoms) != before.natoms:
            raise _refuse(f"an observer changed the atom count ({before.natoms} -> {len(atoms)}): that invalidates "
                          "the plan; build a new run (SPEC §7.7)", code="observer-atom-count")
        if tuple(bool(p) for p in atoms.pbc) != before.pbc:
            raise _refuse("an observer changed the PBC: that invalidates the plan (full 3D PBC, SPEC §4.4)",
                          code="observer-pbc")
        if atoms.constraints:
            raise _refuse("an observer added constraints: that invalidates the plan; FixAtoms is phase 4 (SPEC "
                          "§4.4)", code="observer-constraints")
        if not np.array_equal(atoms.cell.array, before.cell):
            raise _refuse("an observer changed the cell: only the barostat changes it (SPEC §7.7)",
                          code="observer-cell")
        info = {key: atoms.info.get(key) for key in PLAN_INFO_KEYS}
        if info != before.info:
            raise _refuse("an observer changed atoms.info keys the force field reads (charge, spin): that "
                          "invalidates the plan (SPEC §7.7)", code="observer-info")
        for name, now, then in (("species", atoms.numbers, before.numbers),
                                ("masses", atoms.get_masses(), before.masses),
                                ("charges", atoms.get_initial_charges(), before.charges)):
            if not np.array_equal(now, then):
                raise _refuse(f"an observer changed the {name}: in-place events are phase 3 (owner decision D4); "
                              "positions and momenta may change", code=f"observer-{name}")
        if not np.array_equal(atoms.positions, before.positions):
            self._sim.set_positions(atoms.positions)
        if not np.array_equal(atoms.get_momenta(), before.momenta):
            self._sim.set_momenta(atoms.get_momenta())

    # ------------------------------------------------------------------ files

    def _write_frame(self, step, values):
        """One logged step from the asynchronous snapshot: the trajectory frame and the log line."""
        from ase.calculators.singlepoint import SinglePointCalculator

        frame = self._frame
        extra = {}
        if "cell" in values and "stress" in values:
            from ase.stress import full_3x3_to_voigt_6_stress

            frame.set_cell(values["cell"])
            extra["stress"] = full_3x3_to_voigt_6_stress(values["stress"])
        frame.set_positions(values["positions"])
        frame.set_momenta(values["momenta"])
        frame.calc = SinglePointCalculator(frame, energy=float(values["energy"][0]), forces=values["forces"],
                                           **extra)
        if self._trajectory is not None or self._trajectory_writer is not None:
            if self._trajectory_writer is None:
                from ase.io.trajectory import Trajectory

                self._trajectory_writer = Trajectory(str(self._trajectory), mode="a")
            self._trajectory_writer.set_description(self.todict() | {"interval": self.loginterval})
            self._trajectory_writer.write(frame)
            self._last_frame_step = int(step)
        if self._logger is not None:
            self._clock.step = int(step)
            self._logger()


class Langevin(_Dynamics):
    """ASE 3.27's ``ase.md.langevin.Langevin`` on FastMACE's graph engine (Langevin, SPEC §10.2).

    ``Langevin(atoms, timestep, temperature=None, friction=None, fixcm=True, *, temperature_K=None,
    trajectory=None, logfile=None, loginterval=1, append_trajectory=False, model=None,
    force_field=None, head=None, seed=None, output=None, check="skip", dispersion=None)``: ASE's
    arguments and units (``timestep`` and ``1/friction`` in ASE time units, ``temperature`` in eV,
    deprecated as in ASE); ``rng``, a non-default ``comm``/``communicator`` and ``master`` are
    refused. The additions are those of ``fm.md``.
    """

    SUPPORTED = ("atoms", "timestep", "temperature", "friction", "fixcm", "temperature_K", "trajectory", "logfile",
                 "loginterval", "append_trajectory")
    REFUSED = ("rng", "comm", "master")

    def __init__(self, atoms, timestep, temperature=None, friction=None, fixcm=True, *, temperature_K=None,
                 trajectory=None, logfile=None, loginterval=1, append_trajectory=False, model=None, force_field=None,
                 head=None, seed=None, output=None, check="skip", dispersion=None, **kwargs):
        from ase import units
        from ase.md.md import process_temperature

        from fastmace import integrators

        self._refuse_extra(kwargs)
        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        kelvin = process_temperature(temperature, temperature_K, "eV")
        self.temp = units.kB * kelvin  # ASE's attribute, in eV
        self.fr = friction
        self.fix_com = fixcm
        record = integrators.Langevin(timestep_fs=float(timestep) / units.fs, temperature_K=float(kelvin),
                                      friction_per_fs=float(friction) * units.fs, fixcm=bool(fixcm))
        self._setup(atoms, timestep, record, trajectory=trajectory, logfile=logfile, loginterval=loginterval,
                    append_trajectory=append_trajectory, model=model, force_field=force_field, head=head, seed=seed,
                    output=output, check=check, dispersion=dispersion)

    def todict(self) -> dict[str, Any]:
        return super().todict() | {"temperature_K": self.temp / _kB(), "friction": self.fr, "fixcm": self.fix_com}


class LangevinBAOAB(_Dynamics):
    """ASE 3.27's ``ase.md.langevinbaoab.LangevinBAOAB`` on FastMACE's graph engine (BAOAB NPT, §10.2).

    ``LangevinBAOAB(atoms, timestep, *, temperature_K, externalstress, hydrostatic=False, T_tau=None,
    P_tau=None, P_mass=None, P_mass_factor=1.0, disable_cell_langevin=False, trajectory=None,
    logfile=None, loginterval=1, append_trajectory=False, model=None, ...)``, ASE units throughout:
    ``externalstress`` in eV/Å³ (a scalar, 3 diagonal, 6 Voigt components or a symmetric 3×3),
    ``T_tau`` and ``P_tau`` in ASE time. Mapped to ``fm.integrators.LangevinBAOAB`` with
    ``friction_per_fs = fs / T_tau`` and ``cell_mode = "hydrostatic" if hydrostatic else "full"``.

    Refused: no ``externalstress`` (fixed-cell BAOAB: use ``Langevin``), no ``temperature_K``
    (NVE/NPH, phase 4), an explicit ``P_mass`` (``P_mass_factor`` scales ASE's heuristic), ``rng``,
    a non-default ``comm``/``communicator`` and ``master``.
    """

    SUPPORTED = ("atoms", "timestep", "temperature_K", "externalstress", "hydrostatic", "T_tau", "P_tau", "P_mass",
                 "P_mass_factor", "disable_cell_langevin", "trajectory", "logfile", "loginterval", "append_trajectory")
    REFUSED = ("rng", "comm", "master")

    def __init__(self, atoms, timestep, *, temperature_K=None, externalstress=None, hydrostatic=False, T_tau=None,
                 P_tau=None, P_mass=None, P_mass_factor=1.0, disable_cell_langevin=False, trajectory=None,
                 logfile=None, loginterval=1, append_trajectory=False, model=None, force_field=None, head=None,
                 seed=None, output=None, check="skip", dispersion=None, **kwargs):
        from ase import units

        from fastmace import integrators

        self._refuse_extra(kwargs)
        if externalstress is None:
            raise _refuse("LangevinBAOAB without externalstress is fixed-cell BAOAB, which FastMACE does not have: "
                          "use `Langevin` for NVT (SPEC §10.2)", code="baoab-fixed-cell")
        if temperature_K is None:
            raise _refuse("LangevinBAOAB without temperature_K is NVE/NPH: phase 4 (SPEC §10.2)",
                          code="baoab-no-thermostat")
        if P_mass is not None:
            raise _refuse("LangevinBAOAB(P_mass=...) is refused: FastMACE uses ASE's heuristic barostat mass; scale "
                          "it with P_mass_factor", code="baoab-p-mass")
        stress = _voigt_GPa(externalstress, hydrostatic)
        if T_tau is None:
            T_tau = 50.0 * timestep
            warnings.warn(f"Got `temperature_K` but missing `T_tau`, defaulting to 50 * `timstep` = {T_tau}",
                          stacklevel=2)
        if T_tau <= 0:
            raise ValueError(f"Invalid T_tau {T_tau} <= 0")
        if P_tau is not None and P_tau <= 0:
            raise ValueError(f"Invalid P_tau {P_tau} <= 0")
        self.T_tau, self.P_tau, self.hydrostatic = T_tau, P_tau, bool(hydrostatic)
        self.temperature_K = temperature_K
        record = integrators.LangevinBAOAB(
            timestep_fs=float(timestep) / units.fs, temperature_K=float(temperature_K),
            friction_per_fs=units.fs / float(T_tau), external_stress_GPa=stress,
            cell_mode="hydrostatic" if hydrostatic else "full",
            barostat_tau_fs=None if P_tau is None else float(P_tau) / units.fs,
            barostat_mass_factor=float(P_mass_factor), disable_cell_langevin=bool(disable_cell_langevin))
        self._setup(atoms, timestep, record, trajectory=trajectory, logfile=logfile, loginterval=loginterval,
                    append_trajectory=append_trajectory, model=model, force_field=force_field, head=head, seed=seed,
                    output=output, check=check, dispersion=dispersion)


# --------------------------------------------------------------------------- helpers


class _Clock:
    """``MDLogger``'s ``dyn``: the time of the step being logged (the log is written behind the run)."""

    def __init__(self, dt):
        self.dt = dt
        self.step = 0

    def get_time(self):
        return self.step * self.dt


class _Written:
    """What the shim wrote into ``atoms`` before the observers, to compare with afterwards."""

    def __init__(self, atoms, committed, calc):
        self.calc = calc
        self.natoms = len(atoms)
        self.pbc = tuple(bool(p) for p in atoms.pbc)
        self.cell = atoms.cell.array.copy()
        self.positions = atoms.positions.copy()
        self.momenta = atoms.get_momenta()
        self.numbers = atoms.numbers.copy()
        self.masses = atoms.get_masses()
        self.charges = atoms.get_initial_charges()
        self.info = {key: atoms.info.get(key) for key in PLAN_INFO_KEYS}


def _kB():
    from ase import units

    return units.kB


def _model(model, force_field, head):
    """What the ``Simulation`` gets: a ``ForceField`` or a loaded model (a label is loaded here)."""
    if (model is None) == (force_field is None):
        raise TypeError("give exactly one of model= or force_field= (SPEC §7.7)")
    if force_field is not None:
        from fastmace.model.forcefield import ForceField

        if not isinstance(force_field, ForceField):
            raise TypeError(f"force_field must be an fm.ForceField, got {type(force_field).__name__}")
        if head is not None:
            raise ValueError("head= with force_field=: the ForceField's model has its head; pass head= to fm.load")
        return force_field
    if isinstance(model, (str, os.PathLike)):
        from fastmace.ase.calculator import _is_label

        if not isinstance(model, str) or not _is_label(model):
            raise TypeError(f"{model!r} is not a label of the model table; load a checkpoint path or URL with "
                            "fm.load(path, sha256=... or trusted=True) first, the trust decision needs it")
        from fastmace.model import loading

        return loading.load_pristine(model, head)
    if head is not None and head != getattr(model, "head", None):
        raise ValueError(f"head={head!r} does not match the loaded model's head {getattr(model, 'head', None)!r}; "
                         "pass head= to fm.load instead")
    return model


def _dispersion(dispersion, model):
    """``dispersion=`` as ``fm.md``: ``"d3bj"`` expands from the training functional (``ValueError`` when
    it already contains dispersion or is unknown), and a complete record is refused until phase 3b."""
    from fastmace.longrange import d3
    from fastmace.model.forcefield import ForceField

    loaded = model.model if isinstance(model, ForceField) else model
    name = loaded.label or str(loaded.path)
    record, notes = d3.expand(dispersion, functional=loaded.functional, model=name)
    if record is not None:
        text = f"D3: phase 3b; the expanded record is {d3.describe(record)}"
        if notes:
            text += f" ({'; '.join(notes)})"
        raise _refuse(text + " (SPEC §11.2)", code="d3-phase-3b")


def _check_constraints(atoms):
    from ase.constraints import FixAtoms

    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            raise _refuse("FixAtoms is refused until the integrators support it in phase 4 (SPEC §4.4); no other "
                          "tier is substituted", code="constraint-fixatoms")
        raise _refuse(f"the ASE constraint {type(constraint).__name__} is refused in 1.0 (SPEC §4.4)",
                      code="constraint-other")


def _voigt_GPa(externalstress, hydrostatic):
    """ASE's ``externalstress`` (eV/Å³) as the record's GPa: a scalar (hydrostatic) or six Voigt components."""
    import numpy as np
    from ase import units

    stress = np.asarray(externalstress, dtype=float)
    if stress.shape == ():
        stress = stress.reshape(-1)
    if hydrostatic:
        if stress.shape != (1,):
            raise ValueError(f"externalstress must be scalar when hydrostatic, got '{externalstress}' with shape "
                             f"{stress.shape}")
        return float(stress[0]) / units.GPa
    if stress.shape == (1,):
        voigt = (stress[0],) * 3 + (0.0,) * 3
    elif stress.shape == (3,):
        voigt = (*stress, 0.0, 0.0, 0.0)
    elif stress.shape == (6,):
        voigt = tuple(stress)
    elif stress.shape == (3, 3):
        if not np.array_equal(stress, stress.T):
            raise _refuse("externalstress must be a symmetric 3×3 matrix: an antisymmetric part would rotate the "
                          "cell", code="baoab-asymmetric-stress")
        voigt = (stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1])
    else:
        raise ValueError("externalstress must be scalar, 3-vector (diagonal), 6-vector (Voigt), or 3x3 matrix, got "
                         f"'{externalstress}' with shape {stress.shape}")
    return tuple(float(value) / units.GPa for value in voigt)


def _json_safe(record):
    return {key: (value if isinstance(value, (str, bool, int)) or value is None
                  else float(value) if isinstance(value, float) and math.isfinite(value) else repr(value))
            for key, value in record.items()}


__all__ = ("ADDITIONS", "MAPPED", "Langevin", "LangevinBAOAB", "unmapped")
