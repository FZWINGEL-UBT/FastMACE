"""``fastmace quench RUN_DIR``: quench the frames of a finished run (SPEC §7.5, §10.3, §10.5; plan task 4-T09).

Any run with a schedule and trajectory frames whose directory holds a restart (``restart.npz`` and
``simulation.json``, any restart point: the schedule is in it), a protocol run or a plain ``fm.md``
run. The frames are picked as a protocol's quench picks them (``protocols.quench.picks``) and relaxed
at their own cells, with the run's model, precision, D3 record and engine options.

The settings are the protocol's ``[quench]`` table for a protocol run, else the anneal profile's;
``--min-K``, ``--bin-K``, ``--per-bin``, ``--fmax``, ``--max-steps`` and ``--optimizer`` replace single
values. The quench goes into ``RUN_DIR/quench/``, or ``RUN_DIR/quench-NAME/`` with ``--into NAME``, so
one run can be quenched with several optimizers side by side. A complete quench prints its summary;
an interrupted one continues with its own settings.
"""

import json
from pathlib import Path

from fastmace.cli.md import usage_error
from fastmace.protocols.profiles import OPTIMIZERS

HELP = "quench the trajectory frames of a finished run (the frames a protocol's quench picks)"


def add_arguments(parser):
    parser.add_argument("run_dir", metavar="RUN_DIR", help="the run's output directory (with restart.npz)")
    parser.add_argument("--into", metavar="NAME", help="quench into RUN_DIR/quench-NAME/ (default RUN_DIR/quench/)")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, help="the optimizer (LBFGS and BFGS are ASE's)")
    parser.add_argument("--fmax", type=float, metavar="EV_A", help="converged when every force is below this, eV/Å")
    parser.add_argument("--max-steps", type=int, help="at most this many steps per frame")
    parser.add_argument("--min-K", type=float, help="the lowest target temperature whose frames are picked")
    parser.add_argument("--bin-K", type=float, help="the width of a target-temperature bin")
    parser.add_argument("--per-bin", type=int, help="frames per bin and ramp")


_FLAGS = {"min_K": "min_K", "bin_K": "bin_K", "per_bin": "per_bin", "fmax": "fmax", "max_steps": "max_steps",
          "optimizer": "optimizer"}


def _settings(directory, args):
    """The quench settings: the protocol's (or the anneal profile's) with the flags given."""
    from fastmace.protocols import PROFILES, runner
    from fastmace.protocols.profiles import _quench

    base = PROFILES["anneal"].quench
    if (directory / runner.PROTOCOL_RECORD).is_file():
        base = runner.read_record(directory).get("quench") or base
    settings = dict(base)
    settings.update({key: getattr(args, dest) for dest, key in _FLAGS.items() if getattr(args, dest) is not None})
    return _quench(settings)


def _run_record(directory):
    """``(simulation.json's record, the ResolvedSchedule, the trajectory interval, the timestep in fs)``."""
    from fastmace.io import restart
    from fastmace.protocols import ScheduleState
    from fastmace.simulation import ENGINE_NAME, RESTART_NAME, SIMULATION_NAME

    file, record_path = directory / RESTART_NAME, directory / SIMULATION_NAME
    if not file.is_file() or not record_path.is_file():
        raise ValueError(f"{directory} holds no {RESTART_NAME} and {SIMULATION_NAME}: fastmace quench reads the "
                         "run's schedule and model from its restart (Simulation.save_restart writes it)")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    protocol = restart.protocol_record(restart.read_restart(file, engine=ENGINE_NAME))
    if protocol is None:
        raise ValueError(f"{directory}: the run has no schedule; a quench picks frames by the schedule's target")
    frame_every = (record.get("output") or {}).get("trajectory_every")
    if not frame_every:
        raise ValueError(f"{directory}: the run saved no trajectory frames (output trajectory_every)")
    timestep_fs = float(record["integrator"]["fields"]["timestep_fs"])
    return record, ScheduleState.from_restart(protocol).resolved, int(frame_every), timestep_fs


def _print(result, out):
    print(f"quench: {result['frames']} frames with {result['settings']['optimizer']}, final quenched "
          f"{result['final_quenched_eV_atom']:.6f} eV/atom, {result['not_converged']} not converged ({out})")


def run(args):
    from fastmace.protocols import quench

    directory = Path(args.run_dir)
    try:
        name = quench.quench_directory(args.into)
        out = directory / name
        if (out / quench.SUMMARY).is_file():
            result = json.loads((out / quench.SUMMARY).read_text(encoding="utf-8"))
            print(f"{out}: the quench is complete")
            _print(result, out)
            return 0
        record, resolved, frame_every, timestep_fs = _run_record(directory)
        settings = _settings(directory, args)
    except ValueError as error:
        return usage_error("quench", error)
    from fastmace.errors import FastMACEError
    from fastmace.simulation import PlannerOptions, _record_force_field

    field = _record_force_field(record)  # the run's model (by SHA-256), precision and D3 record; no engine
    engine = PlannerOptions.from_record(record.get("planner_options")).engine
    relaxer = quench.relaxer(settings["optimizer"], field.model, precision=field.precision,
                             dispersion=field.dispersion, engine=engine)
    try:
        result = quench.quench_run(directory, relaxer, resolved, frame_every, timestep_fs, settings, name=name)
    except FastMACEError:
        raise  # PlanRefused is a ValueError too; main prints it
    except ValueError as error:  # a quench of another run's picks or settings
        return usage_error("quench", error)
    finally:
        relaxer.close()
    _print(result, out)
    return 0
