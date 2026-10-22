"""``fastmace run CONFIG.toml``: ``fm.md`` from a TOML file (SPEC §7.5; owner decision D5).

The file holds ``fm.md``'s arguments under their own names, as ``fastmace init`` writes them; a
key that is not an ``fm.md`` argument is refused, naming it. Paths are taken as they are, relative
to the directory ``fastmace run`` starts in (as ``fastmace md``'s flags). The arguments that are
records in Python are TOML values or tables:

- ``schedule``: a profile name (``fm.protocols.PROFILES``: slow, fast, ultrafast, anneal) or an
  array of ``[[schedule]]`` segments (``kind = "ramp"`` with ``start_K``, ``end_K`` and ``ps`` or
  ``rate_K_per_ps``; ``kind = "hold"`` with ``temperature_K`` and ``ps``);
- ``integrator``: a table with ``type = "Langevin"`` or ``"LangevinBAOAB"`` and the record's
  fields (``fm.integrators``); ``friction_per_fs`` defaults to ``fm.md``'s friction;
- ``dispersion``: ``"none"``, ``"d3bj"`` or a ``[dispersion]`` table of ``fm.D3`` fields;
- ``mc``: one ``[mc]`` table or ``[[mc]]`` tables, each an ``fm.mc`` record: ``move`` (``"swap"``,
  ``"transmute"`` or ``"displace"``) and that record's fields (``pairs``, ``species``, ``delta_A``,
  ``every``, ``attempts``, ``velocities``, ``mu``, ``formal_charges``, ``adaptive``);
- ``output``: a directory or an ``[output]`` table of ``fm.Output`` fields (the table's
  ``directory`` defaults to ``fm.md``'s, ``./<structure stem>.fastmace/``);
- ``repeat``: an array of three ints.

A ``.yaml``/``.yml`` file is read as YAML with ``safe_load`` only (SPEC §10.5), with the same keys.

Protocols (SPEC §10.5, plan task 3-T14): ``CONFIG`` may also name a protocol shipped with FastMACE
(``fastmace.protocols.runner.protocol_names``), whose relative structure path is a file shipped with
it. A file with a ``[relax]`` or ``[quench]`` table, or a run with ``--steps``, is a protocol run
(``fastmace.protocols.runner``): the relaxation, ``fm.md`` for at most ``--steps`` steps of the
schedule with a restart at the end, and the quench once the schedule is done. ``--resume RUN_DIR``
continues it for at most ``--steps`` more steps. ``--output`` sets the output directory.
"""

import dataclasses
import inspect
import sys
from pathlib import Path

from fastmace.cli.md import run_md, usage_error

HELP = "run MD or a protocol from a TOML file of fm.md arguments (write one with fastmace init)"

INTEGRATOR_TYPES = ("Langevin", "LangevinBAOAB")


def add_arguments(parser):
    parser.add_argument("config", metavar="CONFIG.toml", nargs="?",
                        help="the TOML file ('-' reads standard input) or the name of a shipped protocol")
    parser.add_argument("--resume", metavar="RUN_DIR", help="continue the protocol run in RUN_DIR")
    parser.add_argument("--steps", type=int, help="run at most this many MD steps, then write a restart and stop "
                        "(a protocol run; continue it with --resume)")
    parser.add_argument("--output", metavar="DIR", help="the output directory (in place of the file's)")


def _tomllib():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    return tomllib


def parse_toml(text):
    """The mapping of a TOML text; ``ValueError`` (``TOMLDecodeError``) if it is not TOML."""
    return _tomllib().loads(text)


def parse_config(name, text):
    """The mapping of a config: YAML (``safe_load`` only, SPEC §10.5) for a ``.yaml``/``.yml`` name, else TOML.

    ``ValueError`` if it is not valid, or not a mapping.
    """
    if Path(name).suffix.lower() not in (".yaml", ".yml"):
        return parse_toml(text)
    try:
        import yaml
    except ImportError:
        raise ValueError("reading a YAML config needs PyYAML (pip install pyyaml); TOML configs need nothing "
                         "extra") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"not valid YAML for safe_load: {error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"a config must be a mapping, got {type(data).__name__}")
    return data


def _md_parameters():
    from fastmace.quick import md

    return tuple(inspect.signature(md).parameters)


def _fields(record_type, data, what):
    """``data`` checked against the dataclass ``record_type``'s fields; lists become tuples."""
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be a table, got {data!r}")
    names = {field.name for field in dataclasses.fields(record_type)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown {what} keys {unknown}; known: {sorted(names)}")
    return {key: tuple(value) if isinstance(value, list) else value for key, value in data.items()}


def _schedule(value):
    from fastmace.protocols import PROFILES, Schedule

    if isinstance(value, str):
        if value not in PROFILES:
            raise ValueError(f"schedule = {value!r} is not a profile; profiles: {', '.join(PROFILES)}")
        return PROFILES[value].schedule
    if isinstance(value, list):
        return Schedule.from_data(value)
    raise ValueError(f"schedule must be a profile name or [[schedule]] segments, got {value!r}")


def _integrator(value):
    from fastmace import integrators
    from fastmace.quick import default_friction_per_fs

    if not isinstance(value, dict):
        raise ValueError(f"integrator must be a table with type = {' or '.join(INTEGRATOR_TYPES)}, got {value!r}")
    fields = dict(value)
    kind = fields.pop("type", None)
    if kind not in INTEGRATOR_TYPES:
        raise ValueError(f"integrator type = {kind!r} is not one of {', '.join(INTEGRATOR_TYPES)} (1b; the others "
                         "come with their integrators, SPEC §10.2)")
    record_type = getattr(integrators, kind)
    fields = _fields(record_type, fields, f"integrator ({kind})")
    fields.setdefault("friction_per_fs", default_friction_per_fs())
    return record_type(**fields)


def _dispersion(value):
    from fastmace.longrange.d3 import D3

    if value == "none":
        return None
    if isinstance(value, dict):
        return D3(**_fields(D3, value, "dispersion"))
    return value  # "d3bj", or anything else for fm.md to refuse


def _mc(value):
    blocks = [value] if isinstance(value, dict) else value
    if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
        raise ValueError(f"mc must be [mc] or [[mc]] tables, got {value!r}")
    from fastmace.simulation import move_from_record

    for block in blocks:
        if "move" not in block:
            raise ValueError(f"every mc table needs move = ..., got {block!r}")
    return tuple(move_from_record(block) for block in blocks)


def _output(value):
    from fastmace.io.output import Output

    if isinstance(value, dict):
        return Output(**_fields(Output, value, "output"))
    return value


def _config(key, name):
    """The converter of a ``[key]`` table to the ``fastmace.config`` record ``name`` (the planner options)."""
    def convert(value):
        from fastmace import config

        record_type = getattr(config, name)
        return record_type(**_fields(record_type, value, key))

    return convert


_CONVERT = {"schedule": _schedule, "integrator": _integrator, "dispersion": _dispersion, "mc": _mc,
            "output": _output, "repeat": lambda value: tuple(value) if isinstance(value, list) else value,
            "engine": _config("engine", "EngineConfig"), "neighbours": _config("neighbours", "NeighbourConfig"),
            "fallback": _config("fallback", "FallbackConfig"), "gate": _config("gate", "GateConfig"),
            "runner": _config("runner", "RunnerConfig")}


def md_arguments(data):
    """``(structure, options)`` for ``fm.md(structure, **options)`` from the TOML mapping.

    ``ValueError`` names an unknown key, a missing ``structure`` or ``model``, and a malformed table.
    """
    known = _md_parameters()
    unknown = [key for key in data if key not in known]
    if unknown:
        raise ValueError(f"unknown key{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}; fastmace run takes "
                         f"fm.md's arguments: {', '.join(known)}")
    missing = [key for key in ("structure", "model") if key not in data]
    if missing:
        raise ValueError(f"missing {' and '.join(missing)}")
    options = {}
    for key, value in data.items():
        if key == "structure":
            continue
        value = _CONVERT[key](value) if key in _CONVERT else value
        if value is not None:
            options[key] = value
    return data["structure"], options


def _read(config):
    """``(name, text, shipped data directory or None)`` of ``config``: standard input, a file or a shipped protocol."""
    if config == "-":
        return "<stdin>", sys.stdin.read(), None
    from fastmace.protocols import runner

    path = Path(config)
    if not path.is_file() and config in runner.protocol_names():
        path, _ = runner.protocol_file(config)
        return config, path.read_text(encoding="utf-8"), runner.DATA
    return str(path), path.read_text(encoding="utf-8"), None


def _with_output(data, directory):
    """The output entry with ``directory`` (``--output``) or, for a table without one, ``fm.md``'s default."""
    value = data.get("output")
    if isinstance(value, dict):
        if directory is None and "directory" not in value:
            from fastmace.quick import _output

            directory = _output(None, data.get("structure", "")).directory
        return {**value, "directory": directory} if directory is not None else value
    return directory if directory is not None else value


def run(args):
    if args.steps is not None and args.steps < 1:
        return usage_error("run", f"--steps must be a positive number of steps, got {args.steps}")
    if args.resume is not None:
        if args.config is not None:
            return usage_error("run", "give CONFIG.toml or --resume RUN_DIR, not both")
        from fastmace.protocols import runner

        try:
            runner.read_record(args.resume)
        except ValueError as error:
            return usage_error("run", error)
        return _protocol(lambda: runner.resume(args.resume, steps=args.steps), args)
    if args.config is None:
        return usage_error("run", "give CONFIG.toml or --resume RUN_DIR")
    try:
        name, text, shipped = _read(args.config)
    except OSError as error:
        return usage_error("run", error)
    try:
        from fastmace.protocols import runner

        data = parse_config(name, text)
        if shipped is not None and "structure" in data and not Path(data["structure"]).is_absolute():
            data["structure"] = str(shipped / data["structure"])
        if "output" in data or args.output is not None:
            data["output"] = _with_output(data, args.output)
        stages, rest = runner.split_stages(data)
        structure, options = md_arguments(rest)
    except ValueError as error:
        return usage_error("run", f"{name}: {error}")
    if stages["relax"] is None and stages["quench"] is None and args.steps is None:
        return run_md(structure, options, command="run", debug=args.debug)
    source = args.config if shipped is not None else str(Path(name).resolve()) if name != "<stdin>" else name
    return _protocol(lambda: runner.start(structure, options, steps=args.steps, source=source, **stages), args)


def _protocol(call, args):
    """Run a protocol stage; a ``ValueError`` before the run is a usage error (2), a stopped stage an error (1)."""
    from fastmace.errors import FastMACEError

    try:
        call()
    except FastMACEError:
        raise
    except ValueError as error:
        if args.debug:
            raise
        return usage_error("run", error)
    except RuntimeError as error:
        if args.debug:
            raise
        print(f"fastmace run: {error}", file=sys.stderr)
        return 1
    return 0
