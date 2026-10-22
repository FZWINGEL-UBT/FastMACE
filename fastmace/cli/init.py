"""``fastmace init > run.toml``: a commented TOML of ``fm.md``'s arguments for ``fastmace run``
(SPEC §7.5; owner decision D5).

Every ``fm.md`` argument appears, under its own name: set lines for the values given, commented
lines (``# key = value``) for the rest, each with what it does and its default or an example.
Without flags the set lines are SPEC §7.5's example run; with ``fastmace md``'s flags they are
exactly those flags, so ``fastmace init <flags> > run.toml; fastmace run run.toml`` is
``fastmace md <flags>``. The records (schedule segments, integrator, full D3, MC moves, output
intervals) are tables at the end, as TOML requires.
"""

import dataclasses
import json

from fastmace.cli import md as md_command

HELP = "write a commented TOML of fm.md's arguments for fastmace run (takes fastmace md's flags)"

HEADER = """\
# fastmace run: the arguments of fm.md (SPEC §7.7), written by fastmace init.
# Run it with: fastmace run run.toml
# A commented line shows an argument at its default or with an example; uncomment it to set it.
# Paths are taken relative to the directory fastmace run starts in."""

# The SPEC §7.5 example, written when no flag is given.
EXAMPLE = {"structure": "NaCl.cif", "model": "mh1", "head": "matpes_r2scan", "ensemble": "NVT",
           "temperature_K": 300.0, "steps": 1_000_000, "output": "run/"}

# (fm.md argument, comment, example as TOML text): the top-level keys, in this order; "" is no comment.
SCALARS = (
    ("structure", "Structure file, any format ASE reads (required).", '"NaCl.cif"'),
    ("model", "Model: a label (fastmace models), a path or a URL (required).", '"mh1"'),
    ("head", "Head of a multi-head model; MH-1 has no default head (fastmace models lists them).",
     '"matpes_r2scan"'),
    ("charge", "Total charge and spin multiplicity, for models that take them.", "0"),
    ("spin_multiplicity", "", "1"),
    ("sha256", "A model file outside the label table needs its SHA-256, or trusted = true (it is unpickled).",
     '"' + "0" * 64 + '"'),
    ("trusted", "", "true"),
    ("ensemble", '"NVT" is Langevin, "NPT" the LangevinBAOAB barostat; "NVE" is phase 4.', '"NVT"'),
    ("temperature_K", "Target temperature in K; or a schedule, not both.", "300.0"),
    ("schedule", "A run profile instead of temperature_K (slow, fast, ultrafast, anneal), or [[schedule]] "
                 "segments below.", '"anneal"'),
    ("pressure_GPa", 'NPT only: pressure in GPa and cell mode ("hydrostatic" or "full").', "0.0"),
    ("cell_mode", "", '"hydrostatic"'),
    ("timestep_fs", "Timestep in fs.", "0.5"),
    ("steps", "Number of steps; optional with a schedule (then the whole schedule).", "1000000"),
    ("min_mc_temperature_K", "MC: the lowest MC temperature, required with a schedule (MC runs only at or "
                             "above it).", "300.0"),
    ("repeat", "Supercell repeats; the cell is never expanded silently.", "[1, 1, 1]"),
    ("dispersion", "\"d3bj\": the D3(BJ) record of the head's training functional, refused until phase 3b;\n"
                   "# a full record is the [dispersion] table below.", '"d3bj"'),
    ("seed", "Noise seed; without it one is drawn and recorded in metadata.json.", "1"),
    ("check", '"skip" (default), "gate" (the gate on this structure, opt-in) or "allow_unavailable".', '"skip"'),
    ("output", "Output directory, default ./<structure stem>.fastmace/; or the [output] table below.", '"run/"'),
    ("mode", 'Execution mode: unset lets the planner choose ("graph", or "compiled" where rules 4 and 9 need '
             'it); or "graph", "compiled", "eager", "reference".', '"compiled"'),
    ("allow_reference", "Allow the reference tier where rule 1 needs it (SPEC §9.3).", "false"),
)

# (comment, commented table lines): the records, after every top-level key; "" is an empty comment line.
TABLES = (
    ("Schedule segments: ramp (start_K, end_K, and ps or rate_K_per_ps) and hold (temperature_K, ps).",
     ('[[schedule]]', 'kind = "ramp"', "start_K = 0.0", "end_K = 1000.0", "ps = 20.0", "",
      '[[schedule]]', 'kind = "hold"', "temperature_K = 1000.0", "ps = 10.0")),
    ("An explicit integrator (type Langevin or LangevinBAOAB, with the fields of fm.integrators); it wins over\n"
     "# ensemble and the physics keys above. friction_per_fs defaults to fm.md's (0.02 per ASE time unit).",
     ("[integrator]", 'type = "LangevinBAOAB"', "temperature_K = 300.0", "pressure_GPa = 0.0",
      'cell_mode = "full"')),
    ("A full D3 record (the fields of fm.D3); its functional should be the model's training functional.\n"
     "# s6, s8, a1, a2: the Becke-Johnson numbers; without them, the functional's set (pbe or r2scan).",
     ("[dispersion]", 'functional = "r2scan"', "pair_cutoff_bohr = 40.0", "cn_cutoff_bohr = 40.0",
      "three_body = false", "s6 = 1.0", "s8 = 0.7898", "a1 = 0.4948", "a2 = 5.7308")),
    ("MC moves, one [[mc]] table each (the fields of fm.mc.Swap, Transmute or Displace; experimental).",
     ("[[mc]]", 'move = "swap"', 'pairs = [["Na", "Cl"]]', "every = 100", "attempts = 50",
      'velocities = "exchange"')),
    ("Output with its intervals (the fields of fm.Output).",
     ("[output]", 'directory = "run/"', "trajectory_every = 20", "thermo_every = 20", "restart_every = 10000")),
    ("Planner options (the fields of fastmace.config.EngineConfig and NeighbourConfig); without them the\n"
     "# default planner's.",
     ("[engine]", 'backend = "cueq"', "", "[neighbours]", "skin = 0.3")),
    ("Safety options (fastmace.config.FallbackConfig, GateConfig, RunnerConfig; SPEC §9.4, §12.3, §8.7).",
     ("[fallback]", 'fallback = "auto"', "", "[gate]", "cpu_float64_time_limit_s = 300.0", "",
      "[runner]", "stall_timeout_s = 1800.0")),
)


def add_arguments(parser):
    md_command.add_arguments(parser, required=False)


def value(item):
    """A TOML value: bool, int, float, str, or an array of them."""
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)):
        return repr(item)
    if isinstance(item, str):
        return json.dumps(item, ensure_ascii=False)  # JSON's escapes are TOML basic-string escapes
    if isinstance(item, (list, tuple)):
        return "[" + ", ".join(value(element) for element in item) + "]"
    raise TypeError(f"no TOML value for {item!r}")


def _record_table(name, record):
    """``[name]`` with the record's fields that differ from their defaults (the first field always)."""
    lines = [f"[{name}]"]
    for index, field in enumerate(dataclasses.fields(record)):
        current = getattr(record, field.name)
        if index == 0 or field.default is dataclasses.MISSING or current != field.default:
            lines.append(f"{field.name} = {value(current)}")
    return lines


def text(given):
    """The TOML text: ``given`` (fm.md arguments) set, every other argument commented."""
    lines = [HEADER]
    tables = []
    for name, comment, example in SCALARS:
        if comment:
            lines += ["", f"# {comment}"]
        current = given.get(name)
        if current is None:
            lines.append(f"# {name} = {example}")
        elif dataclasses.is_dataclass(current):
            lines.append(f"# {name} = {example}")
            tables.append(_record_table(name, current))
        else:
            lines.append(f"{name} = {value(current)}")
    for set_table in tables:
        lines += [""] + set_table
    for comment, table in TABLES:
        lines += ["", f"# {comment}"]
        lines += [f"# {line}" if line else "#" for line in table]
    return "\n".join(lines) + "\n"


def run(args):
    try:
        structure, options = md_command.md_arguments(args)
    except ValueError as error:
        return md_command.usage_error("init", error)
    given = dict(options)
    if structure is not None:
        given["structure"] = structure
    if not given:
        given = dict(EXAMPLE)
    print(text(given), end="")
    return 0
