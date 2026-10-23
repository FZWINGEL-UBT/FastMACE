"""``fastmace md``: the CLI face of ``fm.md`` for constant-temperature and NVE runs (SPEC §7.5, §7.7).

Each §7.5 flag is one ``fm.md`` argument (``md_arguments``); a flag not given is left to
``fm.md``'s default, so the CLI has no defaults of its own. ``--dispersion d3bj`` is passed as
``"d3bj"``, which ``fm.md`` expands from the head's training functional; with ``--d3-functional``
the CLI builds the explicit ``fm.D3`` record instead. Either goes into the force field (MACE plus D3).

Schedules, MC moves, explicit integrators and the full D3 record go through ``fastmace init`` and
``fastmace run``. A ``ValueError`` in the arguments is a usage error (exit 2); a ``FastMACEError``
is printed by ``fastmace.cli.main`` (exit 1).
"""

import sys

from fastmace.quick import ENSEMBLES

HELP = "run constant-temperature or NVE MD in one command (the CLI face of fm.md)"

ENSEMBLE_CHOICES = ENSEMBLES
# fastmace.simulation.CHECK_MODES, written out so that --help imports no torch (a test keeps them equal).
CHECK_CHOICES = ("skip", "gate", "allow_unavailable")
CELL_MODE_CHOICES = ("hydrostatic", "full")
DISPERSION_CHOICES = ("none", "d3bj")

# argparse destination -> fm.md argument, for the flags that pass their value unchanged.
_PLAIN = {
    "model": "model",
    "head": "head",
    "charge": "charge",
    "spin_multiplicity": "spin_multiplicity",
    "ensemble": "ensemble",
    "temperature": "temperature_K",
    "pressure": "pressure_GPa",
    "cell_mode": "cell_mode",
    "timestep": "timestep_fs",
    "steps": "steps",
    "seed": "seed",
    "check": "check",
    "output": "output",
    "sha256": "sha256",
}


def add_arguments(parser, *, required=True):
    """The §7.5 flags. ``required=False`` for ``fastmace init``, where every flag is optional."""
    model = parser.add_argument_group("model")
    model.add_argument("--model", required=required, help="model label (fastmace models), path or URL")
    model.add_argument("--head", help="head of a multi-head model (required for MH-1)")
    model.add_argument("--charge", type=int, help="total charge, for models that take it")
    model.add_argument("--spin-multiplicity", type=int, help="spin multiplicity, for models that take it")
    model.add_argument("--sha256", help="expected SHA-256 of a model file outside the label table")
    model.add_argument("--trusted", action="store_true", default=None,
                       help="trust a model file without a known SHA-256 (it is unpickled)")
    run = parser.add_argument_group("run")
    run.add_argument("--structure", required=required, help="structure file (any format ASE reads)")
    run.add_argument("--repeat", type=int, nargs=3, metavar=("NA", "NB", "NC"),
                     help="supercell repeats (default 1 1 1: the cell is never expanded silently)")
    run.add_argument("--ensemble", type=str.upper, choices=ENSEMBLE_CHOICES,
                     help="NVT (Langevin, default), NPT (LangevinBAOAB) or NVE (velocity Verlet from the "
                          "structure's momenta, without --temperature)")
    run.add_argument("--temperature", type=float, metavar="K", help="target temperature in K")
    run.add_argument("--pressure", type=float, metavar="GPA", help="NPT pressure in GPa (default 0)")
    run.add_argument("--cell-mode", choices=CELL_MODE_CHOICES, help="NPT cell mode (default hydrostatic)")
    run.add_argument("--timestep", type=float, metavar="FS", help="timestep in fs (default 0.5)")
    run.add_argument("--steps", type=int, help="number of MD steps")
    run.add_argument("--dispersion", choices=DISPERSION_CHOICES,
                     help="D3(BJ) dispersion: expanded from the head's training functional (SPEC §7.7)")
    run.add_argument("--d3-functional", metavar="NAME", help="the D3 parameter set, with --dispersion d3bj")
    run.add_argument("--seed", type=int, help="noise seed (default: drawn and recorded)")
    run.add_argument("--check", choices=CHECK_CHOICES, help="gate on the structure (default skip; opt-in: gate)")
    run.add_argument("--output", metavar="DIR", help="output directory (default ./<structure stem>.fastmace/)")


def _dispersion(args):
    """``fm.md``'s ``dispersion=`` from ``--dispersion`` and ``--d3-functional``; absent means none."""
    functional = getattr(args, "d3_functional", None)
    if functional is not None and args.dispersion != "d3bj":
        raise ValueError("--d3-functional needs --dispersion d3bj")
    if args.dispersion != "d3bj":
        return None
    if functional is None:
        return "d3bj"
    from fastmace.longrange.d3 import D3

    return D3(functional=functional)


def md_arguments(args):
    """``(structure, options)`` for ``fm.md(structure, **options)`` from the parsed flags.

    Only the flags given appear in ``options``; ``ValueError`` for a flag combination that has no
    ``fm.md`` argument (``--d3-functional`` without ``--dispersion d3bj``).
    """
    options = {name: getattr(args, dest) for dest, name in _PLAIN.items() if getattr(args, dest) is not None}
    if args.trusted:
        options["trusted"] = True
    if args.repeat is not None:
        options["repeat"] = tuple(args.repeat)
    dispersion = _dispersion(args)
    if dispersion is not None:
        options["dispersion"] = dispersion
    return args.structure, options


def usage_error(command, error):
    print(f"fastmace {command}: error: {error}", file=sys.stderr)
    return 2


def run_md(structure, options, *, command, debug=False):
    """Call ``fm.md`` and print its report; a ``ValueError`` in the arguments is a usage error (2).

    ``PlanRefused`` is a ``ValueError`` too, but a ``FastMACEError``: it goes to ``main``.
    """
    from fastmace.errors import FastMACEError
    from fastmace.quick import md

    try:
        result = md(structure, **options)
    except FastMACEError:
        raise
    except ValueError as error:
        if debug:
            raise
        return usage_error(command, error)
    print(result.summary)
    print()
    print(f"Output: {result.output} (metadata.json holds the plan, gate and provenance)")
    return 0


def run(args):
    try:
        structure, options = md_arguments(args)
    except ValueError as error:
        return usage_error("md", error)
    return run_md(structure, options, command="md", debug=args.debug)
