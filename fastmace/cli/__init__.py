"""The fastmace command-line interface (SPEC §7.5).

``main`` dispatches to one module per subcommand (``COMMANDS``). Each module has
``HELP`` (one line), ``add_arguments(parser)`` and ``run(args) -> int``, and
imports nothing heavy at module level, so ``fastmace --help`` stays fast.

A ``FastMACEError`` is printed as its message alone (the §7.7 shape: what, why,
next step, record, ``fastmace doctor``), without a traceback, and the exit code
is 1; ``--debug`` prints the traceback too. Any other exception is a bug and
keeps its traceback.
"""

import argparse
import importlib
import sys
import traceback

import fastmace
from fastmace.errors import FastMACEError

# Subcommand -> module.
COMMANDS = {
    "doctor": "fastmace.cli.doctor",
    "models": "fastmace.cli.models",
    "md": "fastmace.cli.md",
    "init": "fastmace.cli.init",
    "run": "fastmace.cli.run",
    "relax": "fastmace.cli.relax",
    "quench": "fastmace.cli.quench",
}


def _debug_option(parser, default):
    parser.add_argument("--debug", action="store_true", default=default,
                        help="show the traceback of a FastMACE error")


def build_parser():
    parser = argparse.ArgumentParser(prog="fastmace", description=fastmace.__doc__)
    parser.add_argument("--version", action="version", version=f"fastmace {fastmace.__version__}")
    _debug_option(parser, False)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, module_name in COMMANDS.items():
        module = importlib.import_module(module_name)
        command = commands.add_parser(name, help=module.HELP, description=module.HELP)
        # SUPPRESS: a --debug before the command is not reset by the command's own default.
        _debug_option(command, argparse.SUPPRESS)
        module.add_arguments(command)
        command.set_defaults(module=module)
    return parser


def main(argv=None):
    """Run ``fastmace`` with ``argv`` (default: ``sys.argv[1:]``); returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return int(args.module.run(args) or 0)
    except FastMACEError as error:
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return 1
