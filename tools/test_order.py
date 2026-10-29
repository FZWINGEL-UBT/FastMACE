"""Record which CPU tests are slow, so that xdist runs start them first (suite-speed, 27 October 2026).

Reads one or more junit files of a full CPU run (``pytest --junitxml``; with several, a test's longest
time counts) and writes ``tests/_support/data/cpu_durations.json``: seconds per test function (all
parametrisations together, their longest case) for every function that took at least --min-seconds.
``tests/conftest.py`` sorts these first, longest first, in every xdist worker, so that the long tests do
not start last and leave the other workers idle at the end of the run. The file changes only the order:
a stale or missing file costs time, never a test.

    python tools/test_order.py results/<run>/junit.xml
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DURATIONS = REPOSITORY_ROOT / "tests" / "_support" / "data" / "cpu_durations.json"


def _budget():
    spec = importlib.util.spec_from_file_location("_tool_test_budget", REPOSITORY_ROOT / "tools" / "test_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_times(paths, min_seconds: float = 2.0) -> dict[str, float]:
    """{test function node id: longest case's seconds}, for functions of at least ``min_seconds``."""
    budget = _budget()
    functions: dict[str, float] = {}
    for node, seconds in budget.junit_times(paths).items():
        function = budget.function_of(node)
        functions[function] = max(functions.get(function, 0.0), seconds)
    return {name: round(seconds, 1) for name, seconds in sorted(functions.items()) if seconds >= min_seconds}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("junit", nargs="+", type=Path)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DURATIONS)
    args = parser.parse_args(argv)
    times = function_times(args.junit, args.min_seconds)
    args.output.write_text(json.dumps(times, indent=0, sort_keys=True) + "\n", newline="\n")
    print(f"{len(times)} test functions of at least {args.min_seconds:g} s -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
