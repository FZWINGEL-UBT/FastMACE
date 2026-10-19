"""Per-test time budgets of the GPU tiers (plan task 1b-T27; docs/development.md, "Test tiers").

Reads one or more junit files (``pytest --junitxml``; with several, a test's longest time counts) and
the tiers the tests are in now, and lists:

- every fast-tier test (``-m "not slow and not acceptance"``) over --fast-limit seconds (default 60):
  mark it ``slow`` and keep a cheaper test of its behaviour in the fast tier;
- every full-tier test (``-m "not acceptance"``) over --full-limit seconds (default 120) that is not in
  FULL_TIER_ALLOWED: make it cheaper, mark it ``acceptance`` (with ``slow``), or add it to
  FULL_TIER_ALLOWED with the reason it must run at every merge.

Exit status 1 when it lists any test. The tiers come from ``pytest --collect-only`` over tests/gpu in
this Python (a CPU machine is enough), or from --fast-ids/--full-ids files of node ids, one per line.

    python tools/test_budget.py results/<run>/gpu-suite/junit.xml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FAST = "not slow and not acceptance"
FULL = "not acceptance"

# Full-tier tests allowed over the limit, by test function (all parametrisations): the reason they
# run at every merge rather than at acceptance.
FULL_TIER_ALLOWED = {
    "tests/gpu/test_canonical_restart_gpu.py::test_mh1_plain_cueq_float64_resumes_within_1e_12":
        "the only check that a resumed MH-1 engine follows the uninterrupted one within 1e-12 Å (203 s)",
    "tests/gpu/test_planner_gpu.py::test_mh1_plain_cueq_float64_planned_equals_hand_built_and_minimal":
        "planned, hand-built and minimal engines on MH-1, merged from four tests in 1b-T27",
    "tests/gpu/test_planner_gpu.py::test_mh1_hybrid_float32_planned_equals_hand_built_and_minimal":
        "planned, hand-built and minimal engines on MH-1, merged from four tests in 1b-T27",
}


def node_id(classname: str, name: str) -> str:
    """``tests.gpu.test_x`` and ``test_y[a]`` -> ``tests/gpu/test_x.py::test_y[a]``."""
    return classname.replace(".", "/") + ".py::" + name


def junit_times(paths) -> dict[str, float]:
    """{node id: seconds}, the longest over the files; skipped tests are left out."""
    times: dict[str, float] = {}
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            if case.find("skipped") is not None:
                continue
            node = node_id(case.get("classname", ""), case.get("name", ""))
            times[node] = max(times.get(node, 0.0), float(case.get("time", 0.0)))
    return times


def collect(expression: str, python=sys.executable) -> set[str]:
    """Node ids under tests/gpu that ``-m expression`` selects."""
    done = subprocess.run([python, "-m", "pytest", "--collect-only", "-q", "-m", expression, "tests/gpu"],
                          cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False)
    if done.returncode not in (0, 5):  # 5: nothing collected
        raise RuntimeError(f"pytest --collect-only -m {expression!r} failed:\n"
                           f"{done.stdout[-3000:]}{done.stderr[-3000:]}")
    return {line.strip() for line in done.stdout.splitlines() if "::" in line}


def read_ids(path) -> set[str]:
    return {line.strip() for line in Path(path).read_text().splitlines() if "::" in line}


def function_of(node: str) -> str:
    return node.split("[", 1)[0]


def violations(times, fast, full, fast_limit=60.0, full_limit=120.0, allowed=None) -> list[str]:
    """Lines naming each test over its tier's budget, longest first."""
    allowed = FULL_TIER_ALLOWED if allowed is None else allowed
    out = []
    for node, seconds in sorted(times.items(), key=lambda item: -item[1]):
        if node in fast and seconds > fast_limit:
            out.append(f"{seconds:8.1f} s  fast tier, over {fast_limit:g} s: {node}  (mark it slow)")
        elif node in full and node not in fast and seconds > full_limit and function_of(node) not in allowed:
            out.append(f"{seconds:8.1f} s  full tier, over {full_limit:g} s: {node}  "
                       "(mark it acceptance, or add it to FULL_TIER_ALLOWED with a reason)")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("junit", nargs="+", help="junit XML files (pytest --junitxml)")
    parser.add_argument("--fast-ids", help="node ids of the fast tier, one per line (default: collect)")
    parser.add_argument("--full-ids", help="node ids of the full tier, one per line (default: collect)")
    parser.add_argument("--fast-limit", type=float, default=60.0)
    parser.add_argument("--full-limit", type=float, default=120.0)
    args = parser.parse_args(argv)
    times = junit_times(args.junit)
    fast = read_ids(args.fast_ids) if args.fast_ids else collect(FAST)
    full = read_ids(args.full_ids) if args.full_ids else collect(FULL)
    found = violations(times, fast, full, args.fast_limit, args.full_limit)
    measured = [node for node in times if node in full]
    print(f"{len(measured)} full-tier tests measured ({sum(node in fast for node in measured)} fast); "
          f"budgets: fast {args.fast_limit:g} s, full {args.full_limit:g} s")
    for line in found:
        print(line)
    print(f"{len(found)} over budget" if found else "all within budget")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
