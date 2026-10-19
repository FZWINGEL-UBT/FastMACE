"""tools/test_budget.py (plan task 1b-T27): per-test budgets of the GPU tiers, from junit files."""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_tool_test_budget", REPOSITORY_ROOT / "tools" / "test_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget = _load()

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="tests.gpu.test_a" name="test_fast_ok" time="12.0" />
<testcase classname="tests.gpu.test_a" name="test_fast_slow[x]" time="75.5" />
<testcase classname="tests.gpu.test_b" name="test_full_ok" time="110.0" />
<testcase classname="tests.gpu.test_b" name="test_full_long" time="300.0" />
<testcase classname="tests.gpu.test_b" name="test_allowed[p]" time="400.0" />
<testcase classname="tests.gpu.test_c" name="test_acceptance" time="4000.0" />
<testcase classname="tests.gpu.test_c" name="test_skipped" time="900.0"><skipped message="no model"/></testcase>
</testsuite></testsuites>
"""
FAST = ["tests/gpu/test_a.py::test_fast_ok", "tests/gpu/test_a.py::test_fast_slow[x]"]
FULL = FAST + ["tests/gpu/test_b.py::test_full_ok", "tests/gpu/test_b.py::test_full_long",
               "tests/gpu/test_b.py::test_allowed[p]", "tests/gpu/test_c.py::test_skipped"]


def files(tmp_path, junit=JUNIT, fast=FAST, full=FULL):
    (tmp_path / "junit.xml").write_text(junit)
    (tmp_path / "fast.txt").write_text("\n".join(fast) + "\n")
    (tmp_path / "full.txt").write_text("\n".join(full) + "\n")
    return [str(tmp_path / "junit.xml"), "--fast-ids", str(tmp_path / "fast.txt"),
            "--full-ids", str(tmp_path / "full.txt")]


def test_junit_times_name_node_ids_and_leave_out_skipped_tests(tmp_path):
    files(tmp_path)
    times = budget.junit_times([tmp_path / "junit.xml"])
    assert times["tests/gpu/test_a.py::test_fast_slow[x]"] == 75.5
    assert "tests/gpu/test_c.py::test_skipped" not in times


def test_several_junit_files_keep_the_longest_time(tmp_path):
    (tmp_path / "a.xml").write_text(JUNIT)
    (tmp_path / "b.xml").write_text(JUNIT.replace('time="12.0"', 'time="61.0"'))
    times = budget.junit_times([tmp_path / "a.xml", tmp_path / "b.xml"])
    assert times["tests/gpu/test_a.py::test_fast_ok"] == 61.0


def test_violations_per_tier():
    times = {"tests/gpu/test_a.py::test_fast_ok": 12.0, "tests/gpu/test_a.py::test_fast_slow[x]": 75.5,
             "tests/gpu/test_b.py::test_full_ok": 110.0, "tests/gpu/test_b.py::test_full_long": 300.0,
             "tests/gpu/test_b.py::test_allowed[p]": 400.0, "tests/gpu/test_c.py::test_acceptance": 4000.0}
    allowed = {"tests/gpu/test_b.py::test_allowed": "a reason"}
    found = budget.violations(times, set(FAST), set(FULL), allowed=allowed)
    assert len(found) == 2
    assert "full tier, over 120 s: tests/gpu/test_b.py::test_full_long" in found[0]
    assert "fast tier, over 60 s: tests/gpu/test_a.py::test_fast_slow[x]" in found[1]


def test_the_allowlist_names_existing_tests_with_reasons():
    for key, reason in budget.FULL_TIER_ALLOWED.items():
        path, function = key.split("::")
        assert function in (REPOSITORY_ROOT / path).read_text(encoding="utf-8"), key
        assert reason


def test_command_line_exit_status(tmp_path):
    tool = [sys.executable, str(REPOSITORY_ROOT / "tools" / "test_budget.py")]
    over = subprocess.run([*tool, *files(tmp_path)], capture_output=True, text=True, check=False)
    assert over.returncode == 1, over.stdout + over.stderr
    assert "test_full_long" in over.stdout and "test_fast_slow[x]" in over.stdout
    assert "test_acceptance" not in over.stdout and "test_allowed" in over.stdout  # not in FULL_TIER_ALLOWED
    within = subprocess.run([*tool, *files(tmp_path), "--fast-limit", "100", "--full-limit", "500"],
                            capture_output=True, text=True, check=False)
    assert within.returncode == 0 and "all within budget" in within.stdout, within.stdout + within.stderr
