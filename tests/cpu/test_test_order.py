"""tools/test_order.py and the conftest's longest-first order (suite-speed, 27 October 2026)."""

import importlib.util
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_tool_test_order", REPOSITORY_ROOT / "tools" / "test_order.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


order = _load()

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="tests.cpu.test_a" name="test_fast" time="0.5" />
<testcase classname="tests.cpu.test_a" name="test_param[x]" time="3.0" />
<testcase classname="tests.cpu.test_a" name="test_param[y]" time="7.25" />
<testcase classname="tests.cpu.test_b" name="test_long" time="40.0" />
<testcase classname="tests.cpu.test_b" name="test_skipped" time="90.0"><skipped message="no"/></testcase>
</testsuite></testsuites>
"""


def test_functions_keep_their_longest_case_and_drop_fast_and_skipped_tests(tmp_path):
    (tmp_path / "junit.xml").write_text(JUNIT)
    times = order.function_times([tmp_path / "junit.xml"], min_seconds=2.0)
    assert times == {"tests/cpu/test_a.py::test_param": 7.2, "tests/cpu/test_b.py::test_long": 40.0}


def test_main_writes_the_file(tmp_path):
    (tmp_path / "junit.xml").write_text(JUNIT)
    out = tmp_path / "durations.json"
    assert order.main([str(tmp_path / "junit.xml"), "--output", str(out), "--min-seconds", "5"]) == 0
    assert json.loads(out.read_text()) == {"tests/cpu/test_a.py::test_param": 7.2,
                                           "tests/cpu/test_b.py::test_long": 40.0}


def test_the_recorded_file_names_collected_test_functions():
    """Every entry of the checked-in file is a test function that exists, so the order is not silently stale."""
    times = json.loads(order.DURATIONS.read_text())
    assert times and all(seconds >= 2.0 for seconds in times.values())
    for node in times:
        path, function = node.split("::", 1)
        assert f"def {function}(" in (REPOSITORY_ROOT / path).read_text(), node


def test_longest_first_sorts_recorded_functions_and_keeps_the_rest_in_order(monkeypatch, tmp_path):
    import conftest

    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"t.py::slow": 9.0, "t.py::slower": 30.0}))
    monkeypatch.setattr(conftest, "_DURATIONS", durations)

    class Item:
        def __init__(self, nodeid):
            self.nodeid = nodeid

    items = [Item(n) for n in ("t.py::a", "t.py::slow[1]", "t.py::b", "t.py::slower", "t.py::slow[2]", "t.py::c")]
    conftest._longest_first(items)
    assert [i.nodeid for i in items] == ["t.py::slower", "t.py::slow[1]", "t.py::slow[2]", "t.py::a", "t.py::b",
                                         "t.py::c"]
