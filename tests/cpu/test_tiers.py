"""The GPU suite's tier rules (plan task 1b-T27): ``_support.tiers`` and the conductor's checks."""

import os
import subprocess
import sys
from pathlib import Path

from _support import tiers

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_tier_of_each_mark_combination():
    assert tiers.tier([]) == "fast"
    assert tiers.tier(["gpu"]) == "fast"
    assert tiers.tier(["slow"]) == "full"
    assert tiers.tier(["slow", "acceptance"]) == "acceptance"


def test_acceptance_environment_names_each_missing_variable(tmp_path):
    assert len(tiers.acceptance_environment_problems({})) == 3
    environ = {"FASTMACE_BASELINE": str(tmp_path), "FASTMACE_MODELS": str(tmp_path),
               "FASTMACE_HYBRID_PYTHON": str(tmp_path / "missing-python")}
    problems = tiers.acceptance_environment_problems(environ)
    assert len(problems) == 1 and problems[0].startswith("FASTMACE_HYBRID_PYTHON=") and "does not exist" in problems[0]
    environ["FASTMACE_HYBRID_PYTHON"] = sys.executable
    assert tiers.acceptance_environment_problems(environ) == []


def test_which_mark_expressions_select_the_acceptance_tier():
    assert tiers.selects_acceptance("acceptance") and tiers.selects_acceptance("slow")
    assert tiers.selects_acceptance("gpu and acceptance")
    assert not tiers.selects_acceptance("")  # no -m: the tests keep their own skips
    assert not tiers.selects_acceptance(tiers.FAST) and not tiers.selects_acceptance(tiers.FULL)


def collect(*args, env=None):
    return subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *args], cwd=REPOSITORY_ROOT,
                          env=env, capture_output=True, text=True, timeout=600, check=False)


def ids(result):
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return {line for line in result.stdout.splitlines() if "::" in line}


def test_no_gpu_test_is_in_both_the_fast_and_the_acceptance_tier(tmp_path):
    bare = {k: v for k, v in os.environ.items() if k not in tiers.ACCEPTANCE_ENVIRONMENT}
    fast = ids(collect("-m", tiers.FAST, "tests/gpu", env=bare))
    # Stand-ins that exist: collection only (the GPU tests are skipped without CUDA anyway).
    stand_ins = dict(bare, FASTMACE_BASELINE=str(tmp_path), FASTMACE_MODELS=str(tmp_path),
                     FASTMACE_HYBRID_PYTHON=sys.executable)
    acceptance = ids(collect("-m", tiers.ACCEPTANCE, "tests/gpu", env=stand_ins))
    assert fast and acceptance and not fast & acceptance, sorted(fast & acceptance)
    assert any("against_the_research_engine" in node for node in acceptance)
    assert any("ase_equivalence" in node for node in acceptance)


def test_an_acceptance_run_without_its_environment_is_a_usage_error():
    bare = {k: v for k, v in os.environ.items() if k not in tiers.ACCEPTANCE_ENVIRONMENT}
    result = collect("-m", tiers.ACCEPTANCE, "tests/gpu", env=bare)
    assert result.returncode == 4, result.stdout[-3000:] + result.stderr[-3000:]
    assert "FASTMACE_BASELINE is not set" in result.stdout + result.stderr


def test_every_acceptance_test_is_also_slow(tmp_path):
    stand_ins = {k: v for k, v in os.environ.items() if k not in tiers.ACCEPTANCE_ENVIRONMENT}
    stand_ins.update(FASTMACE_BASELINE=str(tmp_path), FASTMACE_MODELS=str(tmp_path),
                     FASTMACE_HYBRID_PYTHON=sys.executable)
    result = collect("-m", "acceptance and not slow", "tests/gpu", env=stand_ins)
    assert result.returncode == 5, result.stdout[-3000:] + result.stderr[-3000:]  # 5: nothing collected


def test_a_whole_run_without_the_environment_starts_under_xdist():
    """No -m: acceptance tests are collected and skip as before; the xdist workers must not crash (1b-T27)."""
    bare = {k: v for k, v in os.environ.items() if k not in tiers.ACCEPTANCE_ENVIRONMENT}
    result = collect("-n", "2", "tests/gpu/test_selfcheck_gpu.py", "tests/cpu/test_tiers.py",
                     env=bare)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "INTERNALERROR" not in result.stdout + result.stderr


# --------------------------------------------------------------------------- the compatibility tier (4-T13)


def test_compat_takes_the_fast_tier_the_listed_tests_and_its_own():
    assert tiers.in_compat("test_smoke.py::test_anything", [])  # fast
    assert tiers.in_compat("test_compat_gpu.py::test_x", ["slow", "compat"])
    assert not tiers.in_compat("test_engine_gpu.py::test_slow_one", ["slow"])
    assert tiers.in_compat("test_loading_gpu.py::test_mh1_plain_cueq_float64", ["slow"])
    # an entry without [...] takes every parametrisation; one with [...] only that one
    assert tiers.in_compat("test_rewrites_mh1_gpu.py::test_mh1_float64_package_agrees_with_the_unrewritten_model[on]",
                           ["slow"])
    assert tiers.in_compat("test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[mp0-medium]", ["slow"])
    assert not tiers.in_compat("test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[omol0-xl]",
                               ["slow", "acceptance"])
    assert not tiers.in_compat("test_loading_gpu.py::test_mh1_plain_cueq_float64_and_more", ["slow"])


def test_every_listed_compat_test_is_collected_and_the_tier_is_the_fast_tier_plus_them(tmp_path):
    bare = {k: v for k, v in os.environ.items() if k not in tiers.ACCEPTANCE_ENVIRONMENT}
    compat = ids(collect("-m", tiers.COMPAT, "tests/gpu", env=bare))  # no acceptance environment needed
    fast = ids(collect("-m", tiers.FAST, "tests/gpu", env=bare))
    slow_compat = ids(collect("-m", f"{tiers.COMPAT} and slow", "tests/gpu", env=bare))
    assert fast <= compat and compat == fast | slow_compat
    for entry in tiers.COMPAT_TESTS:
        node = f"tests/gpu/{entry}"
        assert any(n == node or ("[" not in entry and n.startswith(node + "[")) for n in slow_compat), entry
    assert {n for n in slow_compat if n.startswith("tests/gpu/test_compat_gpu.py::")}, "the tier's own tests"
    assert not any("omol0-xl" in n or "against_the_research_engine" in n for n in compat)


def _compat_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_compat_tool", REPOSITORY_ROOT / "tools/acceptance/compat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_compat_driver_runs_the_tier_and_summarizes_its_junit(tmp_path):
    tool = _compat_tool()
    args = tool.parse_args(["--provider", "colab", "--label", "Colab L4", "--output", str(tmp_path), "--", "-x"])
    command = tool.pytest_command(args)
    assert command[1:6] == ["-m", "pytest", "tests/gpu", "-m", "compat"] and command[-1] == "-x"
    assert f"--junitxml={tmp_path / 'junit.xml'}" in command
    (tmp_path / "junit.xml").write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.gpu.test_a" name="test_ok" time="2.5"/>'
        '<testcase classname="tests.gpu.test_a" name="test_bad" time="1.0"><failure message="boom">x\nlast</failure>'
        '</testcase><testcase classname="tests.gpu.test_b" name="test_skip" time="0"><skipped/></testcase>'
        '</testsuite></testsuites>')
    summary = tool.summarize(tmp_path / "junit.xml")
    assert summary["counts"] == {"passed": 1, "failed": 1, "error": 0, "skipped": 1}
    assert summary["failures"] == [{"test": "tests.gpu.test_a::test_bad", "outcome": "failed", "message": "boom"}]
    assert summary["slowest"][0] == {"test": "tests.gpu.test_a::test_ok", "seconds": 2.5}
    assert tool.summarize(tmp_path / "missing.xml")["junit"] is None
