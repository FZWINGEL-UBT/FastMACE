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


def test_acceptance_without_slow_is_a_problem():
    problems = tiers.marker_problems({"a.py::ok": ["slow", "acceptance"], "a.py::bad": ["acceptance"],
                                      "a.py::fast": ["gpu"]})
    assert len(problems) == 1 and problems[0].startswith("a.py::bad: marked acceptance but not slow")


def test_acceptance_environment_names_each_missing_variable(tmp_path):
    assert len(tiers.acceptance_environment_problems({})) == 3
    environ = {"FASTMACE_BASELINE": str(tmp_path), "FASTMACE_MODELS": str(tmp_path),
               "FASTMACE_HYBRID_PYTHON": str(tmp_path / "missing-python")}
    problems = tiers.acceptance_environment_problems(environ)
    assert len(problems) == 1 and problems[0].startswith("FASTMACE_HYBRID_PYTHON=") and "does not exist" in problems[0]
    environ["FASTMACE_HYBRID_PYTHON"] = sys.executable
    assert tiers.acceptance_environment_problems(environ) == []


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
