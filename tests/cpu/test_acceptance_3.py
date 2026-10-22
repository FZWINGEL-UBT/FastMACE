"""Acceptance tools of plan task 3-T21, without a GPU.

- ``tools/ab_fastmace.py --release p3b``: this checkout (side 3) against main before phase 3 (tag
  ``phase-3-baseline``, 156ab76) in one process, item 10. After decision 37 this checkout's side pins
  ``fixcm_projection="ase"``, the baseline's projection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY_ROOT / "tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TOOLS))
ab_fastmace = _load(TOOLS / "ab_fastmace.py", "_tool_ab_fastmace_3t21")


def _args(tmp_path, *extra):
    return ["--output", str(tmp_path / "out"), "--checkpoint", "m", "--renamed", str(tmp_path / "v"), *extra]


def test_release_p3b_is_main_before_phase_3(tmp_path):
    release = ab_fastmace.RELEASES["p3b"]
    assert (release.tag, release.commit, release.renamed, release.planner, release.current) == (
        "phase-3-baseline", "156ab76", "fastmace_p3b", "minimal", "3")
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b"))
    assert args.sides == ["p3b", "3"] and args.current == "3" and args.expected_commit == "156ab76"


def test_release_p3b_pins_the_ase_projection_on_this_checkout_only(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b", "--sides", "3", "p3b"))
    assert args.integrator_options == {"fixcm_projection": "ase"}
    assert ab_fastmace.Version("3", args, None).minimal and ab_fastmace.Version("p3b", args, None).minimal
    side = ab_fastmace.Version("3", args, sys.modules["fastmace"])
    assert side.integrator_options() == {"fixcm_projection": "ase"}
    assert ab_fastmace.Version("p3b", args, None).integrator_options() == {}


def test_an_explicit_integrator_option_overrides_the_pin(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b", "--integrator-option",
                                        "fixcm_projection=mass_weighted"))
    assert args.integrator_options == {"fixcm_projection": "mass_weighted"}
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b", "--integrator-option", "fixcm_projection"))


def test_other_releases_pin_nothing(tmp_path):
    args = ab_fastmace.parse_args(_args(tmp_path, "--release", "1b"))
    assert args.integrator_options == {}
    with pytest.raises(SystemExit):
        ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b", "--sides", "2", "p3b"))
