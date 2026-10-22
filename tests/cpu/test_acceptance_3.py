"""Acceptance tools of plan task 3-T21, without a GPU.

- ``tools/ab_fastmace.py --release p3b``: this checkout (side 3) against main before phase 3 (tag
  ``phase-3-baseline``, 156ab76) in one process, item 10. After decision 37 this checkout's side pins
  ``fixcm_projection="ase"``, the baseline's projection.
"""

from __future__ import annotations

import importlib.util
import json
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


# --- tools/trajectory_versions.py --against p3b, bitwise ----------------------------------------------------

trajectory_versions = _load(TOOLS / "trajectory_versions.py", "_tool_trajectory_versions_3t21")


def _run_args(tmp_path, *extra):
    return ["run", "--output", str(tmp_path / "out"), "--checkpoint", "m", *extra]


def test_trajectory_run_against_p3b(tmp_path):
    args = trajectory_versions.parse_args(_run_args(tmp_path, "--against", "p3b", "--side", "3", "--device", "cpu",
                                                    "--backend", "auto"))
    assert args.expected_commit == "156ab76" and args.device == "cpu" and not args.reproducible
    args = trajectory_versions.parse_args(_run_args(tmp_path, "--against", "p3b", "--side", "p3b", "--renamed", "v",
                                                    "--reproducible", "--backend", "cueq"))
    assert args.reproducible and args.expected_commit == "156ab76"
    for side in ("1b", "2"):
        with pytest.raises(SystemExit):
            trajectory_versions.parse_args(_run_args(tmp_path, "--against", "p3b", "--side", side, "--renamed", "v"))
    with pytest.raises(SystemExit):
        trajectory_versions.parse_args(_run_args(tmp_path, "--against", "p3b", "--side", "p3b"))
    assert trajectory_versions.parse_args(_run_args(tmp_path, "--side", "2")).expected_commit == "673a8a1"


def _rows(energy, shift=0.0, positions=True):
    return [{"step": s, "target": 300.0, "energy": energy, "temperature": 300.0,
             "positions": [[0.0, 0.0, shift + s * 1e-3]] if positions else None} for s in range(10, 101, 10)]


def _write(directory, side, rows):
    directory.mkdir()
    meta = dict(status="complete", side=side, fastmace_commit=side, backend="auto", steps=100, every=10, seeds=[1],
                targets=[300.0, 600.0], structure="NaCl", model="mh1-r2scan",
                runs={"1": dict(plan=dict(options={"a": True}), rows=rows)})
    (directory / "trajectory.json").write_text(json.dumps(meta))
    return directory


def test_compare_bitwise(tmp_path):
    release = _write(tmp_path / "a", "p3b", _rows(-1.0))
    same = _write(tmp_path / "b", "3", _rows(-1.0))
    moved = _write(tmp_path / "c", "3", _rows(-1.0, 1e-15))
    energy = _write(tmp_path / "d", "3", _rows(-1.0 + 1e-15))
    unrecorded = _write(tmp_path / "e", "3", _rows(-1.0, positions=False))
    result = trajectory_versions.compare(release, same, exact=True)
    assert result["passed"] and result["bitwise"] == {"1": True} and result["positions_max_A"] == {"1": 0.0}
    for other in (moved, energy, unrecorded):
        assert not trajectory_versions.compare(release, other, exact=True)["passed"]


def test_a_cpu_run_plans_both_versions_on_the_cpu(tmp_path, monkeypatch):
    """--device cpu reaches EngineConfig.device on both sides (its default is cuda:0)."""
    seen = []

    class Config:
        @staticmethod
        def EngineConfig(**kwargs):  # noqa: N802 (stands in for the class)
            seen.append(kwargs)
            return kwargs

    class Planner:
        def __init__(self, engine):
            self.engine = engine

        def hardware(self):
            return None

    class Minimal:
        MinimalPlanner = Planner

    for side, device in (("3", "cpu"), ("p3b", "cpu"), ("3", None)):
        args = ab_fastmace.parse_args(_args(tmp_path, "--release", "p3b"))
        args.device = device
        version = ab_fastmace.Version(side, args, None)
        monkeypatch.setattr(version, "module", lambda name: Config if name == "config" else Minimal)
        version.prepare()
    assert [kwargs.get("device") for kwargs in seen] == ["cpu", "cpu", None]
