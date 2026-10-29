"""Packaging audit (6-T01): the wheel and sdist carry the package data and nothing else (SPEC §14)."""

import importlib.util
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("build") is None, reason="the `build` package is not installed"),
]

REQUIRED = [
    "fastmace/__init__.py",
    "fastmace/cli/__init__.py",
    "fastmace/kernels/oeq_streaming/LICENSE",
    "fastmace/kernels/oeq_streaming/NOTICE.md",
    "fastmace/longrange/data/dftd3_params.npz",
    "fastmace/longrange/data/LICENSE",
    "fastmace/longrange/data/NOTICE.md",
    "fastmace/protocols/data/SOURCES.md",
]
FORBIDDEN_TOP = {"results", "tests", "tools", "spikes", "plans", "docs", "constraints"}


def _build(out):
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)], check=True, capture_output=True)
    return out


def _check(names):
    for req in REQUIRED:
        assert req in names, f"missing {req}"
    assert any(n.startswith("fastmace/kernels/oeq_streaming/") and n.endswith(".cuh") for n in names)
    assert any(n.startswith("fastmace/protocols/data/") and n.endswith(".toml") for n in names)
    leaked = {n.split("/")[0] for n in names} & FORBIDDEN_TOP
    assert not leaked, f"leaked into the distribution: {leaked}"


def test_wheel_and_sdist_contents(tmp_path):
    """One build for both checks: two xdist workers building in the same source tree raced on its
    ``*.egg-info`` and ``build/`` (suite-speed, 27 October 2026), and one build is half the time."""
    dist = _build(tmp_path)
    _check_wheel(dist)
    _check_sdist(dist)


def _check_wheel(dist):
    (wheel,) = dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    _check(names)
    assert any(n.endswith(".dist-info/entry_points.txt") for n in names)
    # Every licence and notice (6-T13): the MIT licence, the LGPL text and NOTICE.md naming the LGPL files.
    for name in ("LICENSE", "NOTICE.md", "LICENSES/LGPL-2.1-or-later.txt"):
        assert any(n.endswith(".dist-info/licenses/" + name) for n in names), name


def _check_sdist(dist):
    (sdist,) = dist.glob("*.tar.gz")
    with tarfile.open(sdist) as tf:
        names = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
    _check(names)
    assert {"pyproject.toml", "README.md", "LICENSE", "NOTICE.md", "LICENSES/LGPL-2.1-or-later.txt"} <= names
