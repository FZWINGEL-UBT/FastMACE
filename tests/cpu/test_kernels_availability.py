"""Availability probes of fastmace.kernels: Triton, NVRTC through cuda.bindings, the streaming kernels."""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import fastmace.kernels as kernels

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_availability_is_truthy_only_when_available():
    assert kernels.Availability(True, "ok")
    assert not kernels.Availability(False, "missing")


@pytest.mark.parametrize("probe", [kernels.triton_available, kernels.nvrtc_available, kernels.streaming_available])
def test_probes_return_an_availability_with_a_reason(probe):
    result = probe()
    assert isinstance(result, kernels.Availability)
    assert isinstance(result.available, bool)
    assert result.reason and "\n" not in result.reason


def test_triton_probe_matches_the_installation():
    expected = importlib.util.find_spec("triton") is not None
    assert kernels.triton_available().available is expected


def test_triton_probe_reports_why_it_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)  # makes `import triton` raise ImportError
    result = kernels.triton_available()
    assert not result
    assert result.reason.startswith("triton is not importable (ModuleNotFoundError")


def test_triton_probe_reports_the_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(__version__="9.9.9"))
    result = kernels.triton_available()
    assert result and result.reason == "triton 9.9.9 is importable"


def fake_nvrtc(monkeypatch, version):
    cuda = types.ModuleType("cuda")
    bindings = types.ModuleType("cuda.bindings")
    nvrtc = types.ModuleType("cuda.bindings.nvrtc")
    nvrtc.nvrtcVersion = version
    cuda.bindings, bindings.nvrtc = bindings, nvrtc
    for name, module in (("cuda", cuda), ("cuda.bindings", bindings), ("cuda.bindings.nvrtc", nvrtc)):
        monkeypatch.setitem(sys.modules, name, module)


def test_nvrtc_probe_reports_the_version(monkeypatch):
    fake_nvrtc(monkeypatch, lambda: (0, 13, 0))
    result = kernels.nvrtc_available()
    assert result and result.reason == "NVRTC 13.0 loads through cuda.bindings"


def test_nvrtc_probe_reports_a_library_that_does_not_load(monkeypatch):
    def broken():
        raise RuntimeError("Failed to dlopen libnvrtc.so.13")

    fake_nvrtc(monkeypatch, broken)
    result = kernels.nvrtc_available()
    assert not result and "NVRTC does not load (RuntimeError: Failed to dlopen" in result.reason


def test_nvrtc_probe_reports_missing_bindings(monkeypatch):
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", None)
    result = kernels.nvrtc_available()
    assert not result and result.reason.startswith("cuda.bindings.nvrtc is not importable")


def test_streaming_probe_needs_nvrtc_first(monkeypatch):
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", None)
    assert kernels.streaming_available() == kernels.nvrtc_available()


def test_streaming_probe_names_a_missing_package(monkeypatch):
    fake_nvrtc(monkeypatch, lambda: (0, 13, 0))
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "openequivariance"
                        else real(name, *a))
    result = kernels.streaming_available()
    assert not result and result.reason.startswith("openequivariance is not installed")


def test_streaming_probe_does_not_import_openequivariance(monkeypatch):
    fake_nvrtc(monkeypatch, lambda: (0, 13, 0))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    monkeypatch.delitem(sys.modules, "openequivariance", raising=False)
    result = kernels.streaming_available()
    assert result
    assert result.reason == "NVRTC 13.0 loads through cuda.bindings; jinja2 and openequivariance are installed"
    assert "openequivariance" not in sys.modules


def test_importing_kernels_and_probing_imports_nothing_heavy():
    code = ("import sys, fastmace.kernels as k\n"
            "before = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'triton', 'cuda', 'openequivariance', 'mace'))\n"
            "k.streaming_available()\n"
            "after = sorted(m for m in sys.modules if m.split('.')[0] in ('openequivariance', 'mace'))\n"
            "print(before, after)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY_ROOT), os.environ.get("PYTHONPATH", "")]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY_ROOT, env=env, capture_output=True,
                            text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[] []"
