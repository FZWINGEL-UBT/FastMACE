"""Selection of the prebuilt OEq host extension (fastmace-oeq-ext, plan task 6-T04), mocked on the CPU."""

import json
import sys
import types
from pathlib import Path

import pytest

from fastmace.cli import doctor
from fastmace.errors import OEqRuntimeError
from fastmace.kernels import _oeq_runtime as runtime

PYTHON = f"cp{sys.version_info.major}{sys.version_info.minor}"
VERSIONS = {"torch": "2.14.1+cu130", "openequivariance": "0.7.0"}
LINUX = lambda: ("linux", "x86_64")  # noqa: E731


def make_package(tmp_path, library=True, **changes):
    directory = tmp_path / "fastmace_oeq_ext"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"extension_version": "0.7.0.post1", "torch": "2.14.1+cu130", "openequivariance": "0.7.0",
                "python": PYTHON, "library": "libtorch_tp_jit.so", **changes}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if library:
        (directory / "libtorch_tp_jit.so").write_bytes(b"\x7fELF")
    return directory


def look(directory, environ=None, versions=VERSIONS, platform=LINUX):
    return runtime.prebuilt_extension(environ or {}, versions=versions, package_directory=lambda: directory,
                                      platform=platform)


def test_a_matching_wheel_is_used(tmp_path):
    directory = make_package(tmp_path)
    found = look(directory)
    assert found.usable and found.library == directory / "libtorch_tp_jit.so" and found.reason is None


def test_not_installed_falls_back(tmp_path):
    found = look(None)
    assert not found.usable and "not installed" in found.reason


@pytest.mark.parametrize("key, value", [("torch", "2.14.1+cu128"), ("torch", "2.15.0+cu130"),
                                        ("openequivariance", "0.7.1"), ("python", "cp39")])
def test_a_version_mismatch_falls_back_and_says_which(tmp_path, key, value):
    found = look(make_package(tmp_path, **{key: value}))
    assert not found.usable and key in found.reason and value in found.reason


def test_an_installed_version_mismatch_falls_back(tmp_path):
    found = look(make_package(tmp_path), versions={"torch": "2.14.0+cu130", "openequivariance": "0.7.0"})
    assert not found.usable and "2.14.0+cu130" in found.reason


def test_the_switch_turns_it_off(tmp_path):
    found = look(make_package(tmp_path), environ={runtime.PREBUILT_SWITCH: "0"})
    assert not found.usable and runtime.PREBUILT_SWITCH in found.reason


def test_a_missing_library_or_manifest_falls_back(tmp_path):
    assert "missing" in look(make_package(tmp_path, library=False)).reason
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "manifest.json" in look(empty).reason


def test_other_platforms_fall_back(tmp_path):
    found = look(make_package(tmp_path), platform=lambda: ("linux", "aarch64"))
    assert not found.usable and "aarch64" in found.reason


def test_the_loader_serves_only_oeqs_extension(tmp_path):
    calls = []
    fake = types.SimpleNamespace(load=lambda name, *a, **k: calls.append(("built", name)) or "built")
    library = tmp_path / "libtorch_tp_jit.so"
    wrapper = runtime.install_prebuilt_loader(library, cpp_extension=fake,
                                              load_module=lambda path, name: ("prebuilt", Path(path), name))
    assert fake.load is wrapper
    assert fake.load("libtorch_tp_jit", ["a.cpp"], extra_cflags=["-O3"]) == ("prebuilt", library, "libtorch_tp_jit")
    assert fake.load("other", ["b.cpp"]) == "built" and calls == [("built", "other")]
    assert runtime.install_prebuilt_loader(tmp_path / "x.so", cpp_extension=fake) is wrapper  # idempotent


def _fake_plan(tmp_path, environ):
    return runtime.RuntimePlan(root=tmp_path / "cuda-runtime",
                               environment={"TORCH_EXTENSIONS_DIR": str(tmp_path / "ext"), "CUDA_HOME": "x"})


def _prepare(monkeypatch, tmp_path, found, environ):
    monkeypatch.setattr(runtime, "_check_not_too_late", lambda: None)
    monkeypatch.setattr(runtime, "prebuilt_extension", lambda environ: found)
    monkeypatch.setattr(runtime, "_plan_for", lambda cache, environ: _fake_plan(tmp_path, environ))
    monkeypatch.setattr(runtime, "materialize", lambda plan: None)
    installed = []
    monkeypatch.setattr(runtime, "install_prebuilt_loader", lambda library: installed.append(library))
    built = []
    monkeypatch.setattr(runtime, "build_identity", lambda plan, environ: {"k": 1})
    monkeypatch.setattr(runtime, "reuse_extension",
                        lambda *a: built.append(a) or {"key": "k", "path": "p", "source": "built",
                                                       "seconds": 0.0, "error": None})
    monkeypatch.setattr(runtime, "_check_build_locks", lambda directory, wait: None)
    plan, handles = runtime.prepare_in_process(tmp_path, preload=False, environ=environ)
    return plan, installed, built


def test_prepare_prefers_the_prebuilt_library_and_builds_nothing(monkeypatch, tmp_path):
    library = tmp_path / "libtorch_tp_jit.so"
    found = runtime.Prebuilt(library, {"extension_version": "0.7.0.post1"})
    plan, installed, built = _prepare(monkeypatch, tmp_path, found, {})
    assert installed == [library] and built == []
    assert plan.extension["source"] == "prebuilt" and plan.extension["version"] == "0.7.0.post1"


def test_prepare_falls_back_to_the_toolchain_build_with_the_reason(monkeypatch, tmp_path):
    found = runtime.Prebuilt(None, {}, "fastmace-oeq-ext is not installed")
    plan, installed, built = _prepare(monkeypatch, tmp_path, found, {})
    assert installed == [] and len(built) == 1
    assert plan.extension["source"] == "built" and plan.extension["prebuilt_skipped"] == found.reason


def test_switch_1_refuses_the_toolchain_build(monkeypatch, tmp_path):
    found = runtime.Prebuilt(None, {}, "fastmace-oeq-ext is not installed")
    with pytest.raises(OEqRuntimeError, match="prebuilt extension not usable"):
        _prepare(monkeypatch, tmp_path, found, {runtime.PREBUILT_SWITCH: "1"})


def _doctor(monkeypatch, found, which=None):
    monkeypatch.setattr(doctor, "_find_spec", lambda name: True)
    monkeypatch.setattr(doctor, "_version", lambda *names: "0.7.0")
    monkeypatch.setattr(doctor, "_prebuilt", lambda environ: found)
    monkeypatch.setattr(doctor, "_which", which or (lambda tool: None))  # no g++, no ninja
    monkeypatch.setattr(doctor, "_oeq_runtime_problems", lambda environ: [])
    monkeypatch.setattr(doctor, "_built_extensions", lambda environ: [])
    monkeypatch.setattr(doctor, "_old_build_locks", lambda environ: ([], []))
    return doctor.check_oeq(doctor.Context(environ={}))


def test_doctor_reports_the_prebuilt_extension_and_needs_no_compiler(monkeypatch, tmp_path):
    found = runtime.Prebuilt(tmp_path / "libtorch_tp_jit.so", {"extension_version": "0.7.0.post1"})
    item = _doctor(monkeypatch, found)
    assert item.ok and "prebuilt (fastmace-oeq-ext 0.7.0.post1" in item.detail and "not found" not in item.detail


def test_doctor_reports_the_toolchain_and_why(monkeypatch):
    found = runtime.Prebuilt(None, {}, "fastmace-oeq-ext was built for torch 2.14.1+cu130, installed is 2.15.0")
    item = _doctor(monkeypatch, found)
    assert not item.ok and "host toolchain" in item.detail and found.reason in item.detail
    assert "g++ not found" in item.detail and "fastmace[oeq]" in item.fix
    item = _doctor(monkeypatch, found, which=lambda tool: f"/usr/bin/{tool}")
    assert item.ok and "host toolchain" in item.detail
