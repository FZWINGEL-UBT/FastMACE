"""The OEq runtime plan (fastmace/kernels/_oeq_runtime.py, plan tasks 1a-T00 and 1a-T05): pure environment computation.

A fake cu13 wheel directory and driver in tmp_path; nothing is imported from
torch or OpenEquivariance and nothing touches a GPU.
"""

import os
import sys

import pytest

from fastmace.kernels import _oeq_runtime as runtime_module


def fake_runtime(tmp_path, libraries=runtime_module.RUNTIME_LIBRARIES, headers=("cuda.h", "nvrtc.h")):
    runtime = tmp_path / "env" / "nvidia" / "cu13"
    (runtime / "include" / "cccl").mkdir(parents=True)
    (runtime / "lib").mkdir()
    for header in headers:
        (runtime / "include" / header).write_text("")
    for name in libraries:
        (runtime / "lib" / f"lib{name}.so.13").write_text("")
    driver = tmp_path / "driver" / "libcuda.so.1"
    driver.parent.mkdir()
    driver.write_text("")
    return runtime, driver


def make_plan(tmp_path, environ=None, system=()):
    runtime, driver = fake_runtime(tmp_path)
    cache = tmp_path / "cache"
    extensions = cache / "torch-extensions-x"
    environ = {"PATH": "/usr/bin:/bin"} if environ is None else environ
    tools = tmp_path / "env" / "bin"
    return runtime_module.plan(runtime, driver, cache, environ, extensions, tools, system), runtime, driver, cache


def test_plan_sets_the_variables_of_oeq_no_nvcc(tmp_path):
    plan, runtime, driver, cache = make_plan(tmp_path)
    root = cache / "cuda-runtime"
    env = plan.environment
    assert env["CUDA_HOME"] == env["CUDA_PATH"] == str(root)
    sentinel = str(root / "bin" / "nvcc")
    assert env["CUDACXX"] == env["CMAKE_CUDA_COMPILER"] == env["PYTORCH_NVCC"] == sentinel
    assert env["PATH"].split(os.pathsep)[0] == str(root / "bin")  # the sentinel wins over any nvcc
    assert env["PATH"].split(os.pathsep)[1:] == [str(tmp_path / "env" / "bin"), "/usr/bin", "/bin"]  # ninja
    assert env["CPATH"] == str(runtime / "include" / "cccl")
    assert env["OEQ_JIT_EXTENSION"] == "1"
    assert env["MAX_JOBS"] == "4"
    assert env["TORCH_EXTENSIONS_DIR"] == str(cache / "torch-extensions-x")
    assert env["OEQ_NVCC_ATTEMPTS"] == str(cache / "nvcc-attempts.log")
    assert "LD_LIBRARY_PATH" not in env  # cannot take effect in a running process
    assert "CXX" not in env


def test_plan_links_headers_libraries_and_driver(tmp_path):
    plan, runtime, driver, cache = make_plan(tmp_path)
    lib64 = cache / "cuda-runtime" / "lib64"
    assert plan.links[cache / "cuda-runtime" / "include"] == runtime / "include"
    assert plan.links[lib64 / "libcuda.so"] == driver
    for name in ("cudart", "nvrtc", "cublas", "cublasLt"):
        assert plan.links[lib64 / f"lib{name}.so"] == runtime / "lib" / f"lib{name}.so.13"
    assert [p.name for p in plan.preload] == ["libcudart.so.13", "libnvrtc.so.13", "libcublasLt.so.13",
                                              "libcublas.so.13"]


def test_plan_keeps_user_settings(tmp_path):
    environ = {"PATH": "/a", "CPATH": "/inc", "MAX_JOBS": "2", "LD_LIBRARY_PATH": "/l"}
    plan, runtime, driver, cache = make_plan(tmp_path, environ)
    assert plan.environment["MAX_JOBS"] == "2"
    assert plan.environment["CPATH"].split(os.pathsep) == [str(runtime / "include" / "cccl"), "/inc"]
    assert environ == {"PATH": "/a", "CPATH": "/inc", "MAX_JOBS": "2", "LD_LIBRARY_PATH": "/l"}  # unchanged


def test_plan_wraps_the_compiler_when_a_system_cuda_has_link_libraries(tmp_path):
    system = tmp_path / "usr-lib"
    system.mkdir()
    (system / "libnvrtc.so").write_text("")
    plan, runtime, driver, cache = make_plan(tmp_path, system=(str(tmp_path / "none"), str(system)))
    wrapper = cache / "cuda-runtime" / "bin" / "c++-cuda13"
    assert plan.environment["CXX"] == str(wrapper)
    text, mode = plan.files[wrapper]
    assert f'-L"{cache / "cuda-runtime" / "lib64"}"' in text and mode == 0o755


@pytest.mark.parametrize("missing", ["cuda.h", "nvrtc.h"])
def test_plan_refuses_a_runtime_without_headers(tmp_path, missing):
    runtime, driver = fake_runtime(tmp_path, headers=[h for h in ("cuda.h", "nvrtc.h") if h != missing])
    with pytest.raises(runtime_module.OEqRuntimeError, match=missing):
        runtime_module.plan(runtime, driver, tmp_path / "c", {}, tmp_path / "c" / "x", tmp_path, ())


def test_plan_refuses_a_runtime_without_a_library(tmp_path):
    runtime, driver = fake_runtime(tmp_path, libraries=("cudart", "cublasLt", "cublas"))
    with pytest.raises(runtime_module.OEqRuntimeError, match="libnvrtc.so.13"):
        runtime_module.plan(runtime, driver, tmp_path / "c", {}, tmp_path / "c" / "x", tmp_path, ())


def test_find_driver(tmp_path):
    driver = tmp_path / "libcuda.so.1"
    driver.write_text("")
    assert runtime_module.find_driver({}, (str(tmp_path / "no"), str(driver))) == driver
    assert runtime_module.find_driver({"OEQ_CUDA_DRIVER_LIBRARY": str(driver)}, ()) == driver
    with pytest.raises(runtime_module.OEqRuntimeError, match="OEQ_CUDA_DRIVER_LIBRARY"):
        runtime_module.find_driver({}, (str(tmp_path / "no"),))


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links and sh sentinel are POSIX")
def test_materialize_is_idempotent(tmp_path):
    plan, runtime, driver, cache = make_plan(tmp_path)
    runtime_module.materialize(plan)
    runtime_module.materialize(plan)
    for link, target in plan.links.items():
        assert os.readlink(link) == str(target)
    sentinel = cache / "cuda-runtime" / "bin" / "nvcc"
    assert os.access(sentinel, os.X_OK)
    assert "exit 97" in sentinel.read_text()
    assert sorted(p.name for p in sentinel.parent.iterdir()) == ["nvcc"]  # no temporaries left


def test_prepare_refuses_after_openequivariance_import(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openequivariance", object())
    with pytest.raises(runtime_module.OEqRuntimeError, match="already imported"):
        runtime_module.prepare_in_process(tmp_path, environ={})


def test_promoted_module_has_only_the_in_process_entry_point():
    # 1a-T05 removed the spike's re-exec and launcher candidates (plan, conductor decisions after wave 0).
    for name in ("prepare_by_reexec", "main", "PREPARED_VARIABLE"):
        assert not hasattr(runtime_module, name)
    assert callable(runtime_module.prepare_in_process)
    assert "loader_path" not in runtime_module.RuntimePlan.__dataclass_fields__


@pytest.mark.parametrize("module", ["openequivariance", "torch.utils.cpp_extension"])
def test_prepare_names_the_module_imported_too_early(tmp_path, monkeypatch, module):
    fake = type("Module", (), {"CUDA_HOME": "/usr/local/cuda"})()
    monkeypatch.delitem(sys.modules, "openequivariance", raising=False)
    monkeypatch.setitem(sys.modules, module, fake)
    with pytest.raises(runtime_module.OEqRuntimeError, match=module.replace(".", r"\.")):
        runtime_module.prepare_in_process(tmp_path, environ={})
