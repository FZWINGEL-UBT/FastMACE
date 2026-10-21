"""The OEq runtime plan (fastmace/kernels/_oeq_runtime.py, plan tasks 1a-T00 and 1a-T05): pure environment computation.

A fake cu13 wheel directory and driver in tmp_path; nothing is imported from
torch or OpenEquivariance and nothing touches a GPU.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fastmace.kernels import _oeq_runtime as runtime_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


# ------------------------------------------------------------------ persistent host extension (1b-T24)

IDENTITY_VERSIONS = {"torch": "2.14.1+cu130", "openequivariance": "0.7.0"}


def fake_build(directory):
    """What torch leaves in TORCH_EXTENSIONS_DIR after building OEq's extension (and its stale lock)."""
    build = directory / "libtorch_tp_jit"
    build.mkdir(parents=True)
    for name in ("build.ninja", ".ninja_log", ".ninja_deps", "json11.o", "libtorch_tp_jit.o", "lock"):
        (build / name).write_text(name)
    (build / "libtorch_tp_jit.so").write_text(f"built by {os.getpid()}")


def refuse_build(directory):
    raise AssertionError("rebuilt although a build for this key exists")


def identity_of(plan, versions=IDENTITY_VERSIONS, compiler="g++ (Ubuntu 13.3.0) 13.3.0"):
    return runtime_module.build_identity(plan, {"PATH": "/usr/bin"}, versions=versions,
                                         compiler_version=lambda command, environ: compiler)


def test_build_key_changes_with_torch_cuda_oeq_and_compiler(tmp_path):
    plan, runtime, driver, cache = make_plan(tmp_path)
    key = runtime_module.build_key(identity_of(plan))
    assert key == runtime_module.build_key(identity_of(plan))  # the same build, the same key
    changed = [
        identity_of(plan, versions=dict(IDENTITY_VERSIONS, torch="2.13.0+cu130")),
        identity_of(plan, versions=dict(IDENTITY_VERSIONS, openequivariance="0.7.1")),
        identity_of(plan, compiler="g++ (Ubuntu 14.2.0) 14.2.0"),
    ]
    (runtime / "include" / "cuda.h").write_text("#define CUDA_VERSION 13010\n")  # another CUDA runtime
    changed.append(identity_of(plan))
    keys = [runtime_module.build_key(identity) for identity in changed]
    assert key not in keys and len(set(keys)) == len(keys)


def test_build_key_ignores_shapes_and_the_process(tmp_path):
    plan, runtime, driver, cache = make_plan(tmp_path)
    identity = identity_of(plan)
    text = repr(identity)
    # not the per-process directory; the pid as a whole token, since a short pid can occur inside a hash
    assert not re.search(rf"(?<![0-9a-fA-F]){os.getpid()}(?![0-9a-fA-F])", text) and "torch-extensions-x" not in text
    assert set(identity) == {"python", "prefix", "torch", "openequivariance", "cuda_runtime", "compiler",
                             "environment"}


def test_a_first_build_is_published_and_seeds_this_process(tmp_path):
    cache, extensions = tmp_path / "cache", tmp_path / "cache" / "torch-extensions-a"
    extensions.mkdir(parents=True)
    info = runtime_module.reuse_extension(cache, extensions, {"k": 1}, fake_build)
    assert info["source"] == "built" and info["key"] == runtime_module.build_key({"k": 1})
    published = Path(info["path"])
    assert published.parent == cache / "extensions"
    assert not (published / "libtorch_tp_jit" / "lock").exists()  # torch's stale lock is not kept
    assert (extensions / "libtorch_tp_jit" / "libtorch_tp_jit.so").is_file()
    assert sorted(p.name for p in (cache / "extensions").iterdir()) == [published.name]  # no staging left


def test_a_second_process_reuses_the_build_without_rebuilding(tmp_path):
    cache = tmp_path / "cache"
    first = cache / "torch-extensions-a"
    first.mkdir(parents=True)
    built = runtime_module.reuse_extension(cache, first, {"k": 1}, fake_build)
    so = Path(built["path"]) / "libtorch_tp_jit" / "libtorch_tp_jit.so"
    code = textwrap.dedent(f"""
        import json, sys
        from pathlib import Path
        from fastmace.kernels import _oeq_runtime as r
        def refuse(directory):
            raise SystemExit("rebuilt")
        target = Path({str(cache / "torch-extensions-b")!r})
        target.mkdir()
        print(json.dumps(r.reuse_extension(Path({str(cache)!r}), target, {{"k": 1}}, refuse)))
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY_ROOT, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads(result.stdout.strip().splitlines()[-1])
    assert info["source"] == "reused" and info["path"] == built["path"]
    copied = cache / "torch-extensions-b" / "libtorch_tp_jit"
    assert (copied / "libtorch_tp_jit.so").read_text() == so.read_text()
    # ninja sees the copied outputs as up to date only if their times are kept
    for name in ("libtorch_tp_jit.so", "libtorch_tp_jit.o", "build.ninja", ".ninja_log"):
        assert (copied / name).stat().st_mtime_ns == (so.parent / name).stat().st_mtime_ns
    assert not (copied / "lock").exists()


def test_another_key_builds_again(tmp_path):
    cache = tmp_path / "cache"
    for name, identity in (("a", {"k": 1}), ("b", {"k": 2})):
        (cache / f"torch-extensions-{name}").mkdir(parents=True)
        assert runtime_module.reuse_extension(cache, cache / f"torch-extensions-{name}", identity,
                                              fake_build)["source"] == "built"
    assert len(list((cache / "extensions").iterdir())) == 2


def test_a_failed_build_publishes_nothing(tmp_path):
    cache = tmp_path / "cache"
    (cache / "torch-extensions-a").mkdir(parents=True)

    def broken(directory):
        (directory / "half").write_text("")
        raise runtime_module.OEqRuntimeError("g++ failed")

    info = runtime_module.reuse_extension(cache, cache / "torch-extensions-a", {"k": 1}, broken)
    assert info["source"] == "in-process" and "g++ failed" in info["error"]
    assert list((cache / "extensions").iterdir()) == []  # neither a result nor a staging directory
    assert list((cache / "torch-extensions-a").iterdir()) == []  # this process builds as before 1b-T24

    def empty(directory):
        pass

    info = runtime_module.reuse_extension(cache, cache / "torch-extensions-a", {"k": 1}, empty)
    assert info["source"] == "in-process" and "no extension library" in info["error"]


def test_child_build_imports_oeq_in_a_child_with_the_staging_directory(tmp_path, monkeypatch):
    # stands in for `import openequivariance`: write what torch's build writes, where it is told to
    monkeypatch.setattr(runtime_module, "CHILD_BUILD", textwrap.dedent("""
        import os, pathlib
        build = pathlib.Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "libtorch_tp_jit"
        build.mkdir()
        (build / "libtorch_tp_jit.so").write_text(os.environ["OEQ_JIT_EXTENSION"])
    """))
    cache = tmp_path / "cache"
    (cache / "torch-extensions-a").mkdir(parents=True)
    build = runtime_module.child_build(dict(os.environ, OEQ_JIT_EXTENSION="1"))
    info = runtime_module.reuse_extension(cache, cache / "torch-extensions-a", {"k": 1}, build)
    assert info["source"] == "built", info
    assert (cache / "torch-extensions-a" / "libtorch_tp_jit" / "libtorch_tp_jit.so").read_text() == "1"
    monkeypatch.setattr(runtime_module, "CHILD_BUILD", "raise SystemExit('no OEq here')")
    info = runtime_module.reuse_extension(cache, cache / "torch-extensions-b", {"k": 2},
                                          runtime_module.child_build(dict(os.environ)))
    assert info["source"] == "in-process" and "no OEq here" in info["error"]


CONCURRENT_BUILD = """
import json, os, sys, time
from pathlib import Path
from fastmace.kernels import _oeq_runtime as r
cache = Path(sys.argv[1])
target = cache / f"torch-extensions-{os.getpid()}"
target.mkdir(parents=True)
start = Path(sys.argv[2])
while not start.exists():  # both processes start building at the same time
    time.sleep(0.01)
def build(directory):
    (directory / "libtorch_tp_jit").mkdir()
    time.sleep(1.0)
    (directory / "libtorch_tp_jit" / "libtorch_tp_jit.so").write_text(str(os.getpid()))
info = r.reuse_extension(cache, target, {"k": 1}, build)
info["mine"] = (target / "libtorch_tp_jit" / "libtorch_tp_jit.so").read_text()
print(json.dumps(info))
"""


def test_two_concurrent_first_builds_both_succeed_with_one_result(tmp_path):
    cache, start = tmp_path / "cache", tmp_path / "start"
    children = [subprocess.Popen([sys.executable, "-c", CONCURRENT_BUILD, str(cache), str(start)],
                                 cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(2)]
    time.sleep(0.5)
    start.write_text("")
    outputs = [child.communicate(timeout=120) for child in children]
    assert [child.returncode for child in children] == [0, 0], outputs
    infos = [json.loads(out.strip().splitlines()[-1]) for out, _ in outputs]
    assert {info["source"] for info in infos} <= {"built", "raced"} and infos[0]["path"] == infos[1]["path"]
    published = Path(infos[0]["path"]) / "libtorch_tp_jit" / "libtorch_tp_jit.so"
    assert infos[0]["mine"] == infos[1]["mine"] == published.read_text()  # both use the one published build
    assert [p.name for p in (cache / "extensions").iterdir()] == [Path(infos[0]["path"]).name]


@pytest.mark.parametrize("module", ["openequivariance", "torch.utils.cpp_extension"])
def test_prepare_names_the_module_imported_too_early(tmp_path, monkeypatch, module):
    fake = type("Module", (), {"CUDA_HOME": "/usr/local/cuda"})()
    monkeypatch.delitem(sys.modules, "openequivariance", raising=False)
    monkeypatch.setitem(sys.modules, module, fake)
    with pytest.raises(runtime_module.OEqRuntimeError, match=module.replace(".", r"\.")):
        runtime_module.prepare_in_process(tmp_path, environ={})
