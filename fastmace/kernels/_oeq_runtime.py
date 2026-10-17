"""OpenEquivariance 0.7 host-extension runtime without the oeq_no_nvcc.sh launcher (SPEC §3.2, §14).

Prototype from plan task 1a-T00; the findings are in docs/design/oeq-runtime.md.

OEq 0.7 with ``OEQ_JIT_EXTENSION=1`` builds its small host C++ extension with
``torch.utils.cpp_extension.load`` when ``openequivariance`` is first imported,
and compiles its device kernels with NVRTC. No ``nvcc`` is needed, but the
build needs a CUDA "home" with the runtime headers and link libraries. The
environment's pip wheels (``nvidia/cu13``) have both, under other names than a
toolkit, so a small directory of symbolic links under the cache root stands in
for the toolkit.

Everything here can be done in the running process, before OpenEquivariance and
``torch.utils.cpp_extension`` are imported, except ``LD_LIBRARY_PATH``: the
dynamic loader reads it once, at process start. Instead of it, the cu13
libraries the extension needs are loaded by absolute path with ``RTLD_GLOBAL``
(``preload_libraries``), so the loader finds them by soname when the extension
is loaded.

Three entry points, from the three candidates of the spike:

- ``prepare_in_process(cache)``: candidate (a), the recommended one.
- ``prepare_by_reexec(cache)``: candidate (b), re-executes the interpreter once
  with ``LD_LIBRARY_PATH`` set.
- ``python -m fastmace.kernels._oeq_runtime CACHE -- COMMAND ...``: candidate
  (c), a launcher like the shell script.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

RUNTIME_LIBRARIES = ("cudart", "nvrtc", "cublasLt", "cublas")  # soname major 13, as oeq_no_nvcc.sh
DRIVER_CANDIDATES = ("/usr/lib/wsl/lib/libcuda.so.1", "/lib/x86_64-linux-gnu/libcuda.so.1",
                     "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1")
SYSTEM_LIBRARY_DIRECTORIES = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64")
SENTINEL = """#!/bin/sh
printf 'nvcc invocation blocked: %s\\n' "$*" >> "${OEQ_NVCC_ATTEMPTS:?}"
printf 'nvcc intentionally unavailable: FastMACE builds the OEq host extension without it\\n' >&2
exit 97
"""
PREPARED_VARIABLE = "FASTMACE_OEQ_PREPARED"


class OEqRuntimeError(RuntimeError):
    """The OEq host-extension runtime cannot be prepared."""


@dataclass(frozen=True)
class RuntimePlan:
    """What preparing the OEq runtime does: links to create, variables to set, libraries to preload."""

    root: Path  # the stand-in CUDA home under the cache
    links: dict = field(default_factory=dict)  # link path -> target
    files: dict = field(default_factory=dict)  # path -> (text, mode)
    environment: dict = field(default_factory=dict)  # variable -> value, for this process
    loader_path: str = ""  # LD_LIBRARY_PATH for a new process (re-exec or launcher only)
    preload: tuple = ()  # absolute library paths, in load order


def find_runtime(prefix=None):
    """The cu13 runtime wheel directory (nvidia/cu13) of the environment, with include/ and lib/."""
    prefix = Path(prefix or sys.prefix)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for site in (prefix / "lib" / version / "site-packages", prefix / "lib64" / version / "site-packages"):
        runtime = site / "nvidia" / "cu13"
        if (runtime / "include" / "cuda.h").is_file():
            return runtime
    raise OEqRuntimeError(f"no CUDA 13 runtime wheel (nvidia/cu13 with include/cuda.h) under {prefix}")


def find_driver(environ, candidates=DRIVER_CANDIDATES):
    """The existing libcuda.so.1: OEQ_CUDA_DRIVER_LIBRARY, else the first candidate that exists."""
    configured = environ.get("OEQ_CUDA_DRIVER_LIBRARY")
    for path in ([configured] if configured else candidates):
        if os.path.isfile(path):
            return Path(os.path.realpath(path))
    raise OEqRuntimeError("no libcuda.so.1 found; set OEQ_CUDA_DRIVER_LIBRARY to the driver library")


def plan(runtime, driver, cache, environ, extensions_dir, tools_dir,
         system_library_directories=SYSTEM_LIBRARY_DIRECTORIES):
    """The runtime plan for the cu13 wheel directory ``runtime``, driver ``driver`` and cache ``cache``.

    Pure: reads only the file system (which libraries exist), changes nothing.
    ``extensions_dir`` is the fresh TORCH_EXTENSIONS_DIR of this process.
    ``tools_dir`` is the environment's bin directory, where torch looks for ``ninja`` on PATH.
    """
    runtime, driver, cache = Path(runtime), Path(driver), Path(cache)
    for header in ("cuda.h", "nvrtc.h"):
        if not (runtime / "include" / header).is_file():
            raise OEqRuntimeError(f"{runtime / 'include' / header} is missing")
    root = cache / "cuda-runtime"
    links = {root / "include": runtime / "include", root / "lib64" / "libcuda.so": driver}
    preload = []
    for name in RUNTIME_LIBRARIES:
        library = runtime / "lib" / f"lib{name}.so.13"
        if not library.is_file():
            raise OEqRuntimeError(f"{library} is missing")
        links[root / "lib64" / f"lib{name}.so"] = library
        preload.append(library)
    sentinel = root / "bin" / "nvcc"
    files = {sentinel: (SENTINEL, 0o755)}
    environment = {
        "CUDA_HOME": str(root), "CUDA_PATH": str(root),
        "CUDACXX": str(sentinel), "CMAKE_CUDA_COMPILER": str(sentinel), "PYTORCH_NVCC": str(sentinel),
        "OEQ_NVCC_ATTEMPTS": str(cache / "nvcc-attempts.log"),
        "PATH": os.pathsep.join([str(root / "bin"), str(tools_dir)]
                               + [p for p in environ.get("PATH", "").split(os.pathsep) if p]),
        "CPATH": os.pathsep.join([str(runtime / "include" / "cccl")]
                                 + [p for p in environ.get("CPATH", "").split(os.pathsep) if p]),
        "OEQ_JIT_EXTENSION": "1",
        "MAX_JOBS": environ.get("MAX_JOBS", "4"),
        # OEq 0.7 waits while a "lock" file exists in the build directory; torch 2.14 leaves it
        # behind. A fresh directory per process avoids the 300-second false wait (and rebuilds).
        "TORCH_EXTENSIONS_DIR": str(extensions_dir),
    }
    # OEq links -lcudart -lnvrtc -lcublas after -L<libpython directory>. Where that directory also
    # holds a system CUDA's development links (leto: CUDA 11.5), the extension would link CUDA 11.
    # A compiler wrapper puts the stand-in lib64 first, as oeq_no_nvcc.sh does.
    for directory in system_library_directories:
        if any(os.path.lexists(os.path.join(directory, f"lib{n}.so")) for n in ("nvrtc", "cudart", "cublas")):
            wrapper = root / "bin" / "c++-cuda13"
            files[wrapper] = (f'#!/bin/sh\nexec g++ -L"{root / "lib64"}" "$@"\n', 0o755)
            environment["CXX"] = str(wrapper)
            break
    loader_path = os.pathsep.join([str(runtime / "lib"), str(driver.parent)]
                                  + [p for p in environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p])
    return RuntimePlan(root=root, links=links, files=files, environment=environment,
                       loader_path=loader_path, preload=tuple(preload))


def materialize(runtime_plan):
    """Create the stand-in CUDA home: links and the nvcc sentinel. Idempotent."""
    for path, (text, mode) in runtime_plan.files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text() != text:
            temporary = path.with_name(f".{path.name}.{os.getpid()}")
            temporary.write_text(text)
            temporary.chmod(mode)
            os.replace(temporary, path)
    for link, target in runtime_plan.links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.path.islink(link) and os.readlink(link) == str(target):
            continue
        temporary = link.with_name(f".{link.name}.{os.getpid()}")
        if os.path.lexists(temporary):
            os.unlink(temporary)
        os.symlink(target, temporary)
        os.replace(temporary, link)


def preload_libraries(runtime_plan):
    """Load the cu13 runtime libraries by absolute path with RTLD_GLOBAL; returns the handles."""
    return [ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL) for path in runtime_plan.preload]


def _check_not_too_late():
    if "openequivariance" in sys.modules:
        raise OEqRuntimeError("openequivariance is already imported; prepare the OEq runtime before it")
    extension = sys.modules.get("torch.utils.cpp_extension")
    if extension is not None:
        raise OEqRuntimeError("torch.utils.cpp_extension is already imported and has fixed its CUDA_HOME "
                              f"({extension.CUDA_HOME}); prepare the OEq runtime before it")


def _plan_for(cache, environ):
    cache = Path(os.path.realpath(os.path.expanduser(cache)))
    cache.mkdir(parents=True, exist_ok=True)
    extensions = Path(tempfile.mkdtemp(prefix="torch-extensions-", dir=cache))
    return plan(find_runtime(), find_driver(environ), cache, environ, extensions, Path(sys.executable).parent)


def prepare_in_process(cache, preload=True, environ=None):
    """Candidate (a): prepare the OEq runtime in this process, before OpenEquivariance is imported."""
    environ = os.environ if environ is None else environ
    _check_not_too_late()
    runtime_plan = _plan_for(cache, environ)
    materialize(runtime_plan)
    environ.update(runtime_plan.environment)
    handles = preload_libraries(runtime_plan) if preload else []
    return runtime_plan, handles


def prepare_by_reexec(cache):
    """Candidate (b): re-execute the interpreter once with the OEq runtime environment and LD_LIBRARY_PATH."""
    cache = os.path.realpath(os.path.expanduser(cache))
    if os.environ.get(PREPARED_VARIABLE) == cache:
        return
    _check_not_too_late()
    runtime_plan = _plan_for(cache, os.environ)
    materialize(runtime_plan)
    environment = dict(os.environ, **runtime_plan.environment)
    environment["LD_LIBRARY_PATH"] = runtime_plan.loader_path
    environment[PREPARED_VARIABLE] = cache
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable] + sys.orig_argv[1:], environment)


def main(argv=None):
    """Candidate (c): ``python -m fastmace.kernels._oeq_runtime CACHE -- COMMAND [ARG ...]``."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or argv[1] != "--":
        print("usage: python -m fastmace.kernels._oeq_runtime CACHE -- COMMAND [ARG ...]", file=sys.stderr)
        return 2
    cache = os.path.realpath(os.path.expanduser(argv[0]))
    runtime_plan = _plan_for(cache, os.environ)
    materialize(runtime_plan)
    environment = dict(os.environ, **runtime_plan.environment)
    environment["LD_LIBRARY_PATH"] = runtime_plan.loader_path
    environment[PREPARED_VARIABLE] = cache
    os.execvpe(argv[2], argv[2:], environment)


if __name__ == "__main__":
    sys.exit(main())
