"""OpenEquivariance 0.7 host-extension runtime, prepared in-process (SPEC §3.2, §14).

From plan task 1a-T00 (findings in docs/design/oeq-runtime.md), promoted in
1a-T05; it replaces ResearchModule's ``oeq_no_nvcc.sh`` launcher.

OEq 0.7 with ``OEQ_JIT_EXTENSION=1`` builds its small host C++ extension with
``torch.utils.cpp_extension.load`` when ``openequivariance`` is first imported,
and compiles its device kernels with NVRTC. No ``nvcc`` is needed, but the
build needs a CUDA "home" with the runtime headers and link libraries. The
environment's pip wheels (``nvidia/cu13``) have both, under other names than a
toolkit, so a small directory of symbolic links under the cache root stands in
for the toolkit.

Everything is done in the running process, before OpenEquivariance and
``torch.utils.cpp_extension`` are imported (``import mace.calculators`` imports
both). ``LD_LIBRARY_PATH`` is not needed: torch's RPATH already maps the cu13
libraries. As a guard against a torch build that loads them lazily, they are
also loaded by absolute path with ``RTLD_GLOBAL`` (``preload_libraries``), so the
loader finds them by soname when the extension is loaded.

The entry point is ``prepare_in_process(cache)``; model loading calls it once per
process through ``fastmace._patches.prepare_oeq_runtime``, before its first MACE
import.

The host extension is built once per build identity (plan task 1b-T24): Python
and environment, torch, OpenEquivariance, the CUDA runtime wheel, the compiler
and the build variables (``build_identity``), never shapes or models. The first
process with a new identity builds it in a child interpreter into a staging
directory under ``<cache>/extensions`` and renames that into place. Later
processes copy the finished build into their own fresh ``TORCH_EXTENSIONS_DIR``,
where ninja finds it up to date. Renaming needs no lock, so the cache may lie
on NFS (SPEC §14), and two processes that build at once both use the build that
was renamed first.

Stale build locks (plan task 2-T11). OEq 0.7 waits up to 300 s, silently, while
``<build directory>/lock`` exists; torch's ``FileBaton`` creates that file with
``O_CREAT | O_EXCL`` and holds it open until the build ends, so a killed build
leaves it behind for good. ``stale_build_locks`` finds such locks without
writing anything: older than ``STALE_LOCK_AGE`` and open in no process
(``/proc/<pid>/fd``; without ``/proc`` the holder is unknown and the lock is not
called stale). ``fastmace doctor`` lists them. The preparation waits at most
``lock_wait`` seconds for a lock in this process's extension directory, then
gives up with "OEq unavailable: stale build lock", and the planner falls to
plain cuEq with that reason.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from fastmace.errors import OEqRuntimeError  # noqa: F401 (re-exported: the 1a names)

RUNTIME_LIBRARIES = ("cudart", "nvrtc", "cublasLt", "cublas")  # soname major 13, as oeq_no_nvcc.sh
DRIVER_CANDIDATES = ("/usr/lib/wsl/lib/libcuda.so.1", "/lib/x86_64-linux-gnu/libcuda.so.1",
                     "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1")
SYSTEM_LIBRARY_DIRECTORIES = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64")
SENTINEL = """#!/bin/sh
printf 'nvcc invocation blocked: %s\\n' "$*" >> "${OEQ_NVCC_ATTEMPTS:?}"
printf 'nvcc intentionally unavailable: FastMACE builds the OEq host extension without it\\n' >&2
exit 97
"""
EXTENSIONS = "extensions"  # <cache>/extensions/<key>: one finished TORCH_EXTENSIONS_DIR per build identity
LIBRARY_SUFFIX = ".so"
NOT_COPIED = ("lock",)  # torch's build lock, which OEq 0.7 waits on when torch leaves it behind
BUILD_VARIABLES = ("CUDA_HOME", "CPATH", "CXX", "OEQ_JIT_EXTENSION")  # change what the compiler produces
CHILD_BUILD = "import openequivariance"  # OEq builds its host extension at import
CHILD_BUILD_TIMEOUT = 900
BUILD_LOCK = "lock"  # torch.utils.file_baton.FileBaton's file in <build directory>, OEq 0.7 polls for it
LOCK_PATTERNS = (f"*/{BUILD_LOCK}", f"*/*/{BUILD_LOCK}")  # <root>/<name>/lock and <root>/<py>_<cu>/<name>/lock
STALE_LOCK_AGE = 3600.0  # seconds; a host build takes minutes (CHILD_BUILD_TIMEOUT)
LOCK_WAIT = 30.0  # seconds the preparation waits for a lock in its own extension directory
LOCK_POLL = 1.0
PROC = Path("/proc")


@dataclass(frozen=True)
class RuntimePlan:
    """What preparing the OEq runtime does: links to create, variables to set, libraries to preload."""

    root: Path  # the stand-in CUDA home under the cache
    links: dict = field(default_factory=dict)  # link path -> target
    files: dict = field(default_factory=dict)  # path -> (text, mode)
    environment: dict = field(default_factory=dict)  # variable -> value, for this process
    preload: tuple = ()  # absolute library paths, in load order
    extension: dict = field(default_factory=dict)  # the persistent build: key, path, source, seconds, error


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
        # behind. A fresh directory per process avoids the 300-second false wait; reuse_extension
        # seeds it with the persistent build, so it does not rebuild.
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
    return RuntimePlan(root=root, links=links, files=files, environment=environment,
                       preload=tuple(preload))


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


def _distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _compiler_version(command, environ):
    """The resolved path and first line of ``<command> --version``, looked up on the build's PATH."""
    executable = shutil.which(command, path=environ.get("PATH"))
    if executable is None:
        return f"{command}: not found"
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=60,
                                env=dict(environ), check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return f"{executable}: {error}"
    lines = (result.stdout or result.stderr).strip().splitlines()
    return f"{os.path.realpath(executable)}: {lines[0] if lines else result.returncode}"


def build_identity(runtime_plan, environ, versions=None, compiler_version=_compiler_version):
    """What the OEq host extension depends on, the key of its persistent build: no shapes, no process.

    ``environ`` is the environment before or after ``runtime_plan.environment`` is applied; the plan's
    values win. ``versions`` overrides the installed versions of ``torch`` and ``openequivariance``.
    """
    versions = versions or {name: _distribution_version(name) for name in ("torch", "openequivariance")}
    build_environ = dict(environ, **runtime_plan.environment)
    include = runtime_plan.links[runtime_plan.root / "include"]
    cuda_runtime = {header: hashlib.sha256((include / header).read_bytes()).hexdigest()
                    for header in ("cuda.h", "nvrtc.h")}
    for library in runtime_plan.preload:
        cuda_runtime[library.name] = [os.path.realpath(library), os.path.getsize(library)]
    environment = {name: build_environ.get(name) for name in BUILD_VARIABLES}
    compiler = environment["CXX"] or "c++"  # what torch.utils.cpp_extension calls
    return {"python": [sys.implementation.cache_tag, sys.version.split()[0]],
            "prefix": os.path.realpath(sys.prefix),  # the OEq sources the build refers to
            "torch": versions.get("torch"), "openequivariance": versions.get("openequivariance"),
            "cuda_runtime": cuda_runtime, "compiler": compiler_version(compiler, build_environ),
            "environment": environment}


def build_key(identity):
    """A short stable name for a build identity."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def _has_library(directory):
    return Path(directory).is_dir() and any(Path(directory).rglob(f"*{LIBRARY_SUFFIX}"))


def reuse_extension(cache, extensions_dir, identity, build):
    """Seed ``extensions_dir`` with the persistent build for ``identity``, building and publishing it first.

    ``build(directory)`` builds the extension into the empty ``directory``. The result goes to
    ``<cache>/extensions/<key>`` by renaming a staging directory, so another process sees it whole or
    not at all; a process that loses the race discards its own build and uses the winner's. If the
    build fails, nothing is published and ``extensions_dir`` stays empty: OEq then builds in this
    process, as before 1b-T24. Returns {"key", "path", "source", "seconds", "error"}, with ``source``
    "reused", "built", "raced" (another process published first) or "in-process".
    """
    start = time.perf_counter()
    key = build_key(identity)
    published = Path(cache) / EXTENSIONS / key
    info = {"key": key, "path": str(published), "source": "reused", "seconds": 0.0, "error": None}
    if not _has_library(published):
        published.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=published.parent))
        try:
            build(staging)
            if not _has_library(staging):
                raise OEqRuntimeError(f"the build left no extension library (*{LIBRARY_SUFFIX}) in {staging}")
            for name in NOT_COPIED:
                for path in list(staging.rglob(name)):
                    path.unlink()
            (staging / "identity.json").write_text(json.dumps(identity, indent=1, sort_keys=True) + "\n")
            try:
                os.rename(staging, published)
                info["source"] = "built"
            except OSError:
                if not _has_library(published):
                    raise
                info["source"] = "raced"  # another process renamed its build into place first
        except Exception as error:  # noqa: BLE001  (any failure falls back to the in-process build)
            info.update(source="in-process", error=f"{type(error).__name__}: {error}",
                        seconds=round(time.perf_counter() - start, 3))
            return info
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    # copy2 keeps the file times, so ninja finds the outputs up to date and does not rebuild
    shutil.copytree(published, extensions_dir, dirs_exist_ok=True, copy_function=shutil.copy2,
                    ignore=shutil.ignore_patterns("identity.json", *NOT_COPIED))
    info["seconds"] = round(time.perf_counter() - start, 3)
    return info


def child_build(environ, python=None, timeout=CHILD_BUILD_TIMEOUT):
    """A ``build(directory)`` that imports OpenEquivariance in a child interpreter with ``environ``."""
    def build(directory):
        env = dict(environ, TORCH_EXTENSIONS_DIR=str(directory), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([python or sys.executable, "-c", CHILD_BUILD], cwd=directory, env=env,
                                capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode != 0:
            raise OEqRuntimeError(f"building the OEq host extension failed (exit {result.returncode}): "
                                  + (result.stderr or result.stdout)[-2000:])
    return build


@dataclass(frozen=True)
class BuildLock:
    """A torch-extension build lock: its age in seconds and the processes that hold it open."""

    path: Path
    age: float
    holders: tuple | None  # pids; None: unknown (no readable /proc)
    threshold: float = STALE_LOCK_AGE

    @property
    def stale(self):
        """Old, and no process holds it: a killed build left it."""
        return self.holders == () and self.age > self.threshold


def torch_extensions_root(environ):
    """Where torch builds extensions by default: TORCH_EXTENSIONS_DIR, else <user cache>/torch_extensions."""
    configured = environ.get("TORCH_EXTENSIONS_DIR")
    if configured:
        return Path(os.path.expanduser(configured))
    home = environ.get("HOME") or os.path.expanduser("~")
    return Path(environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")) / "torch_extensions"


def lock_holders(path, proc=PROC):
    """The pids that hold ``path`` open, from ``/proc/<pid>/fd``; None when ``/proc`` cannot be read.

    Only this user's processes on this host are visible: a build on another NFS client is not, which
    the age threshold covers.
    """
    target = os.path.realpath(path)
    try:
        entries = list(Path(proc).iterdir())
    except OSError:
        return None
    holders = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except OSError:  # another user's process, or it has just ended
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(descriptor) == target:
                    holders.append(int(entry.name))
                    break
            except OSError:
                continue
    return sorted(holders)


def _lock_files(roots):
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for pattern in LOCK_PATTERNS:
            found += [path for path in root.glob(pattern) if path.is_file() and path not in found]
    return found


def build_locks(roots, threshold=STALE_LOCK_AGE, now=None, holders=None):
    """Every build lock under ``roots`` (``LOCK_PATTERNS``), with its age and holders. Reads only."""
    now = time.time() if now is None else now
    holders = lock_holders if holders is None else holders
    locks = []
    for path in _lock_files(roots):
        try:
            age = now - path.stat().st_mtime
        except OSError:  # removed meanwhile
            continue
        pids = holders(path)
        locks.append(BuildLock(path=path, age=age, holders=None if pids is None else tuple(pids),
                               threshold=threshold))
    return locks


def stale_build_locks(roots, threshold=STALE_LOCK_AGE, now=None, holders=None):
    """The build locks under ``roots`` older than ``threshold`` seconds that no process holds."""
    return [lock for lock in build_locks(roots, threshold, now, holders) if lock.stale]


def wait_for_build_locks(directory, wait=LOCK_WAIT, poll=LOCK_POLL, clock=time.monotonic, sleep=time.sleep):
    """Wait at most ``wait`` seconds for the build locks under ``directory`` to go; returns those left."""
    locks = _lock_files([directory])
    deadline = clock() + wait
    while locks and clock() < deadline:
        sleep(min(poll, max(deadline - clock(), 0.0)))
        locks = [path for path in locks if os.path.lexists(path)]
    return locks


def _remove_hint(locks):
    return "If no build is running, delete " + ", ".join(str(path) for path in locks)


def _check_build_locks(directory, wait):
    """Before OEq is imported: OEq 0.7 would wait 300 s silently for a lock in ``directory``."""
    locks = _lock_files([directory])
    if not locks:
        return
    warnings.warn(f"torch-extension build lock {', '.join(map(str, locks))}: OpenEquivariance would wait 300 s "
                  f"for it; waiting at most {wait:g} s. {_remove_hint(locks)}.", UserWarning, stacklevel=3)
    left = wait_for_build_locks(directory, wait=wait)
    if left:
        raise OEqRuntimeError(f"OEq unavailable: stale build lock {', '.join(map(str, left))}",
                              why=f"it was still there after {wait:g} s; a killed torch-extension build leaves it",
                              next_step=f"{_remove_hint(left)} and run again (`fastmace doctor` lists build locks).")


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


def prepare_in_process(cache, preload=True, environ=None, build=None, *, lock_wait=LOCK_WAIT):
    """Prepare the OEq runtime in this process, before OpenEquivariance is imported; returns (plan, handles).

    The fresh TORCH_EXTENSIONS_DIR of this process is seeded with the persistent build of this build
    identity (``reuse_extension``); ``build`` replaces the child-interpreter build (``child_build``).
    The plan's ``extension`` records which. A build lock left in that directory is waited for at most
    ``lock_wait`` seconds, with a warning, then ``OEqRuntimeError`` ("OEq unavailable: stale build
    lock"). The handles of the preloaded libraries must stay alive while the process uses OEq.
    """
    environ = os.environ if environ is None else environ
    _check_not_too_late()
    runtime_plan = _plan_for(cache, environ)
    materialize(runtime_plan)
    environ.update(runtime_plan.environment)
    extensions = Path(runtime_plan.environment["TORCH_EXTENSIONS_DIR"])
    extension = reuse_extension(extensions.parent, extensions, build_identity(runtime_plan, environ),
                                build or child_build(dict(environ)))
    _check_build_locks(extensions, lock_wait)
    runtime_plan = dataclasses.replace(runtime_plan, extension=extension)
    handles = preload_libraries(runtime_plan) if preload else []
    return runtime_plan, handles
