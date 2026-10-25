"""``fastmace doctor``: the environment as a checklist (SPEC §7.5, §14).

Each check prints ``[ ok ]``, ``[FAIL]`` (a required check: exit code 1) or
``[warn]`` (an optional one), and every failed item names the command or
document that fixes it. Required: torch, MACE, cuEquivariance with its fused
kernels, the cache root and the file locks. The rest is optional: without a GPU,
OEq, Triton, NVRTC, graph-longrange or torch-dftd a part of FastMACE is
unavailable, and the run report says which.

``doctor`` changes nothing:
- OpenEquivariance and MACE are looked up, never imported: importing either
  builds OEq's host extension. A previously built extension under the cache
  root is reported as built.
- cuEquivariance's ops wheels are checked from their metadata (installed, and
  every shared library they list present), not imported: the import creates a
  Triton cache directory. A library that is present but does not load (CUDA or
  torch ABI mismatch) therefore shows only at the first model load.
- The cache root and the models directory are not created.
- Stale torch-extension build locks are listed with the command that removes
  them, never removed (2-T11).
- The process sets ``CUDA_CACHE_DISABLE=1`` before CUDA starts, so the CUDA
  driver writes no JIT cache. The file-lock check reports the driver-cache
  status of a normal run, recorded at ``import fastmace``.
"""

import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import fastmace
from fastmace import RuntimeInfo, _cache, _file_locks, kernels
from fastmace.errors import CacheLocationError, OEqRuntimeError
from fastmace.model import labels

HELP = "check this installation: GPU, packages, kernels, caches, file locks and model files (changes nothing)"

QUALIFIED_MACE = "0.3.16"  # SPEC §4.1
TORCH_INSTALL = "https://pytorch.org/get-started/locally/"
CUEQ_INSTALL = ("pip install cuequivariance-torch cuequivariance-ops-torch-cu13 "
                "(cuequivariance-ops-torch-cu12 for a CUDA 12 torch)")
OPS_DISTRIBUTIONS = ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-cu13",
                     "cuequivariance-ops-torch-cu12", "cuequivariance-ops-cu12")


def add_arguments(parser):
    parser.add_argument("--no-hash", action="store_true",
                        help="list the model files without computing their SHA-256 (hashing takes seconds per file)")


@dataclass(frozen=True)
class Context:
    environ: object  # a mapping of environment variables
    hash_models: bool = True


@dataclass(frozen=True)
class Item:
    """One checklist line: ``fix`` names the command or document, for a failed item only."""

    name: str
    required: bool
    ok: bool
    detail: str
    fix: str | None = None


# --------------------------------------------------------------------------- probes (mocked in the tests)

def _find_spec(name):
    """Whether a top-level module is installed; looked up, not imported."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _ops_wheels():
    """The installed cuEquivariance ops wheels: {distribution: (version, shared libraries it lists but lacks)}.

    Read from the wheels' metadata, without importing them: importing
    ``cuequivariance_ops_torch`` creates its Triton cache directory.
    """
    wheels = {}
    for name in OPS_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        libraries = [f for f in distribution.files or () if f.name.endswith(".so") or ".so." in f.name]
        wheels[name] = (distribution.version, [str(f) for f in libraries if not Path(f.locate()).is_file()])
    return wheels


def _driver_version():
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        result = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.split()
    return lines[0] if result.returncode == 0 and lines else None


def _cuda():
    try:
        import torch
        available = torch.cuda.is_available()
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else []
        return {"available": available, "devices": devices, "cuda": torch.version.cuda,
                "driver": _driver_version(), "error": None if available else "torch.cuda.is_available() is False"}
    except Exception as error:
        return {"available": False, "devices": [], "cuda": None, "driver": None,
                "error": f"{type(error).__name__}: {error}"}


def _triton():
    return kernels.triton_available()


def _nvrtc():
    return kernels.nvrtc_available()


def _which(name):
    beside = Path(sys.executable).parent / name  # torch finds ninja in the environment's bin directory
    return str(beside) if beside.is_file() else shutil.which(name)


def _oeq_runtime_problems(environ):
    """What OEq's host-toolchain build would miss here (empty: nothing); reads files only."""
    from fastmace.kernels import _oeq_runtime
    problems = []
    for find in (_oeq_runtime.find_runtime, lambda: _oeq_runtime.find_driver(environ)):
        try:
            find()
        except OEqRuntimeError as error:
            problems.append(error.what)
    return problems


def _prebuilt(environ):
    """The prebuilt fastmace-oeq-ext extension, if usable here (6-T04); reads its manifest only."""
    from fastmace.kernels import _oeq_runtime
    return _oeq_runtime.prebuilt_extension(environ)


def _on_nfs(path):
    return _file_locks.on_nfs(path)


def _runtime_info():
    return fastmace.runtime_info()


def _writable(path):
    """Whether ``path``, or its nearest existing parent, can be written to."""
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return os.access(path, os.W_OK)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_files():
    """Released file name -> checkpoint of the label table."""
    return {checkpoint.url.rsplit("/", 1)[-1]: checkpoint for checkpoint in labels.CHECKPOINTS.values()}


# --------------------------------------------------------------------------- checks

class _Result:
    def __init__(self, ok, detail, fix=None):
        self.ok, self.detail, self.fix = ok, detail, fix


def _ok(detail):
    return _Result(True, detail)


def _fail(detail, fix):
    return _Result(False, detail, fix)


def _check(name, required):
    def decorate(function):
        @functools.wraps(function)
        def check(context):
            result = function(context)
            return Item(name=name, required=required, ok=result.ok, detail=result.detail,
                        fix=None if result.ok else result.fix)
        check.title, check.required = name, required
        return check
    return decorate


@_check("GPU and driver", required=False)
def check_gpu(context):
    cuda = _cuda()
    if not cuda["available"]:
        return _fail(f"no CUDA GPU ({cuda['error']}); FastMACE runs MD on NVIDIA GPUs only",
                     "Check `nvidia-smi`; install the NVIDIA driver and a CUDA build of torch "
                     f"({TORCH_INSTALL}).")
    driver = cuda["driver"] or "unknown (nvidia-smi not found)"
    return _ok(f"{', '.join(cuda['devices'])}; driver {driver}; torch CUDA {cuda['cuda']}")


@_check("torch", required=True)
def check_torch(context):
    if not _find_spec("torch"):
        return _fail("torch is not installed", f"pip install torch (a CUDA 13 build: {TORCH_INSTALL})")
    return _ok(f"torch {_version('torch')}")


@_check("MACE", required=True)
def check_mace(context):
    if not _find_spec("mace"):
        return _fail("mace is not installed", f"pip install mace-torch=={QUALIFIED_MACE}")
    version = _version("mace-torch")
    note = "" if version == QUALIFIED_MACE else f" (qualified: {QUALIFIED_MACE}, SPEC §4.1)"
    return _ok(f"mace-torch {version}{note}; looked up, not imported")


@_check("cuEquivariance", required=True)
def check_cueq(context):
    missing = [name for name in ("cuequivariance", "cuequivariance_torch") if not _find_spec(name)]
    if missing:
        return _fail(f"not installed: {', '.join(missing)}", CUEQ_INSTALL)
    versions = (f"cuequivariance {_version('cuequivariance')}, "
                f"cuequivariance-torch {_version('cuequivariance-torch')}")
    wheels = _ops_wheels()
    if not _find_spec("cuequivariance_ops_torch") or not any(name.startswith("cuequivariance-ops-torch")
                                                             for name in wheels):
        return _fail(f"{versions}; the fused-kernel ops package cuequivariance_ops_torch is not installed, so "
                     "MACE would use the naive kernels", CUEQ_INSTALL)
    missing = [f"{name}: {library}" for name, (_, libraries) in wheels.items() for library in libraries]
    if missing:
        return _fail(f"{versions}; the ops wheels lack their fused-kernel libraries ({'; '.join(missing)})",
                     f"Reinstall: {CUEQ_INSTALL}, matching torch's CUDA version")
    ops = ", ".join(f"{name} {version}" for name, (version, _) in wheels.items())
    # Not imported (that writes a Triton cache), so a library that is present but fails to load
    # (a CUDA or torch ABI mismatch) shows only at the first model load, as NaiveCueqError or BackendError.
    return _ok(f"{versions}; fused-kernel libraries present ({ops}); not imported, so a load failure "
               "shows at the first model load")


def _built_extensions(environ):
    try:
        oeq = _cache.cache_root(environ=environ) / "oeq"
    except CacheLocationError:
        return []
    if not oeq.is_dir():
        return []
    return sorted(oeq.glob("torch-extensions-*/**/*.so"), key=lambda path: path.stat().st_mtime, reverse=True)


def _lock_roots(environ):
    """Where torch-extension build locks can lie: torch's default root and FastMACE's OEq cache."""
    from fastmace.kernels import _oeq_runtime
    roots = [_oeq_runtime.torch_extensions_root(environ)]
    try:
        oeq = _cache.cache_root(environ=environ) / "oeq"
        roots += [oeq, oeq / _oeq_runtime.EXTENSIONS]
    except CacheLocationError:
        pass
    return roots


def _old_build_locks(environ):
    """Build locks older than the threshold (2-T11): (stale ones, ones whose holder is unknown). Reads only."""
    from fastmace.kernels import _oeq_runtime
    old = [lock for lock in _oeq_runtime.build_locks(_lock_roots(environ)) if lock.age > lock.threshold]
    return [lock for lock in old if lock.stale], [lock for lock in old if lock.holders is None]


def _age(seconds):
    return f"{seconds / 86400:.1f} days" if seconds >= 86400 else f"{seconds / 3600:.1f} h"


@_check("OpenEquivariance", required=False)
def check_oeq(context):
    if not _find_spec("openequivariance"):
        return _fail("openequivariance is not installed; the hybrid backend is unavailable (plain cuEq runs)",
                     "pip install 'fastmace[oeq]'")
    lines = [f"openequivariance {_version('openequivariance')}; looked up, not imported"]
    prebuilt = _prebuilt(context.environ)
    if prebuilt.usable:
        lines.append(f"host extension: prebuilt (fastmace-oeq-ext {prebuilt.manifest.get('extension_version')}, "
                     f"{prebuilt.library}); no compiler needed")
        problems = []
    else:
        lines.append("host extension: built with the host toolchain (OEQ_JIT_EXTENSION=1, no nvcc); "
                     f"prebuilt not used: {prebuilt.reason}")
        problems = [f"{tool} not found" for tool in ("g++", "ninja") if _which(tool) is None]
    problems += _oeq_runtime_problems(context.environ)
    if problems:
        lines += [f"missing: {problem}" for problem in problems]
        return _fail("\n".join(lines),
                     "pip install 'fastmace[oeq]' (prebuilt fastmace-oeq-ext for torch 2.14.1+cu130, CPython "
                     "3.10-3.12), or see docs/design/oeq-runtime.md (SPEC §14): g++, ninja (pip install ninja), "
                     "the CUDA 13 runtime wheels (nvidia/cu13) and libcuda.so.1 (OEQ_CUDA_DRIVER_LIBRARY).")
    built = _built_extensions(context.environ)
    if prebuilt.usable:
        pass
    elif built:
        lines.append(f"extension built before: {built[0]}")
    else:
        lines.append("extension not built yet: the first hybrid load builds it")
    stale, unknown = _old_build_locks(context.environ)
    lines += [f"holder unknown (no readable /proc): build lock {lock.path}, {_age(lock.age)} old" for lock in unknown]
    if stale:
        lines += [f"stale build lock: {lock.path}, {_age(lock.age)} old, no process holds it; OpenEquivariance "
                  "waits 300 s for it in a process that imports MACE first" for lock in stale]
        return _fail("\n".join(lines), "A killed torch-extension build left it. If no build is running, delete it: "
                                       + "; ".join(f"rm {lock.path}" for lock in stale)
                                       + " (doctor changes nothing, so it does not delete it).")
    return _ok("\n".join(lines))


def _availability(probe, fix):
    availability = probe()
    return _ok(availability.reason) if availability else _fail(availability.reason, fix)


@_check("Triton", required=False)
def check_triton(context):
    return _availability(_triton, "pip install triton (the Linux CUDA torch wheels include it)")


@_check("NVRTC", required=False)
def check_nvrtc(context):
    return _availability(_nvrtc, "pip install cuda-bindings matching torch's CUDA version "
                                 "(NVRTC compiles the streaming kernels)")


def _optional_package(module, distribution, fix, needed_for):
    if not _find_spec(module):
        return _fail(f"{distribution} is not installed; needed for {needed_for}", fix)
    return _ok(f"{distribution} {_version(distribution)}")


@_check("graph-longrange", required=False)
def check_graph_longrange(context):
    return _optional_package("graph_longrange", "graph-longrange",
                             "pip install 'fastmace[polar]' (graph-longrange 0.4.0)", "MACE-POLAR-1")


@_check("torch-dftd", required=False)
def check_torch_dftd(context):
    return _optional_package("torch_dftd", "torch-dftd", "pip install 'fastmace[d3]'", "the D3 reference")


@_check("cache root", required=True)
def check_cache_root(context):
    try:
        root = _cache.cache_root(environ=context.environ)
    except CacheLocationError as error:
        return _fail(error.what, error.next_step)
    source = _cache.ENVIRONMENT_VARIABLE if context.environ.get(_cache.ENVIRONMENT_VARIABLE) else \
        "the user cache directory"
    state = "exists" if root.is_dir() else "not created yet"
    if not _writable(root):
        return _fail(f"{root} ({source}) is not writable",
                     f"Set {_cache.ENVIRONMENT_VARIABLE} to a writable directory outside the Python environment "
                     "(SPEC §14).")
    return _ok(f"{root} ({source}; {state})")


def _yes(value):
    return "yes" if value else "no"


@_check("file locks", required=True)
def check_file_locks(context):
    info = _runtime_info()
    try:
        root_on_nfs = _on_nfs(_cache.cache_root(environ=context.environ))
    except CacheLocationError:
        root_on_nfs = False
    driver_cache = _file_locks.cuda_driver_cache_path(context.environ)
    driver_on_nfs = _on_nfs(driver_cache)
    detail = (f"PyTorch file locks: {info.file_locks} ({info.file_lock_source}); cache root on NFS: "
              f"{_yes(root_on_nfs)}; CUDA driver cache {driver_cache} on NFS: {_yes(driver_on_nfs)} "
              f"({info.cuda_driver_cache})")
    if info.file_locks != "soft" and root_on_nfs:
        return _fail(detail, "Use soft file locks on NFS: `import fastmace` before torch, and do not set "
                             f"{_file_locks.ENVIRONMENT_VARIABLE}=0 (SPEC §14).")
    disabled = info.cuda_driver_cache.startswith("disabled") or \
        info.cuda_driver_cache == "CUDA_CACHE_DISABLE=1 set by the user"
    if driver_on_nfs and not disabled:
        return _fail(detail, "Unset CUDA_CACHE_DISABLE (fastmace then disables the driver cache on NFS), or set "
                             "CUDA_CACHE_DISABLE=1, or CUDA_CACHE_PATH to a local directory (SPEC §14).")
    return _ok(detail)


def _model_directories(environ):
    directories = []
    configured = environ.get("FASTMACE_MODELS")
    if configured:
        directories.append(Path(os.path.expanduser(configured)))
    try:
        directories.append(_cache.cache_dir("models", create=False, environ=environ))
    except CacheLocationError:
        pass
    return directories


@_check("model files", required=False)
def check_models(context):
    directories = _model_directories(context.environ)
    expected = _expected_files()
    names = {}
    for label, entry in labels.LABELS.items():
        names.setdefault(entry.checkpoint.name, []).append(label)
    lines, bad = [], []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            checkpoint = expected.get(path.name)
            if checkpoint is None:
                lines.append(f"{path}: not in the label table (fm.load needs sha256= or trusted=True)")
                continue
            which = ", ".join(names.get(checkpoint.name, [checkpoint.name]))
            if not context.hash_models:
                lines.append(f"{path}: {which} (not hashed: --no-hash)")
            elif _sha256(path) == checkpoint.sha256:
                lines.append(f"{path}: {which}, SHA-256 matches")
            else:
                lines.append(f"{path}: {which}, SHA-256 does not match the label table")
                bad.append(path)
    if not lines:
        places = ", ".join(str(d) for d in directories) or "no directory"
        return _ok(f"no model files in {places}; fm.load(label, download=True) downloads into "
                   "<cache root>/models")
    if bad:
        return _fail("\n".join(lines), "Delete the file and fetch it again with fm.load(label, download=True), "
                                       "which checks the SHA-256 before unpickling.")
    return _ok("\n".join(lines))


# The SPEC §7.5 order.
CHECKS = (check_gpu, check_torch, check_mace, check_cueq, check_oeq, check_triton, check_nvrtc,
          check_graph_longrange, check_torch_dftd, check_cache_root, check_file_locks, check_models)

INDENT = " " * 7


def _item(check, context):
    try:
        return check(context)
    except Exception as error:  # a probe crashed: report it, keep checking
        return Item(name=check.title, required=check.required, ok=False,
                    detail=f"the check itself failed ({type(error).__name__}: {error})",
                    fix="Run `fastmace --debug doctor` and report the traceback.")


def run_checks(context):
    """Print the checklist; returns 1 if a required check failed, else 0."""
    print(f"fastmace doctor: FastMACE {fastmace.__version__}, Python {platform.python_version()}, "
          f"{platform.system()} {platform.machine()}")
    failed = warnings = 0
    for check in CHECKS:
        item = _item(check, context)
        mark = "[ ok ]" if item.ok else "[FAIL]" if item.required else "[warn]"
        failed += not item.ok and item.required
        warnings += not item.ok and not item.required
        first, *rest = item.detail.splitlines() or [""]
        print(f"{mark} {item.name}: {first}")
        for line in rest:
            print(f"{INDENT}{line}")
        if item.fix:
            print(f"{INDENT}fix: {item.fix}")
    plural = "s" if warnings != 1 else ""
    if failed:
        print(f"{failed} required check{'s' if failed != 1 else ''} failed ({warnings} warning{plural}).")
        return 1
    print(f"all required checks passed ({warnings} warning{plural}).")
    return 0


def run(args):
    # Before CUDA starts: the driver then writes no JIT cache, so doctor changes nothing.
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    return run_checks(Context(environ=os.environ, hash_models=not args.no_hash))


__all__ = ["CHECKS", "Context", "Item", "RuntimeInfo", "run", "run_checks"]
