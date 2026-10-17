"""No kernel file locks on network filesystems (SPEC §14 "Caches"), applied at ``import fastmace``.

PyTorch takes its extension-build and Inductor-cache locks through
``filelock.FileLock``, which on Linux is ``UnixFileLock`` (``fcntl.flock``).
On an NFS home every such lock is a LOCK request to the server; on 10 October
(coraline) and 16 October 2026 (leto) the NAS stopped accepting a client's
lock requests altogether until the mount was reset. ``SoftFileLock`` locks by
creating the lock file with ``O_CREAT | O_EXCL`` (atomic on NFS 4, no lock
request) and removes it on release; filelock 3.32 writes the holder's PID and
host into the file and removes the file of a dead process on the same host.

The switch must happen before torch is imported: ``torch.utils._filelock``
and Inductor bind the class at their own import. ``FASTMACE_SOFT_FILE_LOCKS``:

* unset: soft locks when possible. A process that imported torch first keeps
  kernel locks, and ``runtime_info()`` reports that the switch came late.
* ``1``: soft locks; importing fastmace after torch is an error.
* ``0``: kernel locks, and the CUDA driver cache is left alone. If the
  start-up hook below already made the locks soft and torch is not imported
  yet, the kernel-lock class is restored; after torch, the soft locks stay and
  ``runtime_info()`` says so.

The CUDA driver locks its JIT cache's ``index`` file with POSIX locks, 220 to
440 times per run. When that cache lies on NFS and the user has not set
``CUDA_CACHE_DISABLE``, it is switched off (no measurable start-up cost on
leto). NFS is read from the mounts table, never by calling ``stat``, which can
hang on a broken mount.

Our research environments may carry the ``nfs-soft-locks`` start-up hook,
which does the same for every interpreter. Where it already made the locks
soft, this module changes nothing unless ``FASTMACE_SOFT_FILE_LOCKS=0``. The
hook reads its own variables, not ours: ``LANGEVIN_SOFT_FILE_LOCKS=0`` (kernel
locks, also for the research engine) or ``NFS_SOFT_LOCKS_HOOK=0`` switch it
off at interpreter start, which also covers code that imports torch before
fastmace.
"""

import importlib.util
import os
import sys

ENVIRONMENT_VARIABLE = "FASTMACE_SOFT_FILE_LOCKS"
MOUNTS_FILE = "/proc/self/mounts"
HOOK_MODULE = "nfs_soft_locks"


def requested(environ=None):
    """True (=1), False (=0) or None (unset: soft locks when torch is not imported yet)."""
    value = (os.environ if environ is None else environ).get(ENVIRONMENT_VARIABLE, "")
    if value not in ("", "0", "1"):
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be 0, 1 or unset, not {value!r}")
    return None if value == "" else value == "1"


def mode():
    """"soft" when PyTorch's locks are SoftFileLock in this process, else "kernel" (for provenance)."""
    import filelock

    base = filelock.FileLock
    torch_filelock = sys.modules.get("torch.utils._filelock")
    if torch_filelock is not None:
        base = torch_filelock.FileLock
    return "soft" if issubclass(base, filelock.SoftFileLock) else "kernel"


def _hook_active():
    """True if the nfs-soft-locks start-up hook ran in this process and made filelock soft."""
    hook = sys.modules.get(HOOK_MODULE)
    state = getattr(hook, "STATE", None)
    return isinstance(state, dict) and state.get("filelock") == "soft"


def use_soft_file_locks():
    """Make filelock.FileLock (and so PyTorch's locks) SoftFileLock; torch must not be imported yet."""
    import filelock

    if filelock.FileLock is not filelock.SoftFileLock:
        if "torch" in sys.modules:
            raise RuntimeError(
                f"soft file locks ({ENVIRONMENT_VARIABLE}=1) must be set before torch is imported: "
                "import fastmace first"
            )
        filelock.FileLock = filelock.SoftFileLock
        filelock.AsyncFileLock = filelock.AsyncSoftFileLock
    return mode()


def restore_kernel_file_locks():
    """Make filelock.FileLock the platform's kernel-lock class again (undoes a start-up hook's switch).

    Only meaningful before torch is imported; a platform without kernel locks keeps SoftFileLock.
    """
    import filelock

    if sys.platform == "win32":
        names = ("WindowsFileLock", "AsyncWindowsFileLock")
    else:
        if importlib.util.find_spec("fcntl") is None:  # filelock's default there is SoftFileLock
            return mode()
        names = ("UnixFileLock", "AsyncUnixFileLock")
    sync, asynchronous = (getattr(filelock, name, None) for name in names)
    if sync is not None:
        filelock.FileLock = sync
    if asynchronous is not None:
        filelock.AsyncFileLock = asynchronous
    return mode()


def configure_file_locks(environ=None):
    """Apply FASTMACE_SOFT_FILE_LOCKS for this process; returns (source, late).

    source says how the mode came about; late is True when torch was imported
    first and the locks therefore stayed kernel locks.
    """
    import filelock

    wanted = requested(environ)
    if wanted is False:
        if filelock.FileLock is filelock.SoftFileLock and "torch" not in sys.modules:
            restore_kernel_file_locks()
        if mode() == "soft":
            # The start-up hook switched before torch was imported, and torch bound the soft class.
            return f"opted out ({ENVIRONMENT_VARIABLE}=0), but soft since torch was imported", False
        return f"opted out ({ENVIRONMENT_VARIABLE}=0)", False
    if filelock.FileLock is filelock.SoftFileLock and mode() == "soft":
        if _hook_active():
            return f"start-up hook ({HOOK_MODULE})", False
        return "already soft", False
    if wanted is None and "torch" in sys.modules:
        return "kernel: torch was imported before fastmace", True
    use_soft_file_locks()
    return "switched at import", False


def on_nfs(path, mounts_file=MOUNTS_FILE):
    """True if path lies on an NFS mount (the longest matching mount point of mounts_file)."""
    try:
        path = os.path.realpath(path)
        best, best_type = "", ""
        with open(mounts_file) as mounts:
            for line in mounts:
                fields = line.split()
                if len(fields) < 3:
                    continue
                mount_point = fields[1].replace("\\040", " ")
                inside = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
                if inside and len(mount_point) > len(best):
                    best, best_type = mount_point, fields[2]
        return best_type.startswith("nfs")
    except OSError:
        return False


def cuda_driver_cache_path(environ=None):
    """The CUDA driver's JIT cache directory: CUDA_CACHE_PATH, else the driver's default ~/.nv/ComputeCache."""
    environ = os.environ if environ is None else environ
    path = environ.get("CUDA_CACHE_PATH")
    if path:
        return path
    home = environ.get("HOME") or os.path.expanduser("~")
    return os.path.join(home, ".nv", "ComputeCache")


def configure_cuda_driver_cache(environ=None, mounts_file=MOUNTS_FILE):
    """Set CUDA_CACHE_DISABLE=1 in environ when the driver cache is on NFS and the user did not set it.

    Returns a short status for provenance.
    """
    environ = os.environ if environ is None else environ
    if "CUDA_CACHE_DISABLE" in environ:
        return f"CUDA_CACHE_DISABLE={environ['CUDA_CACHE_DISABLE']} set by the user"
    path = cuda_driver_cache_path(environ)
    if on_nfs(path, mounts_file):
        environ["CUDA_CACHE_DISABLE"] = "1"
        return f"disabled ({path} on NFS)"
    return "enabled"


def configure(environ=None):
    """Both rules for this process (fastmace/__init__.py); returns (source, late, cuda status)."""
    environ = os.environ if environ is None else environ
    source, late = configure_file_locks(environ)
    if requested(environ) is False:
        return source, late, f"untouched ({ENVIRONMENT_VARIABLE}=0)"
    return source, late, configure_cuda_driver_cache(environ)
