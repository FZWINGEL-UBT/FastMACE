"""Durable writes: files that survive a crash of the machine, not only of the process (plan task 2-T10).

A flushed file is in the operating system's cache; after a power loss or an NFS client crash it may
be empty or short, and a renamed file may point at nothing. The order that makes a file durable:

1. write a temporary file next to the target, flush it and ``fsync`` it;
2. rename it over the target (``os.replace``, atomic);
3. ``fsync`` the directory, so that the rename itself is on disk.

``atomic_write`` does all three. ``io.output.ChunkedOutput`` fsyncs each closed chunk and its
directory before ``io.restart.write_restart`` writes the restart file this way, so a restart never
counts rows or frames that are not on disk (SPEC §10.6). The directory fsync is POSIX only (Windows
cannot open a directory) and is skipped where the file system refuses it (``EINVAL``, ``ENOTSUP``,
``EOPNOTSUPP``); every other error raises. It is always on: one restart point costs a few fsyncs,
every 10,000 steps by default. This module imports neither torch nor ASE.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Callable
from pathlib import Path
from typing import IO

# A directory fsync that the file system does not support (some FUSE and network mounts).
_UNSUPPORTED = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP})


def fsync_file(path: str | Path) -> None:
    """``fsync`` a closed file by path. Windows needs a descriptor with write access."""
    fd = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_directory(path: str | Path) -> None:
    """``fsync`` a directory, so that the names created, renamed or deleted in it are on disk.

    A no-op on Windows; skipped where the file system does not support it.
    """
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as error:
        if error.errno not in _UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def write_atomically(path: str | Path, write: Callable[[IO[bytes]], None], *,
                     temporary: str | Path | None = None) -> Path:
    """Durable atomic write of ``path``: ``write(handle)`` fills a temporary binary file, which is
    fsynced, renamed over ``path``, and then the directory is fsynced.

    ``temporary`` defaults to ``.<name>.<pid>.tmp`` next to ``path``. If anything fails before the
    rename, the temporary file is removed and ``path`` is left as it was. Returns ``path``.
    """
    path = Path(path)
    temporary = Path(temporary) if temporary is not None else path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    fsync_directory(path.parent)
    return path


def atomic_write(path: str | Path, data: bytes | str) -> Path:
    """Write ``data`` (``str`` as UTF-8, newlines as given) to ``path`` durably and atomically."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return write_atomically(path, lambda handle: handle.write(data))
