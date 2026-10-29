"""The cache root (SPEC §14 "Caches"): Triton, Inductor, OEq, compile and model caches, never inside an environment.

The root is ``FASTMACE_CACHE``, else the user cache directory
(``$XDG_CACHE_HOME/fastmace`` or ``~/.cache/fastmace`` on Linux). It must not
lie inside ``sys.prefix``: environments may be read-only or shared, and a
cache there is lost or corrupted when the environment is rebuilt.

Importing fastmace sets nothing and creates nothing here. An engine asks for a
directory with ``cache_dir(kind)`` when it needs one.
"""

import os
import sys
from pathlib import Path

from fastmace.errors import CacheLocationError  # noqa: F401 (re-exported: the 1a names)

ENVIRONMENT_VARIABLE = "FASTMACE_CACHE"
# "models": downloaded checkpoints, each verified against its SHA-256 before it is kept (SPEC §7.6).
# "radial-tables": fitted and certified radial_tables, keyed by everything they depend on (rewrites.radial.cache_key).
CACHE_KINDS = ("triton", "inductor", "oeq", "compile", "models", "radial-tables")


def _user_cache_directory(environ):
    if sys.platform == "win32":
        base = environ.get("LOCALAPPDATA") or os.path.join(environ.get("HOME") or os.path.expanduser("~"),
                                                           "AppData", "Local")
        return Path(base) / "fastmace" / "Cache"
    home = environ.get("HOME") or os.path.expanduser("~")
    if sys.platform == "darwin":
        return Path(home) / "Library" / "Caches" / "fastmace"
    return Path(environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")) / "fastmace"


def _inside(path, directory):
    directory = Path(os.path.realpath(directory))
    return path == directory or directory in path.parents


def cache_root(environ=None):
    """The cache root as an absolute path; raises CacheLocationError if it lies inside sys.prefix."""
    environ = os.environ if environ is None else environ
    configured = environ.get(ENVIRONMENT_VARIABLE)
    root = Path(os.path.expanduser(configured)) if configured else _user_cache_directory(environ)
    root = Path(os.path.realpath(root))
    if _inside(root, sys.prefix):
        source = ENVIRONMENT_VARIABLE if configured else "the user cache directory"
        raise CacheLocationError(
            f"cache root {root} ({source}) lies inside the Python environment (sys.prefix {sys.prefix}); "
            f"set {ENVIRONMENT_VARIABLE} to a directory outside it"
        )
    return root


def cache_dir(kind, create=True, environ=None):
    """The directory for one kind of cache (CACHE_KINDS) under the cache root, created unless create=False."""
    if kind not in CACHE_KINDS:
        raise ValueError(f"unknown cache kind {kind!r}; expected one of {', '.join(CACHE_KINDS)}")
    path = cache_root(environ) / kind
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
