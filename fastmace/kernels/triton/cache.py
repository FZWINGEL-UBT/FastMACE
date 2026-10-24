"""Where Triton and torch.compile (Inductor) keep their caches (SPEC §14 "Caches").

By default both live under the FastMACE cache root (``fastmace._cache``,
``FASTMACE_CACHE``), so later runs with the same model code, tensor shapes,
torch version and GPU reuse the generated kernels and the compiled force
derivative instead of building them again (about 1.5-2 min per model and
precision on an A4000 when cold). Dynamo still traces the model in every
process, so a warm start is faster, not free. A run may pass its own
directory instead, for example to compile from scratch in its output
directory.

The caches only store generated code; their keys cover everything that
changes that code, so results are the same with a cold or a warm cache. A
cache directory must never lie inside a Python environment.

Nothing here runs at import time. The planner calls ``configure_defaults``
before a plan's first compilation; it fills in the variables the environment
does not set, so an explicitly set ``TORCHINDUCTOR_CACHE_DIR`` or
``TRITON_CACHE_DIR`` wins. ``configure`` sets them all, for a run's own
directory or the test suite's shared root.
"""

import os
import sys
from pathlib import Path

from fastmace._cache import CacheLocationError, cache_dir

VARIABLES = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE",
             "ENABLE_AOT_AUTOGRAD_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE")
DIRECTORIES = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
FLAGS = dict(TORCHINDUCTOR_FX_GRAPH_CACHE="1",  # Inductor graphs (default on)
             ENABLE_AOT_AUTOGRAD_CACHE="1",     # compiled force derivative, torch 2.5
             TORCHINDUCTOR_AUTOGRAD_CACHE="1")  # the same, newer torch


def cache_settings(directory=None, environ=None):
    """Environment variables for the compiler caches.

    directory: None puts the Inductor and Triton caches under the cache root
    (``cache_dir("inductor")``, ``cache_dir("triton")``); a directory puts them
    in its ``inductor`` and ``triton`` subdirectories. Raises CacheLocationError
    if the caches would lie inside sys.prefix.
    """
    if directory is None:
        inductor = cache_dir("inductor", create=False, environ=environ)
        triton = cache_dir("triton", create=False, environ=environ)
    else:
        base = Path(os.path.realpath(os.path.expanduser(directory)))
        prefix = Path(os.path.realpath(sys.prefix))
        if base == prefix or prefix in base.parents:
            raise CacheLocationError(f"compiler cache {base} would lie inside the Python environment "
                                     f"(sys.prefix {sys.prefix})")
        inductor, triton = base / "inductor", base / "triton"
    return dict(TORCHINDUCTOR_CACHE_DIR=str(inductor), TRITON_CACHE_DIR=str(triton), **FLAGS)


def configure(directory=None, environ=None):
    """Set the cache variables in environ (default os.environ; inherited by child processes).

    Creates the two cache directories and returns metadata for the provenance:
    {"inductor": ..., "triton": ..., "shared": True when the cache root is used}.
    Triton and Inductor read the variables when they first need a cache, so call
    this before the first compilation.
    """
    environ = os.environ if environ is None else environ
    settings = cache_settings(directory, environ)
    for key in DIRECTORIES:
        Path(settings[key]).mkdir(parents=True, exist_ok=True)
    environ.update(settings)
    return dict(inductor=settings["TORCHINDUCTOR_CACHE_DIR"], triton=settings["TRITON_CACHE_DIR"],
                shared=directory is None)


def configure_defaults(environ=None):
    """The production default (SPEC §14): every compiler cache variable that ``environ`` does not set yet,
    set to its value under the cache root; a variable already set (by the caller, or a directory from
    ``configure``) wins. Creates nothing: Inductor and Triton create their directories when they first write.

    The planner calls this before a plan's first compilation (``FixedPlanner.configure_process``), so that
    Inductor does not fall back to ``$TMPDIR/torchinductor_$USER`` and Triton to ``~/.triton`` (OPT4-T05b).
    Importing ``torch._dynamo`` (MACE does, while loading) writes Inductor's own default into the environment
    (``torch/_dynamo/package.py``), so ``TORCHINDUCTOR_CACHE_DIR`` equal to that default counts as unset.
    Returns the names it set (empty when everything was set already).
    """
    environ = os.environ if environ is None else environ
    torch_default = _inductor_default()
    missing = [key for key in VARIABLES if key not in environ
               or (key == "TORCHINDUCTOR_CACHE_DIR" and environ[key] == torch_default)]
    if not missing:
        return ()
    settings = cache_settings(None, environ) if any(key in DIRECTORIES for key in missing) else FLAGS
    for key in missing:
        environ[key] = settings[key]
    return tuple(missing)


def _inductor_default():
    """Inductor's own cache directory (``$TMPDIR/torchinductor_$USER``), or None without Inductor."""
    try:
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
    except ImportError:  # pragma: no cover - a torch without Inductor
        return None
    return default_cache_dir()


def cache_stats():
    """Cache hits and misses counted by torch in this process so far."""
    from torch._dynamo.utils import counters
    return {f"{group}.{name}": count for group in ("inductor", "aot_autograd")
            for name, count in counters[group].items() if "cache" in name}
