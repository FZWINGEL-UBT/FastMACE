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

Nothing here runs at import time; ``configure`` sets the variables when an
engine asks for it.
"""

import os
import sys
from pathlib import Path

from fastmace._cache import CacheLocationError, cache_dir

VARIABLES = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE",
             "ENABLE_AOT_AUTOGRAD_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE")


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
    return dict(TORCHINDUCTOR_CACHE_DIR=str(inductor),
                TRITON_CACHE_DIR=str(triton),
                TORCHINDUCTOR_FX_GRAPH_CACHE="1",  # Inductor graphs (default on)
                ENABLE_AOT_AUTOGRAD_CACHE="1",     # compiled force derivative, torch 2.5
                TORCHINDUCTOR_AUTOGRAD_CACHE="1")  # the same, newer torch


def configure(directory=None, environ=None):
    """Set the cache variables in environ (default os.environ; inherited by child processes).

    Creates the two cache directories and returns metadata for the provenance:
    {"inductor": ..., "triton": ..., "shared": True when the cache root is used}.
    Triton and Inductor read the variables when they first need a cache, so call
    this before the first compilation.
    """
    environ = os.environ if environ is None else environ
    settings = cache_settings(directory, environ)
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        Path(settings[key]).mkdir(parents=True, exist_ok=True)
    environ.update(settings)
    return dict(inductor=settings["TORCHINDUCTOR_CACHE_DIR"], triton=settings["TRITON_CACHE_DIR"],
                shared=directory is None)


def cache_stats():
    """Cache hits and misses counted by torch in this process so far."""
    from torch._dynamo.utils import counters
    return {f"{group}.{name}": count for group in ("inductor", "aot_autograd")
            for name, count in counters[group].items() if "cache" in name}
