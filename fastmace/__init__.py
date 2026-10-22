"""FastMACE: fast molecular dynamics with pretrained MACE models on NVIDIA GPUs."""

# First, before anything can import torch: PyTorch's file locks become soft locks and the CUDA
# driver cache is switched off on NFS (SPEC §14, fastmace/_file_locks.py).
from fastmace import _file_locks

_FILE_LOCK_SOURCE, _FILE_LOCK_SWITCH_LATE, _CUDA_DRIVER_CACHE = _file_locks.configure()

from dataclasses import dataclass  # noqa: E402  (after the lock switch above)

__version__ = "0.0.1.dev0"


@dataclass(frozen=True)
class RuntimeInfo:
    """Process-wide import-time settings, for provenance (SPEC §13)."""

    file_locks: str  # "soft" or "kernel": the class PyTorch's locks use now
    file_lock_source: str  # how that came about
    file_lock_switch_late: bool  # torch was imported before fastmace, so the locks stayed kernel locks
    cuda_driver_cache: str  # what happened to the CUDA driver's JIT cache at import


def runtime_info():
    """The file-lock mode and CUDA driver-cache status of this process (a frozen record)."""
    return RuntimeInfo(
        file_locks=_file_locks.mode(),
        file_lock_source=_FILE_LOCK_SOURCE,
        file_lock_switch_late=_FILE_LOCK_SWITCH_LATE,
        cuda_driver_cache=_CUDA_DRIVER_CACHE,
    )


# The public API of SPEC §7 (docs/design/contracts-1b.md), resolved at first use so that
# `import fastmace` imports neither torch nor MACE: name -> (module, attribute or None for the module).
_PUBLIC = {
    "load": ("fastmace.model.loading", "load_pristine"),
    "ForceField": ("fastmace.model.forcefield", "ForceField"),
    "Simulation": ("fastmace.simulation", "Simulation"),
    "Workload": ("fastmace.planner.workload", "Workload"),
    "Output": ("fastmace.io.output", "Output"),
    "integrators": ("fastmace.integrators", None),
    "protocols": ("fastmace.protocols", None),
    "md": ("fastmace.quick", "md"),
    "calc": ("fastmace.ase.calculator", "calc"),
    "D3": ("fastmace.longrange.d3", "D3"),
    "mc": ("fastmace.mc", None),  # phase 3 (docs/design/contracts-3.md)
}

__all__ = ["RuntimeInfo", "__version__", "runtime_info", *_PUBLIC]


def __getattr__(name):
    try:
        module_name, attribute = _PUBLIC[name]
    except KeyError:
        raise AttributeError(f"module 'fastmace' has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    return module if attribute is None else getattr(module, attribute)


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC))
