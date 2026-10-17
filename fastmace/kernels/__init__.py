"""Triton and vendored OEq streaming kernels used by the rewrites (SPEC §5).

Subpackages: ``triton`` (Triton kernels and the compiler cache settings) and
``oeq_streaming`` (OpenEquivariance PR 223 receiver-streaming kernels, BSD 3-Clause,
compiled with NVRTC). ``_oeq_runtime`` prepares OEq's host extension (plan task 1a-T05).

The probes below say whether a kernel family can run here, with a reason string for
the planner and ``fastmace doctor``. They are functions, evaluated when called;
importing this package imports neither triton, ``cuda.bindings`` nor OpenEquivariance,
and sets nothing.
"""

import importlib
import importlib.util
from dataclasses import dataclass

__all__ = ["Availability", "nvrtc_available", "streaming_available", "triton_available"]


@dataclass(frozen=True)
class Availability:
    """Result of a probe: ``available`` and a one-line ``reason`` (why, or why not)."""

    available: bool
    reason: str

    def __bool__(self):
        return self.available


def _failure(name, error):
    return f"{name} is not importable ({type(error).__name__}: {error})"


def triton_available():
    """Whether triton imports (the Triton kernels also need a CUDA tensor to run on)."""
    try:
        triton = importlib.import_module("triton")
    except Exception as error:  # ImportError, or a broken installation raising anything else
        return Availability(False, _failure("triton", error))
    version = getattr(triton, "__version__", "unknown version")
    return Availability(True, f"triton {version} is importable")


def nvrtc_available():
    """Whether NVRTC loads through ``cuda.bindings`` (the streaming kernels compile with it)."""
    try:
        nvrtc = importlib.import_module("cuda.bindings.nvrtc")
    except Exception as error:
        return Availability(False, _failure("cuda.bindings.nvrtc", error))
    try:
        result = nvrtc.nvrtcVersion()
    except Exception as error:  # the bindings import, but libnvrtc does not load
        return Availability(False, f"cuda.bindings imports but NVRTC does not load ({type(error).__name__}: "
                                   f"{error})")
    if int(result[0]) != 0:
        return Availability(False, f"cuda.bindings imports but nvrtcVersion failed ({result[0]})")
    return Availability(True, f"NVRTC {result[1]}.{result[2]} loads through cuda.bindings")


def streaming_available():
    """Whether the receiver-streaming kernels can be built: NVRTC, Jinja2 and OpenEquivariance.

    OpenEquivariance is only looked up, not imported: importing it builds its host
    extension, which must be prepared first (``_oeq_runtime``).
    """
    nvrtc = nvrtc_available()
    if not nvrtc:
        return nvrtc
    for name in ("jinja2", "openequivariance"):
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError) as error:
            return Availability(False, _failure(name, error))
        if not found:
            return Availability(False, f"{name} is not installed (the streaming kernels need it)")
    return Availability(True, f"{nvrtc.reason}; jinja2 and openequivariance are installed")
