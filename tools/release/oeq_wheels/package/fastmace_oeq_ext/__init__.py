"""Prebuilt OpenEquivariance 0.7.0 host extension (``libtorch_tp_jit``) for FastMACE.

FastMACE (``fastmace.kernels._oeq_runtime``) reads ``manifest.json`` next to this file without
importing the package and uses ``libtorch_tp_jit.so`` instead of building OEq's host extension with
g++ and ninja, when the manifest's torch and OpenEquivariance versions match the installed ones.
Nothing here runs at import.
"""

from pathlib import Path

LIBRARY = Path(__file__).with_name("libtorch_tp_jit.so")
MANIFEST = Path(__file__).with_name("manifest.json")
