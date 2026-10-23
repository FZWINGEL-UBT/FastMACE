"""ASE front end: the FastMACE calculator and ASE-compatible integrators (SPEC §7.4, §7.7)."""

from fastmace.ase.calculator import FastMACECalculator, calc

__all__ = ["FastMACECalculator", "calc"]

# The ASE-compatible integrators (SPEC §7.7 level 1, plan task 1b-T21).
from fastmace.ase.md import (  # noqa: E402
    MTKNPT,
    Bussi,
    Inhomogeneous_NPTBerendsen,
    IsotropicMTKNPT,
    Langevin,
    LangevinBAOAB,
    NoseHooverChainNVT,
    NPTBerendsen,
    NVTBerendsen,
    VelocityVerlet,
)

__all__ += ["Langevin", "LangevinBAOAB", "VelocityVerlet", "Bussi", "NoseHooverChainNVT", "IsotropicMTKNPT", "MTKNPT",
            "NVTBerendsen", "NPTBerendsen", "Inhomogeneous_NPTBerendsen"]


def __getattr__(name):
    """An ASE MD class without a mapping raises ``PlanRefused`` listing the mapped ones (SPEC §7.7)."""
    from fastmace.ase import md

    return md.unmapped(name)
