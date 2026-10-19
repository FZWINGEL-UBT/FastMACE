"""ASE front end: the FastMACE calculator and ASE-compatible integrators (SPEC §7.4, §7.7)."""

from fastmace.ase.calculator import FastMACECalculator, calc

__all__ = ["FastMACECalculator", "calc"]
