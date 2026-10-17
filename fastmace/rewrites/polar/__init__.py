"""PolarMACE model rewrites (SPEC §6.2 ``polar_uvu``).

``RECORDS`` maps each rewrite name to a plain dict with the fields of SPEC §6.1's ``Rewrite``
record (plus ``config_field``), for the rewrite registry of plan task 1a-T06 to register. The
GPU-only long-range path itself is not a rewrite; it is ``fastmace.longrange.polar``.
"""

from fastmace.rewrites.polar import uvu

RECORDS = {uvu.NAME: uvu.RECORD}

__all__ = ["RECORDS", "uvu"]
