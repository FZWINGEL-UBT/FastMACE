"""PolarMACE model rewrites (SPEC §6.2 ``polar_uvu``; opt-4 OPT4-T02 ``polar_graph_sums``, decision 58).

``RECORDS`` maps each rewrite name to a plain dict with the fields of SPEC §6.1's ``Rewrite``
record (plus ``config_field``), for the rewrite registry of plan task 1a-T06 to register. The
GPU-only long-range path itself is not a rewrite; it is ``fastmace.longrange.polar``.
"""

from fastmace.rewrites.polar import graph_sums, uvu

RECORDS = {uvu.NAME: uvu.RECORD, graph_sums.NAME: graph_sums.RECORD}

__all__ = ["RECORDS", "graph_sums", "uvu"]
