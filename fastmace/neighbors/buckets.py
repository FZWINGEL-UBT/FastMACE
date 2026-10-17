"""Padded edge capacities and edge buckets (SPEC §8.1 "Edges").

The neighbour list's buffers hold ``capacity`` edges, a multiple of
``NeighbourConfig.edge_multiple`` with ``capacity_headroom`` above the first build, so that
their addresses stay fixed between rebuilds unless the candidates outgrow them. The model reads
a padded prefix of the compacted edges whose length is the edge count rounded up to a bucket.
Each padded size costs one CUDA graph (and one compiled variant), so coarser buckets bound
memory and recompilation; the padding costs at most one bucket of extra work. The bucket is
about ``edge_bucket_fraction`` of the initial edge count.

All arguments are explicit: the callers pass the fields of their ``NeighbourConfig``.
"""

from __future__ import annotations

import math


def edge_bucket(reference_count: int, fraction: float, multiple: int) -> int:
    """Step between padded edge counts: ``fraction * reference_count``, a multiple of ``multiple``.

    ``fraction=0`` gives ``multiple``, the fixed step used before 7 October 2026.
    """
    return multiple * max(1, math.ceil(fraction * reference_count / multiple))


def round_up_edges(count: int, capacity: int, bucket: int) -> int:
    """Padded edge size for ``count`` edges: the next multiple of ``bucket`` (at least one), capped at ``capacity``."""
    return min(capacity, bucket * max(1, math.ceil(count / bucket)))


def edge_capacity(needed: int, headroom: float, multiple: int) -> int:
    """Buffer size for ``needed`` candidate pairs: ``needed * headroom`` rounded up to ``multiple`` (at least one)."""
    return multiple * max(1, math.ceil(int(needed * headroom) / multiple))
