"""Interaction 1's reassociated projection composed with its receiver-species skip (Codex H200 follow-up, item 1;
7 November 2026; ``results/h200-compose-quads-20261107``).

In a density interaction the first interaction's message per degree l is ``left_l @ right_l``
(``FirstInteractionReassociation``), through the linear's ``L_l`` (a per-l ``BlockLinear``, or the identity after
the generic fold), divided by ``density + 1`` and multiplied by the receiver's species matrix ``B_l,z``
(``SpeciesSkip``). The division is per row, so the products compose:
``(left_l right_l L_l / (d + 1)) B_l,z = left_l (right_l L_l B_l,z) / (d + 1)``. ``composed`` builds
``C_l [species, K, mul_out]`` (in float64, stored in the input's dtype) and the reassociation runs
``species_mm(left_l, C_l)``: one per-species product per l instead of the projection, the linear and the skip's
all-species product (3x the columns, forward and backward). The interaction then skips its linear and ``skip_tp``.

Not bitwise: the composed matrices round differently from the two products (a rounding-class change: force/stress
gate). Lab, H200, forward + backward at OMAT-0 medium's shapes: 0.960 -> 0.448 ms (species_mm's CUTLASS launch).

``COMPOSE_SKIP``: None: on for ``COMPOSE_SKIP_GPUS``; True / False: an A/B side's choice. Set at build
(``precision.route``) where the interaction qualifies (``generic.modules.compose_skip_products``).
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels

COMPOSE_SKIP = None
# NVIDIA H200 (7 November 2026, results/h200-compose-quads-20261107): MD step b/a 0.9698 / 0.9738 against the H200
# defaults (OMAT-0 medium, SrTiO3 10x10x10, Benchmark_loup settings, ab / ba), force/stress gate passes (decision 66).
COMPOSE_SKIP_GPUS = GPUModels(("NVIDIA H200",))


# The fast path of ``COMPOSE_SKIP`` (h200-fastpath, bitwise): the composed weights once per species table
# (``FirstInteractionReassociation.bind``, refreshed with the table) instead of in every call, and the four
# products through one ``species_gemm.species_mm_joined`` (rows built outside the opaque product, written straight
# into the joined output, no ``torch.cat``), the division's result viewed rather than copied (``reshape_view``).
COMPOSE_FAST = None
# NVIDIA H200 (8 November 2026, results/h200-fastpath-20261108): MD step b/a 0.9614 / 0.9656 against the H200 defaults (OMAT-0 medium,
# SrTiO3 10x10x10, Benchmark_loup settings, ab / ba); forces within the build's run-to-run spread (decision 66).
COMPOSE_FAST_GPUS = GPUModels(("NVIDIA H200",))
# COUNT_PLAN (8 November 2026, round 3; bitwise): COMPOSE_SKIP's species plan by a counting sort (one-hot prefix
# sums) instead of an int64 radix sort per force call (8 onesweep passes and 10 memsets, ~57 us on the H200). The
# order is the same stable order. None: on for COUNT_PLAN_GPUS; True / False: an A/B side's choice.
COUNT_PLAN = None
# NVIDIA H200 (8 November 2026, results/h200-fastpath-20261108, round 3): MD step b/a 0.9831 / 0.9818 against the
# H200 defaults of round 2 (ab / ba); bitwise (the same order), faster in both orders (decision 66).
COUNT_PLAN_GPUS = GPUModels(("NVIDIA H200",))
# JOINED_LAYOUT (Codex H200 round-6 report, 8 November 2026; bitwise): with COMPOSE_FAST the tabled outer sum writes
# the per-slot sums d-major (its ``inner`` layout with the slots of a receiver as one group), so each path's input is
# a row group of one [atoms * 16, slots * K] matrix: ``species_gemm.species_mm_rows`` reads it in place (no per-path
# transpose copy) and its gradient comes back in the same layout for the outer sum's backward (no repack). The same
# products and sums. None: on for JOINED_LAYOUT_GPUS; True / False: an A/B side's choice.
JOINED_LAYOUT = None
JOINED_LAYOUT_GPUS = GPUModels(())


def compose_fast_default(name=None) -> bool:
    """``COMPOSE_FAST`` (None: ``name`` in ``COMPOSE_FAST_GPUS``)."""
    on = COMPOSE_FAST
    if on is None:
        on = name in COMPOSE_FAST_GPUS
    return bool(on)


def joined_layout_default(name=None) -> bool:
    """``JOINED_LAYOUT`` (None: ``name`` in ``JOINED_LAYOUT_GPUS``)."""
    on = JOINED_LAYOUT
    if on is None:
        on = name in JOINED_LAYOUT_GPUS
    return bool(on)


def count_plan_default(name=None) -> bool:
    """``COUNT_PLAN`` (None: ``name`` in ``COUNT_PLAN_GPUS``)."""
    on = COUNT_PLAN
    if on is None:
        on = name in COUNT_PLAN_GPUS
    return bool(on)


def compose_skip_default(name=None) -> bool:
    """``COMPOSE_SKIP`` (None: ``name`` in ``COMPOSE_SKIP_GPUS``)."""
    on = COMPOSE_SKIP
    if on is None:
        on = name in COMPOSE_SKIP_GPUS
    return bool(on)


def composed(right, ws, linear=None):
    """``right [K, c] @ linear [c, mul_in] @ ws[z] [mul_in, mul_out]`` for every species z (``linear`` None: the
    identity), as ``(C [species, K, mul_out], C transposed [species, mul_out, K])``, both contiguous, computed in
    float64 and returned in ``right``'s dtype."""
    left = right.double() if linear is None else right.double() @ linear.double()
    c = torch.matmul(left[None], ws.double()).to(right.dtype)
    return c.contiguous(), c.transpose(1, 2).contiguous()


__all__ = ["COMPOSE_FAST", "COMPOSE_FAST_GPUS", "COMPOSE_SKIP", "COMPOSE_SKIP_GPUS", "COUNT_PLAN", "COUNT_PLAN_GPUS", "JOINED_LAYOUT",
           "JOINED_LAYOUT_GPUS", "joined_layout_default",
           "compose_fast_default", "compose_skip_default", "composed", "count_plan_default"]
